"""
Multi-Modal Pathway-GAT + Histology (Phikon) Cross-Attention Pipeline
==========================================================================
Combines two previously-separate approaches into one model:

  RNA branch:   Raw genes -> pathway/gene-set scores (via sc.tl.score_genes,
                using Hallmark + C8 + C7 + C6 MSigDB gene sets) -> dynamic-
                depth GATv2 encoder (BatchNorm + ELU) -> RNA embedding
  Image branch: Precomputed Phikon histology embeddings (768-dim, extracted
                separately via extract_histology_embeddings.py) -> linear
                projection -> Image embedding
  Fusion:       Single-direction cross-attention (RNA query attends over
                Image key/value), residual + layernorm -> Fused embedding
  Prediction:   Fused embedding -> MLP -> 44 protein values

Trains on the buffered spatial train/val split (from generate_split.py),
with genuine held-out validation, early stopping, and a final accuracy
report comparing predicted vs actual protein values on held-out bins.

LEAKAGE NOTE: pathway/gene-set SELECTION (which gene sets pass the
variance + correlation-with-protein filters) is fit using ONLY the train
split, then the same selected gene sets + fitted StandardScaler are
applied identically to the val split. Selecting gene sets using a
correlation computed against val-split protein values would leak val
information into feature selection itself, before training even begins -
a different, easy-to-miss leakage channel from the spatial one
generate_split.py already guards against.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------

    python pathway_gat_crossattn_pipeline.py train \
        --train_rna data/rna_train_split.pkl \
        --train_pro data/pro_train_split.pkl \
        --val_rna data/rna_val_split.pkl \
        --val_pro data/pro_val_split.pkl \
        --image_embeddings outputs_histology/embeddings.npy \
        --image_index outputs_histology/embedding_index.csv \
        --gmt_hallmark data/hallmark_pathways.gmt \
        --gmt_c8 data/c8_pathways.gmt \
        --gmt_c7 data/c7_pathways.gmt \
        --gmt_c6 data/c6_pathways.gmt \
        --output_dir outputs_pathway_gat_crossattn \
        --min_genes 10 --min_corr 0.05 \
        --gat_layers 2 --gat_hidden 256 --gat_heads 4 \
        --num_neighbors 15 10 --batch_size 512 \
        --n_epochs 50 --patience 10 --eval_every 1

Requires torch_geometric with torch_sparse/pyg_lib installed (for
NeighborLoader), same as the other GAT scripts in this project.
"""

import argparse
import gc
import os
import pickle
import time

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from joblib import Parallel, delayed

from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader


# ============================================================================
# Shared utilities
# ============================================================================

def log(msg, log_file=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(line + "\n")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def resource_status_str(device):
    parts = []
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        parts.append(f"GPU: {alloc:.1f}/{total:.1f}GB")
    try:
        import psutil
        ram = psutil.virtual_memory()
        parts.append(f"RAM: {ram.used/(1024**3):.1f}/{ram.total/(1024**3):.1f}GB ({ram.percent:.0f}%)")
    except ImportError:
        pass
    return " | ".join(parts) if parts else ""


def load_adata(path):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    return sc.read_h5ad(path)


def build_spatial_knn_graph(coords, k):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    n_nodes = coords.shape[0]
    src = np.repeat(np.arange(n_nodes), k)
    dst = indices[:, 1:].flatten()
    edge_index = np.stack([src, dst], axis=0)
    edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    return torch.tensor(edge_index, dtype=torch.long)


def align_image_embeddings(obs_names, img_embeddings, img_index, log_file=None):
    img_df = pd.DataFrame(img_embeddings, index=img_index["barcode"].values)
    img_df = img_df.reindex(obs_names)
    valid_mask = ~img_df.isna().any(axis=1)
    n_missing = (~valid_mask).sum()
    if n_missing > 0:
        log(f"  WARNING: {n_missing} bins have no matching image embedding - excluding them", log_file)
    return img_df[valid_mask].values.astype(np.float32), valid_mask.values


# ============================================================================
# Pathway scoring (train-fit, leak-safe)
# ============================================================================

def parse_gmt(filepath):
    pathways = {}
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            pathways[parts[0]] = parts[2:]
    return pathways


def match_gene_sets(gmt_paths, rna_gene_set, min_genes):
    """Loads and combines gene sets from multiple GMT files, keeping only
    sets with at least min_genes genes actually present in the RNA panel.
    This step doesn't depend on bin-level values, so it's the same
    regardless of train/val - only the correlation-based SELECTION later
    needs to be train-only."""
    combined_raw = {}
    for path in gmt_paths:
        if path:
            combined_raw.update(parse_gmt(path))
    matched_sets = {}
    for name, genes in combined_raw.items():
        matched = [g for g in genes if g in rna_gene_set]
        if len(matched) >= min_genes:
            matched_sets[name] = matched
    return matched_sets


def _score_one_pathway(X, gene_positions, control_positions):
    """Scores a single gene set: pathway mean minus control mean, across
    all bins. Kept as a standalone top-level function (not a closure) so
    joblib's 'loky' backend can pickle a reference to it for worker
    processes."""
    pathway_mean = X[:, gene_positions].mean(axis=1)
    if control_positions:
        control_mean = X[:, control_positions].mean(axis=1)
    else:
        control_mean = np.zeros(X.shape[0], dtype=np.float32)
    return (pathway_mean - control_mean).astype(np.float32)


def compute_pathway_scores(rna_adata, matched_sets, seed, ctrl_size=50, n_bins=25,
                            device=None, chunk_size=20000, tmp_dir=None, log_file=None):
    """Computes per-bin pathway activity scores: pathway average minus a
    matched control gene set's average (same statistical idea as
    sc.tl.score_genes - corrects for bins having different total RNA
    content, which would otherwise inflate every gene set's score in
    high-count bins regardless of real pathway activity).

    KEY REDESIGN: instead of computing each gene set's score with a
    separate reduction operation (looped, or parallelized across
    processes with joblib - both of which carry real per-task overhead),
    this reformulates ALL gene sets as ONE matrix multiplication:

        score_matrix = X @ M

    where M is a small (n_genes x n_sets) "membership matrix" encoding,
    per gene set, which genes belong to it (weighted 1/|set|) minus which
    genes are in its control set (weighted -1/|control_size|). A single
    matmul computes every gene set's pathway-minus-control score for
    every bin simultaneously. This is exactly the kind of operation GPUs
    (and CPU BLAS libraries) are built to do efficiently - far faster
    than either a per-call sc.tl.score_genes loop or process-based
    parallelization, with no multiprocessing/worker-pool overhead at all.

    Runs on GPU if available (falls back to CPU otherwise), processing
    bins in chunks to keep memory bounded regardless of GPU size.
    """
    if device is None:
        device = get_device()

    X = rna_adata.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    X = X.astype(np.float32, copy=False)

    gene_names = rna_adata.var_names.tolist()
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)

    log(f"  Binning {n_genes} genes by expression level (once, reused for all gene sets)...",
        log_file)
    avg_expr = pd.Series(X.mean(axis=0), index=gene_names)
    n_items = max(1, int(np.round(len(avg_expr) / n_bins)))
    expr_bin = (avg_expr.rank(method="min") // n_items).astype(int)

    bin_to_gene_positions = {}
    for gene, b in expr_bin.items():
        bin_to_gene_positions.setdefault(b, []).append(gene_idx[gene])

    rng = np.random.default_rng(seed)
    all_names = list(matched_sets.keys())
    n_sets = len(all_names)

    # ------------------------------------------------------------------
    # Build the (n_genes x n_sets) membership matrix M in one pass.
    # Column j: +1/|pathway genes| at pathway gene rows,
    #           -1/|control genes| at control gene rows.
    # Then X @ M gives every gene set's (pathway_mean - control_mean)
    # score for every bin, in a single matrix multiply.
    # ------------------------------------------------------------------
    log(f"  Building membership matrix for {n_sets} gene sets...", log_file)
    M = np.zeros((n_genes, n_sets), dtype=np.float32)
    for j, name in enumerate(all_names):
        genes = matched_sets[name]
        gene_positions = [gene_idx[g] for g in genes if g in gene_idx]
        if gene_positions:
            M[gene_positions, j] += 1.0 / len(gene_positions)

        control_positions = set()
        for g in genes:
            if g not in gene_idx:
                continue
            b = expr_bin[g]
            bin_genes = bin_to_gene_positions[b]
            k = min(ctrl_size, len(bin_genes))
            sampled = rng.choice(bin_genes, size=k, replace=False)
            control_positions.update(sampled.tolist())
        if control_positions:
            cp = list(control_positions)
            M[cp, j] -= 1.0 / len(cp)

    log(f"  Computing scores via matrix multiplication on {device} "
        f"(X: {X.shape}, M: {M.shape})...", log_file)
    t_start = time.time()

    M_t = torch.tensor(M, dtype=torch.float32, device=device)
    n_obs = X.shape[0]
    score_matrix = np.zeros((n_obs, n_sets), dtype=np.float32)

    # Process bins in chunks so memory usage stays bounded regardless of
    # dataset size or GPU VRAM - each chunk's X slice + M together are
    # small enough to comfortably fit even on a modest GPU.
    for start in range(0, n_obs, chunk_size):
        end = min(start + chunk_size, n_obs)
        X_chunk = torch.tensor(X[start:end], dtype=torch.float32, device=device)
        chunk_scores = X_chunk @ M_t
        score_matrix[start:end] = chunk_scores.cpu().numpy()
        del X_chunk, chunk_scores

    if device.type == "cuda":
        torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    log(f"  Done in {elapsed:.1f}s ({elapsed/n_sets*1000:.2f}ms/gene set average)", log_file)

    return score_matrix, all_names


def select_informative_pathways(score_matrix_train, Y_train, min_r, var_percentile=25):
    """Selects gene sets by variance (top 75%) AND correlation with at
    least one target protein (|r| >= min_r). FIT ON TRAIN DATA ONLY -
    selecting based on correlation with val-split protein values would
    leak val information into feature selection itself."""
    set_stds = score_matrix_train.std(axis=0)
    var_threshold = np.percentile(set_stds, var_percentile)
    var_mask = set_stds > var_threshold

    n_sets = score_matrix_train.shape[1]
    corr_mask = np.zeros(n_sets, dtype=bool)
    for j in range(n_sets):
        if not var_mask[j]:
            continue
        path_vals = score_matrix_train[:, j]
        if path_vals.std() < 1e-9:
            continue
        for i in range(Y_train.shape[1]):
            pro_vals = Y_train[:, i]
            if pro_vals.std() < 1e-9:
                continue
            r, _ = pearsonr(path_vals, pro_vals)
            if abs(r) >= min_r:
                corr_mask[j] = True
                break

    final_mask = var_mask & corr_mask
    return np.where(final_mask)[0]


def fit_protein_transform(Y_train, cofactor):
    X_arcsinh = np.arcsinh(Y_train / cofactor)
    return X_arcsinh.mean(axis=0), X_arcsinh.std(axis=0)


def apply_protein_transform(Y, cofactor, mean, std, clip_min, clip_max):
    X_arcsinh = np.arcsinh(Y / cofactor)
    X_scaled = (X_arcsinh - mean) / std
    return np.clip(X_scaled, clip_min, clip_max).astype(np.float32)


# ============================================================================
# Model architecture
# ============================================================================

def build_gated_adaptive_edges(coords_px, X_features, k, mpp, sigma_um=15.0,
                                max_distance_um=None, gate_threshold=0.0,
                                chunk_size=200_000, log_file=None):
    """Builds a spatial k-NN graph with two upgrades over a plain fixed-k
    graph:

    1. ADAPTIVE DENSITY (KNN + max distance cap): finds each bin's k
       nearest neighbours by real physical distance (converted from
       pixels to microns via mpp), then drops any neighbour farther than
       max_distance_um. In dense tissue, the k nearest neighbours are all
       close and all survive; in sparse tissue, distant "neighbours" that
       aren't really spatially meaningful get dropped - so message volume
       adapts to local tissue density instead of forcing a fixed count of
       connections everywhere.

    2. TRANSCRIPTOMIC-GATED EDGE WEIGHTS: each surviving edge gets a
       weight combining (a) a physical-distance Gaussian kernel (closer
       = higher weight) and (b) the cosine similarity between the two
       bins' transcriptomic feature vectors (here, their pathway-score
       profiles - the same representation the GAT actually consumes,
       rather than the full 18k-gene space, which would be prohibitively
       expensive to compute pairwise similarity over at this edge count).
       Two physically-adjacent bins with very different transcriptomic
       profiles (e.g. tumour vs immune) get a LOW combined weight even
       though they're spatially close - this weight is passed into the
       GAT as edge_attr, letting the attention mechanism learn to trust
       physically-close-but-transcriptomically-different edges less,
       rather than blurring across what may be a real biological
       boundary. Edges below gate_threshold are removed entirely (hard
       gating); default threshold of 0 means only the soft signal is
       used unless you explicitly raise it.

    Returns (edge_index, edge_weight) - edge_weight has shape (E,) and
    is meant to be reshaped to (E, 1) as edge_attr for GATv2Conv's
    edge_dim.
    """
    from sklearn.neighbors import NearestNeighbors

    coords_um = coords_px.astype(np.float64) * mpp
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords_um)
    dists, indices = nbrs.kneighbors(coords_um)

    n_nodes = coords_um.shape[0]
    src = np.repeat(np.arange(n_nodes), k)
    dst = indices[:, 1:].flatten()
    dist_um = dists[:, 1:].flatten()

    if max_distance_um is not None:
        density_mask = dist_um <= max_distance_um
    else:
        density_mask = np.ones_like(dist_um, dtype=bool)

    n_before_density = len(src)
    src, dst, dist_um = src[density_mask], dst[density_mask], dist_um[density_mask]
    if log_file is not None or True:
        log(f"  Adaptive density filter: {n_before_density:,} -> {len(src):,} candidate edges "
            f"(max_distance_um={max_distance_um})", log_file)

    w_phys = np.exp(-(dist_um ** 2) / (2 * sigma_um ** 2)).astype(np.float32)

    # Normalize feature rows once, so cosine similarity per edge is just
    # a dot product of already-unit-length vectors.
    norms = np.linalg.norm(X_features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = (X_features / norms).astype(np.float32)

    cos_sim = np.zeros(len(src), dtype=np.float32)
    for start in range(0, len(src), chunk_size):
        end = min(start + chunk_size, len(src))
        cos_sim[start:end] = np.sum(X_norm[src[start:end]] * X_norm[dst[start:end]], axis=1)

    # Negative cosine similarity (anti-correlated profiles) is clipped to
    # 0 rather than allowed to flip the sign of the combined weight -
    # we want "how similar" to gate the edge, not to reward opposite-
    # direction profiles as if they were meaningfully connected.
    combined_weight = w_phys * np.clip(cos_sim, 0, None)

    keep_mask = combined_weight >= gate_threshold
    n_before_gate = len(src)
    src, dst, combined_weight = src[keep_mask], dst[keep_mask], combined_weight[keep_mask]
    log(f"  Transcriptomic gating: {n_before_gate:,} -> {len(src):,} edges survive "
        f"(gate_threshold={gate_threshold}, mean edge weight={combined_weight.mean():.3f})",
        log_file)

    # Make undirected: mirror every surviving edge
    edge_index = np.stack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src]),
    ], axis=0)
    edge_weight = np.concatenate([combined_weight, combined_weight])

    return (torch.tensor(edge_index, dtype=torch.long),
            torch.tensor(edge_weight, dtype=torch.float32))


class PathwayGATEncoder(nn.Module):
    """Dynamic-depth GATv2 encoder over pathway-score features (not raw
    genes), with BatchNorm + ELU between layers. Outputs a fusion_dim-sized
    embedding (not final protein predictions directly - that's the cross-
    attention model's job), unlike the original single-modality version
    this is adapted from.

    edge_dim=1 lets each GATv2Conv layer incorporate the combined
    physical+transcriptomic edge weight (from build_gated_adaptive_edges)
    into its attention computation - the network can learn how much to
    trust an edge's prior weight, rather than treating all spatial
    neighbours identically regardless of transcriptomic similarity."""
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, heads, dropout):
        super().__init__()
        assert n_layers >= 1
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        if n_layers == 1:
            self.convs.append(GATv2Conv(in_channels, out_channels, heads=1,
                                          dropout=dropout, concat=False, edge_dim=1))
        else:
            self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads,
                                          dropout=dropout, concat=True, edge_dim=1))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
            for _ in range(n_layers - 2):
                self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels,
                                              heads=heads, dropout=dropout, concat=True, edge_dim=1))
                self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
            self.convs.append(GATv2Conv(hidden_channels * heads, out_channels, heads=1,
                                          dropout=dropout, concat=False, edge_dim=1))
            self.bns.append(nn.BatchNorm1d(out_channels))

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index, edge_attr=edge_attr)
            if i < len(self.bns):
                x = self.bns[i](x)
                x = F.elu(x)
        return x


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads,
                                                  dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, rna_emb, img_emb):
        q = rna_emb.unsqueeze(1)
        kv = img_emb.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        return self.norm(attn_out.squeeze(1) + rna_emb)


class PathwayGATCrossAttnPredictor(nn.Module):
    def __init__(self, n_pathways, img_dim, gat_hidden, gat_layers, gat_heads,
                 fusion_dim, cross_attn_heads, n_proteins, dropout):
        super().__init__()
        self.rna_encoder = PathwayGATEncoder(n_pathways, gat_hidden, fusion_dim,
                                              gat_layers, gat_heads, dropout)
        self.img_projection = nn.Sequential(
            nn.Linear(img_dim, fusion_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.fusion = CrossAttentionFusion(fusion_dim, cross_attn_heads, dropout)
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_proteins)
        )

    def forward(self, x_rna, edge_index, edge_attr, x_img):
        rna_emb = self.rna_encoder(x_rna, edge_index, edge_attr)
        img_emb = self.img_projection(x_img)
        fused = self.fusion(rna_emb, img_emb)
        return self.predictor(fused)


def build_model_from_checkpoint(checkpoint, device):
    a = checkpoint["args"]
    model = PathwayGATCrossAttnPredictor(
        n_pathways=checkpoint["n_pathways"], img_dim=checkpoint["img_dim"],
        gat_hidden=a["gat_hidden"], gat_layers=a["gat_layers"], gat_heads=a["gat_heads"],
        fusion_dim=a["fusion_dim"], cross_attn_heads=a["cross_attn_heads"],
        n_proteins=checkpoint["n_proteins"], dropout=a["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def mini_batch_inference(model, graph_data, num_neighbors, batch_size, device):
    loader = NeighborLoader(graph_data, num_neighbors=num_neighbors,
                             batch_size=batch_size, shuffle=False)
    n_nodes = graph_data.num_nodes
    n_proteins = graph_data.y.shape[1]
    all_preds = np.zeros((n_nodes, n_proteins), dtype=np.float32)
    filled = np.zeros(n_nodes, dtype=bool)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            seed_idx = batch.n_id[:batch.batch_size].cpu().numpy()
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.img_x)
            all_preds[seed_idx] = pred[:batch.batch_size].cpu().numpy()
            filled[seed_idx] = True
    if not filled.all():
        raise RuntimeError(f"{(~filled).sum()} nodes never covered during inference")
    return all_preds


# ============================================================================
# TRAIN
# ============================================================================

def run_train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "training_log.txt")
    open(log_file, "w").close()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    log(f"[TRAIN] Using device: {device}", log_file)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    log(f"Loading train RNA/protein", log_file)
    train_rna = load_adata(args.train_rna)
    train_pro = load_adata(args.train_pro)
    if not (train_rna.obs_names == train_pro.obs_names).all():
        train_pro = train_pro[train_rna.obs_names].copy()
    protein_names = train_pro.var_names.tolist()
    log(f"Train bins: {train_rna.n_obs}", log_file)

    has_val = bool(args.val_rna and args.val_pro)
    if has_val:
        log(f"Loading val RNA/protein", log_file)
        val_rna = load_adata(args.val_rna)
        val_pro = load_adata(args.val_pro)
        if not (val_rna.obs_names == val_pro.obs_names).all():
            val_pro = val_pro[val_rna.obs_names].copy()
        log(f"Val bins: {val_rna.n_obs}", log_file)

    img_embeddings = np.load(args.image_embeddings)
    img_index = pd.read_csv(args.image_index)

    # ------------------------------------------------------------------
    # RNA preprocessing (normalize_total + log1p) - per-bin, no leakage
    # ------------------------------------------------------------------
    log("Preprocessing RNA (normalize_total + log1p)", log_file)
    sc.pp.normalize_total(train_rna, target_sum=1e4)
    sc.pp.log1p(train_rna)
    if has_val:
        sc.pp.normalize_total(val_rna, target_sum=1e4)
        sc.pp.log1p(val_rna)

    # ------------------------------------------------------------------
    # Pathway gene sets: matching is data-independent (just gene name
    # overlap), but SELECTION (variance + correlation filter) must be
    # train-only to avoid leaking val-split protein correlations into
    # feature selection.
    # ------------------------------------------------------------------
    log("Matching pathway gene sets against RNA panel", log_file)
    rna_gene_set = set(train_rna.var_names)
    gmt_paths = [args.gmt_hallmark, args.gmt_c8, args.gmt_c7, args.gmt_c6]
    matched_sets = match_gene_sets(gmt_paths, rna_gene_set, args.min_genes)
    log(f"Matched gene sets (>= {args.min_genes} genes present): {len(matched_sets)}", log_file)

    log("Computing pathway scores for train split", log_file)
    train_score_matrix, all_names = compute_pathway_scores(
        train_rna, matched_sets, args.seed, ctrl_size=args.ctrl_size, n_bins=args.score_n_bins,
        device=device, chunk_size=args.score_chunk_size, tmp_dir=args.output_dir, log_file=log_file
    )

    Y_train_raw = train_pro.X
    Y_train_raw = np.asarray(Y_train_raw.todense()) if hasattr(Y_train_raw, "todense") else np.asarray(Y_train_raw)

    log(f"Selecting informative pathways (TRAIN ONLY - min_corr={args.min_corr})", log_file)
    kept_idx = select_informative_pathways(train_score_matrix, Y_train_raw, args.min_corr)
    names_final = [all_names[i] for i in kept_idx]
    log(f"Pathways after selection: {len(kept_idx)} / {len(all_names)}", log_file)

    train_score_final = train_score_matrix[:, kept_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_score_final).astype(np.float32)

    if has_val:
        log("Computing pathway scores for val split (same gene sets as train)", log_file)
        matched_sets_final = {name: matched_sets[name] for name in names_final}
        val_score_matrix, val_names_order = compute_pathway_scores(
            val_rna, matched_sets_final, args.seed, ctrl_size=args.ctrl_size, n_bins=args.score_n_bins,
            device=device, chunk_size=args.score_chunk_size, tmp_dir=args.output_dir, log_file=log_file
        )
        # val_names_order matches names_final order since matched_sets_final
        # preserves insertion order and compute_pathway_scores iterates dict order
        X_val_scaled = scaler.transform(val_score_matrix).astype(np.float32)

    # ------------------------------------------------------------------
    # Protein transform - fit on train only, applied to both
    # ------------------------------------------------------------------
    log(f"Fitting protein transform (arcsinh cofactor={args.arcsinh_cofactor}) on TRAIN ONLY", log_file)
    marker_mean, marker_std = fit_protein_transform(Y_train_raw, args.arcsinh_cofactor)
    Y_train = apply_protein_transform(Y_train_raw, args.arcsinh_cofactor, marker_mean, marker_std,
                                       args.protein_clip_min, args.protein_clip_max)
    if has_val:
        Y_val_raw = val_pro.X
        Y_val_raw = np.asarray(Y_val_raw.todense()) if hasattr(Y_val_raw, "todense") else np.asarray(Y_val_raw)
        Y_val = apply_protein_transform(Y_val_raw, args.arcsinh_cofactor, marker_mean, marker_std,
                                         args.protein_clip_min, args.protein_clip_max)

    # ------------------------------------------------------------------
    # Build graphs (separate train/val - no cross-split edges)
    # ------------------------------------------------------------------
    log(f"Building train graph (k={args.k_neighbors}, gated edges)", log_file)
    train_img_aligned, train_valid_mask = align_image_embeddings(
        train_rna.obs_names, img_embeddings, img_index, log_file)
    if not train_valid_mask.all():
        X_train_scaled = X_train_scaled[train_valid_mask]
        Y_train = Y_train[train_valid_mask]
        train_rna = train_rna[train_valid_mask].copy()

    train_coords = train_rna.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values
    train_edge_index, train_edge_weight = build_gated_adaptive_edges(
        train_coords, X_train_scaled, args.k_neighbors, args.mpp,
        sigma_um=args.sigma_um, max_distance_um=args.max_edge_distance_um,
        gate_threshold=args.gate_threshold, log_file=log_file
    )
    train_graph = Data(
        x=torch.from_numpy(X_train_scaled), edge_index=train_edge_index,
        edge_attr=train_edge_weight.unsqueeze(-1), y=torch.from_numpy(Y_train)
    )
    train_graph.img_x = torch.from_numpy(train_img_aligned)
    log(f"Train graph: {train_graph.num_nodes} nodes, {train_graph.edge_index.shape[1]} edges", log_file)
    del train_rna, train_pro
    gc.collect()

    if has_val:
        log(f"Building val graph (k={args.k_neighbors}, gated edges) - separate from train graph",
            log_file)
        val_img_aligned, val_valid_mask = align_image_embeddings(
            val_rna.obs_names, img_embeddings, img_index, log_file)
        if not val_valid_mask.all():
            X_val_scaled = X_val_scaled[val_valid_mask]
            Y_val = Y_val[val_valid_mask]
            val_rna = val_rna[val_valid_mask].copy()

        val_coords = val_rna.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values
        val_edge_index, val_edge_weight = build_gated_adaptive_edges(
            val_coords, X_val_scaled, args.k_neighbors, args.mpp,
            sigma_um=args.sigma_um, max_distance_um=args.max_edge_distance_um,
            gate_threshold=args.gate_threshold, log_file=log_file
        )
        val_graph = Data(
            x=torch.from_numpy(X_val_scaled), edge_index=val_edge_index,
            edge_attr=val_edge_weight.unsqueeze(-1), y=torch.from_numpy(Y_val)
        )
        val_graph.img_x = torch.from_numpy(val_img_aligned)
        log(f"Val graph: {val_graph.num_nodes} nodes, {val_graph.edge_index.shape[1]} edges", log_file)
        del val_rna, val_pro
        gc.collect()

    # ------------------------------------------------------------------
    # Model + training
    # ------------------------------------------------------------------
    train_loader = NeighborLoader(train_graph, num_neighbors=args.num_neighbors,
                                    batch_size=args.batch_size, shuffle=True)
    n_batches = (train_graph.num_nodes + args.batch_size - 1) // args.batch_size
    log(f"NeighborLoader: num_neighbors={args.num_neighbors}, batch_size={args.batch_size}, "
        f"~{n_batches} batches/epoch", log_file)

    n_pathways = X_train_scaled.shape[1]
    n_proteins = Y_train.shape[1]

    model = PathwayGATCrossAttnPredictor(
        n_pathways=n_pathways, img_dim=img_embeddings.shape[1],
        gat_hidden=args.gat_hidden, gat_layers=args.gat_layers, gat_heads=args.gat_heads,
        fusion_dim=args.fusion_dim, cross_attn_heads=args.cross_attn_heads,
        n_proteins=n_proteins, dropout=args.dropout,
    ).to(device)
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", log_file)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # ------------------------------------------------------------------
    # Boosted per-protein loss weighting: errors on --boost_proteins count
    # boost_factor times as much toward the training loss as same-sized
    # errors on other proteins. Matching is case-insensitive since
    # protein name casing can vary between the requested list and the
    # actual column names (e.g. "Synd" vs "synd").
    # ------------------------------------------------------------------
    protein_weights = torch.ones(n_proteins, dtype=torch.float32)
    if args.boost_proteins:
        name_to_idx = {name.upper(): i for i, name in enumerate(protein_names)}
        boosted_idx = []
        for bp in args.boost_proteins:
            idx = name_to_idx.get(bp.upper())
            if idx is not None:
                boosted_idx.append(idx)
            else:
                log(f"  WARNING: boosted protein '{bp}' not found in protein_names - skipping",
                    log_file)
        protein_weights[boosted_idx] = args.boost_factor
        log(f"Boosting {len(boosted_idx)} proteins by {args.boost_factor}x in the training loss: "
            f"{[protein_names[i] for i in boosted_idx]}", log_file)
    protein_weights = protein_weights.to(device)

    def weighted_mse(pred, target):
        return (protein_weights * (pred - target) ** 2).mean()

    loss_fn = weighted_mse

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None
    epochs_without_improvement = 0
    stopped_early = False
    avg_loss = None

    def save_checkpoint(path, state_dict, extra_info=None):
        """Saves a fully self-contained, loadable checkpoint (same format
        as the final save) at any point during training - not just at the
        end. Used both for periodic 'latest' checkpoints (crash/interrupt
        protection) and immediate best-model saves (so the best model so
        far is never only sitting in memory)."""
        payload = {
            "model_state_dict": state_dict,
            "args": vars(args),
            "n_pathways": n_pathways,
            "img_dim": img_embeddings.shape[1],
            "n_proteins": n_proteins,
            "protein_names": protein_names,
            "pathway_names_final": names_final,
            "matched_sets_final": {name: matched_sets[name] for name in names_final},
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "protein_marker_mean": marker_mean,
            "protein_marker_std": marker_std,
            "arcsinh_cofactor": args.arcsinh_cofactor,
        }
        if extra_info:
            payload.update(extra_info)
        torch.save(payload, path)

    checkpoint_latest_path = os.path.join(args.output_dir, "checkpoint_latest.pt")
    checkpoint_best_path = os.path.join(args.output_dir, "checkpoint_best.pt")

    log(f"Starting mini-batch training for up to {args.n_epochs} epochs "
        f"(patience={args.patience if has_val else 'N/A'}, "
        f"checkpoint_every={args.checkpoint_every} epochs)", log_file)
    start = time.time()

    for epoch in range(args.n_epochs):
        model.train()
        epoch_loss, n_batches_done = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.img_x)
            loss = loss_fn(pred[:batch.batch_size], batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches_done += 1
        avg_loss = epoch_loss / n_batches_done
        scheduler.step(avg_loss)

        if has_val and ((epoch + 1) % args.eval_every == 0 or epoch == args.n_epochs - 1):
            val_preds = mini_batch_inference(model, val_graph, args.num_neighbors,
                                               args.inference_batch_size, device)
            val_y = val_graph.y.numpy()
            val_loss = float(np.mean((val_preds - val_y) ** 2))
            mean_corr = np.mean([spearmanr(val_preds[:, i], val_y[:, i]).correlation
                                  for i in range(val_y.shape[1])])

            improved = val_loss < (best_val_loss - args.min_delta)
            marker = ""
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
                marker = " *best*"
                # Save the best checkpoint to disk IMMEDIATELY, not just
                # held in memory - protects against losing the best model
                # if the process crashes/gets killed before training ends.
                save_checkpoint(checkpoint_best_path, best_state_dict,
                                 {"best_val_loss": best_val_loss, "best_epoch": best_epoch,
                                  "final_train_loss": avg_loss})
            else:
                epochs_without_improvement += 1

            log(f"Epoch {epoch+1}/{args.n_epochs} - Train Loss: {avg_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - Val MeanCorr: {mean_corr:.4f}{marker} - "
                f"{resource_status_str(device)}", log_file)

            if epochs_without_improvement >= args.patience:
                log(f"Early stopping (best val loss {best_val_loss:.4f} at epoch {best_epoch})", log_file)
                stopped_early = True
                break
        else:
            log(f"Epoch {epoch+1}/{args.n_epochs} - Train Loss: {avg_loss:.4f} - "
                f"{resource_status_str(device)}", log_file)

        # Periodic rolling checkpoint (overwrites each time) - protects
        # against losing progress on a long run even between validation
        # evaluations, and regardless of whether this epoch improved.
        if (epoch + 1) % args.checkpoint_every == 0:
            current_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(checkpoint_latest_path, current_state,
                             {"best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
                              "best_epoch": best_epoch, "final_train_loss": avg_loss,
                              "last_completed_epoch": epoch + 1})
            log(f"  Checkpoint saved: {checkpoint_latest_path} (epoch {epoch+1})", log_file)

    elapsed = time.time() - start
    log(f"Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min). Stopped early: {stopped_early}", log_file)

    if has_val and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        log(f"Restored best model (epoch {best_epoch}, val loss {best_val_loss:.4f})", log_file)

        final_val_preds = mini_batch_inference(model, val_graph, args.num_neighbors,
                                                 args.inference_batch_size, device)
        final_val_actual = val_graph.y.numpy()
        per_protein_corr = {
            name: spearmanr(final_val_preds[:, i], final_val_actual[:, i]).correlation
            for i, name in enumerate(protein_names)
        }
        corr_series = pd.Series(per_protein_corr).sort_values(ascending=False)

        log("=" * 60, log_file)
        log("FINAL VALIDATION ACCURACY REPORT (predicted vs actual)", log_file)
        log("=" * 60, log_file)
        log(f"Best epoch: {best_epoch}", log_file)
        log(f"Mean Spearman correlation: {corr_series.mean():.4f}", log_file)
        log(f"Top 5:\n{corr_series.head()}", log_file)
        log(f"Bottom 5:\n{corr_series.tail()}", log_file)
        corr_series.to_csv(os.path.join(args.output_dir, "val_accuracy_per_protein.csv"), header=True)
    else:
        best_epoch = args.n_epochs
        best_val_loss = None

    model_path = os.path.join(args.output_dir, "model_weights.pt")
    save_checkpoint(model_path, model.state_dict(),
                     {"final_train_loss": avg_loss, "best_val_loss": best_val_loss,
                      "best_epoch": best_epoch})
    log(f"Saved model weights to {model_path}", log_file)
    log("Done.", log_file)


def run_predict(args):
    device = get_device()
    log(f"[PREDICT] Using device: {device}")

    log(f"Loading checkpoint from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    protein_names = checkpoint["protein_names"]
    pathway_names_final = checkpoint["pathway_names_final"]
    matched_sets_final = checkpoint["matched_sets_final"]
    scaler_mean = checkpoint["scaler_mean"]
    scaler_scale = checkpoint["scaler_scale"]
    marker_mean = checkpoint["protein_marker_mean"]
    marker_std = checkpoint["protein_marker_std"]
    cofactor = checkpoint["arcsinh_cofactor"]
    a = checkpoint["args"]
    log(f"Checkpoint: n_pathways={checkpoint['n_pathways']}, img_dim={checkpoint['img_dim']}, "
        f"n_proteins={checkpoint['n_proteins']}, final_train_loss={checkpoint.get('final_train_loss')}, "
        f"best_val_loss={checkpoint.get('best_val_loss')}")

    model = build_model_from_checkpoint(checkpoint, device)
    log("Model loaded and set to eval mode")

    log(f"Loading test RNA from {args.test_rna}")
    test_rna = load_adata(args.test_rna)
    log(f"Test RNA shape (raw): {test_rna.shape}")

    log("Preprocessing RNA (normalize_total + log1p)")
    sc.pp.normalize_total(test_rna, target_sum=1e4)
    sc.pp.log1p(test_rna)

    log(f"Computing pathway scores for test bins using the SAME {len(pathway_names_final)} "
        f"gene sets identified during training")
    test_score_matrix, _ = compute_pathway_scores(
        test_rna, matched_sets_final, a["seed"], ctrl_size=a["ctrl_size"], n_bins=a["score_n_bins"],
        device=device, chunk_size=a["score_chunk_size"], tmp_dir=args.output_dir_tmp,
    )

    log("Applying saved StandardScaler (fit during training) to test pathway scores")
    X_test_scaled = ((test_score_matrix - scaler_mean) / scaler_scale).astype(np.float32)

    log(f"Loading test image embeddings from {args.test_image_embeddings}")
    img_embeddings = np.load(args.test_image_embeddings)
    img_index = pd.read_csv(args.test_image_index)
    test_img_aligned, valid_mask = align_image_embeddings(test_rna.obs_names, img_embeddings, img_index)
    if not valid_mask.all():
        X_test_scaled = X_test_scaled[valid_mask]
        test_rna = test_rna[valid_mask].copy()

    log(f"Building spatial k-NN graph for test bins (k={args.k_neighbors}, gated edges, "
        f"using training's saved mpp/sigma_um/gate settings for consistency)")
    coords = test_rna.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values
    edge_index, edge_weight = build_gated_adaptive_edges(
        coords, X_test_scaled, args.k_neighbors, a["mpp"],
        sigma_um=a["sigma_um"], max_distance_um=a["max_edge_distance_um"],
        gate_threshold=a["gate_threshold"],
    )

    log(f"Running mini-batch inference on {test_rna.n_obs} bins")
    dummy_y = torch.zeros((X_test_scaled.shape[0], len(protein_names)), dtype=torch.float32)
    infer_graph = Data(x=torch.from_numpy(X_test_scaled), edge_index=edge_index,
                        edge_attr=edge_weight.unsqueeze(-1), y=dummy_y)
    infer_graph.img_x = torch.from_numpy(test_img_aligned)

    preds = mini_batch_inference(model, infer_graph, a["num_neighbors"], args.inference_batch_size, device)
    log("Inference complete")

    if args.inverse_transform:
        log("Applying inverse transform (arcsinh + z-score -> raw protein intensity units), "
            "using the SAME marker_mean/marker_std fit on the training data")
        arcsinh_recovered = preds * marker_std + marker_mean
        preds = np.sinh(arcsinh_recovered) * cofactor
        log("NOTE: values originally clipped to [protein_clip_min, protein_clip_max] during "
            "training cannot be perfectly recovered")
    else:
        log("NOTE: predictions are in arcsinh + z-scored + clipped space, not raw protein units")

    log("Building output dataframe")
    if "array_row" not in test_rna.obs.columns or "array_col" not in test_rna.obs.columns:
        raise ValueError("Expected 'array_row'/'array_col' columns not found in test data")
    out_df = pd.DataFrame({
        "barcode": test_rna.obs_names,
        "pxl_row_in_fullres": test_rna.obs["array_row"].values,
        "pxl_col_in_fullres": test_rna.obs["array_col"].values,
    })
    pred_df = pd.DataFrame(preds, columns=protein_names, index=test_rna.obs_names)
    out_df = pd.concat([out_df.set_index("barcode"), pred_df], axis=1).reset_index().rename(
        columns={"index": "barcode"}
    )
    out_df.to_csv(args.output_path, index=False)
    log(f"Saved predictions to {args.output_path}")
    log("Done.")


def build_parser():
    parser = argparse.ArgumentParser(description="Pathway-GAT + Phikon cross-attention pipeline")
    sub = parser.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("train")
    p.add_argument("--train_rna", required=True)
    p.add_argument("--train_pro", required=True)
    p.add_argument("--val_rna", default=None)
    p.add_argument("--val_pro", default=None)
    p.add_argument("--image_embeddings", required=True)
    p.add_argument("--image_index", required=True)
    p.add_argument("--gmt_hallmark", default=None)
    p.add_argument("--gmt_c8", default=None)
    p.add_argument("--gmt_c7", default=None)
    p.add_argument("--gmt_c6", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--min_genes", type=int, default=10)
    p.add_argument("--min_corr", type=float, default=0.05)
    p.add_argument("--ctrl_size", type=int, default=50,
                    help="Number of control genes sampled per pathway gene, matched by "
                         "expression level (default: 50, same as scanpy's default)")
    p.add_argument("--score_n_bins", type=int, default=25,
                    help="Number of expression-level bins used for control gene matching "
                         "(default: 25, same as scanpy's default)")
    p.add_argument("--score_chunk_size", type=int, default=20000,
                    help="Number of bins processed per chunk during pathway score matrix "
                         "multiplication (default: 20000) - keeps memory bounded regardless "
                         "of GPU size or dataset scale; lower this if you hit GPU OOM")
    p.add_argument("--k_neighbors", type=int, default=15)
    p.add_argument("--num_neighbors", type=int, nargs="+", default=[15, 10])
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--inference_batch_size", type=int, default=1024)
    p.add_argument("--gat_hidden", type=int, default=256)
    p.add_argument("--gat_layers", type=int, default=2)
    p.add_argument("--gat_heads", type=int, default=4)
    p.add_argument("--fusion_dim", type=int, default=256)
    p.add_argument("--cross_attn_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--checkpoint_every", type=int, default=5,
                    help="Save a rolling 'checkpoint_latest.pt' every N epochs regardless of "
                         "improvement, so a crash or interruption doesn't lose all progress "
                         "on a long run. The best-so-far model is ALSO saved immediately as "
                         "'checkpoint_best.pt' whenever validation improves (default: 5).")
    p.add_argument("--arcsinh_cofactor", type=float, default=5.0)
    p.add_argument("--protein_clip_min", type=float, default=-5.0)
    p.add_argument("--protein_clip_max", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mpp", type=float, default=0.8820219467631594,
                    help="Microns-per-pixel for the training slide, used to convert pixel "
                         "coordinates to real physical (micron) distances for edge gating "
                         "(default matches the value used elsewhere in this project)")
    p.add_argument("--sigma_um", type=float, default=15.0,
                    help="Gaussian kernel width (microns) for the physical-distance edge "
                         "weight component - smaller values make weight fall off faster "
                         "with distance (default: 15.0)")
    p.add_argument("--max_edge_distance_um", type=float, default=None,
                    help="Adaptive density cap: drop any k-NN edge farther than this many "
                         "microns, so sparse tissue regions naturally get fewer edges "
                         "instead of being force-connected to distant bins (default: None, "
                         "no cap - pure KNN)")
    p.add_argument("--gate_threshold", type=float, default=0.0,
                    help="Transcriptomic gating: edges with combined "
                         "(physical_weight * transcriptomic_similarity) below this value "
                         "are removed entirely (hard gating). Default 0.0 means only the "
                         "soft signal (via edge_attr) is used, no edges are hard-dropped "
                         "based on transcriptomic dissimilarity alone.")
    p.add_argument("--boost_proteins", type=str, nargs="*",
                    default=["MAP2", "synd", "IDH1", "CD44", "CD14", "FIBR", "CD21",
                             "FOXP3", "CXCR5", "CD4"],
                    help="Protein names whose training-loss error is multiplied by "
                         "--boost_factor (case-insensitive matching). Pass an empty list "
                         "(--boost_proteins) to disable boosting entirely.")
    p.add_argument("--boost_factor", type=float, default=3.0,
                    help="Loss multiplier for --boost_proteins (default: 3.0). Higher "
                         "values push harder on the boosted proteins at the cost of the "
                         "other proteins' accuracy - watch the final per-protein report "
                         "if you raise this.")

    p2 = sub.add_parser("predict")
    p2.add_argument("--model_path", required=True)
    p2.add_argument("--test_rna", required=True)
    p2.add_argument("--test_image_embeddings", required=True)
    p2.add_argument("--test_image_index", required=True)
    p2.add_argument("--output_path", required=True)
    p2.add_argument("--output_dir_tmp", default=".",
                     help="Directory for temporary files created during pathway scoring")
    p2.add_argument("--k_neighbors", type=int, default=15)
    p2.add_argument("--inference_batch_size", type=int, default=1024)
    p2.add_argument("--inverse_transform", action="store_true")

    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "predict":
        run_predict(args)


if __name__ == "__main__":
    main()
