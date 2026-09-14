"""
Pathway Score + Cross-Attention Training Pipeline
==================================================
Trains on spatial train/val split using pathway scores + Phikon image
embeddings. Reports both mean Pearson r and top-10 SCC per epoch.

USAGE:
    python pathway_crossattn_train.py \\
        --train_rna  data/rna_train_split.pkl \\
        --train_pro  data/pro_train_split.pkl \\
        --val_rna    data/rna_val_split.pkl \\
        --val_pro    data/pro_val_split.pkl \\
        --gmt_h      data/hallmark_pathways.gmt \\
        --gmt_c8     data/c8_pathways.gmt \\
        --image_embeddings data/embeddings_gat_test.npy \\
        --image_index      data/embedding_index_gat_test.csv \\
        --output_dir outputs_pathway_crossattn \\
        --k_neighbors 15 --num_neighbors 15 10 \\
        --batch_size 512 --n_epochs 50 --patience 10
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader


# ============================================================================
# Utilities
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
    return torch.device("cpu")


def resource_status_str(device):
    parts = []
    if device.type == "cuda":
        alloc    = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        total    = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        parts.append(f"GPU:{alloc:.1f}/{total:.1f}GB(res {reserved:.1f}GB)")
    try:
        import psutil
        ram = psutil.virtual_memory()
        parts.append(f"RAM:{ram.used/(1024**3):.1f}/{ram.total/(1024**3):.1f}GB({ram.percent:.0f}%)")
    except ImportError:
        pass
    return " | ".join(parts) if parts else ""


def load_adata(path):
    """Load AnnData from .h5ad or pickled .pkl"""
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    return sc.read_h5ad(path)


# ============================================================================
# Pathway scoring
# ============================================================================

def parse_gmt(filepath):
    pathways = {}
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                pathways[parts[0]] = parts[2:]
    return pathways


def compute_pathway_scores(rna_adata, matched_sets, all_names,
                            chunk_size=5000, log_fn=None):
    n_bins = rna_adata.n_obs
    n_sets = len(all_names)

    adata_gene_idx = {g: i for i, g
                      in enumerate(rna_adata.var_names)}
    col_idx = {}
    for name, genes in matched_sets.items():
        present     = [g for g in genes if g in adata_gene_idx]
        col_idx[name] = [adata_gene_idx[g] for g in present]

    score_matrix = np.zeros((n_bins, n_sets), dtype=np.float32)

    for chunk_start in range(0, n_bins, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_bins)
        chunk = rna_adata.X[chunk_start:chunk_end]
        if sp.issparse(chunk):
            chunk = chunk.toarray()
        chunk = np.array(chunk, dtype=np.float32)

        for j, name in enumerate(all_names):
            cols = col_idx[name]
            if len(cols) > 0:
                score_matrix[chunk_start:chunk_end, j] = \
                    chunk[:, cols].mean(axis=1)

        if log_fn and (
            chunk_start % 25000 == 0 or chunk_end == n_bins
        ):
            log_fn(f"  {chunk_end/n_bins*100:.0f}% — "
                   f"{chunk_end:,}/{n_bins:,}")

    return score_matrix


def build_pathway_features(rna_adata, gmt_h_path, gmt_c8_path,
                            min_genes=10, var_pct=25, min_r=0.05,
                            Y_for_filter=None, scaler=None,
                            matched_sets=None, names_final=None,
                            log_fn=None):
    """
    Full pathway scoring pipeline.
    Training:  matched_sets=None → parse, match, filter, fit scaler
    Val/Test:  matched_sets provided → use same sets, apply scaler
    """
    if matched_sets is None:
        if log_fn:
            log_fn("Parsing GMT files...")
        hallmark = parse_gmt(gmt_h_path)
        c8       = parse_gmt(gmt_c8_path)
        combined = {**hallmark, **c8}

        rna_gene_set = set(rna_adata.var_names)
        matched_sets = {}
        for name, genes in combined.items():
            matched = [g for g in genes if g in rna_gene_set]
            if len(matched) >= min_genes:
                matched_sets[name] = matched

        all_names = list(matched_sets.keys())
        if log_fn:
            log_fn(f"Matched gene sets: {len(matched_sets)}")
    else:
        all_names = names_final

    # Compute scores
    if log_fn:
        log_fn(f"Computing {len(all_names)} pathway scores "
               f"for {rna_adata.n_obs:,} bins...")
    t0 = time.time()
    score_matrix = compute_pathway_scores(
        rna_adata, matched_sets, all_names, log_fn=log_fn
    )
    if log_fn:
        log_fn(f"Scores done in {(time.time()-t0)/60:.1f} min")

    # Filter (training only)
    if names_final is None:
        if log_fn:
            log_fn("Filtering pathway scores...")
        n_sets        = len(all_names)
        set_stds      = score_matrix.std(axis=0)
        var_threshold = np.percentile(set_stds, var_pct)
        var_mask      = set_stds > var_threshold

        corr_mask = np.zeros(n_sets, dtype=bool)
        if Y_for_filter is not None:
            for j in range(n_sets):
                if not var_mask[j]:
                    continue
                path_vals = score_matrix[:, j]
                if path_vals.std() < 1e-9:
                    continue
                for i in range(Y_for_filter.shape[1]):
                    pro_vals = Y_for_filter[:, i]
                    if pro_vals.std() < 1e-9:
                        continue
                    r, _ = pearsonr(path_vals, pro_vals)
                    if abs(r) >= min_r:
                        corr_mask[j] = True
                        break
        else:
            corr_mask = var_mask.copy()

        kept_idx     = np.where(var_mask & corr_mask)[0]
        names_final  = [all_names[i] for i in kept_idx]
        matched_sets = {
            names_final[i]: matched_sets[names_final[i]]
            for i in range(len(names_final))
        }
        score_matrix = score_matrix[:, kept_idx]
        if log_fn:
            log_fn(f"Features after filter: {len(kept_idx)}")
            log_fn(f"Score matrix: {score_matrix.shape}")

    # Scale
    if scaler is None:
        scaler        = StandardScaler()
        scores_scaled = scaler.fit_transform(
            score_matrix
        ).astype(np.float32)
    else:
        scores_scaled = scaler.transform(
            score_matrix
        ).astype(np.float32)

    return scores_scaled, matched_sets, names_final, scaler


# ============================================================================
# Protein normalisation
# ============================================================================

def fit_protein_transform(adata_train, cofactor):
    X = adata_train.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") \
        else np.asarray(X)
    X_arcsinh   = np.arcsinh(X / cofactor)
    marker_mean = X_arcsinh.mean(axis=0)
    marker_std  = X_arcsinh.std(axis=0)
    return marker_mean, marker_std


def apply_protein_transform(adata, cofactor, marker_mean,
                             marker_std, clip_min, clip_max):
    adata = adata.copy()
    X = adata.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") \
        else np.asarray(X)
    X_arcsinh = np.arcsinh(X / cofactor)
    X_scaled  = (X_arcsinh - marker_mean) / marker_std
    X_clipped = np.clip(X_scaled, clip_min, clip_max)
    adata.X   = X_clipped.astype(np.float32)
    return adata


# ============================================================================
# Graph construction
# ============================================================================

def build_spatial_knn_graph(coords, k):
    nbrs       = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    n_nodes    = coords.shape[0]
    src        = np.repeat(np.arange(n_nodes), k)
    dst        = indices[:, 1:].flatten()
    edge_index = np.stack([src, dst], axis=0)
    edge_index = np.concatenate(
        [edge_index, edge_index[::-1]], axis=1
    )
    return torch.tensor(edge_index, dtype=torch.long)


def align_image_embeddings(obs_names, img_embeddings, img_index):
    img_df     = pd.DataFrame(
        img_embeddings,
        index=img_index["barcode"].values
    )
    img_df     = img_df.reindex(obs_names)
    valid_mask = ~img_df.isna().any(axis=1)
    n_missing  = (~valid_mask).sum()
    if n_missing > 0:
        log(f"  WARNING: {n_missing} bins have no matching "
            f"image embedding — excluding them")
    aligned = img_df[valid_mask].values.astype(np.float32)
    return aligned, valid_mask.values


def build_graph(pathway_scores, pro_adata,
                obs_df, img_embeddings, img_index,
                k_neighbors):
    obs_names = obs_df.index.tolist()

    img_aligned, valid_mask = align_image_embeddings(
        obs_names, img_embeddings, img_index
    )

    if not valid_mask.all():
        pathway_scores = pathway_scores[valid_mask]
        obs_df         = obs_df[valid_mask]
        obs_names      = obs_df.index.tolist()
        if pro_adata is not None:
            pro_adata = pro_adata[obs_names].copy()

    coords     = obs_df[["pxl_row_in_fullres",
                          "pxl_col_in_fullres"]].values
    edge_index = build_spatial_knn_graph(coords, k_neighbors)

    X = pathway_scores.astype(np.float32)
    Y = pro_adata.X
    Y = np.asarray(Y.todense()) if hasattr(Y, "todense") \
        else np.asarray(Y)
    Y = Y.astype(np.float32)

    data       = Data(
        x          = torch.from_numpy(X),
        edge_index = edge_index,
        y          = torch.from_numpy(Y)
    )
    data.img_x = torch.from_numpy(img_aligned)
    return data, X.shape[1], Y.shape[1]


# ============================================================================
# Model
# ============================================================================

class RNAGATEncoder(nn.Module):
    def __init__(self, n_pathway_scores, hidden_dim,
                 out_dim, heads, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU()

        self.gat1 = GATv2Conv(
            n_pathway_scores, hidden_dim,
            heads=heads, dropout=dropout, concat=True
        )
        self.bn1  = nn.BatchNorm1d(hidden_dim * heads)

        self.gat2 = GATv2Conv(
            hidden_dim * heads, out_dim,
            heads=1, dropout=dropout, concat=False
        )
        self.bn2  = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = dim,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, rna_emb, img_emb):
        q        = rna_emb.unsqueeze(1)
        kv       = img_emb.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        fused    = self.norm(
            attn_out.squeeze(1) + rna_emb
        )
        return fused


class PathwayCrossAttnPredictor(nn.Module):
    def __init__(self, n_pathway_scores, img_dim,
                 gat_hidden_dim, gat_heads,
                 fusion_dim, cross_attn_heads,
                 n_proteins, dropout):
        super().__init__()

        self.rna_encoder = RNAGATEncoder(
            n_pathway_scores = n_pathway_scores,
            hidden_dim       = gat_hidden_dim,
            out_dim          = fusion_dim,
            heads            = gat_heads,
            dropout          = dropout
        )
        self.img_projection = nn.Sequential(
            nn.Linear(img_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fusion = CrossAttentionFusion(
            dim     = fusion_dim,
            n_heads = cross_attn_heads,
            dropout = dropout
        )
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_proteins)
        )

    def forward(self, x_pathway, edge_index, x_img):
        rna_emb = self.rna_encoder(x_pathway, edge_index)
        img_emb = self.img_projection(x_img)
        fused   = self.fusion(rna_emb, img_emb)
        return self.predictor(fused)


# ============================================================================
# Mini-batch inference
# ============================================================================

def mini_batch_inference(model, graph_data,
                          num_neighbors, batch_size, device):
    loader    = NeighborLoader(
        graph_data,
        num_neighbors = num_neighbors,
        batch_size    = batch_size,
        shuffle       = False,
        num_workers   = 0
    )
    n_nodes    = graph_data.num_nodes
    n_proteins = graph_data.y.shape[1]
    all_preds  = np.zeros((n_nodes, n_proteins), dtype=np.float32)
    filled     = np.zeros(n_nodes, dtype=bool)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            seed_idx = batch.n_id[:batch.batch_size].cpu().numpy()
            batch    = batch.to(device)
            pred     = model(batch.x, batch.edge_index, batch.img_x)
            all_preds[seed_idx] = pred[:batch.batch_size].cpu().numpy()
            filled[seed_idx]    = True

    if not filled.all():
        raise RuntimeError(
            f"{(~filled).sum()} nodes never covered during inference"
        )
    return all_preds


# ============================================================================
# Validation metrics
# ============================================================================

def compute_val_metrics(val_preds, val_y, protein_names, log_fn=None):
    """
    Computes for each protein:
      - Pearson r
      - Spearman SCC

    Reports:
      - Overall mean Pearson r (all 44 proteins)
      - Overall mean SCC      (all 44 proteins)
      - Top 10 SCC proteins
      - Bottom 10 SCC proteins
    """
    n_proteins = val_y.shape[1]

    pearson_scores  = []
    spearman_scores = []
    per_protein     = []

    for i in range(n_proteins):
        pred_i = val_preds[:, i]
        true_i = val_y[:, i]

        if pred_i.std() > 1e-9 and true_i.std() > 1e-9:
            r_p, _ = pearsonr(pred_i, true_i)
            r_s    = spearmanr(pred_i, true_i).correlation
        else:
            r_p = 0.0
            r_s = 0.0

        pearson_scores.append(float(r_p))
        spearman_scores.append(float(r_s))
        per_protein.append({
            "protein"   : protein_names[i],
            "pearson_r" : float(r_p),
            "scc"       : float(r_s),
        })

    mean_pearson  = float(np.mean(pearson_scores))
    mean_scc      = float(np.mean(spearman_scores))

    df = pd.DataFrame(per_protein).sort_values(
        "scc", ascending=False
    )

    top10    = df.head(10)
    bottom10 = df.tail(10)

    if log_fn:
        log_fn("-" * 55)
        log_fn(f"Mean Pearson r (all 44): {mean_pearson:.4f}")
        log_fn(f"Mean SCC      (all 44): {mean_scc:.4f}")
        log_fn("")
        log_fn("Top 10 proteins by SCC:")
        for _, row in top10.iterrows():
            log_fn(f"  {row['protein']:<22} "
                   f"SCC={row['scc']:.4f}  "
                   f"Pearson={row['pearson_r']:.4f}")
        log_fn("")
        log_fn("Bottom 10 proteins by SCC:")
        for _, row in bottom10.iterrows():
            log_fn(f"  {row['protein']:<22} "
                   f"SCC={row['scc']:.4f}  "
                   f"Pearson={row['pearson_r']:.4f}")
        log_fn("-" * 55)

    return mean_pearson, mean_scc, df


# ============================================================================
# TRAIN
# ============================================================================

def run_train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "training_log.txt")
    open(log_file, "w").close()

    def llog(msg):
        log(msg, log_file)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()

    llog(f"Device: {device}")
    if device.type == "cuda":
        llog(f"GPU : {torch.cuda.get_device_name(0)}")
        llog(f"VRAM: "
             f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # ── Load data ──────────────────────────────────────────────────────────
    llog(f"Loading train RNA from {args.train_rna}")
    train_rna = load_adata(args.train_rna)
    llog(f"Loading train protein from {args.train_pro}")
    train_pro = load_adata(args.train_pro)
    if not (train_rna.obs_names == train_pro.obs_names).all():
        train_pro = train_pro[train_rna.obs_names].copy()
    protein_names = train_pro.var_names.tolist()
    llog(f"Train bins   : {train_rna.n_obs:,}")
    llog(f"Genes        : {train_rna.n_vars:,}")
    llog(f"Proteins     : {len(protein_names)}")

    has_val = bool(args.val_rna and args.val_pro)
    if has_val:
        llog(f"Loading val RNA from {args.val_rna}")
        val_rna = load_adata(args.val_rna)
        llog(f"Loading val protein from {args.val_pro}")
        val_pro = load_adata(args.val_pro)
        if not (val_rna.obs_names == val_pro.obs_names).all():
            val_pro = val_pro[val_rna.obs_names].copy()
        llog(f"Val bins     : {val_rna.n_obs:,}")
    else:
        llog("No val split provided")

    # ── RNA normalisation ──────────────────────────────────────────────────
    llog("Normalising RNA (normalize_total + log1p)...")
    sc.pp.normalize_total(train_rna, target_sum=1e4)
    sc.pp.log1p(train_rna)
    if has_val:
        sc.pp.normalize_total(val_rna, target_sum=1e4)
        sc.pp.log1p(val_rna)

    # ── Protein normalisation ──────────────────────────────────────────────
    llog(f"Normalising protein "
         f"(arcsinh cofactor={args.arcsinh_cofactor} + z-score + clip)...")
    marker_mean, marker_std = fit_protein_transform(
        train_pro, args.arcsinh_cofactor
    )
    train_pro = apply_protein_transform(
        train_pro, args.arcsinh_cofactor,
        marker_mean, marker_std,
        args.protein_clip_min, args.protein_clip_max
    )
    if has_val:
        val_pro = apply_protein_transform(
            val_pro, args.arcsinh_cofactor,
            marker_mean, marker_std,
            args.protein_clip_min, args.protein_clip_max
        )

    # Y for filter
    Y_train = train_pro.X
    Y_train = np.asarray(Y_train.todense()) \
        if hasattr(Y_train, "todense") else np.asarray(Y_train)
    Y_train = Y_train.astype(np.float32)

    # ── Pathway scores ─────────────────────────────────────────────────────
    llog("=" * 55)
    llog("Computing pathway scores — TRAINING")
    llog("=" * 55)
    train_scores, matched_sets, names_final, scaler = \
        build_pathway_features(
            rna_adata    = train_rna,
            gmt_h_path   = args.gmt_h,
            gmt_c8_path  = args.gmt_c8,
            min_genes    = args.min_genes,
            var_pct      = args.var_pct,
            min_r        = args.min_r,
            Y_for_filter = Y_train,
            log_fn       = llog
        )
    n_pathway_scores = train_scores.shape[1]

    # Save scaler + names for later use
    import joblib
    joblib.dump(
        scaler,
        os.path.join(args.output_dir, "scaler.pkl")
    )
    np.save(
        os.path.join(args.output_dir, "names_final.npy"),
        np.array(names_final)
    )
    llog(f"Pathway features: {n_pathway_scores}")

    if has_val:
        llog("=" * 55)
        llog("Computing pathway scores — VALIDATION")
        llog("=" * 55)
        val_scores, _, _, _ = build_pathway_features(
            rna_adata    = val_rna,
            gmt_h_path   = args.gmt_h,
            gmt_c8_path  = args.gmt_c8,
            matched_sets = matched_sets,
            names_final  = names_final,
            scaler       = scaler,
            log_fn       = llog
        )

    # ── Image embeddings ───────────────────────────────────────────────────
    llog(f"Loading image embeddings from {args.image_embeddings}")
    img_embeddings = np.load(args.image_embeddings)
    img_index      = pd.read_csv(args.image_index)
    llog(f"Image embeddings: {img_embeddings.shape}")

    # ── Build graphs ───────────────────────────────────────────────────────
    llog(f"Building train graph (k={args.k_neighbors})...")
    train_graph, n_feat, n_proteins = build_graph(
        pathway_scores = train_scores,
        pro_adata      = train_pro,
        obs_df         = train_rna.obs,
        img_embeddings = img_embeddings,
        img_index      = img_index,
        k_neighbors    = args.k_neighbors
    )
    llog(f"Train graph: {train_graph.num_nodes:,} nodes, "
         f"{train_graph.edge_index.shape[1]:,} edges")
    del train_rna, train_pro
    gc.collect()

    if has_val:
        llog(f"Building val graph (k={args.k_neighbors})...")
        val_graph, _, _ = build_graph(
            pathway_scores = val_scores,
            pro_adata      = val_pro,
            obs_df         = val_rna.obs,
            img_embeddings = img_embeddings,
            img_index      = img_index,
            k_neighbors    = args.k_neighbors
        )
        llog(f"Val graph: {val_graph.num_nodes:,} nodes, "
             f"{val_graph.edge_index.shape[1]:,} edges")
        del val_rna, val_pro
        gc.collect()

    # ── Model ──────────────────────────────────────────────────────────────
    model = PathwayCrossAttnPredictor(
        n_pathway_scores = n_pathway_scores,
        img_dim          = img_embeddings.shape[1],
        gat_hidden_dim   = args.gat_hidden_dim,
        gat_heads        = args.gat_heads,
        fusion_dim       = args.fusion_dim,
        cross_attn_heads = args.cross_attn_heads,
        n_proteins       = n_proteins,
        dropout          = args.dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    llog(f"Model parameters : {n_params:,}")
    llog(f"  Input (pathway): {n_pathway_scores}")
    llog(f"  Hidden         : {args.gat_hidden_dim} × {args.gat_heads} heads")
    llog(f"  Fusion dim     : {args.fusion_dim}")
    llog(f"  Output         : {n_proteins} proteins")

    # ── DataLoader ─────────────────────────────────────────────────────────
    train_loader = NeighborLoader(
        train_graph,
        num_neighbors = args.num_neighbors,
        batch_size    = args.batch_size,
        shuffle       = True,
        num_workers   = 0
    )
    n_batches = (train_graph.num_nodes + args.batch_size - 1) \
        // args.batch_size
    llog(f"Train loader: ~{n_batches} batches/epoch")

    # ── Optimiser ──────────────────────────────────────────────────────────
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay
    )
    loss_fn = nn.MSELoss()

    # ── Training state ─────────────────────────────────────────────────────
    best_val_loss              = float("inf")
    best_mean_scc              = -999.0
    best_epoch                 = -1
    best_state_dict            = None
    epochs_without_improvement = 0
    stopped_early              = False
    avg_loss                   = None
    history                    = []

    llog("=" * 55)
    llog("Starting training")
    llog("=" * 55)
    t_start = time.time()

    for epoch in range(args.n_epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        epoch_loss     = 0.0
        n_batches_done = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            pred      = model(batch.x, batch.edge_index, batch.img_x)
            seed_pred = pred[:batch.batch_size]
            seed_y    = batch.y[:batch.batch_size]
            loss      = loss_fn(seed_pred, seed_y)
            loss.backward()
            optimiser.step()
            epoch_loss     += loss.item()
            n_batches_done += 1

        avg_loss = epoch_loss / n_batches_done

        # ── Validate ───────────────────────────────────────────────────────
        if has_val and (
            (epoch + 1) % args.eval_every == 0
            or epoch == args.n_epochs - 1
        ):
            val_preds = mini_batch_inference(
                model, val_graph,
                args.num_neighbors,
                args.inference_batch_size, device
            )
            val_y    = val_graph.y.numpy()
            val_loss = float(np.mean((val_preds - val_y) ** 2))

            # ── Compute BOTH metrics ───────────────────────────────────────
            mean_pearson = float(np.mean([
                pearsonr(val_preds[:, i], val_y[:, i])[0]
                for i in range(val_y.shape[1])
                if val_preds[:, i].std() > 1e-9
                and val_y[:, i].std() > 1e-9
            ]))

            mean_scc = float(np.mean([
                spearmanr(val_preds[:, i], val_y[:, i]).correlation
                for i in range(val_y.shape[1])
                if val_preds[:, i].std() > 1e-9
                and val_y[:, i].std() > 1e-9
            ]))

            # Top 10 SCC
            scc_per_protein = {
                protein_names[i]: spearmanr(
                    val_preds[:, i], val_y[:, i]
                ).correlation
                for i in range(val_y.shape[1])
                if val_preds[:, i].std() > 1e-9
                and val_y[:, i].std() > 1e-9
            }
            scc_series = pd.Series(scc_per_protein).sort_values(
                ascending=False
            )
            top10_scc = scc_series.head(10)

            improved = val_loss < (best_val_loss - args.min_delta)
            marker   = ""
            if improved:
                best_val_loss  = val_loss
                best_mean_scc  = mean_scc
                best_epoch     = epoch + 1
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                epochs_without_improvement = 0
                marker = " ← best"
            else:
                epochs_without_improvement += 1

            history.append({
                "epoch"       : epoch + 1,
                "train_loss"  : avg_loss,
                "val_loss"    : val_loss,
                "mean_pearson": mean_pearson,
                "mean_scc"    : mean_scc,
            })

            llog(f"Epoch {epoch+1:03d}/{args.n_epochs}  "
                 f"train={avg_loss:.4f}  "
                 f"val={val_loss:.4f}  "
                 f"Pearson_r={mean_pearson:.4f}  "
                 f"SCC={mean_scc:.4f}  "
                 f"best_SCC={best_mean_scc:.4f}"
                 f"{marker}  "
                 f"{resource_status_str(device)}")

            llog(f"  Top 10 SCC: " + "  ".join(
                [f"{p}={v:.3f}"
                 for p, v in top10_scc.items()]
            ))

            if epochs_without_improvement >= args.patience:
                llog(f"Early stopping at epoch {epoch+1} "
                     f"(best val loss {best_val_loss:.4f} "
                     f"at epoch {best_epoch})")
                stopped_early = True
                break
        else:
            history.append({
                "epoch"      : epoch + 1,
                "train_loss" : avg_loss,
            })
            llog(f"Epoch {epoch+1:03d}/{args.n_epochs}  "
                 f"train={avg_loss:.4f}  "
                 f"{resource_status_str(device)}")

    elapsed = time.time() - t_start
    llog(f"Training done in {elapsed/60:.1f} min. "
         f"Early stopped: {stopped_early}")

    # ── Restore best model ─────────────────────────────────────────────────
    if has_val and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        llog(f"Restored best weights "
             f"(epoch {best_epoch}, "
             f"val loss {best_val_loss:.4f})")

        # ── Final full accuracy report ─────────────────────────────────────
        llog("=" * 55)
        llog("FINAL VALIDATION ACCURACY REPORT")
        llog("=" * 55)

        final_preds  = mini_batch_inference(
            model, val_graph,
            args.num_neighbors,
            args.inference_batch_size, device
        )
        final_actual = val_graph.y.numpy()

        mean_pearson_final, mean_scc_final, full_df = \
            compute_val_metrics(
                final_preds, final_actual,
                protein_names, log_fn=llog
            )

        llog(f"Best epoch          : {best_epoch}")
        llog(f"Final Mean Pearson r: {mean_pearson_final:.4f}")
        llog(f"Final Mean SCC      : {mean_scc_final:.4f}")

        # Save per-protein report
        report_path = os.path.join(
            args.output_dir, "val_accuracy_per_protein.csv"
        )
        full_df.to_csv(report_path, index=False)
        llog(f"Saved per-protein report to {report_path}")

    # ── Save model ─────────────────────────────────────────────────────────
    model_path = os.path.join(
        args.output_dir, "model_weights.pt"
    )
    torch.save({
        "model_state_dict"   : model.state_dict(),
        "args"               : vars(args),
        "n_pathway_scores"   : n_pathway_scores,
        "img_dim"            : img_embeddings.shape[1],
        "n_proteins"         : n_proteins,
        "protein_names"      : protein_names,
        "names_final"        : names_final,
        "marker_mean"        : marker_mean,
        "marker_std"         : marker_std,
        "arcsinh_cofactor"   : args.arcsinh_cofactor,
        "final_train_loss"   : avg_loss,
        "best_val_loss"      : best_val_loss,
        "best_mean_scc"      : best_mean_scc,
        "best_epoch"         : best_epoch,
    }, model_path)
    llog(f"Saved model to {model_path}")

    # Save training history
    pd.DataFrame(history).to_csv(
        os.path.join(args.output_dir, "training_history.csv"),
        index=False
    )
    llog("Saved training_history.csv")
    llog("Done.")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pathway Score + Cross-Attention training pipeline"
    )

    # Data
    parser.add_argument("--train_rna",  type=str, required=True)
    parser.add_argument("--train_pro",  type=str, required=True)
    parser.add_argument("--val_rna",    type=str, default=None)
    parser.add_argument("--val_pro",    type=str, default=None)
    parser.add_argument("--gmt_h",      type=str, required=True)
    parser.add_argument("--gmt_c8",     type=str, required=True)
    parser.add_argument("--image_embeddings", type=str, required=True)
    parser.add_argument("--image_index",      type=str, required=True)
    parser.add_argument("--output_dir",       type=str, required=True)

    # Pathway scoring
    parser.add_argument("--min_genes", type=int,   default=10)
    parser.add_argument("--var_pct",   type=float, default=25.0)
    parser.add_argument("--min_r",     type=float, default=0.05)

    # Graph
    parser.add_argument("--k_neighbors",   type=int,        default=15)
    parser.add_argument("--num_neighbors", type=int, nargs="+",
                         default=[15, 10])

    # Model
    parser.add_argument("--gat_hidden_dim",   type=int,   default=256)
    parser.add_argument("--gat_heads",        type=int,   default=4)
    parser.add_argument("--fusion_dim",       type=int,   default=256)
    parser.add_argument("--cross_attn_heads", type=int,   default=4)
    parser.add_argument("--dropout",          type=float, default=0.2)

    # Training
    parser.add_argument("--batch_size",           type=int,   default=512)
    parser.add_argument("--inference_batch_size",  type=int,   default=1024)
    parser.add_argument("--lr",                   type=float, default=3e-4)
    parser.add_argument("--weight_decay",         type=float, default=1e-5)
    parser.add_argument("--n_epochs",             type=int,   default=50)
    parser.add_argument("--patience",             type=int,   default=10)
    parser.add_argument("--min_delta",            type=float, default=1e-4)
    parser.add_argument("--eval_every",           type=int,   default=1)
    parser.add_argument("--seed",                 type=int,   default=42)

    # Protein transform
    parser.add_argument("--arcsinh_cofactor", type=float, default=5.0)
    parser.add_argument("--protein_clip_min", type=float, default=-5.0)
    parser.add_argument("--protein_clip_max", type=float, default=5.0)

    args = parser.parse_args()
    run_train(args)


if __name__ == "__main__":
    main()
