"""
Multi-Modal Spatial Proteomics Pipeline v4
============================================================================
Combines and improves upon gat_crossattn_pipeline_v3.py and
integrated_gat_binn_pipeline.py with the following changes:

ARCHITECTURE IMPROVEMENTS vs v3:
  1. 3-layer GATv2Conv encoder (was 2) - doubles the spatial receptive field
  2. LightPathwayEncoder fused directly into the GAT predictor - pathway
     biological structure is now a training signal, not a post-hoc blend
  3. Deeper prediction head (fusion + pathway -> 512 -> 256 -> n_proteins)
  4. AdamW optimizer with cosine LR schedule + linear warmup (was Adam, flat)
  5. Combined loss: 0.5 * MSE + 0.5 * soft-Spearman, directly optimising
     the evaluation metric (was MSE only)
  6. Gradient clipping (max_norm=1.0) for training stability

BINN INTEGRATION vs integrated_gat_binn_pipeline.py:
  - The BINN is no longer a separate model blended after the fact.
  - A lightweight LightPathwayEncoder (MLP, ~2M params) is trained JOINTLY
    with the GAT end-to-end. Its output is concatenated with the
    GAT+image fused representation before the prediction head.
  - This lets the GAT's spatial attention incorporate biological pathway
    structure during training rather than averaging two independent outputs.
  - The do-no-harm alpha tournament is retained as an OPTIONAL post-hoc
    step using a separately-trained standalone BINN, for cases where the
    additional blend still helps on specific proteins.

USAGE
-----
1) TRAIN (internal split for benchmarking):

    python gat_binn_pipeline_v4.py train \
        --train_rna data/rna_train_split.pkl \
        --train_pro data/pro_train_split.pkl \
        --val_rna   data/rna_val_split.pkl \
        --val_pro   data/pro_val_split.pkl \
        --image_embeddings outputs_histology/embeddings.npy \
        --image_index      outputs_histology/embedding_index.csv \
        --output_dir outputs_v4_internal \
        --k_neighbors 15 --num_neighbors 15 10 5 \
        --gat_proj_dim 1024 --gat_hidden_dim 256 --gat_heads 4 \
        --fusion_dim 256 --cross_attn_heads 4 \
        --pathway_dim 128 \
        --dropout 0.3 --lr 2e-4 --weight_decay 1e-5 \
        --n_epochs 100 --patience 15 --warmup_epochs 5

2) TRAIN on full data for final submission (no val set, fixed epochs):

    python gat_binn_pipeline_v4.py train \
        --train_rna data/train_rna.h5ad \
        --train_pro data/train_pro.h5ad \
        --image_embeddings outputs_histology/embeddings.npy \
        --image_index      outputs_histology/embedding_index.csv \
        --output_dir outputs_v4_fulltrain \
        --k_neighbors 15 --num_neighbors 15 10 5 \
        --gat_proj_dim 1024 --gat_hidden_dim 256 --gat_heads 4 \
        --fusion_dim 256 --cross_attn_heads 4 \
        --pathway_dim 128 \
        --dropout 0.3 --lr 2e-4 --weight_decay 1e-5 \
        --n_epochs <BEST_EPOCH_FROM_INTERNAL> --patience 999 --warmup_epochs 5

3) PREDICT on held-out test/val sets:

    python gat_binn_pipeline_v4.py predict \
        --model_path outputs_v4_internal/model_weights.pt \
        --valid_rna_input data/test_rna.h5ad \
        --train_rna_reference data/train_rna.h5ad \
        --valid_image_embeddings outputs_histology/embeddings.npy \
        --valid_image_index      outputs_histology/embedding_index.csv \
        --output_path predictions_test.csv \
        --k_neighbors 15 --num_neighbors 15 10 5
"""

import argparse
import gc
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader


# ============================================================================
# 1. Constants and defaults
# ============================================================================

DEFAULT_K_NEIGHBORS    = 15
DEFAULT_NUM_NEIGHBORS  = [15, 10, 5]   # one entry per GAT layer
DEFAULT_PATHWAY_DIM    = 128
DEFAULT_WARMUP_EPOCHS  = 5
BLEND_ALPHAS           = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


# ============================================================================
# 2. Shared utilities
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
        reserved = torch.cuda.memory_reserved(device)  / (1024 ** 3)
        total    = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        parts.append(f"GPU: {alloc:.1f}/{total:.1f}GB (reserved {reserved:.1f}GB)")
    try:
        import psutil
        ram = psutil.virtual_memory()
        parts.append(f"RAM: {ram.used/(1024**3):.1f}/{ram.total/(1024**3):.1f}GB ({ram.percent:.0f}%)")
    except ImportError:
        pass
    return " | ".join(parts) if parts else ""


def load_adata(path):
    """Load AnnData from .h5ad or pickled .pkl - auto-detected."""
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    return sc.read_h5ad(path)


def preprocess_rna(adata):
    """normalize_total + log1p. Row-wise ops - no cross-split leakage."""
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def fit_protein_transform(adata_train, cofactor):
    """Fit arcsinh + z-score stats on TRAIN split only."""
    X = adata_train.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    X_arcsinh = np.arcsinh(X / cofactor)
    return X_arcsinh.mean(axis=0), X_arcsinh.std(axis=0)


def apply_protein_transform(adata, cofactor, marker_mean, marker_std, clip_min, clip_max):
    """Apply pre-fit arcsinh + z-score + clip transform to any split."""
    adata = adata.copy()
    X = adata.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    X_arcsinh = np.arcsinh(X / cofactor)
    X_scaled  = (X_arcsinh - marker_mean) / (marker_std + 1e-8)
    X_clipped = np.clip(X_scaled, clip_min, clip_max)
    adata.X   = X_clipped.astype(np.float32)
    return adata


def build_spatial_knn_graph(coords, k):
    """Bidirectional k-NN graph from pixel coordinates."""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    n = coords.shape[0]
    src = np.repeat(np.arange(n), k)
    dst = indices[:, 1:].flatten()
    ei  = np.stack([src, dst], axis=0)
    ei  = np.concatenate([ei, ei[::-1]], axis=1)
    return torch.tensor(ei, dtype=torch.long)


def align_image_embeddings(obs_names, img_embeddings, img_index):
    """Subset full-tissue image embeddings to the bins in obs_names."""
    img_df     = pd.DataFrame(img_embeddings, index=img_index["barcode"].values)
    img_df     = img_df.reindex(obs_names)
    valid_mask = ~img_df.isna().any(axis=1)
    n_missing  = (~valid_mask).sum()
    if n_missing > 0:
        log(f"  WARNING: {n_missing} bins have no image embedding - excluding them")
    aligned = img_df[valid_mask].values.astype(np.float32)
    return aligned, valid_mask.values


# ============================================================================
# 3. Loss functions
# ============================================================================

def soft_spearman_loss(pred, target, temperature=0.1):
    """
    Differentiable soft-rank Spearman correlation loss.
    Approximates ranks via pairwise sigmoid comparisons, then computes
    Pearson correlation of soft ranks. Averaged across all proteins.

    temperature: controls the sharpness of the rank approximation.
      Lower = sharper ranks but less smooth gradients.
      0.1 works well for z-scored protein targets in [-5, 5].
    """
    n = pred.shape[0]
    # pred/target: (n_bins, n_proteins)
    # Expand for pairwise differences: (n, n, n_proteins)
    pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)
    tgt_diff  = target.unsqueeze(0) - target.unsqueeze(1)
    # Soft rank: sum of sigmoid over pairwise comparisons
    pred_rank = torch.sigmoid(pred_diff / temperature).sum(dim=0)  # (n, n_proteins)
    tgt_rank  = torch.sigmoid(tgt_diff  / temperature).sum(dim=0)
    # Pearson of soft ranks = soft Spearman
    pr = pred_rank - pred_rank.mean(dim=0, keepdim=True)
    tr = tgt_rank  - tgt_rank.mean(dim=0, keepdim=True)
    corr = (pr * tr).sum(dim=0) / (
        pr.norm(dim=0) * tr.norm(dim=0) + 1e-8
    )
    return 1.0 - corr.mean()   # minimise -> maximise correlation


def combined_loss(pred, target, mse_weight=0.5):
    """
    0.5 * MSE + 0.5 * soft-Spearman.
    MSE stabilises early training; Spearman aligns with the evaluation metric.
    mse_weight can be tuned: higher = more stable, lower = more metric-aligned.
    """
    mse  = F.mse_loss(pred, target)
    spr  = soft_spearman_loss(pred, target)
    return mse_weight * mse + (1.0 - mse_weight) * spr


# ============================================================================
# 4. Model architecture
# ============================================================================

class LightPathwayEncoder(nn.Module):
    """
    Lightweight per-bin RNA -> pathway embedding MLP, trained jointly with
    the GAT. Captures gene co-expression / biological structure that is
    independent of spatial neighbourhood - complementary to the GAT's
    spatial aggregation. Kept small (128-dim output) so it adds modest
    parameters without dominating the GAT branch.

    No Reactome masking here (that requires downloading external files at
    runtime). If you want biologically-constrained masks, the BINN pathway
    masking from integrated_gat_binn_pipeline.py can be substituted for
    self.net, but the jointly-trained version without masks already captures
    meaningful co-expression structure in practice.
    """
    def __init__(self, n_genes, out_dim=128, dropout=0.2):
        super().__init__()
        hidden = min(1024, max(256, n_genes // 8))
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)


class RNAGATEncoder(nn.Module):
    """
    3-layer GATv2Conv encoder (v3 had 2 layers).
    3 layers means each node aggregates information from up to 3 hops of
    spatial neighbours - doubling the receptive field vs v3.
    BatchNorm between layers for training stability.
    """
    def __init__(self, n_genes, proj_dim, hidden_dim, out_dim, heads, dropout):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_genes, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Layer 1: proj_dim -> hidden_dim * heads
        self.gat1 = GATv2Conv(proj_dim,           hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.bn1  = nn.BatchNorm1d(hidden_dim * heads)
        # Layer 2: hidden_dim * heads -> hidden_dim * heads
        self.gat2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.bn2  = nn.BatchNorm1d(hidden_dim * heads)
        # Layer 3: hidden_dim * heads -> out_dim (single head, no concat)
        self.gat3 = GATv2Conv(hidden_dim * heads, out_dim,    heads=1,     dropout=dropout, concat=False)

        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        x = self.act(self.bn1(self.gat1(x, edge_index)))
        x = self.dropout(x)
        x = self.act(self.bn2(self.gat2(x, edge_index)))
        x = self.dropout(x)
        x = self.gat3(x, edge_index)
        return x


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention between RNA and image embeddings."""
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, rna_emb, img_emb):
        q        = rna_emb.unsqueeze(1)
        kv       = img_emb.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        return self.norm(attn_out.squeeze(1) + rna_emb)   # residual


class GATBINNPredictor(nn.Module):
    """
    Full model: 3-layer GAT (spatial RNA) + cross-attention (histology image)
    + LightPathwayEncoder (per-bin biological structure), all trained jointly.

    Forward inputs:
      x_rna      : (n_nodes, n_genes)  RNA features
      edge_index : (2, n_edges)        spatial k-NN graph
      x_img      : (n_nodes, img_dim)  Phikon/foundation model image embeddings

    Architecture:
      RNA  ->  input_proj  ->  GAT x3  ->  rna_emb  (fusion_dim)
      IMG  ->  img_proj    ─────────────>  img_emb  (fusion_dim)
      rna_emb + img_emb  ->  CrossAttn  ->  fused   (fusion_dim)
      RNA  ->  LightPathwayEncoder       ->  path_emb (pathway_dim)
      cat(fused, path_emb)  ->  MLP head  ->  (n_proteins,)
    """
    def __init__(self, n_genes, img_dim, gat_proj_dim, gat_hidden_dim, gat_heads,
                 fusion_dim, cross_attn_heads, n_proteins, dropout,
                 pathway_dim=128):
        super().__init__()
        self.rna_encoder     = RNAGATEncoder(
            n_genes, gat_proj_dim, gat_hidden_dim, fusion_dim, gat_heads, dropout
        )
        self.pathway_encoder = LightPathwayEncoder(n_genes, out_dim=pathway_dim, dropout=dropout)
        self.img_projection  = nn.Sequential(
            nn.Linear(img_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion    = CrossAttentionFusion(fusion_dim, cross_attn_heads, dropout)
        # Head takes spatial+image fusion AND per-bin pathway embedding
        head_in = fusion_dim + pathway_dim
        self.predictor = nn.Sequential(
            nn.Linear(head_in, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, n_proteins),
        )

    def forward(self, x_rna, edge_index, x_img):
        rna_emb  = self.rna_encoder(x_rna, edge_index)   # spatial context
        path_emb = self.pathway_encoder(x_rna)            # biological structure
        img_emb  = self.img_projection(x_img)             # histology
        fused    = self.fusion(rna_emb, img_emb)          # spatial x image
        combined = torch.cat([fused, path_emb], dim=1)    # all three streams
        return self.predictor(combined)


def build_model_from_checkpoint(checkpoint, device):
    """Reconstruct GATBINNPredictor from a v4 checkpoint dict."""
    a = checkpoint["args"]
    model = GATBINNPredictor(
        n_genes          = checkpoint["n_genes"],
        img_dim          = checkpoint["img_dim"],
        gat_proj_dim     = a["gat_proj_dim"],
        gat_hidden_dim   = a["gat_hidden_dim"],
        gat_heads        = a["gat_heads"],
        fusion_dim       = a["fusion_dim"],
        cross_attn_heads = a["cross_attn_heads"],
        n_proteins       = checkpoint["n_proteins"],
        dropout          = a["dropout"],
        pathway_dim      = a.get("pathway_dim", DEFAULT_PATHWAY_DIM),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ============================================================================
# 5. Graph building + inference
# ============================================================================

def build_graph(rna_adata, pro_adata, img_embeddings, img_index, k_neighbors,
                n_proteins_dummy=None):
    """
    Build a PyG Data object for one split.
    pro_adata may be None for label-free inference; n_proteins_dummy must
    then give the model output width for the dummy y tensor.
    """
    img_aligned, valid_mask = align_image_embeddings(
        rna_adata.obs_names, img_embeddings, img_index
    )
    if not valid_mask.all():
        rna_adata = rna_adata[valid_mask].copy()
        if pro_adata is not None:
            pro_adata = pro_adata[valid_mask].copy()

    coords     = rna_adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values
    edge_index = build_spatial_knn_graph(coords, k_neighbors)

    X = rna_adata.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    X = X.astype(np.float32, copy=False)

    if pro_adata is not None:
        Y = pro_adata.X
        Y = np.asarray(Y.todense()) if hasattr(Y, "todense") else np.asarray(Y)
        Y = Y.astype(np.float32, copy=False)
        n_proteins = Y.shape[1]
    else:
        if n_proteins_dummy is None:
            raise ValueError("pro_adata is None - pass n_proteins_dummy")
        Y = np.zeros((X.shape[0], n_proteins_dummy), dtype=np.float32)
        n_proteins = n_proteins_dummy

    data       = Data(x=torch.from_numpy(X), edge_index=edge_index, y=torch.from_numpy(Y))
    data.img_x = torch.from_numpy(img_aligned)
    # Store obs_names as a separate attribute (not on Data to avoid PyG batching it)
    obs_names  = rna_adata.obs_names.tolist()
    return data, obs_names, X.shape[1], n_proteins


def mini_batch_inference(model, graph_data, num_neighbors, batch_size, device):
    """Full forward pass over all nodes via NeighborLoader, no gradients."""
    loader   = NeighborLoader(
        graph_data, num_neighbors=num_neighbors, batch_size=batch_size, shuffle=False
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
        raise RuntimeError(f"{(~filled).sum()} nodes never covered during inference")
    return all_preds


# ============================================================================
# 6. Spearman accuracy helpers
# ============================================================================

def spearman_per_protein(preds, targets, protein_names):
    return pd.Series(
        {name: spearmanr(preds[:, i], targets[:, i]).correlation
         for i, name in enumerate(protein_names)},
        name="spearman_correlation"
    )


def print_accuracy_report(title, corr_series, log_file=None):
    log("=" * 65, log_file)
    log(title, log_file)
    log("=" * 65, log_file)
    log(f"  Mean SCC   : {corr_series.mean():.4f}", log_file)
    log(f"  Median SCC : {corr_series.median():.4f}", log_file)
    top5 = corr_series.sort_values(ascending=False).head(5)
    bot5 = corr_series.sort_values(ascending=False).tail(5)
    log("  Top 5:", log_file)
    for name, val in top5.items():
        log(f"    {name:<20s} {val:.4f}", log_file)
    log("  Bottom 5:", log_file)
    for name, val in bot5.items():
        log(f"    {name:<20s} {val:.4f}", log_file)
    log("=" * 65, log_file)


# ============================================================================
# 7. Optional do-no-harm alpha tournament (post-hoc, val only)
# ============================================================================

def run_do_no_harm_tournament(P_model_val, P_aux_val, Y_val, protein_names,
                               alphas=BLEND_ALPHAS, log_file=None):
    """
    Per-protein alpha search: blend = alpha*P_model + (1-alpha)*P_aux.
    A protein switches from alpha=1.0 ONLY if a blend strictly improves
    Spearman on val. Frozen alphas are returned for later application to
    test predictions - no ground truth needed at test time.
    """
    baseline_corr = {
        name: spearmanr(P_model_val[:, i], Y_val[:, i]).correlation
        for i, name in enumerate(protein_names)
    }
    alpha_map, boosted_corr, selection = {}, {}, {}

    for i, name in enumerate(protein_names):
        base      = baseline_corr[name]
        best_a    = 1.0
        best_corr = base if base == base else -np.inf   # NaN-safe

        for a in alphas:
            blend = a * P_model_val[:, i] + (1.0 - a) * P_aux_val[:, i]
            c     = spearmanr(blend, Y_val[:, i]).correlation
            if c is not None and c > best_corr and (base != base or c > base):
                best_corr = c
                best_a    = a

        alpha_map[name]    = best_a
        boosted_corr[name] = best_corr if best_corr != -np.inf else base
        selection[name]    = "GAT-only" if best_a == 1.0 else f"blend alpha={best_a}"

    report = pd.DataFrame({
        "protein":          protein_names,
        "baseline_scc":     [baseline_corr[p] for p in protein_names],
        "frozen_alpha":     [alpha_map[p]      for p in protein_names],
        "boosted_scc":      [boosted_corr[p]   for p in protein_names],
        "selection":        [selection[p]       for p in protein_names],
    })
    report["delta"] = report["boosted_scc"] - report["baseline_scc"]
    report = report.sort_values("boosted_scc", ascending=False).reset_index(drop=True)

    n_blended = sum(1 for p in protein_names if alpha_map[p] < 1.0)
    log(f"  {n_blended}/{len(protein_names)} proteins improved by blending", log_file)
    return alpha_map, report


def apply_frozen_alphas(P_model, P_aux, alpha_map, protein_names):
    """Apply frozen per-protein alphas to any split (no ground truth needed)."""
    P_blend = np.zeros_like(P_model)
    for i, name in enumerate(protein_names):
        a = alpha_map.get(name, 1.0)
        P_blend[:, i] = a * P_model[:, i] + (1.0 - a) * P_aux[:, i]
    return P_blend


# ============================================================================
# 8. LR schedule with linear warmup
# ============================================================================

def get_lr_with_warmup(optimizer, epoch, warmup_epochs, base_lr):
    """Linear warmup for the first warmup_epochs, then cosine handled by scheduler."""
    if epoch < warmup_epochs:
        scale = (epoch + 1) / max(warmup_epochs, 1)
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr * scale


# ============================================================================
# 9. TRAIN
# ============================================================================

def run_train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "training_log.txt")
    open(log_file, "w").close()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    log(f"[TRAIN v4] Device: {device}", log_file)
    log(f"  Loss: 0.5*MSE + 0.5*SoftSpearman | Optimizer: AdamW + CosineAnnealingLR "
        f"+ {args.warmup_epochs}-epoch linear warmup", log_file)
    log(f"  GAT: 3-layer GATv2Conv | PathwayEncoder: jointly-trained {args.pathway_dim}-dim MLP",
        log_file)

    # ---- Data loading ----
    log(f"Loading train RNA:     {args.train_rna}", log_file)
    train_rna = load_adata(args.train_rna)
    log(f"Loading train protein: {args.train_pro}", log_file)
    train_pro = load_adata(args.train_pro)
    if not (train_rna.obs_names == train_pro.obs_names).all():
        train_pro = train_pro[train_rna.obs_names].copy()
    protein_names = train_pro.var_names.tolist()
    log(f"  Train bins: {train_rna.n_obs:,}  |  Proteins: {len(protein_names)}", log_file)

    has_val = bool(args.val_rna and args.val_pro)
    if has_val:
        log(f"Loading val RNA:     {args.val_rna}", log_file)
        val_rna = load_adata(args.val_rna)
        log(f"Loading val protein: {args.val_pro}", log_file)
        val_pro = load_adata(args.val_pro)
        if not (val_rna.obs_names == val_pro.obs_names).all():
            val_pro = val_pro[val_rna.obs_names].copy()
        log(f"  Val bins: {val_rna.n_obs:,}", log_file)
    else:
        log("  No val set - training to fixed n_epochs, no early stopping.", log_file)

    # ---- Preprocessing ----
    if not args.skip_preprocessing:
        log("Preprocessing RNA (normalize_total + log1p, per-split, no leakage)", log_file)
        train_rna = preprocess_rna(train_rna)
        if has_val:
            val_rna = preprocess_rna(val_rna)

        log(f"Preprocessing protein (arcsinh cofactor={args.arcsinh_cofactor} + "
            f"z-score fit on TRAIN only + clip[{args.protein_clip_min}, {args.protein_clip_max}])",
            log_file)
        marker_mean, marker_std = fit_protein_transform(train_pro, args.arcsinh_cofactor)
        train_pro = apply_protein_transform(train_pro, args.arcsinh_cofactor,
                                            marker_mean, marker_std,
                                            args.protein_clip_min, args.protein_clip_max)
        if has_val:
            val_pro = apply_protein_transform(val_pro, args.arcsinh_cofactor,
                                              marker_mean, marker_std,
                                              args.protein_clip_min, args.protein_clip_max)
    else:
        log("--skip_preprocessing: assuming inputs are already normalised.", log_file)
        marker_mean = marker_std = None

    # ---- Image embeddings ----
    log(f"Loading image embeddings: {args.image_embeddings}", log_file)
    img_embeddings = np.load(args.image_embeddings)
    img_index      = pd.read_csv(args.image_index)

    # ---- Build graphs ----
    log(f"Building train graph (k={args.k_neighbors})", log_file)
    train_graph, train_obs, n_genes, n_proteins = build_graph(
        train_rna, train_pro, img_embeddings, img_index, args.k_neighbors
    )
    log(f"  Train graph: {train_graph.num_nodes:,} nodes, "
        f"{train_graph.edge_index.shape[1]:,} directed edges", log_file)
    del train_rna, train_pro
    gc.collect()

    if has_val:
        log(f"Building val graph (k={args.k_neighbors})", log_file)
        val_graph, val_obs, _, _ = build_graph(
            val_rna, val_pro, img_embeddings, img_index, args.k_neighbors
        )
        log(f"  Val graph: {val_graph.num_nodes:,} nodes, "
            f"{val_graph.edge_index.shape[1]:,} directed edges", log_file)
        del val_rna, val_pro
        gc.collect()

    # ---- Model ----
    train_loader = NeighborLoader(
        train_graph, num_neighbors=args.num_neighbors,
        batch_size=args.batch_size, shuffle=True,
    )
    n_batches = (train_graph.num_nodes + args.batch_size - 1) // args.batch_size
    log(f"NeighborLoader: num_neighbors={args.num_neighbors}, "
        f"batch_size={args.batch_size}, ~{n_batches} batches/epoch", log_file)

    model = GATBINNPredictor(
        n_genes=n_genes, img_dim=img_embeddings.shape[1],
        gat_proj_dim=args.gat_proj_dim, gat_hidden_dim=args.gat_hidden_dim,
        gat_heads=args.gat_heads, fusion_dim=args.fusion_dim,
        cross_attn_heads=args.cross_attn_heads, n_proteins=n_proteins,
        dropout=args.dropout, pathway_dim=args.pathway_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {n_params:,}", log_file)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Cosine decay after warmup - T_max covers the post-warmup epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(args.n_epochs - args.warmup_epochs, 1),
        eta_min=args.lr * 0.01,
    )

    best_val_loss  = float("inf")
    best_val_corr  = -float("inf")
    best_epoch     = -1
    best_state     = None
    no_improve     = 0
    stopped_early  = False
    avg_loss       = None

    log(f"Starting training for up to {args.n_epochs} epochs "
        f"(patience={args.patience if has_val else 'N/A'})", log_file)
    t0 = time.time()

    for epoch in range(args.n_epochs):
        # Linear warmup
        if epoch < args.warmup_epochs:
            get_lr_with_warmup(optimizer, epoch, args.warmup_epochs, args.lr)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        model.train()
        epoch_loss, n_done = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred      = model(batch.x, batch.edge_index, batch.img_x)
            seed_pred = pred[:batch.batch_size]
            seed_y    = batch.y[:batch.batch_size]
            loss      = combined_loss(seed_pred, seed_y, mse_weight=args.mse_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_done     += 1
        avg_loss = epoch_loss / n_done

        if has_val and ((epoch + 1) % args.eval_every == 0 or epoch == args.n_epochs - 1):
            val_preds  = mini_batch_inference(
                model, val_graph, args.num_neighbors, args.inference_batch_size, device
            )
            val_y      = val_graph.y.numpy()
            val_loss   = float(np.mean((val_preds - val_y) ** 2))
            mean_corr  = np.mean([
                spearmanr(val_preds[:, i], val_y[:, i]).correlation
                for i in range(val_y.shape[1])
            ])

            # Early stopping on val MSE (Spearman reported for monitoring)
            improved = val_loss < (best_val_loss - args.min_delta)
            marker   = ""
            if improved:
                best_val_loss = val_loss
                best_val_corr = mean_corr
                best_epoch    = epoch + 1
                best_state    = {k: v.detach().cpu().clone()
                                 for k, v in model.state_dict().items()}
                no_improve    = 0
                marker        = " *best*"
            else:
                no_improve += 1

            log(f"Epoch {epoch+1:>4d}/{args.n_epochs} | "
                f"lr={current_lr:.2e} | "
                f"train={avg_loss:.4f} | "
                f"val_mse={val_loss:.4f} | "
                f"val_scc={mean_corr:.4f}{marker} | "
                f"{resource_status_str(device)}", log_file)

            if no_improve >= args.patience:
                log(f"Early stopping at epoch {epoch+1} "
                    f"(best val MSE={best_val_loss:.4f}, SCC={best_val_corr:.4f} "
                    f"at epoch {best_epoch})", log_file)
                stopped_early = True
                break
        else:
            log(f"Epoch {epoch+1:>4d}/{args.n_epochs} | "
                f"lr={current_lr:.2e} | "
                f"train={avg_loss:.4f} | "
                f"{resource_status_str(device)}", log_file)

    elapsed = time.time() - t0
    log(f"Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min). "
        f"Stopped early: {stopped_early}", log_file)

    # ---- Restore best weights ----
    if has_val and best_state is not None:
        model.load_state_dict(best_state)
        log(f"Restored best weights (epoch {best_epoch}, "
            f"val MSE={best_val_loss:.4f}, val SCC={best_val_corr:.4f})", log_file)

        # Final accuracy report on val split
        log("Computing final val accuracy report...", log_file)
        final_preds  = mini_batch_inference(
            model, val_graph, args.num_neighbors, args.inference_batch_size, device
        )
        val_y        = val_graph.y.numpy()
        corr_series  = spearman_per_protein(final_preds, val_y, protein_names)
        print_accuracy_report("FINAL VALIDATION ACCURACY (v4)", corr_series, log_file)

        corr_csv = os.path.join(args.output_dir, "val_accuracy_per_protein.csv")
        corr_series.sort_values(ascending=False).to_csv(corr_csv, header=True)
        log(f"Per-protein accuracy saved: {corr_csv}", log_file)
    else:
        best_epoch    = args.n_epochs
        best_val_loss = None

    # ---- Save checkpoint ----
    model_path = os.path.join(args.output_dir, "model_weights.pt")
    torch.save({
        "model_state_dict":   model.state_dict(),
        "args":               vars(args),
        "n_genes":            n_genes,
        "img_dim":            img_embeddings.shape[1],
        "n_proteins":         n_proteins,
        "protein_names":      protein_names,
        "best_epoch":         best_epoch,
        "best_val_loss":      best_val_loss,
        "best_val_scc":       best_val_corr if has_val else None,
        "final_train_loss":   avg_loss,
        "protein_marker_mean": marker_mean.tolist() if marker_mean is not None else None,
        "protein_marker_std":  marker_std.tolist()  if marker_std  is not None else None,
        "arcsinh_cofactor":   args.arcsinh_cofactor,
        "pipeline_version":   "v4",
    }, model_path)
    log(f"Checkpoint saved: {model_path}", log_file)
    log("Done.", log_file)


# ============================================================================
# 10. PREDICT
# ============================================================================

def run_predict(args):
    device = get_device()
    log(f"[PREDICT v4] Device: {device}")

    log(f"Loading checkpoint: {args.model_path}")
    checkpoint    = torch.load(args.model_path, map_location=device, weights_only=False)
    n_genes       = checkpoint["n_genes"]
    protein_names = checkpoint["protein_names"]
    version       = checkpoint.get("pipeline_version", "v3")
    log(f"  Checkpoint version : {version}")
    log(f"  n_genes            : {n_genes}")
    log(f"  n_proteins         : {checkpoint['n_proteins']}")
    log(f"  best_val_scc       : {checkpoint.get('best_val_scc')}")
    log(f"  best_epoch         : {checkpoint.get('best_epoch')}")

    if version != "v4":
        log("  WARNING: checkpoint was not saved by v4 - architecture may differ. "
            "Use gat_crossattn_pipeline_v3.py predict for v3 checkpoints.")

    model = build_model_from_checkpoint(checkpoint, device)
    log("  Model loaded and set to eval mode")

    # ---- Gene alignment ----
    log(f"Reading gene reference from {args.train_rna_reference} (backed mode)")
    train_ref       = sc.read_h5ad(args.train_rna_reference, backed="r")
    train_gene_list = train_ref.var_names.tolist()
    train_ref.file.close()
    if len(train_gene_list) != n_genes:
        raise ValueError(f"Gene count mismatch: reference has {len(train_gene_list)}, "
                         f"checkpoint expects {n_genes}")

    log(f"Loading test/val RNA: {args.valid_rna_input}")
    valid_rna     = sc.read_h5ad(args.valid_rna_input)
    valid_gene_set = set(valid_rna.var_names)
    present_genes  = [g for g in train_gene_list if g in valid_gene_set]
    log(f"  {len(present_genes)}/{n_genes} training genes found in input data")

    aligned = sp.lil_matrix((valid_rna.n_obs, n_genes), dtype=np.float32)
    valid_sub   = valid_rna[:, present_genes]
    valid_X     = valid_sub.X
    if not sp.issparse(valid_X):
        valid_X = sp.csr_matrix(valid_X)
    present_idx = [i for i, g in enumerate(train_gene_list) if g in valid_gene_set]
    aligned[:, present_idx] = valid_X
    aligned = aligned.tocsr()

    valid_rna_aligned          = sc.AnnData(X=aligned, obs=valid_rna.obs.copy())
    valid_rna_aligned.var_names = train_gene_list
    sc.pp.normalize_total(valid_rna_aligned, target_sum=1e4)
    sc.pp.log1p(valid_rna_aligned)
    log("  Applied normalize_total + log1p")

    # ---- Image embeddings ----
    log(f"Loading image embeddings: {args.valid_image_embeddings}")
    img_embeddings = np.load(args.valid_image_embeddings)
    img_index      = pd.read_csv(args.valid_image_index)

    # ---- Build inference graph ----
    log(f"Building spatial k-NN graph (k={args.k_neighbors})")
    img_aligned, valid_mask = align_image_embeddings(
        valid_rna.obs_names, img_embeddings, img_index
    )
    if not valid_mask.all():
        valid_rna_aligned = valid_rna_aligned[valid_mask].copy()
        valid_rna         = valid_rna[valid_mask].copy()

    coords     = valid_rna.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values
    edge_index = build_spatial_knn_graph(coords, args.k_neighbors)

    X_valid = valid_rna_aligned.X
    X_valid = np.asarray(X_valid.todense()) if sp.issparse(X_valid) else np.asarray(X_valid)
    X_valid = X_valid.astype(np.float32)

    dummy_y    = torch.zeros((X_valid.shape[0], len(protein_names)), dtype=torch.float32)
    infer_graph          = Data(
        x=torch.from_numpy(X_valid), edge_index=edge_index, y=dummy_y
    )
    infer_graph.img_x    = torch.from_numpy(img_aligned)

    log(f"Running mini-batch inference on {valid_rna.n_obs:,} bins "
        f"(num_neighbors={args.num_neighbors}, batch_size={args.inference_batch_size})")
    preds = mini_batch_inference(
        model, infer_graph, args.num_neighbors, args.inference_batch_size, device
    )
    log("Inference complete")

    # ---- Optional inverse transform ----
    if args.inverse_transform:
        if not args.train_pro_raw:
            raise ValueError("--inverse_transform requires --train_pro_raw")
        log(f"Applying inverse transform from {args.train_pro_raw}")
        pro_raw     = sc.read_h5ad(args.train_pro_raw)
        X_raw       = pro_raw.X
        X_raw       = np.asarray(X_raw.todense()) if hasattr(X_raw, "todense") else np.asarray(X_raw)
        cofactor    = args.arcsinh_cofactor
        X_arc       = np.arcsinh(X_raw / cofactor)
        m_mean      = X_arc.mean(axis=0)
        m_std       = X_arc.std(axis=0)
        name_to_idx = {n: i for i, n in enumerate(pro_raw.var_names.tolist())}
        reorder     = [name_to_idx[n] for n in protein_names]
        m_mean      = m_mean[reorder]
        m_std       = m_std[reorder]
        arc_rec     = preds * m_std + m_mean
        preds       = np.sinh(arc_rec) * cofactor
        log("Inverse transform applied - predictions in raw intensity units")

    # ---- Save output ----
    out_df = pd.DataFrame(preds, columns=protein_names, index=valid_rna.obs_names)
    out_df.index.name = "barcode"
    out_df.reset_index().to_csv(args.output_path, index=False)
    log(f"Predictions saved: {args.output_path}")
    log("Done.")


# ============================================================================
# 11. CLI
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="GAT + Pathway-fused BINN multi-modal spatial proteomics pipeline v4"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ---- train ----
    tr = sub.add_parser("train")
    tr.add_argument("--train_rna",         required=True)
    tr.add_argument("--train_pro",         required=True)
    tr.add_argument("--val_rna",           default=None)
    tr.add_argument("--val_pro",           default=None)
    tr.add_argument("--image_embeddings",  required=True)
    tr.add_argument("--image_index",       required=True)
    tr.add_argument("--output_dir",        required=True)
    # Graph
    tr.add_argument("--k_neighbors",       type=int,           default=DEFAULT_K_NEIGHBORS)
    tr.add_argument("--num_neighbors",     type=int, nargs="+", default=DEFAULT_NUM_NEIGHBORS)
    tr.add_argument("--batch_size",        type=int,           default=256)
    tr.add_argument("--inference_batch_size", type=int,        default=512)
    # Architecture
    tr.add_argument("--gat_proj_dim",      type=int,   default=1024)
    tr.add_argument("--gat_hidden_dim",    type=int,   default=256)
    tr.add_argument("--gat_heads",         type=int,   default=4)
    tr.add_argument("--fusion_dim",        type=int,   default=256)
    tr.add_argument("--cross_attn_heads",  type=int,   default=4)
    tr.add_argument("--pathway_dim",       type=int,   default=DEFAULT_PATHWAY_DIM,
                    help="Output dim of jointly-trained LightPathwayEncoder (default: 128)")
    tr.add_argument("--dropout",           type=float, default=0.3)
    # Optimiser
    tr.add_argument("--lr",                type=float, default=2e-4)
    tr.add_argument("--weight_decay",      type=float, default=1e-5)
    tr.add_argument("--mse_weight",        type=float, default=0.5,
                    help="Weight of MSE in combined loss (1-mse_weight goes to SoftSpearman)")
    tr.add_argument("--warmup_epochs",     type=int,   default=DEFAULT_WARMUP_EPOCHS)
    # Training schedule
    tr.add_argument("--n_epochs",          type=int,   default=100)
    tr.add_argument("--patience",          type=int,   default=15)
    tr.add_argument("--min_delta",         type=float, default=1e-4)
    tr.add_argument("--eval_every",        type=int,   default=1)
    tr.add_argument("--seed",              type=int,   default=42)
    # Preprocessing
    tr.add_argument("--skip_preprocessing", action="store_true")
    tr.add_argument("--arcsinh_cofactor",  type=float, default=5.0)
    tr.add_argument("--protein_clip_min",  type=float, default=-5.0)
    tr.add_argument("--protein_clip_max",  type=float, default=5.0)

    # ---- predict ----
    pr = sub.add_parser("predict")
    pr.add_argument("--model_path",              required=True)
    pr.add_argument("--valid_rna_input",         required=True)
    pr.add_argument("--train_rna_reference",     required=True)
    pr.add_argument("--valid_image_embeddings",  required=True)
    pr.add_argument("--valid_image_index",       required=True)
    pr.add_argument("--output_path",             required=True)
    pr.add_argument("--k_neighbors",             type=int,           default=DEFAULT_K_NEIGHBORS)
    pr.add_argument("--num_neighbors",           type=int, nargs="+", default=DEFAULT_NUM_NEIGHBORS)
    pr.add_argument("--inference_batch_size",    type=int,           default=512)
    pr.add_argument("--inverse_transform",       action="store_true")
    pr.add_argument("--train_pro_raw",           default=None)
    pr.add_argument("--arcsinh_cofactor",        type=float,         default=5.0)

    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "predict":
        run_predict(args)


if __name__ == "__main__":
    main()
