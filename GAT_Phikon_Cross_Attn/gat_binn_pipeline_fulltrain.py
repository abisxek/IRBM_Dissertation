"""
GAT + Pathway-fused BINN Pipeline v4 — Full-Data Train + Portal Submission
============================================================================
This script is the submission-ready version of gat_binn_pipeline_v4.py.

Key differences from the internal-split version:
  - Trains on the FULL train_rna.h5ad + train_pro.h5ad (all ~166K bins)
  - Val set is OPTIONAL — if provided, used for early stopping only
  - After training, runs predict on test_rna.h5ad automatically
  - Output CSV matches v2 portal format exactly:
      barcode | pxl_row_in_fullres | pxl_col_in_fullres | protein_1 ... protein_44
  - Predictions saved in RAW protein intensity units (inverse transform applied)
  - Checkpoint saved after training — subsequent runs skip training entirely
    and go straight to prediction using the saved checkpoint

OOM GUARDS:
  - Chunked sparse-to-dense conversion (never materialises full RNA matrix at once)
  - Backed h5ad reading for gene reference (zero RAM copy)
  - Aggressive gc.collect() + cuda cache clearing between phases
  - Conservative default batch sizes (can be increased if GPU has headroom)
  - NeighborLoader keeps graph on CPU, moves only sampled subgraphs to GPU

USAGE
-----
First run (train + predict):

    python gat_binn_pipeline_v4_fulltrain.py \
        --train_rna  data/train_rna.h5ad \
        --train_pro  data/train_pro.h5ad \
        --test_rna   data/test_rna.h5ad \
        --image_embeddings       outputs_histology/embeddings.npy \
        --image_index            outputs_histology/embedding_index.csv \
        --test_image_embeddings  outputs_histology_test/embeddings.npy \
        --test_image_index       outputs_histology_test/embedding_index.csv \
        --output_dir             outputs_v4_fulltrain \
        --output_csv             outputs_v4_fulltrain/test_predictions.csv \
        --train_pro_raw          data/train_pro.h5ad

Subsequent runs (checkpoint found → skip training, only predict):

    python gat_binn_pipeline_v4_fulltrain.py \
        --train_rna  data/train_rna.h5ad \
        --train_pro  data/train_pro.h5ad \
        --test_rna   data/test_rna.h5ad \
        --image_embeddings       outputs_histology/embeddings.npy \
        --image_index            outputs_histology/embedding_index.csv \
        --test_image_embeddings  outputs_histology_test/embeddings.npy \
        --test_image_index       outputs_histology_test/embedding_index.csv \
        --output_dir             outputs_v4_fulltrain \
        --output_csv             outputs_v4_fulltrain/test_predictions.csv \
        --train_pro_raw          data/train_pro.h5ad

    (Identical command - the script detects the checkpoint automatically)

Force retrain even if checkpoint exists:

    Add --force_retrain to the command above.
"""

import argparse
import gc
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
# Constants
# ============================================================================

CHECKPOINT_NAME     = "model_weights_fulltrain.pt"
DEFAULT_K_NEIGHBORS = 15
DEFAULT_NUM_NEIGHBORS = [15, 10, 5]

# Favor speed over full FP32 precision for matmuls on Ampere+ GPUs (TF32).
# This is a global, harmless default and pairs well with torch.compile.
torch.set_float32_matmul_precision("high")

# NeighborLoader produces subgraphs with a different number of nodes/edges
# on every batch. torch.compile's dynamo tracer will otherwise recompile a
# fresh graph every step (or even every batch), which can make things slower
# than eager mode. Suppressing dynamo errors makes any unsupported op (e.g.
# some torch_geometric/torch_scatter kernels) silently fall back to eager
# for just that sub-graph instead of crashing the whole run.
try:
    import torch._dynamo as _dynamo
    _dynamo.config.suppress_errors = True
except Exception:
    _dynamo = None


def maybe_compile_model(model, args, device, log_file=None):
    """
    Wrap `model` with torch.compile() for faster training/inference, if
    requested and supported. Safe no-op fallback on any failure (e.g. old
    PyTorch, unsupported platform, or a compile-time error) so the script
    never hard-fails just because compilation isn't available.

    Because NeighborLoader yields variably-shaped subgraphs each batch, we
    compile with dynamic=True so dynamo builds shape-agnostic guards instead
    of recompiling for every distinct (n_nodes, n_edges) it sees.
    """
    if not getattr(args, "compile", True):
        return model
    if hasattr(model, "_orig_mod"):
        # Already compiled — avoid double-wrapping.
        return model
    if not hasattr(torch, "compile"):
        log("torch.compile unavailable (requires PyTorch 2.0+) — running eager.",
            log_file)
        return model

    try:
        compiled = torch.compile(
            model,
            mode=getattr(args, "compile_mode", "default"),
            dynamic=True,
        )
        log(f"torch.compile enabled (mode={getattr(args, 'compile_mode', 'default')}, "
            f"dynamic=True) on device={device}", log_file)
        return compiled
    except Exception as e:
        log(f"torch.compile failed to initialise ({e}) — falling back to eager.",
            log_file)
        return model


def raw_state_dict(model):
    """Return the underlying (uncompiled) state_dict, stripping any
    torch.compile OptimizedModule wrapper so checkpoints stay portable
    (loadable regardless of whether the loading run uses --compile)."""
    return (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict()


def save_checkpoint(path, model, args, n_genes, img_dim, n_proteins, protein_names,
                     epoch_completed, best_epoch, best_val_loss, best_val_corr,
                     avg_loss, marker_mean, marker_std, training_complete,
                     log_file=None):
    """
    Write a checkpoint atomically (save to a .tmp file, then os.replace).
    Atomicity matters here because this is called after EVERY epoch — if the
    process is killed mid-write (e.g. the user Ctrl-C's a run that's taking
    too long), a partial/corrupt file must never land at `path`, since that
    would break the "checkpoint exists -> skip training -> predict" path on
    the next invocation.

    `training_complete=False` marks a checkpoint saved mid-training (after
    some epoch N of args.n_epochs, before the loop finished); `True` marks
    the checkpoint saved once the training loop has actually finished (and,
    if a val set was provided, best weights have been restored). Either kind
    can be loaded straight into run_predict — mid-train checkpoints just
    reflect whatever the model learned up through the last completed epoch.
    """
    tmp_path = str(path) + ".tmp"
    torch.save({
        "model_state_dict":    raw_state_dict(model),
        "args":                vars(args),
        "n_genes":             n_genes,
        "img_dim":             img_dim,
        "n_proteins":          n_proteins,
        "protein_names":       protein_names,
        "epoch_completed":     epoch_completed,
        "best_epoch":          best_epoch,
        "best_val_loss":       best_val_loss,
        "best_val_scc":        best_val_corr,
        "final_train_loss":    avg_loss,
        "marker_mean":         marker_mean.tolist(),
        "marker_std":          marker_std.tolist(),
        "arcsinh_cofactor":    args.arcsinh_cofactor,
        "pipeline_version":    "v4_fulltrain",
        "training_complete":   training_complete,
    }, tmp_path)
    os.replace(tmp_path, path)  # atomic on POSIX and on Windows (py3.3+)
    tag = "FINAL" if training_complete else f"mid-train, epoch {epoch_completed}"
    log(f"Checkpoint saved [{tag}]: {path}", log_file)


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


def free_memory(device, *tensors):
    """Delete tensors, collect garbage, clear CUDA cache."""
    for t in tensors:
        del t
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def resource_status(device):
    parts = []
    if device.type == "cuda":
        alloc    = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device)  / 1024**3
        total    = torch.cuda.get_device_properties(device).total_memory / 1024**3
        parts.append(f"GPU {alloc:.1f}/{total:.1f}GB (rsv {reserved:.1f}GB)")
    try:
        import psutil
        ram = psutil.virtual_memory()
        parts.append(f"RAM {ram.used/1024**3:.1f}/{ram.total/1024**3:.1f}GB")
    except ImportError:
        pass
    return " | ".join(parts)


def load_adata(path):
    """Load AnnData from .h5ad or .pkl — auto-detected."""
    path = str(path)
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    return sc.read_h5ad(path)


def sparse_to_dense_chunked(X, chunk_size=10000):
    """
    Convert a sparse matrix to float32 dense in chunks to avoid
    materialising the full matrix at once (OOM guard).
    Returns a numpy float32 array.
    """
    if not sp.issparse(X):
        arr = np.asarray(X)
        return arr.astype(np.float32, copy=False)
    n_rows = X.shape[0]
    out = np.empty((n_rows, X.shape[1]), dtype=np.float32)
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        chunk = X[start:end]
        out[start:end] = np.asarray(chunk.todense(), dtype=np.float32)
    return out


def build_spatial_knn_graph(coords, k):
    """Bidirectional k-NN graph from pixel coordinates. Stays on CPU."""
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    _, indices = nbrs.kneighbors(coords)
    n = coords.shape[0]
    src = np.repeat(np.arange(n), k)
    dst = indices[:, 1:].flatten()
    ei  = np.stack([src, dst], axis=0)
    ei  = np.concatenate([ei, ei[::-1]], axis=1)
    return torch.tensor(ei, dtype=torch.long)


def align_image_embeddings(obs_names, img_embeddings, img_index, log_file=None):
    """
    Align image embeddings to obs_names order.
    Returns (aligned_float32_array, valid_boolean_mask).
    Bins with no embedding are excluded with a warning.
    """
    img_df     = pd.DataFrame(img_embeddings,
                               index=img_index["barcode"].values)
    img_df     = img_df.reindex(obs_names)
    valid_mask = ~img_df.isna().any(axis=1).values
    n_miss     = (~valid_mask).sum()
    if n_miss > 0:
        log(f"  WARNING: {n_miss} bins have no image embedding — excluded", log_file)
    return img_df[valid_mask].values.astype(np.float32), valid_mask


def fit_protein_transform(adata_train, cofactor):
    """
    Fit arcsinh + z-score on TRAIN split only.
    Returns (mean, std) arrays of shape (n_proteins,).
    """
    X = sparse_to_dense_chunked(adata_train.X)
    X_arc = np.arcsinh(X / cofactor)
    return X_arc.mean(axis=0).astype(np.float32), X_arc.std(axis=0).astype(np.float32)


def apply_protein_transform(adata, cofactor, marker_mean, marker_std,
                             clip_min, clip_max):
    """Apply pre-fit arcsinh + z-score + clip. Returns new AnnData."""
    adata = adata.copy()
    X     = sparse_to_dense_chunked(adata.X)
    X     = np.arcsinh(X / cofactor)
    X     = (X - marker_mean) / (marker_std + 1e-8)
    X     = np.clip(X, clip_min, clip_max).astype(np.float32)
    adata.X = X
    return adata


# ============================================================================
# Model architecture (v4)
# ============================================================================

class LightPathwayEncoder(nn.Module):
    """
    Per-bin RNA -> biological pathway embedding MLP.
    Trained jointly with the GAT — captures gene co-expression structure
    that is independent of spatial neighbourhood. Kept small (~128-dim)
    so it complements without dominating the GAT branch.
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
    3-layer GATv2Conv spatial encoder.
    BatchNorm + GELU between layers for stability.
    """
    def __init__(self, n_genes, proj_dim, hidden_dim, out_dim, heads, dropout):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_genes, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gat1 = GATv2Conv(proj_dim,           hidden_dim, heads=heads,
                               dropout=dropout, concat=True)
        self.bn1  = nn.BatchNorm1d(hidden_dim * heads)
        self.gat2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads,
                               dropout=dropout, concat=True)
        self.bn2  = nn.BatchNorm1d(hidden_dim * heads)
        self.gat3 = GATv2Conv(hidden_dim * heads, out_dim,    heads=1,
                               dropout=dropout, concat=False)
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
    """RNA (query) attends over image (key/value) with residual + LayerNorm."""
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, rna_emb, img_emb):
        q   = rna_emb.unsqueeze(1)
        kv  = img_emb.unsqueeze(1)
        out, _ = self.attn(q, kv, kv)
        return self.norm(out.squeeze(1) + rna_emb)


class GATBINNPredictor(nn.Module):
    """
    Three-stream model:
      Stream 1 — RNAGATEncoder   : spatial RNA via 3-layer GATv2
      Stream 2 — CrossAttnFusion : Phikon histology fused into RNA stream
      Stream 3 — LightPathwayEncoder : per-bin biological pathway structure
    All three streams trained jointly end-to-end.
    """
    def __init__(self, n_genes, img_dim, gat_proj_dim, gat_hidden_dim,
                 gat_heads, fusion_dim, cross_attn_heads, n_proteins,
                 dropout, pathway_dim=128):
        super().__init__()
        self.rna_encoder     = RNAGATEncoder(
            n_genes, gat_proj_dim, gat_hidden_dim, fusion_dim, gat_heads, dropout
        )
        self.pathway_encoder = LightPathwayEncoder(
            n_genes, out_dim=pathway_dim, dropout=dropout
        )
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion    = CrossAttentionFusion(fusion_dim, cross_attn_heads, dropout)
        head_in        = fusion_dim + pathway_dim
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
        rna_emb  = self.rna_encoder(x_rna, edge_index)
        path_emb = self.pathway_encoder(x_rna)
        img_emb  = self.img_proj(x_img)
        fused    = self.fusion(rna_emb, img_emb)
        return self.predictor(torch.cat([fused, path_emb], dim=1))


def build_model(checkpoint, device, args=None, log_file=None):
    """Reconstruct model from a saved checkpoint dict.

    `args` (the CURRENT run's CLI args, not the checkpoint's) controls
    whether the reconstructed model is wrapped with torch.compile — the
    checkpoint's state_dict is always saved in raw (uncompiled) form so it
    loads cleanly either way.
    """
    a = checkpoint["args"]
    m = GATBINNPredictor(
        n_genes          = checkpoint["n_genes"],
        img_dim          = checkpoint["img_dim"],
        gat_proj_dim     = a["gat_proj_dim"],
        gat_hidden_dim   = a["gat_hidden_dim"],
        gat_heads        = a["gat_heads"],
        fusion_dim       = a["fusion_dim"],
        cross_attn_heads = a["cross_attn_heads"],
        n_proteins       = checkpoint["n_proteins"],
        dropout          = a["dropout"],
        pathway_dim      = a.get("pathway_dim", 128),
    ).to(device)
    m.load_state_dict(checkpoint["model_state_dict"])
    if args is not None:
        m = maybe_compile_model(m, args, device, log_file)
    return m


# ============================================================================
# Loss
# ============================================================================

def combined_loss(pred, target, mse_weight=0.5, temperature=0.1):
    """
    0.5 * MSE  +  0.5 * soft-Spearman.
    MSE stabilises early training; Spearman aligns with evaluation metric.
    soft_spearman is O(n^2) per batch — keep batch_size <= 256 for speed.
    If mse_weight=1.0, pure MSE is used (faster, no Spearman term).
    """
    mse = F.mse_loss(pred, target)
    if mse_weight >= 1.0:
        return mse
    # Soft Spearman via pairwise sigmoid ranks
    pd_ = pred.unsqueeze(0)   - pred.unsqueeze(1)    # (n,n,p)
    td_ = target.unsqueeze(0) - target.unsqueeze(1)
    pr  = torch.sigmoid(pd_ / temperature).sum(0)    # (n,p)
    tr  = torch.sigmoid(td_ / temperature).sum(0)
    pr  = pr - pr.mean(0, keepdim=True)
    tr  = tr - tr.mean(0, keepdim=True)
    corr = (pr * tr).sum(0) / (pr.norm(0) * tr.norm(0) + 1e-8)
    spr  = 1.0 - corr.mean()
    return mse_weight * mse + (1.0 - mse_weight) * spr


# ============================================================================
# Graph builder
# ============================================================================

def build_graph(rna_adata, pro_adata, img_embeddings, img_index,
                k_neighbors, n_proteins_dummy=None, log_file=None):
    """
    Build a CPU-resident PyG Data object.
    pro_adata may be None for label-free inference (pass n_proteins_dummy).
    Returns (Data, obs_names_list, n_genes, n_proteins).
    """
    img_aligned, valid_mask = align_image_embeddings(
        rna_adata.obs_names, img_embeddings, img_index, log_file
    )
    if not valid_mask.all():
        rna_adata = rna_adata[valid_mask].copy()
        if pro_adata is not None:
            pro_adata = pro_adata[valid_mask].copy()

    coords     = rna_adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values.astype(np.float32)
    edge_index = build_spatial_knn_graph(coords, k_neighbors)

    log(f"  Converting RNA to dense (chunked, OOM-safe)...", log_file)
    X = sparse_to_dense_chunked(rna_adata.X)   # (n, n_genes) float32

    if pro_adata is not None:
        Y = sparse_to_dense_chunked(pro_adata.X)
        n_proteins = Y.shape[1]
    else:
        n_proteins = n_proteins_dummy
        Y = np.zeros((X.shape[0], n_proteins), dtype=np.float32)

    data       = Data(
        x          = torch.from_numpy(X),
        edge_index = edge_index,
        y          = torch.from_numpy(Y),
    )
    data.img_x = torch.from_numpy(img_aligned)
    obs_names  = rna_adata.obs_names.tolist()

    # Free dense arrays — they now live inside the Data object as tensors
    del X, Y, img_aligned
    gc.collect()

    return data, obs_names, rna_adata.n_vars, n_proteins


# ============================================================================
# Inference
# ============================================================================

def run_inference(model, graph_data, num_neighbors, batch_size, device, log_file=None):
    """Mini-batch inference via NeighborLoader — no gradients, OOM-safe."""
    loader    = NeighborLoader(
        graph_data, num_neighbors=num_neighbors,
        batch_size=batch_size, shuffle=False
    )
    n_nodes    = graph_data.num_nodes
    n_proteins = graph_data.y.shape[1]
    preds      = np.zeros((n_nodes, n_proteins), dtype=np.float32)
    filled     = np.zeros(n_nodes, dtype=bool)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            idx   = batch.n_id[:batch.batch_size].cpu().numpy()
            batch = batch.to(device)
            out   = model(batch.x, batch.edge_index, batch.img_x)
            preds[idx] = out[:batch.batch_size].cpu().numpy()
            filled[idx] = True
            # Free batch from GPU immediately
            del batch, out
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not filled.all():
        raise RuntimeError(f"{(~filled).sum()} bins never covered during inference")
    return preds


# ============================================================================
# Inverse transform  (z-score + arcsinh -> raw intensity)
# ============================================================================

def inverse_protein_transform(preds, marker_mean, marker_std,
                               cofactor, protein_names,
                               train_pro_raw_path):
    """
    Recover raw protein intensity from model predictions.
    Recomputes transform stats from train_pro_raw if marker_mean/std
    not available in checkpoint (backwards compat with v2/v3).
    """
    if marker_mean is None or marker_std is None:
        pro_raw    = sc.read_h5ad(train_pro_raw_path)
        X_raw      = sparse_to_dense_chunked(pro_raw.X)
        X_arc      = np.arcsinh(X_raw / cofactor)
        marker_mean = X_arc.mean(axis=0).astype(np.float32)
        marker_std  = X_arc.std(axis=0).astype(np.float32)
        raw_names   = pro_raw.var_names.tolist()
        reorder     = [raw_names.index(p) for p in protein_names]
        marker_mean = marker_mean[reorder]
        marker_std  = marker_std[reorder]
        del pro_raw, X_raw, X_arc
        gc.collect()

    arc_recovered = preds * marker_std + marker_mean
    return np.sinh(arc_recovered) * cofactor


# ============================================================================
# TRAIN
# ============================================================================

def run_train(args, device, log_file):
    log("=" * 65, log_file)
    log("PHASE 1: TRAINING on full dataset", log_file)
    log("=" * 65, log_file)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Load data ----
    log(f"Loading train RNA : {args.train_rna}", log_file)
    train_rna = load_adata(args.train_rna)
    log(f"  Shape: {train_rna.shape}", log_file)

    log(f"Loading train protein: {args.train_pro}", log_file)
    train_pro = load_adata(args.train_pro)
    if not (train_rna.obs_names == train_pro.obs_names).all():
        train_pro = train_pro[train_rna.obs_names].copy()
    protein_names = train_pro.var_names.tolist()
    log(f"  Proteins: {len(protein_names)}", log_file)

    has_val = bool(args.val_rna and args.val_pro)
    if has_val:
        log(f"Loading val RNA : {args.val_rna}", log_file)
        val_rna = load_adata(args.val_rna)
        log(f"Loading val protein: {args.val_pro}", log_file)
        val_pro = load_adata(args.val_pro)
        if not (val_rna.obs_names == val_pro.obs_names).all():
            val_pro = val_pro[val_rna.obs_names].copy()
        log(f"  Val bins: {val_rna.n_obs:,}", log_file)
    else:
        log("  No val set provided — early stopping disabled, training for "
            f"fixed {args.n_epochs} epochs.", log_file)

    # ---- Preprocessing ----
    log("Preprocessing RNA: normalize_total + log1p (per-split, no leakage)", log_file)
    sc.pp.normalize_total(train_rna, target_sum=1e4)
    sc.pp.log1p(train_rna)
    if has_val:
        sc.pp.normalize_total(val_rna, target_sum=1e4)
        sc.pp.log1p(val_rna)

    log(f"Preprocessing protein: arcsinh(cofactor={args.arcsinh_cofactor}) + "
        f"z-score (fit on train) + clip[{args.protein_clip_min}, {args.protein_clip_max}]",
        log_file)
    marker_mean, marker_std = fit_protein_transform(train_pro, args.arcsinh_cofactor)
    train_pro = apply_protein_transform(
        train_pro, args.arcsinh_cofactor, marker_mean, marker_std,
        args.protein_clip_min, args.protein_clip_max
    )
    if has_val:
        val_pro = apply_protein_transform(
            val_pro, args.arcsinh_cofactor, marker_mean, marker_std,
            args.protein_clip_min, args.protein_clip_max
        )

    # ---- Image embeddings ----
    log(f"Loading image embeddings: {args.image_embeddings}", log_file)
    img_embeddings = np.load(args.image_embeddings)
    img_index      = pd.read_csv(args.image_index)

    # ---- Build train graph ----
    log(f"Building train spatial graph (k={args.k_neighbors})", log_file)
    train_graph, train_obs, n_genes, n_proteins = build_graph(
        train_rna, train_pro, img_embeddings, img_index,
        args.k_neighbors, log_file=log_file
    )
    log(f"  Train graph: {train_graph.num_nodes:,} nodes, "
        f"{train_graph.edge_index.shape[1]:,} edges", log_file)
    del train_rna, train_pro
    free_memory(device)

    # ---- Build val graph ----
    if has_val:
        log(f"Building val spatial graph (k={args.k_neighbors})", log_file)
        val_graph, val_obs, _, _ = build_graph(
            val_rna, val_pro, img_embeddings, img_index,
            args.k_neighbors, log_file=log_file
        )
        log(f"  Val graph: {val_graph.num_nodes:,} nodes, "
            f"{val_graph.edge_index.shape[1]:,} edges", log_file)
        del val_rna, val_pro
        free_memory(device)

    del img_embeddings
    gc.collect()

    # ---- DataLoader ----
    train_loader = NeighborLoader(
        train_graph, num_neighbors=args.num_neighbors,
        batch_size=args.batch_size, shuffle=True,
    )
    n_batches = (train_graph.num_nodes + args.batch_size - 1) // args.batch_size
    log(f"NeighborLoader: num_neighbors={args.num_neighbors}, "
        f"batch_size={args.batch_size}, ~{n_batches} batches/epoch", log_file)

    # ---- Model ----
    model = GATBINNPredictor(
        n_genes          = n_genes,
        img_dim          = train_graph.img_x.shape[1],
        gat_proj_dim     = args.gat_proj_dim,
        gat_hidden_dim   = args.gat_hidden_dim,
        gat_heads        = args.gat_heads,
        fusion_dim       = args.fusion_dim,
        cross_attn_heads = args.cross_attn_heads,
        n_proteins       = n_proteins,
        dropout          = args.dropout,
        pathway_dim      = args.pathway_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {n_params:,}", log_file)
    log(f"Loss: MSE weight={args.mse_weight} "
        f"({'pure MSE' if args.mse_weight >= 1.0 else 'MSE + soft-Spearman'})", log_file)

    # ---- torch.compile (PyTorch 2.x graph capture + fusion) ----
    model = maybe_compile_model(model, args, device, log_file)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(args.n_epochs - args.warmup_epochs, 1),
        eta_min=args.lr * 0.01,
    )

    best_val_loss = float("inf")
    best_val_corr = -float("inf")
    best_epoch    = -1
    best_state    = None
    no_improve    = 0
    avg_loss      = None

    checkpoint_path = os.path.join(args.output_dir, CHECKPOINT_NAME)

    log(f"Starting training: n_epochs={args.n_epochs}, patience={args.patience}, "
        f"warmup={args.warmup_epochs}, min_delta={args.min_delta}", log_file)
    t0 = time.time()

    for epoch in range(args.n_epochs):

        # Linear warmup
        if epoch < args.warmup_epochs:
            scale = (epoch + 1) / max(args.warmup_epochs, 1)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * scale
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        model.train()
        epoch_loss, n_done = 0.0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred      = model(batch.x, batch.edge_index, batch.img_x)
            loss      = combined_loss(
                pred[:batch.batch_size], batch.y[:batch.batch_size],
                mse_weight=args.mse_weight
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_done     += 1
            del batch, pred, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

        avg_loss = epoch_loss / n_done

        # ---- Validation ----
        if has_val and ((epoch + 1) % args.eval_every == 0 or epoch == args.n_epochs - 1):
            val_preds = run_inference(
                model, val_graph, args.num_neighbors,
                args.inference_batch_size, device, log_file
            )
            val_y      = val_graph.y.numpy()
            val_loss   = float(np.mean((val_preds - val_y) ** 2))
            mean_corr  = float(np.mean([
                spearmanr(val_preds[:, i], val_y[:, i]).correlation
                for i in range(val_y.shape[1])
            ]))
            improved = val_loss < (best_val_loss - args.min_delta)
            marker   = ""
            if improved:
                best_val_loss = val_loss
                best_val_corr = mean_corr
                best_epoch    = epoch + 1
                best_state    = {k: v.detach().cpu().clone()
                                 for k, v in raw_state_dict(model).items()}
                no_improve    = 0
                marker        = " *best*"
            else:
                no_improve   += 1

            log(f"Epoch {epoch+1:>4}/{args.n_epochs} | lr={current_lr:.2e} | "
                f"train={avg_loss:.4f} | val_mse={val_loss:.4f} | "
                f"val_scc={mean_corr:.4f}{marker} | {resource_status(device)}", log_file)

            if no_improve >= args.patience:
                log(f"Early stopping at epoch {epoch+1} "
                    f"(best epoch={best_epoch}, val_mse={best_val_loss:.4f}, "
                    f"val_scc={best_val_corr:.4f})", log_file)
                break
        else:
            log(f"Epoch {epoch+1:>4}/{args.n_epochs} | lr={current_lr:.2e} | "
                f"train={avg_loss:.4f} | {resource_status(device)}", log_file)

        # ---- Per-epoch checkpoint (resilience against a killed/timed-out run) ----
        # Overwrites the same CHECKPOINT_NAME path every `checkpoint_every` epochs
        # (default: every epoch) so that if the process is killed at any point,
        # main()'s "checkpoint exists -> skip training -> predict" path already
        # has the latest completed epoch's weights to work with. Written
        # atomically (see save_checkpoint) so a mid-write kill can't corrupt it.
        if (epoch + 1) % max(args.checkpoint_every, 1) == 0 or epoch == args.n_epochs - 1:
            save_checkpoint(
                path              = checkpoint_path,
                model             = model,
                args              = args,
                n_genes           = n_genes,
                img_dim           = train_graph.img_x.shape[1],
                n_proteins        = n_proteins,
                protein_names     = protein_names,
                epoch_completed   = epoch + 1,
                best_epoch        = best_epoch if best_epoch != -1 else epoch + 1,
                best_val_loss     = best_val_loss if best_val_loss != float("inf") else None,
                best_val_corr     = best_val_corr if best_val_corr != -float("inf") else None,
                avg_loss          = avg_loss,
                marker_mean       = marker_mean,
                marker_std        = marker_std,
                training_complete = False,
                log_file          = log_file,
            )

    elapsed = time.time() - t0
    log(f"Training complete in {elapsed/60:.1f} min", log_file)

    # ---- Restore best weights ----
    if has_val and best_state is not None:
        (model._orig_mod if hasattr(model, "_orig_mod") else model).load_state_dict(best_state)
        log(f"Restored best weights from epoch {best_epoch}", log_file)
    else:
        best_epoch    = args.n_epochs
        best_val_loss = None
        best_val_corr = None

    # ---- Save final checkpoint (overwrites the last mid-train one) ----
    save_checkpoint(
        path              = checkpoint_path,
        model             = model,
        args              = args,
        n_genes           = n_genes,
        img_dim           = train_graph.img_x.shape[1],
        n_proteins        = n_proteins,
        protein_names     = protein_names,
        epoch_completed   = epoch + 1,
        best_epoch        = best_epoch,
        best_val_loss     = best_val_loss,
        best_val_corr     = best_val_corr,
        avg_loss          = avg_loss,
        marker_mean       = marker_mean,
        marker_std        = marker_std,
        training_complete = True,
        log_file          = log_file,
    )

    del train_graph
    if has_val:
        del val_graph
    free_memory(device)

    return model, checkpoint_path, protein_names, marker_mean, marker_std, n_proteins


# ============================================================================
# PREDICT
# ============================================================================

def run_predict(args, device, log_file, model, protein_names,
                marker_mean, marker_std, n_proteins):
    log("=" * 65, log_file)
    log("PHASE 2: PREDICTION on test set", log_file)
    log("=" * 65, log_file)

    log(f"Loading test RNA: {args.test_rna}", log_file)
    test_rna = sc.read_h5ad(args.test_rna)
    log(f"  Test bins: {test_rna.n_obs:,}", log_file)

    # ---- Gene alignment (same as v2) ----
    log("Aligning test genes to training gene panel...", log_file)
    # Read gene list from checkpoint args rather than reloading train_rna
    checkpoint_path = os.path.join(args.output_dir, CHECKPOINT_NAME)
    ckpt            = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    n_genes         = ckpt["n_genes"]

    # We need the exact gene list — load train_rna in backed mode (zero RAM copy)
    log(f"  Reading gene reference: {args.train_rna} (backed, no RAM copy)", log_file)
    train_ref       = sc.read_h5ad(args.train_rna, backed="r")
    train_gene_list = train_ref.var_names.tolist()
    train_ref.file.close()

    test_gene_set    = set(test_rna.var_names)
    present_genes    = [g for g in train_gene_list if g in test_gene_set]
    present_idx      = [i for i, g in enumerate(train_gene_list) if g in test_gene_set]
    log(f"  {len(present_genes)}/{n_genes} training genes found in test data", log_file)

    # Build aligned sparse matrix without materialising full dense test matrix
    log("  Building aligned gene matrix (sparse, OOM-safe)...", log_file)
    test_sub = test_rna[:, present_genes]
    X_sub    = test_sub.X
    if not sp.issparse(X_sub):
        X_sub = sp.csr_matrix(X_sub)
    aligned = sp.lil_matrix((test_rna.n_obs, n_genes), dtype=np.float32)
    aligned[:, present_idx] = X_sub
    aligned = aligned.tocsr()

    test_rna_aligned          = sc.AnnData(X=aligned, obs=test_rna.obs.copy())
    test_rna_aligned.var_names = train_gene_list
    del aligned, X_sub, test_sub
    gc.collect()

    # ---- Preprocess test RNA ----
    log("  Applying normalize_total + log1p to test RNA", log_file)
    sc.pp.normalize_total(test_rna_aligned, target_sum=1e4)
    sc.pp.log1p(test_rna_aligned)

    # ---- Image embeddings ----
    log(f"Loading test image embeddings: {args.test_image_embeddings}", log_file)
    test_img_embeddings = np.load(args.test_image_embeddings)
    test_img_index      = pd.read_csv(args.test_image_index)

    # ---- Build test graph ----
    log(f"Building test spatial graph (k={args.k_neighbors})", log_file)
    test_graph, test_obs, _, _ = build_graph(
        test_rna_aligned, None, test_img_embeddings, test_img_index,
        args.k_neighbors, n_proteins_dummy=n_proteins, log_file=log_file
    )
    log(f"  Test graph: {test_graph.num_nodes:,} nodes, "
        f"{test_graph.edge_index.shape[1]:,} edges", log_file)
    del test_rna_aligned, test_img_embeddings
    free_memory(device)

    # ---- Inference ----
    log(f"Running inference (batch_size={args.inference_batch_size})...", log_file)
    preds = run_inference(
        model, test_graph, args.num_neighbors,
        args.inference_batch_size, device, log_file
    )
    log("Inference complete", log_file)
    del test_graph
    free_memory(device)

    # ---- Inverse transform -> raw protein intensity (v2 format) ----
    log("Applying inverse transform (model space -> raw protein intensity)...", log_file)
    preds_raw = inverse_protein_transform(
        preds, marker_mean, marker_std,
        args.arcsinh_cofactor, protein_names, args.train_pro_raw
    )

    # ---- Build output CSV — exact v2 format ----
    # barcode | pxl_row_in_fullres | pxl_col_in_fullres | protein_1 ... protein_44
    log("Building output CSV (v2 portal format)...", log_file)

    # test_obs may have been filtered (bins with no image embedding excluded)
    # Reconstruct obs from original test_rna obs filtered to same mask
    img_index_barcodes = pd.read_csv(args.test_image_index)["barcode"].values
    img_df             = pd.DataFrame(
        np.load(args.test_image_embeddings),
        index=img_index_barcodes
    )
    valid_mask = ~img_df.reindex(test_rna.obs_names).isna().any(axis=1).values
    test_obs_filtered  = test_rna.obs[valid_mask]

    pred_df = pd.DataFrame(
        preds_raw,
        columns  = protein_names,
        index    = test_obs_filtered.index,
    )
    out_df = pd.DataFrame({
        "barcode":             test_obs_filtered.index,
        "pxl_row_in_fullres": test_obs_filtered["pxl_row_in_fullres"].values,
        "pxl_col_in_fullres": test_obs_filtered["pxl_col_in_fullres"].values,
    }).set_index("barcode")

    out_df = pd.concat([out_df, pred_df], axis=1).reset_index().rename(
        columns={"index": "barcode"}
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    log(f"Predictions saved: {args.output_csv}", log_file)
    log(f"  Shape: {out_df.shape}  "
        f"(rows=bins, cols=barcode+coords+{len(protein_names)} proteins)", log_file)
    log("Done.", log_file)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GAT+BINN v4 full-data train + portal prediction"
    )

    # ---- I/O ----
    parser.add_argument("--train_rna",   required=True,
                        help="Full training RNA h5ad (raw counts)")
    parser.add_argument("--train_pro",   required=True,
                        help="Full training protein h5ad (raw counts)")
    parser.add_argument("--test_rna",    required=True,
                        help="Test RNA h5ad for submission predictions")
    parser.add_argument("--train_pro_raw", default=None,
                        help="Raw train protein h5ad for inverse transform. "
                             "Defaults to --train_pro if not set.")
    parser.add_argument("--image_embeddings",      required=True,
                        help="Train Phikon embeddings .npy")
    parser.add_argument("--image_index",           required=True,
                        help="Train image embedding index CSV")
    parser.add_argument("--test_image_embeddings", required=True,
                        help="Test Phikon embeddings .npy")
    parser.add_argument("--test_image_index",      required=True,
                        help="Test image embedding index CSV")
    parser.add_argument("--output_dir",  required=True,
                        help="Directory to save checkpoint and logs")
    parser.add_argument("--output_csv",  required=True,
                        help="Path for final submission CSV")

    # ---- Optional val for early stopping ----
    parser.add_argument("--val_rna",  default=None,
                        help="Optional val RNA for early stopping")
    parser.add_argument("--val_pro",  default=None,
                        help="Optional val protein for early stopping")

    # ---- Graph ----
    parser.add_argument("--k_neighbors",     type=int,           default=DEFAULT_K_NEIGHBORS)
    parser.add_argument("--num_neighbors",   type=int, nargs="+", default=DEFAULT_NUM_NEIGHBORS,
                        help="NeighborLoader fan-out per GAT layer (one per layer, "
                             "default: 15 10 5 for 3-layer GAT)")

    # ---- Batch sizes (conservative defaults to avoid OOM) ----
    parser.add_argument("--batch_size",           type=int, default=128,
                        help="Training batch size (default 128 — conservative for OOM safety; "
                             "increase to 256 if GPU has headroom)")
    parser.add_argument("--inference_batch_size", type=int, default=256,
                        help="Inference batch size (default 256; can be larger than "
                             "training since no gradients held in memory)")

    # ---- Architecture ----
    parser.add_argument("--gat_proj_dim",     type=int,   default=1024)
    parser.add_argument("--gat_hidden_dim",   type=int,   default=256)
    parser.add_argument("--gat_heads",        type=int,   default=4)
    parser.add_argument("--fusion_dim",       type=int,   default=256)
    parser.add_argument("--cross_attn_heads", type=int,   default=4)
    parser.add_argument("--pathway_dim",      type=int,   default=128,
                        help="LightPathwayEncoder output dim (default 128)")
    parser.add_argument("--dropout",          type=float, default=0.3)

    # ---- Optimiser ----
    parser.add_argument("--lr",           type=float, default=3e-4,
                        help="Peak learning rate (default 3e-4, matches v2)")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--mse_weight",   type=float, default=1.0,
                        help="Weight of MSE in combined loss. Default 1.0 = pure MSE "
                             "(fast, stable). Set 0.5 to add soft-Spearman term "
                             "(slower due to O(n^2) pairwise computation).")

    # ---- Training schedule ----
    parser.add_argument("--n_epochs",      type=int,   default=50,
                        help="Max epochs (default 50, matching v2 spirit with headroom)")
    parser.add_argument("--patience",      type=int,   default=10,
                        help="Early stopping patience on val MSE (only active if "
                             "--val_rna/--val_pro provided)")
    parser.add_argument("--min_delta",     type=float, default=1e-4)
    parser.add_argument("--eval_every",    type=int,   default=1)
    parser.add_argument("--warmup_epochs", type=int,   default=5)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--checkpoint_every", type=int, default=1,
                        help="Save a resumable checkpoint every N epochs (default: 1, i.e. "
                             "every epoch). The checkpoint is written atomically and always "
                             "overwrites the same file, so if the process is killed at any "
                             "point, the next invocation (without --force_retrain) will skip "
                             "training and go straight to prediction using the weights from "
                             "the last completed checkpointed epoch.")

    # ---- Protein preprocessing ----
    parser.add_argument("--arcsinh_cofactor",  type=float, default=5.0)
    parser.add_argument("--protein_clip_min",  type=float, default=-5.0)
    parser.add_argument("--protein_clip_max",  type=float, default=5.0)

    # ---- Control ----
    parser.add_argument("--force_retrain", action="store_true",
                        help="Retrain even if a checkpoint already exists in output_dir")

    # ---- PyTorch 2 compilation ----
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                        help="Wrap the model with torch.compile() for faster train/inference "
                             "(PyTorch 2.0+). Enabled by default; pass --no-compile to disable "
                             "(e.g. if compilation fails on your environment or CUDA toolchain).")
    parser.add_argument("--compile_mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode. 'default' is the safest/fastest-to-compile "
                             "choice for a model with variably-shaped graph batches. "
                             "'reduce-overhead' uses CUDA graphs (needs fixed shapes — usually "
                             "not a win here since NeighborLoader batches vary in size). "
                             "'max-autotune' spends more time autotuning kernels for a bit more "
                             "throughput at the cost of much longer warmup.")

    args = parser.parse_args()

    # Default train_pro_raw to train_pro if not set
    if not args.train_pro_raw:
        args.train_pro_raw = args.train_pro

    os.makedirs(args.output_dir, exist_ok=True)
    log_file        = os.path.join(args.output_dir, "run_log.txt")
    open(log_file, "w").close()
    device          = get_device()
    checkpoint_path = os.path.join(args.output_dir, CHECKPOINT_NAME)

    log(f"GAT+BINN Pipeline v4 — Full-Data Train + Portal Submission", log_file)
    log(f"Device : {device}", log_file)
    log(f"Output : {args.output_dir}", log_file)

    # ================================================================
    # Decide: train from scratch or load existing checkpoint
    # ================================================================
    if os.path.exists(checkpoint_path) and not args.force_retrain:
        log(f"Checkpoint found: {checkpoint_path}", log_file)
        log("Skipping training — loading checkpoint for prediction.", log_file)
        log("(Pass --force_retrain to retrain from scratch.)", log_file)

        ckpt          = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model         = build_model(ckpt, device, args=args, log_file=log_file)
        model.eval()
        protein_names = ckpt["protein_names"]
        n_proteins    = ckpt["n_proteins"]
        marker_mean   = np.array(ckpt["marker_mean"], dtype=np.float32) \
                        if ckpt.get("marker_mean") else None
        marker_std    = np.array(ckpt["marker_std"],  dtype=np.float32) \
                        if ckpt.get("marker_std")  else None
        is_complete   = ckpt.get("training_complete", True)  # older checkpoints: assume complete
        n_epochs_done = ckpt.get("epoch_completed", ckpt.get("best_epoch"))
        if not is_complete:
            log(f"  NOTE: this checkpoint was saved MID-TRAINING, after epoch "
                f"{n_epochs_done}/{args.n_epochs} (the previous run did not finish). "
                f"Predictions below will use those partially-trained weights.", log_file)
        log(f"  n_genes={ckpt['n_genes']}  n_proteins={n_proteins}  "
            f"epoch_completed={n_epochs_done}  training_complete={is_complete}  "
            f"best_epoch={ckpt.get('best_epoch')}  "
            f"best_val_scc={ckpt.get('best_val_scc')}", log_file)
    else:
        if args.force_retrain and os.path.exists(checkpoint_path):
            log("--force_retrain set — ignoring existing checkpoint.", log_file)

        model, checkpoint_path, protein_names, marker_mean, marker_std, n_proteins = \
            run_train(args, device, log_file)
        model.eval()

    # ================================================================
    # Predict
    # ================================================================
    run_predict(args, device, log_file, model, protein_names,
                marker_mean, marker_std, n_proteins)


if __name__ == "__main__":
    main()
