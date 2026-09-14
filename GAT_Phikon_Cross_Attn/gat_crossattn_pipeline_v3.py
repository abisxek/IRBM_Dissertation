"""
Multi-Modal RNA (GAT) + Histology (Phikon) Cross-Attention Pipeline (v3)
============================================================================
Same architecture as gat_crossattn_pipeline_v2.py (matches the proj1024
checkpoint that scored 0.67), but TRAIN mode now uses the buffered spatial
train/val split (from generate_split.py) for genuine held-out evaluation
during training - early stopping and best-checkpoint selection are now
possible, since the split guarantees a real physical buffer (in microns)
between every train bin and every val bin, preventing GAT message-passing
from leaking information across the split boundary.

Architecture: unchanged from v2 (RNA -> proj -> GATv2 x2 branch, Phikon
embeddings -> proj branch, single-direction cross-attention fusion, MLP
prediction head). PREDICT mode is unchanged from v2.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------

1) TRAIN with the buffered spatial split (run generate_split.py first):

    python gat_crossattn_pipeline_v3.py train \
        --train_rna data/rna_train_split.pkl \
        --train_pro data/pro_train_split.pkl \
        --val_rna data/rna_val_split.pkl \
        --val_pro data/pro_val_split.pkl \
        --image_embeddings outputs_histology/embeddings.npy \
        --image_index outputs_histology/embedding_index.csv \
        --output_dir outputs_gat_crossattn_valsplit \
        --k_neighbors 15 --num_neighbors 15 10 --batch_size 512 \
        --gat_proj_dim 1024 --gat_hidden_dim 256 --gat_heads 4 \
        --n_epochs 50 --patience 10 --eval_every 1

   --train_rna/--train_pro/--val_rna/--val_pro accept EITHER .h5ad or
   .pkl (pickled AnnData) files - auto-detected by extension. The image
   embeddings file should be the one covering the FULL original bin set
   (both train and val bins are subsets of it) - each split's bins are
   re-aligned to it by barcode automatically.

   --val_rna/--val_pro are optional: omit them to fall back to full-data
   training with no validation (same behavior as v2), useful if you ever
   want a final "use every bin" run once hyperparameters are chosen.

   PREPROCESSING IS AUTOMATIC: since generate_split.py's output .pkl
   files are RAW (unnormalized), this script now applies RNA
   (normalize_total + log1p) and protein (arcsinh + z-score + clip)
   preprocessing internally before training. The protein transform's
   mean/std are fit on the TRAIN split ONLY and then applied identically
   to val, to avoid leaking val-set statistics into the normalization.
   Pass --skip_preprocessing if your inputs are already normalized.

   RESOURCE MONITORING: each epoch's log line includes current GPU
   memory (allocated/reserved/total) and system RAM usage, so you can
   watch resource consumption directly in training_log.txt without a
   separate monitoring pane.

2) PREDICT: identical to v2, see gat_crossattn_pipeline_v2.py usage.
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
from scipy.stats import spearmanr

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
    """Returns a short 'GPU: X.XGB / RAM: X.XGB' style string for
    appending to log lines, so resource usage is visible inline in the
    training log without needing a separate monitoring pane."""
    parts = []
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        parts.append(f"GPU: {alloc:.1f}/{total:.1f}GB (reserved {reserved:.1f}GB)")
    try:
        import psutil
        ram = psutil.virtual_memory()
        used_gb = ram.used / (1024 ** 3)
        total_gb = ram.total / (1024 ** 3)
        parts.append(f"RAM: {used_gb:.1f}/{total_gb:.1f}GB ({ram.percent:.0f}%)")
    except ImportError:
        pass  # psutil not installed - skip RAM reporting rather than fail
    return " | ".join(parts) if parts else ""


def load_adata(path):
    """Load an AnnData object from either a .h5ad file or a pickled
    AnnData (.pkl), auto-detected by file extension."""
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    return sc.read_h5ad(path)


def preprocess_rna(adata):
    """Standard RNA preprocessing: normalize_total + log1p. Both are
    computed per-bin (row-wise), so applying this independently to train
    and val splits does not leak any cross-split information - unlike
    the protein z-scoring below, there are no dataset-wide statistics
    involved here."""
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def fit_protein_transform(adata_train, cofactor):
    """Computes arcsinh + per-marker mean/std STRICTLY from the training
    split. These statistics must never be computed using validation-split
    data - doing so would leak val-set information into the
    normalization itself (a subtler form of leakage than the spatial
    one generate_split.py already guards against)."""
    X = adata_train.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    X_arcsinh = np.arcsinh(X / cofactor)
    marker_mean = X_arcsinh.mean(axis=0)
    marker_std = X_arcsinh.std(axis=0)
    return marker_mean, marker_std


def apply_protein_transform(adata, cofactor, marker_mean, marker_std, clip_min, clip_max):
    """Applies a previously-fit arcsinh + z-score transform (from
    fit_protein_transform, computed on train only) to any split - train
    or val. Using the SAME fitted statistics for both splits is what
    keeps this leakage-free while still normalizing val data consistently
    with train."""
    adata = adata.copy()
    X = adata.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    X_arcsinh = np.arcsinh(X / cofactor)
    X_scaled = (X_arcsinh - marker_mean) / marker_std
    X_clipped = np.clip(X_scaled, clip_min, clip_max)
    adata.X = X_clipped.astype(np.float32)
    return adata


def build_spatial_knn_graph(coords, k):
    """Build a k-NN graph from physical (row, col) pixel coordinates,
    restricted to the bins actually passed in (e.g. only train bins, or
    only val bins) - so no edges ever cross between splits."""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    n_nodes = coords.shape[0]
    src = np.repeat(np.arange(n_nodes), k)
    dst = indices[:, 1:].flatten()
    edge_index = np.stack([src, dst], axis=0)
    edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    return torch.tensor(edge_index, dtype=torch.long)


def align_image_embeddings(obs_names, img_embeddings, img_index):
    """Subset/reorder a full-tissue image embeddings array down to just
    the bins present in obs_names (e.g. only train-split or only
    val-split barcodes), preserving obs_names' order.

    Returns (aligned_array, valid_mask) - valid_mask is True for bins
    that had a matching embedding. A handful of bins can be missing here
    even though they're present in the raw split files: generate_split.py
    filters only by in_tissue, while the embeddings were extracted from
    train_rna_processed.h5ad which additionally went through QC filtering
    (min_genes) - so a few low-quality bins dropped during QC can still
    appear in the raw split. These are excluded (not errored on) since
    they represent a negligible fraction of bins and were already
    considered low-quality by the original QC step.
    """
    img_df = pd.DataFrame(img_embeddings, index=img_index["barcode"].values)
    img_df = img_df.reindex(obs_names)
    valid_mask = ~img_df.isna().any(axis=1)
    n_missing = (~valid_mask).sum()
    if n_missing > 0:
        missing_barcodes = img_df.index[~valid_mask].tolist()
        log(f"  WARNING: {n_missing} bins have no matching image embedding "
            f"(e.g. {missing_barcodes[:5]}) - excluding them from this split "
            f"(likely bins dropped during QC when embeddings were originally "
            f"extracted, but not filtered out by generate_split.py's in_tissue-"
            f"only filter)")
    aligned = img_df[valid_mask].values.astype(np.float32)
    return aligned, valid_mask.values


# ============================================================================
# Model architecture (matches the original proj1024 checkpoint exactly)
# ============================================================================

class RNAGATEncoder(nn.Module):
    def __init__(self, n_genes, proj_dim, hidden_dim, out_dim, heads, dropout):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_genes, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gat1 = GATv2Conv(proj_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATv2Conv(hidden_dim * heads, out_dim, heads=1, dropout=dropout, concat=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        x = self.gat1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
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
        fused = self.norm(attn_out.squeeze(1) + rna_emb)
        return fused


class GATCrossAttnPredictor(nn.Module):
    def __init__(self, n_genes, img_dim, gat_proj_dim, gat_hidden_dim, gat_heads,
                 fusion_dim, cross_attn_heads, n_proteins, dropout):
        super().__init__()
        self.rna_encoder = RNAGATEncoder(n_genes, gat_proj_dim, gat_hidden_dim,
                                          fusion_dim, gat_heads, dropout)
        self.img_projection = nn.Sequential(
            nn.Linear(img_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fusion = CrossAttentionFusion(fusion_dim, cross_attn_heads, dropout)
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_proteins)
        )

    def forward(self, x_rna, edge_index, x_img):
        rna_emb = self.rna_encoder(x_rna, edge_index)
        img_emb = self.img_projection(x_img)
        fused = self.fusion(rna_emb, img_emb)
        pred = self.predictor(fused)
        return pred


def build_model_from_checkpoint(checkpoint, device):
    model_args = checkpoint["args"]
    model = GATCrossAttnPredictor(
        n_genes=checkpoint["n_genes"],
        img_dim=checkpoint["img_dim"],
        gat_proj_dim=model_args["gat_proj_dim"],
        gat_hidden_dim=model_args["gat_hidden_dim"],
        gat_heads=model_args["gat_heads"],
        fusion_dim=model_args["fusion_dim"],
        cross_attn_heads=model_args["cross_attn_heads"],
        n_proteins=checkpoint["n_proteins"],
        dropout=model_args["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ============================================================================
# Shared: build a Data graph + NeighborLoader-based mini-batch predictor
# ============================================================================

def build_graph(rna_adata, pro_adata, img_embeddings, img_index, k_neighbors):
    """Builds a PyG Data object (RNA features, protein targets, image
    embeddings, spatial graph) for a single split (train-only or
    val-only bins) - graph edges never cross into the other split since
    only this split's bins/coordinates are passed in."""
    img_aligned, valid_mask = align_image_embeddings(rna_adata.obs_names, img_embeddings, img_index)

    if not valid_mask.all():
        rna_adata = rna_adata[valid_mask].copy()
        pro_adata = pro_adata[valid_mask].copy()

    coords = rna_adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values
    edge_index = build_spatial_knn_graph(coords, k_neighbors)

    X = rna_adata.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    X = X.astype(np.float32, copy=False)

    Y = pro_adata.X
    Y = np.asarray(Y.todense()) if hasattr(Y, "todense") else np.asarray(Y)
    Y = Y.astype(np.float32, copy=False)

    data = Data(x=torch.from_numpy(X), edge_index=edge_index, y=torch.from_numpy(Y))
    data.img_x = torch.from_numpy(img_aligned)
    return data, X.shape[1], Y.shape[1]


def mini_batch_inference(model, graph_data, num_neighbors, batch_size, device):
    """Runs a full forward pass over every node in graph_data via
    mini-batch neighbor sampling (no gradients), returning predictions
    in the original node order. Used both for validation-loss evaluation
    during training and could be reused for held-out test prediction."""
    loader = NeighborLoader(
        graph_data, num_neighbors=num_neighbors, batch_size=batch_size, shuffle=False
    )
    n_nodes = graph_data.num_nodes
    n_proteins = graph_data.y.shape[1]
    all_preds = np.zeros((n_nodes, n_proteins), dtype=np.float32)
    filled = np.zeros(n_nodes, dtype=bool)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            seed_idx = batch.n_id[:batch.batch_size].cpu().numpy()
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.img_x)
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

    log(f"Loading train RNA from {args.train_rna}", log_file)
    train_rna = load_adata(args.train_rna)
    log(f"Loading train protein from {args.train_pro}", log_file)
    train_pro = load_adata(args.train_pro)
    if not (train_rna.obs_names == train_pro.obs_names).all():
        train_pro = train_pro[train_rna.obs_names].copy()
    protein_names = train_pro.var_names.tolist()
    log(f"Train bins: {train_rna.n_obs}", log_file)

    has_val = bool(args.val_rna and args.val_pro)
    if has_val:
        log(f"Loading val RNA from {args.val_rna}", log_file)
        val_rna = load_adata(args.val_rna)
        log(f"Loading val protein from {args.val_pro}", log_file)
        val_pro = load_adata(args.val_pro)
        if not (val_rna.obs_names == val_pro.obs_names).all():
            val_pro = val_pro[val_rna.obs_names].copy()
        log(f"Val bins: {val_rna.n_obs}", log_file)
    else:
        log("No --val_rna/--val_pro provided - training on train split only, "
            "no early stopping or validation metrics available.", log_file)

    if not args.skip_preprocessing:
        log(f"Preprocessing RNA: normalize_total(target_sum=1e4) + log1p "
            f"(applied independently per-split, no leakage risk)", log_file)
        train_rna = preprocess_rna(train_rna)
        if has_val:
            val_rna = preprocess_rna(val_rna)

        log(f"Preprocessing protein: arcsinh(cofactor={args.arcsinh_cofactor}) + "
            f"per-marker z-score + clip[{args.protein_clip_min}, {args.protein_clip_max}]. "
            f"Mean/std fit on TRAIN split ONLY, then applied identically to val "
            f"(fitting on combined train+val would leak val statistics into the "
            f"normalization itself).", log_file)
        marker_mean, marker_std = fit_protein_transform(train_pro, args.arcsinh_cofactor)
        train_pro = apply_protein_transform(train_pro, args.arcsinh_cofactor, marker_mean,
                                             marker_std, args.protein_clip_min, args.protein_clip_max)
        if has_val:
            val_pro = apply_protein_transform(val_pro, args.arcsinh_cofactor, marker_mean,
                                               marker_std, args.protein_clip_min, args.protein_clip_max)
        log("Preprocessing complete.", log_file)
    else:
        log("--skip_preprocessing set: assuming train_rna/train_pro/val_rna/val_pro "
            "are already normalized (e.g. previously-processed .h5ad files, not raw "
            "split .pkl files).", log_file)

    log(f"Loading image embeddings from {args.image_embeddings}", log_file)
    img_embeddings = np.load(args.image_embeddings)
    img_index = pd.read_csv(args.image_index)

    log(f"Building train graph (k={args.k_neighbors})", log_file)
    train_graph, n_genes, n_proteins = build_graph(
        train_rna, train_pro, img_embeddings, img_index, args.k_neighbors
    )
    log(f"Train graph: {train_graph.num_nodes} nodes, "
        f"{train_graph.edge_index.shape[1]} directed edges", log_file)
    del train_rna, train_pro
    gc.collect()

    if has_val:
        log(f"Building val graph (k={args.k_neighbors}) - separate from train graph, "
            f"so no edges connect train and val bins (each was already guaranteed "
            f">= buffer distance apart by generate_split.py)", log_file)
        val_graph, _, _ = build_graph(
            val_rna, val_pro, img_embeddings, img_index, args.k_neighbors
        )
        log(f"Val graph: {val_graph.num_nodes} nodes, "
            f"{val_graph.edge_index.shape[1]} directed edges", log_file)
        del val_rna, val_pro
        gc.collect()

    train_loader = NeighborLoader(
        train_graph, num_neighbors=args.num_neighbors,
        batch_size=args.batch_size, shuffle=True,
    )
    n_batches = (train_graph.num_nodes + args.batch_size - 1) // args.batch_size
    log(f"NeighborLoader: num_neighbors={args.num_neighbors}, "
        f"batch_size={args.batch_size}, ~{n_batches} batches/epoch", log_file)

    model = GATCrossAttnPredictor(
        n_genes=n_genes, img_dim=img_embeddings.shape[1],
        gat_proj_dim=args.gat_proj_dim, gat_hidden_dim=args.gat_hidden_dim,
        gat_heads=args.gat_heads, fusion_dim=args.fusion_dim,
        cross_attn_heads=args.cross_attn_heads, n_proteins=n_proteins,
        dropout=args.dropout
    ).to(device)
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", log_file)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None
    epochs_without_improvement = 0
    stopped_early = False
    avg_loss = None

    log(f"Starting mini-batch training for up to {args.n_epochs} epochs "
        f"(patience={args.patience if has_val else 'N/A - no val set'})", log_file)
    start = time.time()

    for epoch in range(args.n_epochs):
        model.train()
        epoch_loss, n_batches_done = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.img_x)
            seed_pred = pred[:batch.batch_size]
            seed_y = batch.y[:batch.batch_size]
            loss = loss_fn(seed_pred, seed_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches_done += 1
        avg_loss = epoch_loss / n_batches_done

        if has_val and ((epoch + 1) % args.eval_every == 0 or epoch == args.n_epochs - 1):
            val_preds = mini_batch_inference(
                model, val_graph, args.num_neighbors, args.inference_batch_size, device
            )
            val_y = val_graph.y.numpy()
            val_loss = float(np.mean((val_preds - val_y) ** 2))

            mean_corr = np.mean([
                spearmanr(val_preds[:, i], val_y[:, i]).correlation
                for i in range(val_y.shape[1])
            ])

            improved = val_loss < (best_val_loss - args.min_delta)
            marker = ""
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
                marker = " *best*"
            else:
                epochs_without_improvement += 1

            log(f"Epoch {epoch + 1}/{args.n_epochs} - Train Loss: {avg_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - Val MeanCorr: {mean_corr:.4f}{marker} - "
                f"{resource_status_str(device)}", log_file)

            if epochs_without_improvement >= args.patience:
                log(f"Early stopping: no improvement for {args.patience} eval rounds "
                    f"(best val loss {best_val_loss:.4f} at epoch {best_epoch})", log_file)
                stopped_early = True
                break
        else:
            log(f"Epoch {epoch + 1}/{args.n_epochs} - Train Loss: {avg_loss:.4f} "
                f"({n_batches_done} batches) - {resource_status_str(device)}", log_file)

    elapsed = time.time() - start
    log(f"Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min). "
        f"Stopped early: {stopped_early}", log_file)

    if has_val and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        log(f"Restored best model weights (epoch {best_epoch}, val loss {best_val_loss:.4f})",
            log_file)

        # ------------------------------------------------------------------
        # Final accuracy report: predicted vs ACTUAL protein values on the
        # held-out val split, using the best (restored) model. This is the
        # real answer to "how well is my model performing" - genuine ground
        # truth comparison, not a training-set metric.
        # ------------------------------------------------------------------
        log("Computing final accuracy report (predicted vs actual, val split, "
            "best model)", log_file)
        final_val_preds = mini_batch_inference(
            model, val_graph, args.num_neighbors, args.inference_batch_size, device
        )
        final_val_actual = val_graph.y.numpy()

        per_protein_corr = {}
        for i, name in enumerate(protein_names):
            corr = spearmanr(final_val_preds[:, i], final_val_actual[:, i]).correlation
            per_protein_corr[name] = corr
        corr_series = pd.Series(per_protein_corr, name="spearman_correlation").sort_values(ascending=False)

        log("=" * 60, log_file)
        log("FINAL VALIDATION ACCURACY REPORT (predicted vs actual)", log_file)
        log("=" * 60, log_file)
        log(f"Best epoch: {best_epoch}", log_file)
        log(f"Mean Spearman correlation across {len(protein_names)} proteins: "
            f"{corr_series.mean():.4f}", log_file)
        log(f"Median Spearman correlation: {corr_series.median():.4f}", log_file)
        log(f"Top 5 best-predicted proteins:\n{corr_series.head()}", log_file)
        log(f"Bottom 5 worst-predicted proteins:\n{corr_series.tail()}", log_file)
        log("=" * 60, log_file)

        corr_csv_path = os.path.join(args.output_dir, "val_accuracy_per_protein.csv")
        corr_series.to_csv(corr_csv_path, header=True)
        log(f"Saved per-protein accuracy report to {corr_csv_path}", log_file)
    elif has_val:
        log("No best checkpoint was recorded despite having a validation set - "
            "this shouldn't normally happen; skipping final accuracy report.", log_file)
    else:
        best_epoch = args.n_epochs
        best_val_loss = None
        log("No validation set used - saving final-epoch weights (not necessarily best "
            "for generalization). No accuracy report available without ground truth "
            "to compare against.", log_file)

    model_path = os.path.join(args.output_dir, "model_weights.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "n_genes": n_genes,
        "img_dim": img_embeddings.shape[1],
        "n_proteins": n_proteins,
        "protein_names": protein_names,
        "final_train_loss": avg_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "protein_marker_mean": marker_mean if not args.skip_preprocessing else None,
        "protein_marker_std": marker_std if not args.skip_preprocessing else None,
        "arcsinh_cofactor": args.arcsinh_cofactor,
    }, model_path)
    log(f"Saved model weights to {model_path}", log_file)
    log("Done.", log_file)


# ============================================================================
# PREDICT (unchanged from v2)
# ============================================================================

def run_predict(args):
    device = get_device()
    log(f"[PREDICT] Using device: {device}")

    log(f"Loading checkpoint from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    n_genes = checkpoint["n_genes"]
    protein_names = checkpoint["protein_names"]
    log(f"Checkpoint: n_genes={n_genes}, img_dim={checkpoint['img_dim']}, "
        f"n_proteins={checkpoint['n_proteins']}, "
        f"final_train_loss={checkpoint.get('final_train_loss')}, "
        f"best_val_loss={checkpoint.get('best_val_loss')}, "
        f"best_epoch={checkpoint.get('best_epoch')}")

    model = build_model_from_checkpoint(checkpoint, device)
    log("Model loaded and set to eval mode")

    log(f"Reading gene reference from {args.train_rna_reference} (backed mode)")
    train_ref = sc.read_h5ad(args.train_rna_reference, backed="r")
    train_gene_list = train_ref.var_names.tolist()
    train_ref.file.close()
    if len(train_gene_list) != n_genes:
        raise ValueError(f"Gene count mismatch: reference has {len(train_gene_list)}, "
                          f"checkpoint expects {n_genes}")

    log(f"Loading validation RNA data from {args.valid_rna_input}")
    valid_rna = sc.read_h5ad(args.valid_rna_input)
    log(f"Validation RNA shape (raw): {valid_rna.shape}")

    valid_gene_set = set(valid_rna.var_names)
    present_train_genes = [g for g in train_gene_list if g in valid_gene_set]
    log(f"{len(present_train_genes)}/{n_genes} training genes found in validation data")

    aligned = sp.lil_matrix((valid_rna.n_obs, n_genes), dtype=np.float32)
    valid_sub = valid_rna[:, present_train_genes]
    valid_X = valid_sub.X
    if not sp.issparse(valid_X):
        valid_X = sp.csr_matrix(valid_X)
    present_idx = [i for i, g in enumerate(train_gene_list) if g in valid_gene_set]
    aligned[:, present_idx] = valid_X
    aligned = aligned.tocsr()

    valid_rna_aligned = sc.AnnData(X=aligned, obs=valid_rna.obs.copy())
    valid_rna_aligned.var_names = train_gene_list
    sc.pp.normalize_total(valid_rna_aligned, target_sum=1e4)
    sc.pp.log1p(valid_rna_aligned)
    log("Applied normalize_total + log1p to validation RNA data")

    X_valid = valid_rna_aligned.X
    X_valid = np.asarray(X_valid.todense()) if sp.issparse(X_valid) else np.asarray(X_valid)
    X_valid = X_valid.astype(np.float32)

    log(f"Loading validation image embeddings from {args.valid_image_embeddings}")
    img_embeddings = np.load(args.valid_image_embeddings)
    img_index = pd.read_csv(args.valid_image_index)
    if not (img_index["barcode"].values == valid_rna.obs_names.values).all():
        log("Reindexing validation image embeddings")
        img_df = pd.DataFrame(img_embeddings, index=img_index["barcode"].values)
        img_df = img_df.reindex(valid_rna.obs_names)
        if img_df.isna().any().any():
            raise ValueError("Some validation bins have no matching image embedding")
        img_embeddings = img_df.values
    img_embeddings = img_embeddings.astype(np.float32)

    log(f"Building spatial k-NN graph for validation bins (k={args.k_neighbors})")
    coords = valid_rna.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values
    edge_index = build_spatial_knn_graph(coords, args.k_neighbors)

    log(f"Running mini-batch inference on {valid_rna.n_obs} bins "
        f"(num_neighbors={args.num_neighbors}, inference_batch_size={args.inference_batch_size})")

    X_valid_cpu = torch.from_numpy(X_valid)
    X_img_cpu = torch.from_numpy(img_embeddings)
    dummy_y = torch.zeros((X_valid.shape[0], len(protein_names)), dtype=torch.float32)

    infer_graph = Data(x=X_valid_cpu, edge_index=edge_index, y=dummy_y)
    infer_graph.img_x = X_img_cpu

    preds = mini_batch_inference(model, infer_graph, args.num_neighbors,
                                  args.inference_batch_size, device)
    log("Inference complete")

    if args.inverse_transform:
        if not args.train_pro_raw:
            raise ValueError("--inverse_transform requires --train_pro_raw")
        log(f"Applying inverse transform using raw reference {args.train_pro_raw}")
        pro_raw = sc.read_h5ad(args.train_pro_raw)
        X_raw = pro_raw.X
        X_raw = np.asarray(X_raw.todense()) if hasattr(X_raw, "todense") else np.asarray(X_raw)

        cofactor = args.arcsinh_cofactor
        X_arcsinh = np.arcsinh(X_raw / cofactor)
        marker_mean = X_arcsinh.mean(axis=0)
        marker_std = X_arcsinh.std(axis=0)

        raw_protein_names = pro_raw.var_names.tolist()
        name_to_idx = {n: i for i, n in enumerate(raw_protein_names)}
        reorder = [name_to_idx[n] for n in protein_names]
        marker_mean = marker_mean[reorder]
        marker_std = marker_std[reorder]

        arcsinh_recovered = preds * marker_std + marker_mean
        preds = np.sinh(arcsinh_recovered) * cofactor
        log("Inverse transform applied - predictions now in raw protein intensity units")
    else:
        log("NOTE: predictions are in arcsinh + z-scored + clipped space")

    log("Building output dataframe")
    if "array_row" not in valid_rna.obs.columns or "array_col" not in valid_rna.obs.columns:
        raise ValueError("Expected 'array_row'/'array_col' columns not found")
    out_df = pd.DataFrame({
        "barcode": valid_rna.obs_names,
        "pxl_row_in_fullres": valid_rna.obs["array_row"].values,
        "pxl_col_in_fullres": valid_rna.obs["array_col"].values,
    })
    pred_df = pd.DataFrame(preds, columns=protein_names, index=valid_rna.obs_names)
    out_df = pd.concat([out_df.set_index("barcode"), pred_df], axis=1).reset_index().rename(
        columns={"index": "barcode"}
    )
    out_df.to_csv(args.output_path, index=False)
    log(f"Saved predictions to {args.output_path}")
    log("Done.")


# ============================================================================
# CLI
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Train or predict with the GAT (RNA) + Phikon (histology) "
                    "cross-attention multi-modal protein prediction model."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ---- train subcommand ----
    p_train = subparsers.add_parser("train", help="Train with genuine held-out validation")
    p_train.add_argument("--train_rna", type=str, required=True,
                          help="Train RNA - .h5ad or .pkl (e.g. rna_train_split.pkl)")
    p_train.add_argument("--train_pro", type=str, required=True,
                          help="Train protein - .h5ad or .pkl (e.g. pro_train_split.pkl)")
    p_train.add_argument("--val_rna", type=str, default=None,
                          help="Val RNA - .h5ad or .pkl (e.g. rna_val_split.pkl). "
                               "Omit for full-data training with no validation.")
    p_train.add_argument("--val_pro", type=str, default=None,
                          help="Val protein - .h5ad or .pkl (e.g. pro_val_split.pkl)")
    p_train.add_argument("--image_embeddings", type=str, required=True,
                          help="Embeddings covering the FULL original bin set - "
                               "train/val bins are re-aligned to it by barcode")
    p_train.add_argument("--image_index", type=str, required=True)
    p_train.add_argument("--output_dir", type=str, required=True)
    p_train.add_argument("--k_neighbors", type=int, default=15)
    p_train.add_argument("--num_neighbors", type=int, nargs="+", default=[15, 10])
    p_train.add_argument("--batch_size", type=int, default=512)
    p_train.add_argument("--inference_batch_size", type=int, default=1024,
                          help="Batch size for validation-set mini-batch inference")
    p_train.add_argument("--gat_proj_dim", type=int, default=512)
    p_train.add_argument("--gat_hidden_dim", type=int, default=256)
    p_train.add_argument("--gat_heads", type=int, default=4)
    p_train.add_argument("--fusion_dim", type=int, default=256)
    p_train.add_argument("--cross_attn_heads", type=int, default=4)
    p_train.add_argument("--dropout", type=float, default=0.4)
    p_train.add_argument("--weight_decay", type=float, default=1e-5)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--n_epochs", type=int, default=50)
    p_train.add_argument("--patience", type=int, default=10,
                          help="Early stopping patience, in evaluation rounds (only "
                               "used if --val_rna/--val_pro provided)")
    p_train.add_argument("--min_delta", type=float, default=1e-4)
    p_train.add_argument("--eval_every", type=int, default=1,
                          help="Evaluate on val set every N epochs (default: every epoch)")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--skip_preprocessing", action="store_true",
                          help="Skip RNA/protein preprocessing - use only if your input "
                               "files are ALREADY normalized (e.g. previously-processed "
                               ".h5ad files). Raw split .pkl files from generate_split.py "
                               "require preprocessing (default: preprocessing is applied).")
    p_train.add_argument("--arcsinh_cofactor", type=float, default=5.0,
                          help="Cofactor for the protein arcsinh transform (default: 5.0)")
    p_train.add_argument("--protein_clip_min", type=float, default=-5.0,
                          help="Lower clip bound for z-scored protein values (default: -5.0)")
    p_train.add_argument("--protein_clip_max", type=float, default=5.0,
                          help="Upper clip bound for z-scored protein values (default: 5.0)")

    # ---- predict subcommand (unchanged from v2) ----
    p_pred = subparsers.add_parser("predict", help="Generate predictions on new RNA-only data")
    p_pred.add_argument("--model_path", type=str, required=True)
    p_pred.add_argument("--valid_rna_input", type=str, required=True)
    p_pred.add_argument("--train_rna_reference", type=str, required=True)
    p_pred.add_argument("--valid_image_embeddings", type=str, required=True)
    p_pred.add_argument("--valid_image_index", type=str, required=True)
    p_pred.add_argument("--output_path", type=str, required=True)
    p_pred.add_argument("--k_neighbors", type=int, default=15)
    p_pred.add_argument("--num_neighbors", type=int, nargs="+", default=[15, 10])
    p_pred.add_argument("--inference_batch_size", type=int, default=1024)
    p_pred.add_argument("--inverse_transform", action="store_true")
    p_pred.add_argument("--train_pro_raw", type=str, default=None)
    p_pred.add_argument("--arcsinh_cofactor", type=float, default=5.0)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "predict":
        run_predict(args)


if __name__ == "__main__":
    main()
