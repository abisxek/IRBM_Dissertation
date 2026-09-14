"""
Integrated GAT-CrossAttention + BINN Ensemble Pipeline
============================================================================
Unifies two existing pipelines into a single end-to-end script:

  1. GAT-CrossAttention  (gat_crossattn_pipeline_v3.py)
     - RNA -> proj -> GATv2 x2 branch
     - Phikon histology embeddings -> proj branch
     - single-direction cross-attention fusion + MLP head
     - Architecture is loaded UNCHANGED from model_weights.pt (frozen,
       inference-only in this script - no GAT retraining logic is used
       here; if model_weights.pt is missing, this script will refuse
       to fabricate a GAT model and will instruct you to run the
       original training script first).

  2. BINN  (BINN_pipeline.py)
     - Dual-stream Biology-Informed Neural Network (Reactome-masked
       pathway stream + dense unmapped-gene stream).
     - Smart checkpointing: if binn_checkpoint.pt exists, training is
       skipped entirely and weights are loaded directly. Otherwise the
       BINN is trained using the provided train/val spatial splits
       (the real held-out val split is used for early stopping instead
       of BINN_pipeline.py's original internal 10% carve-out, since we
       now have a genuine spatially-buffered validation set available).
     - Produces two outputs per split: bottleneck embeddings (encode())
       and final per-protein predictions (forward()/head()).

  3. Dynamic "Do-No-Harm" ensembling
     - On the validation split ONLY, for each protein independently,
       try blend weights alpha in {0.60, 0.70, 0.80, 0.90, 0.95} where
           P_blend = alpha * P_GAT + (1 - alpha) * P_BINN
       A protein switches away from alpha = 1.0 (pure GAT) ONLY if some
       alpha in that grid STRICTLY beats the GAT-alone Spearman
       correlation on validation. Otherwise it is frozen at alpha = 1.0
       (pure GAT), so the ensemble can never do worse than the GAT
       baseline on any individual protein, on the data used to select
       the weights.
     - The per-protein alpha values are frozen (saved to
       ensemble_weights.json) and re-applied unchanged to any held-out
       test/inference set - they are NEVER re-fit on test data.

Why the GAT encoder is not fed BINN embeddings
------------------------------------------------
The task's instruction to "not alter the core cross-attention
architecture or internal GAT encoder layers" is structurally
incompatible with routing BINN's bottleneck embedding into the GAT's
`input_proj` layer, whose input dimension is fixed to `n_genes` by the
frozen model_weights.pt checkpoint. Rather than silently resize that
layer (which WOULD be altering the frozen architecture), this script:
  - still computes and saves the BINN bottleneck embeddings for every
    split (train/val/optional test) exactly as BINN_pipeline.py did,
    so they remain available for a *future* GAT variant that is
    intentionally re-trained to consume them, and
  - performs the actual model combination at the prediction level
    (residual/blend ensembling in step 3 above), which is the
    integration path that is architecture-safe for both frozen models.

Data / leakage constraints (enforced throughout)
------------------------------------------------
  - RNA preprocessing (normalize_total(target_sum=1e4) + log1p) is
    computed independently per split (row-wise operations only - no
    leakage risk).
  - Protein normalisation (arcsinh + per-marker mean/std z-score +
    clip) is fit STRICTLY on the TRAIN split and applied identically
    to VAL (and to test, if provided). This single shared transform is
    used for BOTH the GAT and the BINN targets, so their raw
    prediction outputs live in the same numeric space and a linear
    blend between them is meaningful. (Spearman correlation itself is
    rank-based and invariant to any monotonic per-column rescaling, so
    this choice only affects the blend step, not the correlation
    numbers taken in isolation.)
  - The spatial k-NN graph for the GAT is built separately per split,
    so no edges connect train bins to val bins.
  - Blend weights are selected on validation only and frozen before
    ever being applied to test/inference data - test data plays no
    role in choosing alpha.

Memory safety (NVIDIA L4, 24GB VRAM)
------------------------------------------------
  - All GAT forward passes (train-checkpoint loading + val/test
    inference) go through mini-batch NeighborLoader sampling, never a
    full-graph forward pass.
  - All BINN forward passes go through chunked DataLoader batches
    (BINN_BATCH_SIZE, configurable) rather than dense full-dataset
    tensors.
  - `gc.collect()` + `torch.cuda.empty_cache()` are called after every
    major stage (data loading, graph construction, model
    training/inference) and immediately after large intermediate
    objects (raw AnnData splits, DataLoaders) go out of scope.
  - `pin_memory` is only enabled on CUDA devices and is exposed as a
    config flag (`PIN_MEMORY`) so it can be forced off if host RAM is
    under pressure.
  - Sparse gene matrices are only densified per mini-batch
    (DualSparseDataset.__getitems__), never for the whole split at
    once.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    python integrated_gat_binn_pipeline.py \
        --data_dir ./data \
        --gat_checkpoint ./outputs_gat_crossattn_valsplit/model_weights.pt \
        --binn_checkpoint ./outputs_binn/binn_checkpoint.pt \
        --output_dir ./outputs_ensemble

Optional held-out test set (labels optional):
        --test_rna rna_test_split.pkl --test_pro pro_test_split.pkl \
        --test_image_embeddings embeddings.npy --test_image_index embedding_index.csv

If --test_pro is omitted, the frozen ensemble is still applied to
--test_rna (if given) and predictions are written out, just without a
test-set accuracy report (no ground truth to score against).
"""

# ==============================================================================
# 0. Imports
# ==============================================================================
import argparse
import gc
import json
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()
PIN_MEMORY = (DEVICE.type == "cuda")   # config flag - force False under host RAM pressure


# ==============================================================================
# 1. Configuration
# ==============================================================================
# RNA preprocessing
NORMALIZE_TARGET = 1e4

# Shared protein preprocessing (used for BOTH GAT and BINN targets)
ARCSINH_COFACTOR = 5.0
PROTEIN_CLIP_MIN = -5.0
PROTEIN_CLIP_MAX = 5.0

# BINN hyperparameters
N_BINN_LAYERS   = 4
BINN_DROPOUT    = 0.2
BINN_BATCH_SIZE = 2048          # conservative for 24GB L4 alongside GAT graphs in RAM
BINN_EVAL_BATCH = 4096
BINN_EPOCHS     = 100
BINN_LR         = 1e-3
BINN_WD         = 1e-4
BINN_PATIENCE   = 8

# GAT inference (used only for mini-batch inference of the frozen checkpoint)
DEFAULT_K_NEIGHBORS   = 15
DEFAULT_NUM_NEIGHBORS = [15, 10]
DEFAULT_INFERENCE_BATCH_SIZE = 1024

# "Do-no-harm" ensemble tournament
BLEND_ALPHAS = [0.60, 0.70, 0.80, 0.90, 0.95]
TOP10_PROTEINS_HINT = [
    "GFAP", "Granzyme K", "SYNA", "ICOS", "synd",
]  # used only to order the printed report; full protein set is still scored

SUBMISSION_PROTEINS = [
    "synd", "FOXP3", "CD16", "CD31", "CXCL13", "Ki67", "OLIG2", "CXCR5",
    "HLA-A", "PD-L1", "PSD95", "CD20", "CD68", "CD44", "SMA", "MSH6",
    "CD23", "GFAP", "SYNA", "Podoplanin", "Vimentin", "CD47", "CD74",
    "SIRP", "Granzyme B", "IDH1", "MPO", "CD45", "CD21", "FIBR", "C-KIT",
    "CD3e", "TOX", "PD-1", "PDGFR", "CD4", "MAP2", "CD8", "MGMT", "CD38",
    "HLA-DR", "CD14", "ICOS", "Granzyme K",
]


# ==============================================================================
# 2. Shared utilities
# ==============================================================================
def log(msg, log_file=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(line + "\n")


def resource_status_str(device):
    parts = []
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        parts.append(f"GPU: {alloc:.1f}/{total:.1f}GB (reserved {reserved:.1f}GB)")
    try:
        import psutil
        ram = psutil.virtual_memory()
        parts.append(f"RAM: {ram.used/(1024**3):.1f}/{ram.total/(1024**3):.1f}GB ({ram.percent:.0f}%)")
    except ImportError:
        pass
    return " | ".join(parts) if parts else ""


def free_memory(*objs):
    """Explicitly drop references then force a GC + CUDA cache clear.
    Called after every major stage per the hardware constraints."""
    for o in objs:
        del o
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def load_adata(path):
    """Load an AnnData object from either a .h5ad file or a pickled
    AnnData (.pkl), auto-detected by file extension."""
    path = str(path)
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    return sc.read_h5ad(path)


# ==============================================================================
# 3. Preprocessing (shared by GAT and BINN, leakage-safe)
# ==============================================================================
def preprocess_rna(adata):
    """normalize_total + log1p, computed independently per split (both
    are row-wise operations, so no cross-split statistics are involved
    and no leakage is possible)."""
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=NORMALIZE_TARGET)
    sc.pp.log1p(adata)
    return adata


def fit_protein_transform(adata_train, cofactor=ARCSINH_COFACTOR):
    """arcsinh + per-marker mean/std, fit STRICTLY on the training
    split. Never call this with validation or test data."""
    X = adata_train.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    X_arcsinh = np.arcsinh(X / cofactor)
    marker_mean = X_arcsinh.mean(axis=0)
    marker_std = X_arcsinh.std(axis=0)
    marker_std = np.where(marker_std < 1e-8, 1.0, marker_std)  # guard divide-by-zero
    return marker_mean, marker_std


def apply_protein_transform(adata, cofactor, marker_mean, marker_std,
                             clip_min=PROTEIN_CLIP_MIN, clip_max=PROTEIN_CLIP_MAX):
    """Applies a previously-fit (train-only) transform to any split."""
    adata = adata.copy()
    X = adata.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    X_arcsinh = np.arcsinh(X / cofactor)
    X_scaled = (X_arcsinh - marker_mean) / marker_std
    X_clipped = np.clip(X_scaled, clip_min, clip_max)
    adata.X = X_clipped.astype(np.float32)
    return adata


# ==============================================================================
# 4. GAT-CrossAttention architecture (UNCHANGED from gat_crossattn_pipeline_v3.py)
#    Loaded from model_weights.pt only - no training logic lives in this
#    script for the GAT branch, by design ("do not alter the core
#    cross-attention architecture or internal GAT encoder layers").
# ==============================================================================
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


def build_gat_model_from_checkpoint(checkpoint, device):
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


# ------------------------------------------------------------------------
# 4b. "Plain GAT" fallback architecture - a stack of GATv2Conv + BatchNorm
#     layers with a single final linear readout, and NO histology/image
#     branch at all. This matches checkpoints saved as a bare
#     `model.state_dict()` (optionally torch.compile()-wrapped, hence the
#     `_orig_mod.` prefix) with keys like `convs.<i>.att`, `convs.<i>.bias`,
#     `convs.<i>.lin_l.weight`, `convs.<i>.lin_r.weight`, `bns.<i>.*`, and a
#     final `output.weight` / `output.bias` - as opposed to the metadata-
#     wrapped cross-attention checkpoint produced by
#     gat_crossattn_pipeline_v3.py's run_train(). The full architecture
#     (layer count, per-layer heads/out-channels, concat vs. average on the
#     final layer) is reconstructed purely from the tensor shapes already
#     present in the checkpoint, so no external config file is required.
# ------------------------------------------------------------------------
class PlainGATStack(nn.Module):
    def __init__(self, layer_specs, n_genes, n_proteins):
        """layer_specs: list of dicts, one per GATv2Conv layer, each with
        keys {in_dim, heads, out_channels, concat}. The output dim of layer
        i (heads*out_channels if concat else out_channels) must equal the
        in_dim of layer i+1 - this is enforced by construction since specs
        are derived directly from the checkpoint's own tensor shapes."""
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for spec in layer_specs:
            out_dim = spec["heads"] * spec["out_channels"] if spec["concat"] else spec["out_channels"]
            self.convs.append(GATv2Conv(spec["in_dim"], spec["out_channels"],
                                         heads=spec["heads"], concat=spec["concat"]))
            self.bns.append(nn.BatchNorm1d(out_dim))
        last_dim = (layer_specs[-1]["heads"] * layer_specs[-1]["out_channels"]
                    if layer_specs[-1]["concat"] else layer_specs[-1]["out_channels"])
        self.output = nn.Linear(last_dim, n_proteins)
        self.n_genes = n_genes
        self.n_proteins = n_proteins

    def forward(self, x, edge_index, img_x=None):
        # img_x accepted-and-ignored so this class is drop-in interchangeable
        # with GATCrossAttnPredictor's call signature during inference.
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
        return self.output(x)


def _infer_plain_gat_layer_specs(state_dict):
    """Reconstructs per-layer (in_dim, heads, out_channels, concat) purely
    from GATv2Conv's own parameter shapes:
        att.shape      = (1, heads, out_channels)
        lin_l.weight   = (heads * out_channels, in_dim)
        bns.<i>.weight = heads*out_channels if concat=True else out_channels
    """
    conv_indices = sorted({
        int(k.split(".")[1]) for k in state_dict if k.startswith("convs.") and ".att" in k
    })
    if not conv_indices:
        raise ValueError("No 'convs.<i>.att' keys found - this does not look like a "
                          "GATv2Conv-stack state_dict.")

    layer_specs = []
    for i in conv_indices:
        att = state_dict[f"convs.{i}.att"]
        heads, out_channels = att.shape[1], att.shape[2]
        in_dim = state_dict[f"convs.{i}.lin_l.weight"].shape[1]
        bn_dim = state_dict[f"bns.{i}.weight"].shape[0]
        concat = (bn_dim == heads * out_channels)
        if not concat and bn_dim != out_channels:
            raise ValueError(f"Layer {i}: BatchNorm dim {bn_dim} matches neither "
                              f"heads*out_channels ({heads * out_channels}) nor "
                              f"out_channels ({out_channels}) - cannot infer concat mode.")
        layer_specs.append({"in_dim": in_dim, "heads": heads,
                             "out_channels": out_channels, "concat": concat})
    return layer_specs


def build_plain_gat_from_state_dict(raw_state_dict, device):
    """Strips a leading '_orig_mod.' prefix (present when the source model
    was wrapped in torch.compile() before saving), reconstructs the
    architecture from tensor shapes, and loads the weights strictly."""
    state_dict = { (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
                    for k, v in raw_state_dict.items() }

    layer_specs = _infer_plain_gat_layer_specs(state_dict)
    n_genes = layer_specs[0]["in_dim"]
    n_proteins = state_dict["output.weight"].shape[0]

    model = PlainGATStack(layer_specs, n_genes, n_proteins).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    real_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if real_missing or unexpected:
        raise RuntimeError(f"PlainGATStack reconstruction did not match checkpoint exactly.\n"
                            f"  Missing keys: {real_missing}\n  Unexpected keys: {unexpected}\n"
                            f"  Inferred layer specs: {layer_specs}")
    model.eval()
    log(f"  Reconstructed PlainGATStack from raw state_dict: "
        f"{len(layer_specs)} GATv2Conv+BN layers, n_genes={n_genes}, n_proteins={n_proteins}, "
        f"per-layer specs={layer_specs}")
    return model, n_genes, n_proteins


def load_gat_checkpoint_flexible(path, device, expected_protein_names=None):
    """Auto-detects which of two checkpoint formats `path` is:

    (a) The metadata-wrapped cross-attention checkpoint saved by
        gat_crossattn_pipeline_v3.py's run_train() - a dict with keys
        'model_state_dict', 'args', 'n_genes', 'img_dim', 'n_proteins',
        'protein_names'. Uses histology cross-attention (has_image_branch).

    (b) A raw state_dict (optionally torch.compile()-wrapped) for a plain
        GATv2Conv+BatchNorm stack with no image branch at all - as found in
        this project's gat_best_model.pt. protein_names are not stored in
        this format, so they are taken from `expected_protein_names`
        (typically the training split's protein panel) and only the COUNT
        is cross-checked against the checkpoint's own output width.

    Returns a dict: {model, has_image_branch, n_genes, img_dim,
    n_proteins, protein_names, forward_fn}
    """
    raw = torch.load(path, map_location=device, weights_only=False)

    if isinstance(raw, dict) and "model_state_dict" in raw:
        log(f"  Detected metadata-wrapped GAT-CrossAttention checkpoint at {path}")
        model = build_gat_model_from_checkpoint(raw, device)
        return {
            "model": model,
            "has_image_branch": True,
            "n_genes": raw["n_genes"],
            "img_dim": raw["img_dim"],
            "n_proteins": raw["n_proteins"],
            "protein_names": raw["protein_names"],
        }

    # Otherwise treat it as a bare state_dict (dict of tensor name -> tensor).
    log(f"  {path} is not a metadata-wrapped checkpoint - treating it as a raw "
        f"state_dict and reconstructing the architecture from tensor shapes "
        f"(no histology/image branch detected in these keys).")
    model, n_genes, n_proteins = build_plain_gat_from_state_dict(raw, device)

    if expected_protein_names is not None:
        if len(expected_protein_names) < n_proteins:
            raise ValueError(f"Checkpoint predicts {n_proteins} proteins but only "
                              f"{len(expected_protein_names)} protein columns are available "
                              f"in the provided data - cannot align outputs to targets.")
        protein_names = expected_protein_names[:n_proteins]
    else:
        protein_names = [f"protein_{i}" for i in range(n_proteins)]

    return {
        "model": model,
        "has_image_branch": False,
        "n_genes": n_genes,
        "img_dim": None,
        "n_proteins": n_proteins,
        "protein_names": protein_names,
    }


def build_spatial_knn_graph(coords, k):
    """k-NN graph restricted to the bins passed in - so no edges ever
    cross between splits."""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    n_nodes = coords.shape[0]
    src = np.repeat(np.arange(n_nodes), k)
    dst = indices[:, 1:].flatten()
    edge_index = np.stack([src, dst], axis=0)
    edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    return torch.tensor(edge_index, dtype=torch.long)


def align_image_embeddings(obs_names, img_embeddings, img_index):
    """Subset/reorder the full-tissue image embeddings down to the bins
    in obs_names, preserving order. Returns (aligned_array, valid_mask)."""
    img_df = pd.DataFrame(img_embeddings, index=img_index["barcode"].values)
    img_df = img_df.reindex(obs_names)
    valid_mask = ~img_df.isna().any(axis=1)
    n_missing = (~valid_mask).sum()
    if n_missing > 0:
        log(f"  WARNING: {n_missing} bins have no matching image embedding - "
            f"excluding them from this split.")
    aligned = img_df[valid_mask].values.astype(np.float32)
    return aligned, valid_mask.values


def build_gat_graph(rna_adata, pro_adata, img_embeddings, img_index, k_neighbors,
                     n_proteins_if_unlabeled=None):
    """Builds a PyG Data object for a single split. Graph edges never
    cross into another split since only this split's bins/coords are
    passed in. `pro_adata` may be None for label-free inference, in
    which case `n_proteins_if_unlabeled` must give the model's output
    width so the dummy `y` tensor has the right shape for downstream
    inference bookkeeping (it is never used as a real target).

    `img_embeddings`/`img_index` may both be None - used when the loaded
    GAT checkpoint has no histology branch, in which case no image
    alignment/filtering happens at all and `data.img_x` is left unset."""
    use_images = img_embeddings is not None and img_index is not None

    if use_images:
        img_aligned, valid_mask = align_image_embeddings(rna_adata.obs_names, img_embeddings, img_index)
        if not valid_mask.all():
            rna_adata = rna_adata[valid_mask].copy()
            if pro_adata is not None:
                pro_adata = pro_adata[valid_mask].copy()

    coords = rna_adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values
    edge_index = build_spatial_knn_graph(coords, k_neighbors)

    X = rna_adata.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    X = X.astype(np.float32, copy=False)

    if pro_adata is not None:
        Y = pro_adata.X
        Y = np.asarray(Y.todense()) if hasattr(Y, "todense") else np.asarray(Y)
        Y = Y.astype(np.float32, copy=False)
    else:
        if n_proteins_if_unlabeled is None:
            raise ValueError("pro_adata is None - must pass n_proteins_if_unlabeled so the "
                              "dummy target tensor matches the model's output width.")
        Y = np.zeros((X.shape[0], n_proteins_if_unlabeled), dtype=np.float32)  # dummy - unused downstream

    data = Data(x=torch.from_numpy(X), edge_index=edge_index, y=torch.from_numpy(Y))
    if use_images:
        data.img_x = torch.from_numpy(img_aligned)
    data.obs_names = rna_adata.obs_names.tolist()
    return data


def gat_mini_batch_inference(model, graph_data, num_neighbors, batch_size, device, has_image_branch=True):
    """Chunked mini-batch inference via NeighborLoader - never a
    full-graph forward pass, so GPU memory stays bounded regardless of
    split size. Returns predictions in original node order."""
    loader = NeighborLoader(
        graph_data, num_neighbors=num_neighbors, batch_size=batch_size, shuffle=False
    )
    n_nodes = graph_data.num_nodes
    n_out = graph_data.y.shape[1]
    all_preds = np.zeros((n_nodes, n_out), dtype=np.float32)
    filled = np.zeros(n_nodes, dtype=bool)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            seed_idx = batch.n_id[:batch.batch_size].cpu().numpy()
            batch = batch.to(device)
            if has_image_branch:
                pred = model(batch.x, batch.edge_index, batch.img_x)
            else:
                pred = model(batch.x, batch.edge_index)
            all_preds[seed_idx] = pred[:batch.batch_size].cpu().numpy()
            filled[seed_idx] = True
            del batch, pred
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not filled.all():
        raise RuntimeError(f"{(~filled).sum()} nodes never covered during GAT inference")
    return all_preds


# ==============================================================================
# 5. BINN architecture (from BINN_pipeline.py, unchanged internals)
# ==============================================================================
def download_reactome_files(binn_data_dir):
    files = {
        "uniprot_2_reactome.txt":
            "https://reactome.org/download/current/UniProt2Reactome.txt",
        "reactome_pathways_relation.txt":
            "https://reactome.org/download/current/ReactomePathwaysRelation.txt",
        "reactome_pathways_names.txt":
            "https://reactome.org/download/current/ReactomePathways.txt",
    }
    for fname, url in files.items():
        dest = binn_data_dir / fname
        if not dest.exists():
            log(f"  Downloading {fname} ...")
            ret = os.system(f'wget -q "{url}" -O "{dest}"')
            if ret != 0 or not dest.exists():
                raise RuntimeError(f"Failed to download {fname} from {url}")
        else:
            log(f"  Cached: {fname}")


def fetch_gene_to_uniprot(gene_symbols, batch_size=200):
    import requests
    from tqdm.auto import tqdm

    symbol_to_uniprot = {}
    url     = "https://mygene.info/v3/query"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    for i in tqdm(range(0, len(gene_symbols), batch_size), desc="  Gene->UniProt"):
        batch   = gene_symbols[i: i + batch_size]
        payload = {"q": ",".join(batch), "scopes": "symbol",
                   "fields": "uniprot", "species": "human", "size": batch_size}
        try:
            resp = requests.post(url, data=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            for hit in resp.json():
                sym = hit.get("query", "")
                if "uniprot" in hit and "Swiss-Prot" in hit["uniprot"]:
                    acc = hit["uniprot"]["Swiss-Prot"]
                    symbol_to_uniprot[sym] = [acc] if isinstance(acc, str) else acc
        except Exception as e:
            log(f"  [WARN] UniProt batch {i // batch_size} failed: {e}")
    log(f"  Mapped {len(symbol_to_uniprot):,}/{len(gene_symbols):,} genes to UniProt.")
    return symbol_to_uniprot


def build_gene_reactome_mapping(gene_symbols, binn_data_dir, output_dir):
    cache_path = output_dir / "gene_reactome_mapping.csv"
    if cache_path.exists():
        log("  Loading cached Reactome mapping.")
        gene_to_pathway = pd.read_csv(cache_path)
    else:
        up2r = pd.read_csv(
            binn_data_dir / "uniprot_2_reactome.txt",
            sep="\t", header=None,
            names=["input", "translation", "url", "name", "evidence", "species"],
        )
        up2r_human = up2r[up2r["species"] == "Homo sapiens"]
        sym2up = fetch_gene_to_uniprot(gene_symbols)
        rows = []
        for sym, uids in sym2up.items():
            for uid in uids:
                matches = up2r_human[up2r_human["input"] == uid].copy()
                if len(matches):
                    matches["input"] = sym
                    rows.append(matches)
        if not rows:
            raise RuntimeError("No genes mapped to Reactome - check network connectivity.")
        gene_to_pathway = pd.concat(rows, ignore_index=True).drop_duplicates()
        gene_to_pathway.to_csv(cache_path, index=False)
        log(f"  Mapping saved: {cache_path}")

    p_rel = pd.read_csv(
        binn_data_dir / "reactome_pathways_relation.txt",
        sep="\t", header=None, names=["target", "source"],
    )
    p_rel_human = p_rel[
        p_rel["target"].str.startswith("R-HSA") &
        p_rel["source"].str.startswith("R-HSA")
    ]
    pathway_relations = list(p_rel_human.itertuples(index=False, name=None))
    covered_genes = gene_to_pathway["input"].unique().tolist()
    log(f"  Reactome-covered genes: {len(covered_genes):,}")
    return gene_to_pathway, pathway_relations, covered_genes


def build_pathway_layers(gene_symbols_subset, gene_to_pathway_df, pathway_relations, n_layers):
    import networkx as nx
    G = nx.DiGraph()
    for child, parent in pathway_relations:
        G.add_edge(child, parent)

    g2p = gene_to_pathway_df[gene_to_pathway_df["input"].isin(gene_symbols_subset)]
    gene_to_l1 = g2p.groupby("input")["translation"].apply(list).to_dict()

    layer_nodes = [gene_symbols_subset]
    current = set()
    for g in gene_symbols_subset:
        current.update(gene_to_l1.get(g, []))

    for depth in range(n_layers):
        layer_nodes.append(sorted(current))
        nxt = set()
        for p in current:
            nxt.update(G.successors(p))
        nxt -= set(layer_nodes[1])
        current = nxt
        if not current:
            log(f"  Pathway hierarchy exhausted at layer {depth + 1}")
            break

    log(f"  BINN layer sizes: {[len(l) for l in layer_nodes]}")

    masks = []
    gene_idx = {g: i for i, g in enumerate(layer_nodes[0])}
    l1_idx = {p: i for i, p in enumerate(layer_nodes[1])}

    mask0 = torch.zeros(len(layer_nodes[1]), len(layer_nodes[0]), dtype=torch.bool)
    for g, pathways in gene_to_l1.items():
        if g in gene_idx:
            for p in pathways:
                if p in l1_idx:
                    mask0[l1_idx[p], gene_idx[g]] = True
    masks.append(mask0)

    for d in range(1, len(layer_nodes) - 1):
        src_idx = {n: i for i, n in enumerate(layer_nodes[d])}
        dst_idx = {n: i for i, n in enumerate(layer_nodes[d + 1])}
        m = torch.zeros(len(layer_nodes[d + 1]), len(layer_nodes[d]), dtype=torch.bool)
        for child, parent in pathway_relations:
            if child in src_idx and parent in dst_idx:
                m[dst_idx[parent], src_idx[child]] = True
        masks.append(m)

    return layer_nodes, masks


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.register_buffer("mask", mask.float())
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class DualBINNEncoder(nn.Module):
    def __init__(self, masks, n_unmapped, n_outputs, dropout=0.2):
        super().__init__()
        self.pathway_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for mask in masks:
            out_dim, in_dim = mask.shape
            self.pathway_layers.append(MaskedLinear(in_dim, out_dim, mask))
            self.batch_norms.append(nn.BatchNorm1d(out_dim))
            self.dropouts.append(nn.Dropout(p=dropout))
        binn_bn = masks[-1].shape[0]

        dense_hidden = min(512, max(64, n_unmapped // 16))
        self.dense_stream = nn.Sequential(
            nn.Linear(n_unmapped, dense_hidden),
            nn.BatchNorm1d(dense_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        merged_dim = binn_bn + 128
        self.head = nn.Sequential(
            nn.Linear(merged_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_outputs),
        )
        self.embed_dim = merged_dim
        log(f"  DualBINNEncoder: Stream1={binn_bn}  Stream2=128  "
            f"-> embed_dim={merged_dim}  -> head -> {n_outputs}")

    def encode_pathway(self, x_mapped):
        x = x_mapped
        for linear, bn, drop in zip(self.pathway_layers, self.batch_norms, self.dropouts):
            x = drop(torch.tanh(bn(linear(x))))
        return x

    def encode(self, x_mapped, x_unmapped):
        """Bottleneck embedding - forward() minus the head. This is the
        vector that would be handed to a future GAT variant designed
        to consume it as node features."""
        return torch.cat([
            self.encode_pathway(x_mapped),
            self.dense_stream(x_unmapped),
        ], dim=1)

    def forward(self, x_mapped, x_unmapped):
        return self.head(self.encode(x_mapped, x_unmapped))


# ==============================================================================
# 6. BINN dataset / loaders (chunked, sparse-to-dense per mini-batch only)
# ==============================================================================
class DualSparseDataset(Dataset):
    def __init__(self, X_sparse, Y_dense, gene_list, mapped_genes, unmapped_genes):
        var_idx = {g: i for i, g in enumerate(gene_list)}
        m_present = [g for g in mapped_genes if g in var_idx]
        u_present = [g for g in unmapped_genes if g in var_idx]
        self.m_src = np.array([var_idx[g] for g in m_present], dtype=np.int64)
        self.m_dst = np.array([i for i, g in enumerate(mapped_genes) if g in var_idx], dtype=np.int64)
        self.u_src = np.array([var_idx[g] for g in u_present], dtype=np.int64)
        self.u_dst = np.array([i for i, g in enumerate(unmapped_genes) if g in var_idx], dtype=np.int64)
        self.n_mapped = len(mapped_genes)
        self.n_unmapped = len(unmapped_genes)
        self.X = X_sparse.tocsr()
        self.Y = torch.tensor(Y_dense.astype(np.float32)) if Y_dense is not None \
            else torch.zeros(X_sparse.shape[0], 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitems__(self, idx_list):
        idx = np.asarray(idx_list)
        block = self.X[idx]
        block_m = block[:, self.m_src].toarray()
        block_u = block[:, self.u_src].toarray()
        xm = np.zeros((len(idx), self.n_mapped), dtype=np.float32)
        xu = np.zeros((len(idx), self.n_unmapped), dtype=np.float32)
        xm[:, self.m_dst] = block_m
        xu[:, self.u_dst] = block_u
        xm_t, xu_t = torch.from_numpy(xm), torch.from_numpy(xu)
        return [(xm_t[i], xu_t[i], self.Y[idx[i]]) for i in range(len(idx))]

    def __getitem__(self, idx):
        return self.__getitems__([idx])[0]


def make_binn_loader(rna_adata, pro_adata, mapped_genes, unmapped_genes,
                      batch_size, shuffle):
    X = rna_adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    Y = None
    if pro_adata is not None:
        Y = pro_adata.X
        if sp.issparse(Y):
            Y = Y.toarray()
    ds = DualSparseDataset(X, Y, rna_adata.var_names.tolist(), mapped_genes, unmapped_genes)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                       pin_memory=PIN_MEMORY, persistent_workers=False)


# ==============================================================================
# 7. BINN training (early stopping on the REAL held-out val split)
# ==============================================================================
def train_binn(model, train_loader, val_loader, n_epochs, lr, weight_decay,
                patience, device, log_file=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    pat_ctr = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        for Xm, Xu, Yb in train_loader:
            Xm = Xm.to(device, non_blocking=True)
            Xu = Xu.to(device, non_blocking=True)
            Yb = Yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(Xm, Xu), Yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xm, Xu, Yb in val_loader:
                Xm = Xm.to(device, non_blocking=True)
                Xu = Xu.to(device, non_blocking=True)
                Yb = Yb.to(device, non_blocking=True)
                val_losses.append(criterion(model(Xm, Xu), Yb).item())

        t_loss, v_loss = np.mean(train_losses), np.mean(val_losses)
        if epoch % 5 == 0 or epoch == 1:
            log(f"  BINN Epoch {epoch:>3d}/{n_epochs}  train={t_loss:.4f}  val={v_loss:.4f}"
                + (" *" if v_loss < best_val else "") + f"  {resource_status_str(device)}", log_file)

        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat_ctr = 0
        else:
            pat_ctr += 1
            if pat_ctr >= patience:
                log(f"  BINN early stopping at epoch {epoch} (best val MSE={best_val:.4f})", log_file)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        log(f"  BINN best checkpoint restored (val MSE={best_val:.4f})", log_file)
    return model, best_val


@torch.no_grad()
def binn_predict_and_embed(model, loader, device, n_rows, embed_dim, n_outputs, desc=""):
    """Single chunked pass producing BOTH the bottleneck embedding and
    the final prediction for every row - the 'dual output' the BINN
    stage must supply. Never materializes more than one batch on GPU
    at a time."""
    from tqdm.auto import tqdm
    model.eval()
    embeddings = np.zeros((n_rows, embed_dim), dtype=np.float32)
    predictions = np.zeros((n_rows, n_outputs), dtype=np.float32)
    row = 0
    for Xm, Xu, _ in tqdm(loader, desc=f"  BINN inference {desc}"):
        bsz = Xm.shape[0]
        Xm = Xm.to(device, non_blocking=True)
        Xu = Xu.to(device, non_blocking=True)
        z = model.encode(Xm, Xu)
        pred = model.head(z)
        embeddings[row:row + bsz] = z.cpu().numpy()
        predictions[row:row + bsz] = pred.cpu().numpy()
        row += bsz
        del Xm, Xu, z, pred
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return embeddings, predictions


# ==============================================================================
# 8. Dynamic "Do-No-Harm" post-processing tournament
# ==============================================================================
def spearman_per_protein(P, Y, protein_names):
    """Per-protein Spearman correlation. Returns a dict name -> corr
    (NaN-safe: a constant column yields NaN and is reported as such,
    never silently dropped)."""
    out = {}
    for i, name in enumerate(protein_names):
        c = spearmanr(P[:, i], Y[:, i]).correlation
        out[name] = c if c is not None else float("nan")
    return out


def run_do_no_harm_tournament(P_gat_val, P_binn_val, Y_val, protein_names,
                               alphas=BLEND_ALPHAS, log_file=None):
    """For each protein independently: start from the GAT-alone
    baseline (alpha = 1.0). Only switch to a smaller alpha if it
    STRICTLY beats that baseline on validation. Freeze the winning
    alpha per protein. This can never make a protein's validation
    score worse than pure GAT, on the split the weights are chosen on.

    Returns
    -------
    alpha_map        dict protein_name -> frozen alpha in (0, 1]
    report_df        DataFrame with baseline / boosted corr per protein
    """
    baseline_corr = spearman_per_protein(P_gat_val, Y_val, protein_names)

    alpha_map = {}
    boosted_corr = {}
    chosen_alpha_source = {}  # for the report: which alpha won and why

    for i, name in enumerate(protein_names):
        base = baseline_corr[name]
        best_alpha = 1.0
        best_corr = base if base == base else -np.inf  # NaN-safe compare

        for a in alphas:
            blend = a * P_gat_val[:, i] + (1.0 - a) * P_binn_val[:, i]
            c = spearmanr(blend, Y_val[:, i]).correlation
            if c is None:
                continue
            # Strict improvement required, and never accept a worse-than-baseline
            # alpha even if it happens to beat a NaN/degenerate baseline.
            if c > best_corr and (base != base or c > base):
                best_corr = c
                best_alpha = a

        alpha_map[name] = best_alpha
        boosted_corr[name] = best_corr if best_corr != -np.inf else base
        chosen_alpha_source[name] = "GAT-only (no alpha improved on baseline)" \
            if best_alpha == 1.0 else f"blend alpha={best_alpha}"

    report_df = pd.DataFrame({
        "protein": protein_names,
        "gat_baseline_scc": [baseline_corr[p] for p in protein_names],
        "frozen_alpha": [alpha_map[p] for p in protein_names],
        "boosted_scc": [boosted_corr[p] for p in protein_names],
        "selection": [chosen_alpha_source[p] for p in protein_names],
    })
    report_df["delta"] = report_df["boosted_scc"] - report_df["gat_baseline_scc"]
    report_df = report_df.sort_values("boosted_scc", ascending=False).reset_index(drop=True)
    return alpha_map, report_df


def apply_frozen_alphas(P_gat, P_binn, alpha_map, protein_names):
    """Applies previously-frozen per-protein alpha weights to ANY new
    prediction pair (val, at freeze time, or a genuine held-out test
    set later). Alphas are never re-fit here - they are read-only."""
    P_blend = np.zeros_like(P_gat)
    for i, name in enumerate(protein_names):
        a = alpha_map.get(name, 1.0)
        P_blend[:, i] = a * P_gat[:, i] + (1.0 - a) * P_binn[:, i]
    return P_blend


def print_accuracy_report(title, corr_dict, protein_names, log_file=None):
    series = pd.Series(corr_dict).reindex(protein_names)
    log("=" * 65, log_file)
    log(title, log_file)
    log("=" * 65, log_file)
    log(f"  Mean SCC across {len(protein_names)} proteins  : {series.mean():.4f}", log_file)
    log(f"  Median SCC                                     : {series.median():.4f}", log_file)
    top10_present = [p for p in TOP10_PROTEINS_HINT if p in series.index]
    if top10_present:
        top10 = series.reindex(top10_present)
        log(f"  Mean SCC over top-10 hint markers present ({len(top10_present)}) : {top10.mean():.4f}", log_file)
        for p in top10_present:
            log(f"      {p:<15s} {top10[p]:.4f}", log_file)
    log("=" * 65, log_file)


# ==============================================================================
# 9. Split alignment helper - ensures GAT and BINN see the IDENTICAL row
#    set/order for any given split, so their predictions can be blended
#    element-wise and compared against the same Y.
# ==============================================================================
def align_split_to_images(rna_adata, pro_adata, img_embeddings, img_index):
    """No-op passthrough (returns rna/pro unchanged, img=None) when the
    active GAT checkpoint has no image branch - see `has_image_branch` in
    main(). Only performs real filtering/reordering when embeddings are
    actually supplied."""
    if img_embeddings is None or img_index is None:
        return rna_adata, pro_adata, None
    img_aligned, valid_mask = align_image_embeddings(rna_adata.obs_names, img_embeddings, img_index)
    if not valid_mask.all():
        rna_adata = rna_adata[valid_mask].copy()
        if pro_adata is not None:
            pro_adata = pro_adata[valid_mask].copy()
    return rna_adata, pro_adata, img_aligned


def subset_proteins(pro_adata, protein_names):
    if pro_adata is None:
        return None
    return pro_adata[:, protein_names].copy()


# ==============================================================================
# 10. Main orchestration
# ==============================================================================
def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Unified GAT-CrossAttention + BINN ensemble pipeline with a "
                    "validation-only 'do-no-harm' blend-weight tournament."
    )
    p.add_argument("--data_dir", type=str, default=".")
    p.add_argument("--train_rna", type=str, default=None,
                    help="default: <data_dir>/rna_train_split_small.pkl (internal split)")
    p.add_argument("--train_pro", type=str, default=None,
                    help="default: <data_dir>/pro_train_split_small.pkl (internal split)")
    p.add_argument("--val_rna", type=str, default=None,
                    help="default: <data_dir>/rna_val_split_small.pkl (internal split)")
    p.add_argument("--val_pro", type=str, default=None,
                    help="default: <data_dir>/pro_val_split_small.pkl (internal split)")
    p.add_argument("--image_embeddings", type=str, default=None,
                    help="default: <data_dir>/embeddings_gat_test.npy. Ignored entirely if the "
                         "loaded GAT checkpoint has no histology branch.")
    p.add_argument("--image_index", type=str, default=None,
                    help="default: <data_dir>/embedding_index_gat_test.csv. Ignored entirely if "
                         "the loaded GAT checkpoint has no histology branch.")

    p.add_argument("--gat_checkpoint", type=str, default="gat_best_model.pt",
                    help="Auto-detected format: either the metadata-wrapped cross-attention "
                         "checkpoint from gat_crossattn_pipeline_v3.py, or a raw state_dict for "
                         "a plain GATv2Conv+BN stack with no image branch (architecture is "
                         "reconstructed from the checkpoint's own tensor shapes in that case).")
    p.add_argument("--binn_checkpoint", type=str, default="binn_checkpoint.pt")
    p.add_argument("--binn_data_dir", type=str, default="./binn_data",
                    help="Cache dir for Reactome raw files + gene mapping CSV")

    p.add_argument("--output_dir", type=str, default="./outputs_ensemble")

    p.add_argument("--k_neighbors", type=int, default=DEFAULT_K_NEIGHBORS)
    p.add_argument("--num_neighbors", type=int, nargs="+", default=DEFAULT_NUM_NEIGHBORS)
    p.add_argument("--inference_batch_size", type=int, default=DEFAULT_INFERENCE_BATCH_SIZE)

    p.add_argument("--binn_batch_size", type=int, default=BINN_BATCH_SIZE)
    p.add_argument("--binn_eval_batch_size", type=int, default=BINN_EVAL_BATCH)
    p.add_argument("--binn_epochs", type=int, default=BINN_EPOCHS)
    p.add_argument("--binn_lr", type=float, default=BINN_LR)
    p.add_argument("--binn_wd", type=float, default=BINN_WD)
    p.add_argument("--binn_patience", type=int, default=BINN_PATIENCE)
    p.add_argument("--binn_n_layers", type=int, default=N_BINN_LAYERS)
    p.add_argument("--binn_dropout", type=float, default=BINN_DROPOUT)

    p.add_argument("--arcsinh_cofactor", type=float, default=ARCSINH_COFACTOR)
    p.add_argument("--protein_clip_min", type=float, default=PROTEIN_CLIP_MIN)
    p.add_argument("--protein_clip_max", type=float, default=PROTEIN_CLIP_MAX)

    p.add_argument("--test_rna", type=str, default=None,
                    help="Optional held-out test/inference RNA split (.pkl or .h5ad)")
    p.add_argument("--test_pro", type=str, default=None,
                    help="Optional held-out test protein split - if given, a scored "
                         "test-set report is also printed (frozen alphas are still "
                         "NEVER re-fit on this data)")
    p.add_argument("--test_image_embeddings", type=str, default=None,
                    help="default: same as --image_embeddings")
    p.add_argument("--test_image_index", type=str, default=None,
                    help="default: same as --image_index")

    p.add_argument("--no_pin_memory", action="store_true",
                    help="Force pin_memory=False on all DataLoaders (host RAM safety)")
    return p


def resolve_path(data_dir, explicit, default_name):
    if explicit is not None:
        return explicit
    return str(Path(data_dir) / default_name)


def main():
    global PIN_MEMORY

    args = build_arg_parser().parse_args()
    if args.no_pin_memory:
        PIN_MEMORY = False

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "pipeline_log.txt")
    open(log_file, "w").close()

    train_rna_path = resolve_path(args.data_dir, args.train_rna, "rna_train_split_small.pkl")
    train_pro_path = resolve_path(args.data_dir, args.train_pro, "pro_train_split_small.pkl")
    val_rna_path   = resolve_path(args.data_dir, args.val_rna,   "rna_val_split_small.pkl")
    val_pro_path   = resolve_path(args.data_dir, args.val_pro,   "pro_val_split_small.pkl")
    img_emb_path   = resolve_path(args.data_dir, args.image_embeddings, "embeddings_gat_test.npy")
    img_idx_path   = resolve_path(args.data_dir, args.image_index, "embedding_index_gat_test.csv")

    # --------------------------------------------------------------------
    # [0] Upfront validation - fail fast with clear errors, before any
    #     expensive computation begins. Image embeddings/index are checked
    #     for existence here IF the paths were explicitly given or default
    #     files happen to exist, but they are only ever actually REQUIRED
    #     once we know (after loading the checkpoint) whether the GAT
    #     branch has a histology fusion input at all.
    # --------------------------------------------------------------------
    log("=" * 65, log_file)
    log("UPFRONT VALIDATION", log_file)
    log("=" * 65, log_file)
    required = {
        "Train RNA": train_rna_path, "Train protein": train_pro_path,
        "Val RNA": val_rna_path, "Val protein": val_pro_path,
        "GAT checkpoint": args.gat_checkpoint,
    }
    errors = [f"Missing {label}: {path}" for label, path in required.items() if not Path(path).exists()]
    if errors:
        for e in errors:
            log(f"  [ERROR] {e}", log_file)
        log("Aborting - the GAT branch is inference-only in this script and "
            "requires an existing trained checkpoint (run the original GAT "
            "training script first if it doesn't exist).", log_file)
        raise SystemExit(1)
    log(f"  All required files found. Device: {DEVICE}", log_file)
    log(f"  PIN_MEMORY={PIN_MEMORY}", log_file)
    log("=" * 65 + "\n", log_file)

    binn_data_dir = Path(args.binn_data_dir)
    binn_data_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)

    # --------------------------------------------------------------------
    # [1] Load train / val splits FIRST (before the checkpoint), so that if
    #     the GAT checkpoint turns out to be a raw state_dict with no
    #     stored protein names, we already have the real protein panel to
    #     assign names from.
    # --------------------------------------------------------------------
    log("[1] Loading train/val splits", log_file)
    train_rna = load_adata(train_rna_path)
    train_pro = load_adata(train_pro_path)
    if not (train_rna.obs_names == train_pro.obs_names).all():
        train_pro = train_pro[train_rna.obs_names].copy()
    val_rna = load_adata(val_rna_path)
    val_pro = load_adata(val_pro_path)
    if not (val_rna.obs_names == val_pro.obs_names).all():
        val_pro = val_pro[val_rna.obs_names].copy()
    log(f"  Train bins: {train_rna.n_obs:,}  |  Val bins: {val_rna.n_obs:,}", log_file)

    raw_protein_panel = [p for p in train_pro.var_names.tolist() if p in set(val_pro.var_names)]

    # --------------------------------------------------------------------
    # [2] Load GAT checkpoint - format auto-detected (see
    #     load_gat_checkpoint_flexible docstring). Architecture is frozen/
    #     unchanged in both cases; this script never trains the GAT.
    # --------------------------------------------------------------------
    log("\n[2] Loading GAT checkpoint (frozen - no retraining in this script)", log_file)
    gat_info = load_gat_checkpoint_flexible(args.gat_checkpoint, DEVICE,
                                             expected_protein_names=raw_protein_panel)
    gat_model = gat_info["model"]
    gat_protein_names = gat_info["protein_names"]
    gat_n_genes = gat_info["n_genes"]
    has_image_branch = gat_info["has_image_branch"]
    log(f"  GAT checkpoint: n_genes={gat_n_genes}, n_proteins={gat_info['n_proteins']}, "
        f"has_image_branch={has_image_branch}", log_file)
    log(f"  GAT model parameters: {sum(p.numel() for p in gat_model.parameters()):,}", log_file)

    if train_rna.shape[1] != gat_n_genes:
        raise ValueError(
            f"Gene panel mismatch: train_rna has {train_rna.shape[1]} genes but the "
            f"GAT checkpoint was built on {gat_n_genes}. The frozen GAT architecture "
            f"cannot be altered to fit a different gene panel - re-generate the split "
            f"files with the same gene panel used for training."
        )

    # --------------------------------------------------------------------
    # [3] Preprocess (leakage-safe) now that we know the checkpoint's
    #     protein ordering/count.
    # --------------------------------------------------------------------
    log("\n[3] Preprocessing train/val splits", log_file)
    log("  RNA: normalize_total(1e4) + log1p (independent per split)", log_file)
    train_rna = preprocess_rna(train_rna)
    val_rna = preprocess_rna(val_rna)

    log("  Protein: arcsinh + z-score + clip, fit on TRAIN split only", log_file)
    marker_mean, marker_std = fit_protein_transform(train_pro, args.arcsinh_cofactor)
    train_pro = apply_protein_transform(train_pro, args.arcsinh_cofactor, marker_mean, marker_std,
                                         args.protein_clip_min, args.protein_clip_max)
    val_pro = apply_protein_transform(val_pro, args.arcsinh_cofactor, marker_mean, marker_std,
                                       args.protein_clip_min, args.protein_clip_max)
    log("  Protein transform applied identically to train and val (fit on train only).", log_file)

    protein_names = [p for p in gat_protein_names
                      if p in set(train_pro.var_names) and p in set(val_pro.var_names)]
    if len(protein_names) != len(gat_protein_names):
        missing = set(gat_protein_names) - set(protein_names)
        log(f"  [WARN] {len(missing)} checkpoint proteins missing from train/val data: {missing}", log_file)
    log(f"  Ensembling over {len(protein_names)} shared proteins.", log_file)
    train_pro = subset_proteins(train_pro, protein_names)
    val_pro = subset_proteins(val_pro, protein_names)
    gat_col_idx = [gat_protein_names.index(p) for p in protein_names]

    free_memory()

    # --------------------------------------------------------------------
    # [4] Image embeddings - ONLY loaded/required if the checkpoint's
    #     architecture actually has a histology fusion branch.
    # --------------------------------------------------------------------
    img_embeddings, img_index = None, None
    if has_image_branch:
        log("\n[4] Loading image embeddings (checkpoint has a histology branch)", log_file)
        if not (Path(img_emb_path).exists() and Path(img_idx_path).exists()):
            raise SystemExit(
                f"GAT checkpoint requires histology embeddings but they were not found:\n"
                f"  Image embeddings: {img_emb_path}\n  Image index: {img_idx_path}\n"
                f"Pass --image_embeddings/--image_index explicitly if they live elsewhere."
            )
        img_embeddings = np.load(img_emb_path)
        img_index = pd.read_csv(img_idx_path)
    else:
        log("\n[4] Checkpoint has no histology branch - skipping image embeddings entirely "
            "(none required for this run).", log_file)

    train_rna_a, train_pro_a, _ = align_split_to_images(train_rna, train_pro, img_embeddings, img_index)
    val_rna_a, val_pro_a, _ = align_split_to_images(val_rna, val_pro, img_embeddings, img_index)
    log(f"  Aligned train bins: {train_rna_a.n_obs:,}  |  Aligned val bins: {val_rna_a.n_obs:,}", log_file)
    del train_rna, train_pro, val_rna, val_pro
    free_memory()

    Y_val = np.asarray(val_pro_a.X.todense()) if hasattr(val_pro_a.X, "todense") else np.asarray(val_pro_a.X)
    Y_val = Y_val.astype(np.float32)

    # --------------------------------------------------------------------
    # [5] GAT branch - mini-batch inference on val
    # --------------------------------------------------------------------
    log("\n[5] GAT branch inference on validation", log_file)
    val_graph = build_gat_graph(val_rna_a, val_pro_a, img_embeddings, img_index, args.k_neighbors)
    log(f"  Val graph: {val_graph.num_nodes} nodes, {val_graph.edge_index.shape[1]} directed edges", log_file)

    P_gat_val_full = gat_mini_batch_inference(
        gat_model, val_graph, args.num_neighbors, args.inference_batch_size, DEVICE,
        has_image_branch=has_image_branch,
    )
    P_gat_val = P_gat_val_full[:, gat_col_idx] if P_gat_val_full.shape[1] == len(gat_protein_names) \
        else P_gat_val_full
    log(f"  GAT val predictions: {P_gat_val.shape}", log_file)
    free_memory(val_graph)

    # --------------------------------------------------------------------
    # [6] BINN branch - Reactome mapping + pathway layers (cached)
    # --------------------------------------------------------------------
    log("\n[6] BINN branch - Reactome pathway mapping", log_file)
    all_genes = train_rna_a.var_names.tolist()
    download_reactome_files(binn_data_dir)
    gene_to_pathway_df, pathway_relations, covered_genes = \
        build_gene_reactome_mapping(all_genes, binn_data_dir, output_dir)
    covered_set = set(covered_genes)
    mapped_genes = [g for g in all_genes if g in covered_set]
    unmapped_genes = [g for g in all_genes if g not in covered_set]
    log(f"  Mapped genes: {len(mapped_genes):,}  |  Unmapped genes: {len(unmapped_genes):,}", log_file)

    log("\n  Building BINN architecture", log_file)
    layer_nodes, masks = build_pathway_layers(mapped_genes, gene_to_pathway_df,
                                              pathway_relations, args.binn_n_layers)
    masks_dev = [m.to(DEVICE) for m in masks]
    binn_model = DualBINNEncoder(masks=masks_dev, n_unmapped=len(unmapped_genes),
                                  n_outputs=len(protein_names), dropout=args.binn_dropout).to(DEVICE)

    # --------------------------------------------------------------------
    # [7] BINN checkpoint - smart reuse or train on the real train/val split
    # --------------------------------------------------------------------
    log("\n[7] BINN training / checkpoint reuse", log_file)
    binn_ckpt_path = Path(args.binn_checkpoint)
    if binn_ckpt_path.exists():
        log(f"  Existing checkpoint found - loading: {binn_ckpt_path}", log_file)
        ckpt = torch.load(binn_ckpt_path, map_location=DEVICE, weights_only=False)
        if ckpt.get("mapped_genes") != mapped_genes or ckpt.get("protein_names") != protein_names:
            raise ValueError(
                "binn_checkpoint.pt was trained on a different gene panel or protein "
                "set than the current data. Delete the checkpoint to retrain, or point "
                "--binn_checkpoint at the correct file."
            )
        binn_model.load_state_dict(ckpt["model_state"])
        log("  Checkpoint loaded. Skipping BINN training entirely.", log_file)
    else:
        log("  No checkpoint found - training BINN using the real spatial train/val "
            "split for early stopping (not an internal carve-out).", log_file)
        train_loader = make_binn_loader(train_rna_a, train_pro_a, mapped_genes, unmapped_genes,
                                         batch_size=args.binn_batch_size, shuffle=True)
        val_loader_for_es = make_binn_loader(val_rna_a, val_pro_a, mapped_genes, unmapped_genes,
                                              batch_size=args.binn_eval_batch_size, shuffle=False)
        binn_model, best_val_mse = train_binn(
            binn_model, train_loader, val_loader_for_es,
            n_epochs=args.binn_epochs, lr=args.binn_lr, weight_decay=args.binn_wd,
            patience=args.binn_patience, device=DEVICE, log_file=log_file,
        )
        torch.save({
            "model_state": binn_model.state_dict(),
            "mapped_genes": mapped_genes,
            "unmapped_genes": unmapped_genes,
            "layer_nodes": [list(l) for l in layer_nodes],
            "protein_names": protein_names,
            "embed_dim": binn_model.embed_dim,
            "best_val_mse": best_val_mse,
        }, binn_ckpt_path)
        log(f"  BINN checkpoint saved: {binn_ckpt_path}", log_file)
        free_memory(train_loader, val_loader_for_es)

    # --------------------------------------------------------------------
    # [8] BINN dual outputs: embeddings + predictions, train + val
    # --------------------------------------------------------------------
    log("\n[8] BINN dual-output inference (bottleneck embeddings + predictions)", log_file)
    embed_dim = binn_model.embed_dim
    n_proteins = len(protein_names)

    val_embed_loader = make_binn_loader(val_rna_a, val_pro_a, mapped_genes, unmapped_genes,
                                         batch_size=args.binn_eval_batch_size, shuffle=False)
    binn_val_embeddings, P_binn_val = binn_predict_and_embed(
        binn_model, val_embed_loader, DEVICE, val_rna_a.n_obs, embed_dim, n_proteins, desc="val"
    )
    np.save(output_dir / "binn_embeddings_val.npy", binn_val_embeddings)
    with open(output_dir / "binn_embed_barcodes_val.pkl", "wb") as f:
        pickle.dump(val_rna_a.obs_names.tolist(), f)
    free_memory(val_embed_loader, binn_val_embeddings)

    train_embed_loader = make_binn_loader(train_rna_a, train_pro_a, mapped_genes, unmapped_genes,
                                           batch_size=args.binn_eval_batch_size, shuffle=False)
    binn_train_embeddings, _P_binn_train = binn_predict_and_embed(
        binn_model, train_embed_loader, DEVICE, train_rna_a.n_obs, embed_dim, n_proteins, desc="train"
    )
    np.save(output_dir / "binn_embeddings_train.npy", binn_train_embeddings)
    with open(output_dir / "binn_embed_barcodes_train.pkl", "wb") as f:
        pickle.dump(train_rna_a.obs_names.tolist(), f)
    free_memory(train_embed_loader, binn_train_embeddings, _P_binn_train)
    free_memory(train_rna_a, train_pro_a)

    log(f"  BINN val predictions: {P_binn_val.shape}", log_file)

    # --------------------------------------------------------------------
    # [9] Do-No-Harm tournament on validation, freeze per-protein alpha
    # --------------------------------------------------------------------
    log("\n[9] Running do-no-harm ensemble tournament on validation", log_file)
    alpha_map, report_df = run_do_no_harm_tournament(
        P_gat_val, P_binn_val, Y_val, protein_names, alphas=BLEND_ALPHAS, log_file=log_file
    )
    report_path = output_dir / "tournament_report.csv"
    report_df.to_csv(report_path, index=False)
    log(f"  Tournament report saved: {report_path}", log_file)

    weights_path = output_dir / "ensemble_weights.json"
    with open(weights_path, "w") as f:
        json.dump({
            "alpha_map": alpha_map,
            "protein_names": protein_names,
            "blend_formula": "P_blend = alpha * P_GAT + (1 - alpha) * P_BINN",
            "alphas_tried": BLEND_ALPHAS,
            "fit_on_split": "validation",
            "gat_checkpoint": str(args.gat_checkpoint),
            "gat_has_image_branch": has_image_branch,
            "binn_checkpoint": str(args.binn_checkpoint),
        }, f, indent=2)
    log(f"  Frozen per-protein alpha weights saved: {weights_path}", log_file)

    P_blend_val = apply_frozen_alphas(P_gat_val, P_binn_val, alpha_map, protein_names)

    baseline_corr = spearman_per_protein(P_gat_val, Y_val, protein_names)
    boosted_corr = spearman_per_protein(P_blend_val, Y_val, protein_names)

    print_accuracy_report("BASELINE - GAT alone (validation)", baseline_corr, protein_names, log_file)
    print_accuracy_report("BOOSTED - frozen do-no-harm ensemble (validation)", boosted_corr, protein_names, log_file)

    n_boosted = sum(1 for p in protein_names if alpha_map[p] < 1.0)
    log(f"\n  {n_boosted}/{len(protein_names)} proteins used a blend alpha < 1.0; "
        f"the remaining {len(protein_names) - n_boosted} were kept at pure-GAT "
        f"(alpha = 1.0) because no blend strictly improved on the GAT baseline.", log_file)
    mean_base = pd.Series(baseline_corr).reindex(protein_names).mean()
    mean_boost = pd.Series(boosted_corr).reindex(protein_names).mean()
    log(f"  Mean SCC:  baseline GAT = {mean_base:.4f}   ->   boosted ensemble = {mean_boost:.4f}"
        f"   (delta = {mean_boost - mean_base:+.4f})", log_file)

    free_memory(gat_model, binn_model)

    # --------------------------------------------------------------------
    # [10] Optional held-out test/inference set - frozen alphas applied,
    #      NEVER re-fit here.
    # --------------------------------------------------------------------
    if args.test_rna is not None:
        log("\n[10] Applying frozen ensemble to held-out test/inference set", log_file)
        test_img_embeddings, test_img_index = None, None
        if has_image_branch:
            test_img_emb_path = args.test_image_embeddings or img_emb_path
            test_img_idx_path = args.test_image_index or img_idx_path
            test_img_embeddings = np.load(test_img_emb_path)
            test_img_index = pd.read_csv(test_img_idx_path)

        test_rna = load_adata(args.test_rna)
        test_rna = preprocess_rna(test_rna)
        if test_rna.shape[1] != gat_n_genes:
            raise ValueError(f"Test gene panel ({test_rna.shape[1]}) does not match "
                              f"GAT checkpoint ({gat_n_genes}).")

        test_pro = None
        has_test_labels = args.test_pro is not None
        if has_test_labels:
            test_pro = load_adata(args.test_pro)
            if not (test_rna.obs_names == test_pro.obs_names).all():
                test_pro = test_pro[test_rna.obs_names].copy()
            test_pro = apply_protein_transform(test_pro, args.arcsinh_cofactor, marker_mean, marker_std,
                                                args.protein_clip_min, args.protein_clip_max)
            test_pro = subset_proteins(test_pro, protein_names)

        test_rna_a, test_pro_a, _ = align_split_to_images(test_rna, test_pro, test_img_embeddings, test_img_index)
        free_memory(test_rna, test_pro)

        # reload frozen GAT for test inference
        gat_info_test = load_gat_checkpoint_flexible(args.gat_checkpoint, DEVICE,
                                                       expected_protein_names=raw_protein_panel)
        gat_model = gat_info_test["model"]
        test_graph = build_gat_graph(test_rna_a, test_pro_a,
                                      test_img_embeddings, test_img_index, args.k_neighbors,
                                      n_proteins_if_unlabeled=len(protein_names))
        P_gat_test_full = gat_mini_batch_inference(
            gat_model, test_graph, args.num_neighbors, args.inference_batch_size, DEVICE,
            has_image_branch=has_image_branch,
        )
        P_gat_test = P_gat_test_full[:, gat_col_idx] if P_gat_test_full.shape[1] == len(gat_protein_names) \
            else P_gat_test_full
        free_memory(test_graph, gat_model)

        binn_model = DualBINNEncoder(masks=masks_dev, n_unmapped=len(unmapped_genes),
                                      n_outputs=len(protein_names), dropout=args.binn_dropout).to(DEVICE)
        ckpt = torch.load(binn_ckpt_path, map_location=DEVICE, weights_only=False)
        binn_model.load_state_dict(ckpt["model_state"])
        test_loader = make_binn_loader(test_rna_a, test_pro_a, mapped_genes, unmapped_genes,
                                        batch_size=args.binn_eval_batch_size, shuffle=False)
        binn_test_embeddings, P_binn_test = binn_predict_and_embed(
            binn_model, test_loader, DEVICE, test_rna_a.n_obs, embed_dim, n_proteins, desc="test"
        )
        np.save(output_dir / "binn_embeddings_test.npy", binn_test_embeddings)
        with open(output_dir / "binn_embed_barcodes_test.pkl", "wb") as f:
            pickle.dump(test_rna_a.obs_names.tolist(), f)
        free_memory(test_loader, binn_test_embeddings, binn_model)

        P_blend_test = apply_frozen_alphas(P_gat_test, P_binn_test, alpha_map, protein_names)

        out_df = pd.DataFrame(P_blend_test, columns=protein_names, index=test_rna_a.obs_names)
        out_df.index.name = "barcode"
        out_path = output_dir / "test_predictions_boosted_ensemble.csv"
        out_df.reset_index().to_csv(out_path, index=False)
        log(f"  Boosted ensemble test predictions saved: {out_path}", log_file)

        if has_test_labels:
            Y_test = np.asarray(test_pro_a.X.todense()) if hasattr(test_pro_a.X, "todense") \
                else np.asarray(test_pro_a.X)
            Y_test = Y_test.astype(np.float32)
            test_baseline = spearman_per_protein(P_gat_test, Y_test, protein_names)
            test_boosted = spearman_per_protein(P_blend_test, Y_test, protein_names)
            print_accuracy_report("TEST SET - BASELINE (GAT alone)", test_baseline, protein_names, log_file)
            print_accuracy_report("TEST SET - BOOSTED (frozen ensemble, alphas NOT re-fit here)",
                                   test_boosted, protein_names, log_file)

    log("\nDone.", log_file)


if __name__ == "__main__":
    main()
