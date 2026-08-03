"""
Unified BINN → GAT Spatial Multi-Omics Pipeline
================================================
Stage 1 — BINN feature extraction
    - Trains a dual-stream Biology-Informed Neural Network on ALL training bins
      with early stopping on a small held-aside val fraction. The BINN is a
      feature extractor; the val split here is only to guard against wasted
      compute on a diverging run — it does not affect gene selection, which
      uses SHAP on the full training set after the best checkpoint is restored.
    - SHAP attributes importance to Reactome-mapped genes; gradient magnitude
      ranks unmapped genes. An elbow algorithm per stream finds the cutoff;
      the union forms the selected gene panel.

Stage 2 — GAT prediction
    - GAT trains on ALL training bins (no internal split). The external
      valid_rna.h5ad is used only for generating the submission CSV — there
      are no protein labels for it so it cannot be used for early stopping.
    - Early stopping in the GAT uses a small spatial block held out from
      training data (GAT_VAL_FRACTION).
    - Predictions are written in the exact submission format required:
        barcode, pxl_row_in_fullres, pxl_col_in_fullres, <protein columns>

Upfront validation
    - ALL file paths and required obs columns are checked before any
      computation begins. The run aborts immediately with a clear error
      message if anything is missing, rather than failing 4 hours in.

Key design decisions
--------------------
- One RNA normalisation path: raw counts → normalize_total(1e4) → log1p.
  Applied once, flagged in .uns['is_preprocessed'] to prevent repeats.
- One protein normalisation (arcsinh(X/150)) across both BINN and GAT.
- Graph built on full training set; node masks control train vs val nodes
  so edges are never severed at block boundaries.
- Submission CSV: barcodes from valid_rna.h5ad obs_names, spatial coords
  from obs['pxl_row_in_fullres'] / obs['pxl_col_in_fullres'], protein cols
  in the order defined by SUBMISSION_PROTEINS.
"""

# ==============================================================================
# 0. Imports
# ==============================================================================
import gc
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import requests
from tqdm.auto import tqdm

import anndata as ad
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.loader import NeighborLoader
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
import shap
from kneed import KneeLocator

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. Configuration
# ==============================================================================
BASE_DIR      = Path("/home/ubuntu/data")
BINN_DATA_DIR = BASE_DIR / "binn_data"
OUTPUT_DIR    = BASE_DIR / "outputs"
MODELS_DIR    = OUTPUT_DIR / "models"

TRAIN_RNA_PATH = BASE_DIR / "train_rna.h5ad"
TRAIN_PRO_PATH = BASE_DIR / "train_pro.h5ad"
VALID_RNA_PATH = BASE_DIR / "valid_rna.h5ad"
TEST_RNA_PATH  = BASE_DIR / "test_rna.h5ad"

# Pixel → micron conversion factors
TRAIN_MICRONS_PER_PIXEL = 0.8820219467631594
TEST_MICRONS_PER_PIXEL  = 0.883043249671293

# RNA preprocessing
NORMALIZE_TARGET = 1e4
MIN_COUNTS_GENE  = 5

# BINN
N_BINN_LAYERS   = 4
DROPOUT_RATE    = 0.2
BINN_BATCH_SIZE = 4096
BINN_EPOCHS     = 100       # ceiling; early stopping will cut this short
BINN_LR         = 1e-3
BINN_WD         = 1e-4
BINN_VAL_FRAC   = 0.10      # fraction held aside for BINN early stopping only
BINN_PATIENCE   = 8         # epochs without improvement before stopping

# SHAP
N_SHAP_BACKGROUND = 256
N_SHAP_EXPLAIN    = 512
SHAP_NSAMPLES     = 50   # was SHAP default of 200 — main memory driver
SHAP_CHUNK_SIZE   = 64   # explain instances processed per shap_values() call

# GAT
GAT_HIDDEN_DIM   = 256
GAT_NUM_LAYERS   = 5
GAT_RADIUS_UM    = 50
GAT_EPOCHS       = 400
GAT_LR           = 1e-3
GAT_DROPOUT      = 0.2
GAT_PATIENCE     = 20
GAT_VAL_FRACTION = 0.10     # spatial block for GAT early stopping

PROTEIN_COFACTOR = 150.0

# Submission protein column order (from the provided example)
SUBMISSION_PROTEINS = [
    "synd", "FOXP3", "CD16", "CD31", "CXCL13", "Ki67", "OLIG2", "CXCR5",
    "HLA-A", "PD-L1", "PSD95", "CD20", "CD68", "CD44", "SMA", "MSH6",
    "CD23", "GFAP", "SYNA", "Podoplanin", "Vimentin", "CD47", "CD74",
    "SIRP", "Granzyme B", "IDH1", "MPO", "CD45", "CD21", "FIBR", "C-KIT",
    "CD3e", "TOX", "PD-1", "PDGFR", "CD4", "MAP2", "CD8", "MGMT", "CD38",
    "HLA-DR", "CD14", "ICOS", "Granzyme K",
]

# Required obs columns in each RNA file
_REQUIRED_OBS = ["pxl_row_in_fullres", "pxl_col_in_fullres",
                 "array_row", "array_col"]


# ==============================================================================
# 2. Upfront validation — fail fast before any compute
# ==============================================================================

def validate_environment():
    """
    Check every file path, required obs column, GPU availability, and output
    directory writeability before a single byte of data is processed.
    Prints a clear summary and aborts if anything is wrong.
    """
    errors = []

    # --- Required input files -------------------------------------------------
    required_files = {
        "Train RNA"    : TRAIN_RNA_PATH,
        "Train protein": TRAIN_PRO_PATH,
        "Valid RNA"    : VALID_RNA_PATH,
        "Test RNA"     : TEST_RNA_PATH,
    }
    for label, path in required_files.items():
        if not path.exists():
            errors.append(f"Missing {label}: {path}")

    # --- Output directories ---------------------------------------------------
    for d in [OUTPUT_DIR, MODELS_DIR, BINN_DATA_DIR]:
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_test"
            probe.touch(); probe.unlink()
        except Exception as e:
            errors.append(f"Cannot write to {d}: {e}")

    # --- obs columns in each RNA file -----------------------------------------
    for label, path in [("Train RNA", TRAIN_RNA_PATH),
                         ("Valid RNA", VALID_RNA_PATH),
                         ("Test RNA",  TEST_RNA_PATH)]:
        if not path.exists():
            continue   # already flagged above
        try:
            adata = ad.read_h5ad(path, backed="r")
            missing = [c for c in _REQUIRED_OBS if c not in adata.obs.columns]
            if missing:
                errors.append(
                    f"{label} missing obs columns: {missing}")
            adata.file.close()
        except Exception as e:
            errors.append(f"Cannot read {label} ({path}): {e}")

    # --- Protein file obs columns ---------------------------------------------
    if TRAIN_PRO_PATH.exists():
        try:
            pro = ad.read_h5ad(TRAIN_PRO_PATH, backed="r")
            available = set(pro.var_names.tolist())
            missing_p = [p for p in SUBMISSION_PROTEINS if p not in available]
            if missing_p:
                # warn only — some proteins may be named differently
                print(f"  [WARN] {len(missing_p)} submission proteins not found "
                      f"in train_pro var_names: {missing_p[:5]}{'...' if len(missing_p)>5 else ''}")
            pro.file.close()
        except Exception as e:
            errors.append(f"Cannot read Train protein ({TRAIN_PRO_PATH}): {e}")

    # --- GPU ------------------------------------------------------------------
    if not torch.cuda.is_available():
        print("  [WARN] No CUDA GPU detected — running on CPU (will be very slow).")

    # --- Result ---------------------------------------------------------------
    print("\n" + "=" * 65)
    print("UPFRONT VALIDATION")
    print("=" * 65)
    if errors:
        for e in errors:
            print(f"  [ERROR] {e}")
        print("\nAborting — fix the above errors before re-running.\n")
        sys.exit(1)
    else:
        print("  All checks passed. Starting pipeline.")
    print("=" * 65 + "\n")


# ==============================================================================
# 3. Shared preprocessing utilities
# ==============================================================================

def load_and_align(rna_path, pro_path=None, in_tissue_only=True):
    """
    Load RNA (and optionally protein). Aligns on common obs_names explicitly.
    """
    print(f"  Loading RNA: {rna_path}")
    rna = ad.read_h5ad(rna_path, backed="r")
    rna_obs = (
        rna.obs_names[rna.obs["in_tissue"] == 1]
        if in_tissue_only and "in_tissue" in rna.obs.columns
        else rna.obs_names
    )

    pro = None
    if pro_path is not None:
        print(f"  Loading protein: {pro_path}")
        pro = ad.read_h5ad(pro_path, backed="r")
        pro_obs = (
            pro.obs_names[pro.obs["in_tissue"] == 1]
            if in_tissue_only and "in_tissue" in pro.obs.columns
            else pro.obs_names
        )
        common = rna_obs.intersection(pro_obs)
        print(f"  Common in-tissue bins: {len(common):,}")
        rna = rna[common].to_memory()
        pro = pro[common].to_memory()
    else:
        rna = rna[rna_obs].to_memory()

    if not sp.issparse(rna.X):
        rna.X = sp.csr_matrix(rna.X)
    if pro is not None and not sp.issparse(pro.X):
        pro.X = sp.csr_matrix(pro.X)

    print(f"  RNA: {rna.shape}  |  Protein: {pro.shape if pro is not None else 'N/A'}")
    return rna, pro


def normalise_rna(adata, target_sum=NORMALIZE_TARGET, min_counts=MIN_COUNTS_GENE):
    """
    Gene QC → library-size normalisation → log1p. Sets is_preprocessed flag.
    Never call on an already-preprocessed object.
    """
    if adata.uns.get("is_preprocessed"):
        print("  RNA already preprocessed — skipping.")
        return adata
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    sc.pp.filter_genes(adata, min_counts=min_counts)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum, inplace=True)
    sc.pp.log1p(adata, chunked=True, chunk_size=10_000)
    adata.X = adata.X.tocsr()
    adata.uns["is_preprocessed"]  = True
    adata.uns["normalize_target"] = target_sum
    print(f"  After gene QC + normalisation: {adata.shape}")
    return adata


def normalise_protein_arcsinh(adata, cofactor=PROTEIN_COFACTOR):
    """arcsinh(X / cofactor). Used identically in BINN and GAT."""
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    adata.X = np.arcsinh(X.astype(np.float32) / cofactor)
    adata.uns["protein_normalised"] = "arcsinh"
    adata.uns["protein_cofactor"]   = cofactor
    return adata


def apply_gene_panel(source_path, selected_genes, output_path, importance_df=None):
    """
    Slice source h5ad to selected_genes, normalise from raw counts, write output.
    Idempotent: returns cached file if output already exists.
    """
    if output_path.exists():
        print(f"  Cached — loading: {output_path}")
        return ad.read_h5ad(output_path)

    print(f"  Applying gene panel → {output_path.name} ...")
    src = ad.read_h5ad(source_path, backed="r")
    present = [g for g in selected_genes if g in src.var_names]
    if len(present) < len(selected_genes):
        print(f"  [WARN] {len(selected_genes) - len(present)} genes absent — skipped.")

    out = src[:, present].to_memory()

    # Always normalise from raw counts to avoid double-normalisation
    if "counts" in out.layers:
        out.X = out.layers["counts"].copy()
    else:
        print("  [WARN] No raw counts layer — normalising from .X as-is.")

    out.layers["counts"] = (out.X.copy() if sp.issparse(out.X)
                            else sp.csr_matrix(out.X))
    sc.pp.normalize_total(out, target_sum=NORMALIZE_TARGET, inplace=True)
    sc.pp.log1p(out)
    out.X = out.X.tocsr() if sp.issparse(out.X) else out.X
    out.uns["is_preprocessed"]  = True
    out.uns["normalize_target"] = NORMALIZE_TARGET

    if importance_df is not None:
        imp = importance_df.set_index("gene")
        out.var["shap_importance"] = [
            imp.loc[g, "shap_importance"] if g in imp.index else np.nan
            for g in out.var_names
        ]
        out.var["shap_rank"] = [
            int(imp.loc[g, "shap_rank"]) if g in imp.index else -1
            for g in out.var_names
        ]

    out.uns["selected_genes"]    = present
    out.uns["n_input_genes"]     = src.shape[1]
    out.uns["n_selected_genes"]  = len(present)
    out.write_h5ad(output_path)
    print(f"  Written: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return out


# ==============================================================================
# 4. Reactome pathway mapping
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
            print(f"  Downloading {fname} ...")
            ret = os.system(f'wget -q "{url}" -O "{dest}"')
            if ret != 0 or not dest.exists():
                raise RuntimeError(f"Failed to download {fname} from {url}")
        else:
            print(f"  Cached: {fname}")


def fetch_gene_to_uniprot(gene_symbols, batch_size=200):
    symbol_to_uniprot = {}
    url     = "https://mygene.info/v3/query"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    for i in tqdm(range(0, len(gene_symbols), batch_size), desc="  Gene→UniProt"):
        batch   = gene_symbols[i : i + batch_size]
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
            print(f"  [WARN] Batch {i // batch_size} failed: {e}")
    print(f"  Mapped {len(symbol_to_uniprot):,}/{len(gene_symbols):,} genes to UniProt.")
    return symbol_to_uniprot


def build_gene_reactome_mapping(gene_symbols, binn_data_dir):
    cache_path = OUTPUT_DIR / "gene_reactome_mapping.csv"
    if cache_path.exists():
        print(f"  Loading cached Reactome mapping.")
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
            raise RuntimeError("No genes mapped to Reactome — check network.")
        gene_to_pathway = pd.concat(rows, ignore_index=True).drop_duplicates()
        gene_to_pathway.to_csv(cache_path, index=False)
        print(f"  Mapping saved: {cache_path}")

    p_rel = pd.read_csv(
        binn_data_dir / "reactome_pathways_relation.txt",
        sep="\t", header=None, names=["target", "source"],
    )
    p_rel_human = p_rel[
        p_rel["target"].str.startswith("R-HSA") &
        p_rel["source"].str.startswith("R-HSA")
    ]
    pathway_relations = list(p_rel_human.itertuples(index=False, name=None))
    covered_genes     = gene_to_pathway["input"].unique().tolist()
    print(f"  Reactome-covered genes: {len(covered_genes):,}")
    return gene_to_pathway, pathway_relations, covered_genes


# ==============================================================================
# 5. Dual-stream BINN
# ==============================================================================

def build_pathway_layers(gene_symbols_subset, gene_to_pathway_df,
                         pathway_relations, n_layers):
    import networkx as nx
    G = nx.DiGraph()
    for child, parent in pathway_relations:
        G.add_edge(child, parent)

    g2p        = gene_to_pathway_df[gene_to_pathway_df["input"].isin(gene_symbols_subset)]
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
            print(f"  Hierarchy exhausted at layer {depth + 1}")
            break

    print(f"  Layer sizes: {[len(l) for l in layer_nodes]}")

    masks    = []
    gene_idx = {g: i for i, g in enumerate(layer_nodes[0])}
    l1_idx   = {p: i for i, p in enumerate(layer_nodes[1])}

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
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.register_buffer("mask", mask.float())
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class DualBINNEncoder(nn.Module):
    def __init__(self, masks, n_unmapped, n_outputs, dropout=0.2):
        super().__init__()
        self.pathway_layers = nn.ModuleList()
        self.batch_norms    = nn.ModuleList()
        self.dropouts       = nn.ModuleList()
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
        print(f"  DualBINNEncoder: Stream1={binn_bn}  Stream2=128  → head → {n_outputs}")

    def encode_pathway(self, x_mapped):
        x = x_mapped
        for linear, bn, drop in zip(self.pathway_layers, self.batch_norms, self.dropouts):
            x = drop(torch.tanh(bn(linear(x))))
        return x

    def forward(self, x_mapped, x_unmapped):
        return self.head(
            torch.cat([self.encode_pathway(x_mapped),
                       self.dense_stream(x_unmapped)], dim=1)
        )


class DualSparseDataset(Dataset):
    """Sparse RNA → dense gene-subset batches, built per-BATCH via __getitems__.
    Avoids both the original bug (per-row Python loop -> slow, GPU idle) and
    the whole-dataset densify (-> OOM). Only one batch is ever dense in memory
    at a time; PyTorch calls __getitems__ with a full batch of indices when
    the Dataset defines it (torch >= 1.13), instead of __getitem__ per index.
    """
    def __init__(self, X_sparse, Y_dense, gene_list, mapped_genes, unmapped_genes):
        var_idx   = {g: i for i, g in enumerate(gene_list)}
        m_present = [g for g in mapped_genes   if g in var_idx]
        u_present = [g for g in unmapped_genes if g in var_idx]
        self.m_src = np.array([var_idx[g] for g in m_present], dtype=np.int64)
        self.m_dst = np.array([i for i, g in enumerate(mapped_genes)   if g in var_idx], dtype=np.int64)
        self.u_src = np.array([var_idx[g] for g in u_present], dtype=np.int64)
        self.u_dst = np.array([i for i, g in enumerate(unmapped_genes) if g in var_idx], dtype=np.int64)
        self.n_mapped   = len(mapped_genes)
        self.n_unmapped = len(unmapped_genes)
        self.X = X_sparse.tocsr()
        self.Y = torch.tensor(Y_dense.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitems__(self, idx_list):
        idx     = np.asarray(idx_list)
        block   = self.X[idx]                      # one vectorized row gather (sparse)
        block_m = block[:, self.m_src].toarray()    # (batch, n_mapped_present) — dense only for this batch
        block_u = block[:, self.u_src].toarray()    # (batch, n_unmapped_present)

        xm = np.zeros((len(idx), self.n_mapped),   dtype=np.float32)
        xu = np.zeros((len(idx), self.n_unmapped), dtype=np.float32)
        xm[:, self.m_dst] = block_m
        xu[:, self.u_dst] = block_u

        xm_t, xu_t = torch.from_numpy(xm), torch.from_numpy(xu)
        return [(xm_t[i], xu_t[i], self.Y[idx[i]]) for i in range(len(idx))]

    def __getitem__(self, idx):
        return self.__getitems__([idx])[0]

def make_binn_dataloaders(rna_adata, pro_adata, mapped_genes, unmapped_genes,
                          batch_size=BINN_BATCH_SIZE, val_frac=BINN_VAL_FRAC):
    """
    Two DataLoaders: train (shuffled) and val (fixed), split randomly from
    the full training set. The val split is used ONLY for BINN early stopping
    — it does not affect SHAP, which runs on the train loader after training.
    """

    n_workers = 2 if DEVICE.type == "cuda" else 0

    X = rna_adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    Y = pro_adata.X
    if sp.issparse(Y):
        Y = Y.toarray()

    ds      = DualSparseDataset(X, Y, rna_adata.var_names.tolist(),
                                 mapped_genes, unmapped_genes)
    n_val   = int(len(ds) * val_frac)
    n_train = len(ds) - n_val
    gen     = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val],
                                                      generator=gen)
    pin = DEVICE.type == "cuda"
    n_workers = 4 if DEVICE.type == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                                shuffle=True, num_workers=n_workers, pin_memory=pin,
                                persistent_workers=(n_workers > 0))
    val_loader   = DataLoader(val_ds, batch_size=batch_size * 2,
                                shuffle=False, num_workers=n_workers, pin_memory=pin,
                                persistent_workers=(n_workers > 0))
    print(f"  BINN train: {n_train:,} bins  |  val: {n_val:,} bins")
    return train_loader, val_loader


def train_binn(model, train_loader, val_loader, n_epochs, lr, weight_decay,
               patience, device):
    """
    BINN training with early stopping on val MSE.
    Best checkpoint is restored after stopping so SHAP runs on the best model.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)
    criterion = nn.MSELoss()

    best_val    = float("inf")
    best_state  = None
    pat_ctr     = 0

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

        t_loss = np.mean(train_losses)
        v_loss = np.mean(val_losses)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  BINN Epoch {epoch:>3d}/{n_epochs}  "
                  f"train={t_loss:.4f}  val={v_loss:.4f}"
                  + (" *" if v_loss < best_val else ""))

        if v_loss < best_val:
            best_val   = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat_ctr    = 0
        else:
            pat_ctr += 1
            if pat_ctr >= patience:
                print(f"  Early stopping at epoch {epoch}  "
                      f"(best val MSE={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Best checkpoint restored (val MSE={best_val:.4f})")
    return model


# ==============================================================================
# 6. Gene importance & selection
# ==============================================================================

def compute_shap_importance(model, loader, mapped_genes, device=DEVICE):
    """
    Gradient × Input attribution for mapped genes, computed in mini-batches.
    Replaces shap.GradientExplainer which OOMs on large gene sets.
    Produces per-gene importance scores equivalent in quality for gene selection.
    """
    model.eval()
    accum   = torch.zeros(len(mapped_genes), device=device)
    n_total = 0

    for Xm, Xu, Yb in loader:
        Xm = Xm.to(device).requires_grad_(True)
        Xu = Xu.to(device)
        Yb = Yb.to(device)

        out  = model(Xm, Xu)                        # (batch, n_proteins)
        loss = out.sum(dim=1).mean()                 # scalar — all outputs contribute
        loss.backward()

        with torch.no_grad():
            # gradient × input: shape (batch, n_mapped_genes)
            gi = (Xm.grad * Xm.detach()).abs()
            accum   += gi.sum(dim=0)
            n_total += Xm.shape[0]

        Xm.grad = None
        if n_total >= N_SHAP_EXPLAIN:
            break

    importance = (accum / n_total).cpu().numpy()

    assert importance.shape[0] == len(mapped_genes), \
        f"Importance length {importance.shape[0]} != {len(mapped_genes)} mapped genes"

    return pd.DataFrame({
        "gene":            mapped_genes,
        "shap_importance": importance,          # column name kept so downstream code is unchanged
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)


def compute_gradient_importance(model, loader, unmapped_genes, device=DEVICE):
    model.eval()
    grad_accum = torch.zeros(len(unmapped_genes), device=device)
    count = 0
    for Xm, Xu, Yb in loader:
        Xm = Xm.to(device)
        Xu = Xu.to(device).requires_grad_(True)
        Yb = Yb.to(device)
        loss = F.mse_loss(model(Xm, Xu), Yb)
        loss.backward()
        grad_accum += Xu.grad.abs().mean(dim=0)
        count += 1
        if count >= 20:
            break
    grad_imp = (grad_accum / count).cpu().numpy()
    return pd.DataFrame({
        "gene": unmapped_genes,
        "grad_importance": grad_imp,
    }).sort_values("grad_importance", ascending=False).reset_index(drop=True)


def select_genes_elbow(importance_series, label=""):
    vals = importance_series.values
    knee = KneeLocator(range(1, len(vals) + 1), vals,
                       curve="convex", direction="decreasing",
                       interp_method="polynomial")
    if knee.knee is not None:
        cutoff = knee.knee
        print(f"  {label}: elbow at rank {cutoff}")
    else:
        threshold = vals.mean() + vals.std()
        cutoff    = max(1, int((vals >= threshold).sum()))
        print(f"  {label}: no elbow — using mean+1σ cutoff ({cutoff} genes)")
    return cutoff


def run_gene_selection(model, train_loader, mapped_genes, unmapped_genes):
    print("  Running SHAP on mapped genes ...")
    shap_df = compute_shap_importance(model, train_loader, mapped_genes)
    shap_df.to_csv(OUTPUT_DIR / "gene_shap_importance.csv", index=False)

    print("  Running gradient importance on unmapped genes ...")
    grad_df = compute_gradient_importance(model, train_loader, unmapped_genes)
    grad_df.to_csv(OUTPUT_DIR / "gene_grad_importance.csv", index=False)

    n_mapped   = select_genes_elbow(shap_df["shap_importance"],  "Mapped stream")
    n_unmapped = select_genes_elbow(grad_df["grad_importance"],  "Unmapped stream")

    sel_mapped   = shap_df.head(n_mapped)["gene"].tolist()
    sel_unmapped = grad_df.head(n_unmapped)["gene"].tolist()
    selected     = list(dict.fromkeys(sel_mapped + sel_unmapped))

    print(f"  Gene panel: {len(sel_mapped)} mapped + {len(sel_unmapped)} "
          f"unmapped = {len(selected)} total")

    shap_df["shap_rank"] = range(1, len(shap_df) + 1)
    imp_df = shap_df[shap_df["gene"].isin(selected)].copy()
    return selected, imp_df


# ==============================================================================
# 7. Spatial utilities
# ==============================================================================

def compute_microns(adata, microns_per_pixel):
    rows = adata.obs["pxl_row_in_fullres"].values.astype(np.float64)
    cols = adata.obs["pxl_col_in_fullres"].values.astype(np.float64)
    coords = np.zeros((len(rows), 2))
    coords[:, 0] = rows * microns_per_pixel
    coords[:, 1] = cols * microns_per_pixel
    return coords


def spatial_block_split(adata, fraction, n_grid_x=10, n_grid_y=10, seed=SEED):
    rows  = adata.obs["array_row"].values
    cols  = adata.obs["array_col"].values
    rb    = np.linspace(rows.min(), rows.max() + 1, n_grid_x + 1)
    cb    = np.linspace(cols.min(), cols.max() + 1, n_grid_y + 1)
    ri    = np.clip(np.digitize(rows, rb) - 1, 0, n_grid_x - 1)
    ci    = np.clip(np.digitize(cols, cb) - 1, 0, n_grid_y - 1)
    gids  = ri * n_grid_y + ci
    n_grids   = n_grid_x * n_grid_y
    n_holdout = max(1, int(np.ceil(n_grids * fraction)))
    rng   = np.random.default_rng(seed=seed)
    val_ids = rng.choice(np.arange(n_grids), size=n_holdout, replace=False)
    holdout = np.isin(gids, val_ids)
    return ~holdout, holdout


def build_spatial_graph(coords, radius=GAT_RADIUS_UM):
    nbrs = NearestNeighbors(radius=radius, metric="euclidean").fit(coords)
    _, indices = nbrs.radius_neighbors(coords)
    edges = set()
    for i, nbrs_i in enumerate(indices):
        for j in nbrs_i:
            if i != j:
                edges.add((i, j)); edges.add((j, i))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(list(edges), dtype=torch.long).t().contiguous()


def extract_proteins(adata_pro, protein_names):
    cols = []
    for p in protein_names:
        col = (adata_pro[:, p].X if p in adata_pro.var_names
               else np.zeros((adata_pro.n_obs, 1)))
        col = col.toarray() if sp.issparse(col) else np.asarray(col)
        cols.append(col.reshape(-1, 1))
    return np.hstack(cols)


# ==============================================================================
# 8. GATv2 model & loss
# ==============================================================================

class GATv2Regressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers=GAT_NUM_LAYERS, dropout=GAT_DROPOUT):
        super().__init__()
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.input_lin   = nn.Linear(input_dim, hidden_dim)
        self.gat_layers  = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim)
                                          for _ in range(num_layers)])
        self.residuals   = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
                                          for _ in range(num_layers)])
        self.output_lin  = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.input_lin(x))
        for gat, ln, res in zip(self.gat_layers, self.layer_norms, self.residuals):
            r = x
            if edge_index.numel() > 0:
                x = gat(x, edge_index)
            x = F.relu(ln(x + res(r)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output_lin(x)


def pearson_proxy_loss(y_pred, y_true):
    yp    = y_pred - y_pred.mean(dim=0)
    yt    = y_true - y_true.mean(dim=0)
    cov   = (yp * yt).sum(dim=0)
    denom = (torch.sqrt((yp**2).sum(dim=0) + 1e-8) *
             torch.sqrt((yt**2).sum(dim=0) + 1e-8) + 1e-8)
    return 1 - (cov / denom).mean()


def eval_spearman(pred_arcsinh, true_arcsinh, cofactor=PROTEIN_COFACTOR):
    pred   = np.sinh(np.clip(pred_arcsinh, -8, 8)) * cofactor
    true   = np.sinh(np.clip(true_arcsinh, -8, 8)) * cofactor
    scores = []
    for i in range(true.shape[1]):
        if np.std(true[:, i]) > 0 and np.std(pred[:, i]) > 0:
            scores.append(spearmanr(true[:, i], pred[:, i])[0])
        else:
            scores.append(0.0)
    return float(np.mean(scores))


# ==============================================================================
# 9. GAT training (full training data, spatial val block for early stopping)
# ==============================================================================
def train_gat(model, full_data, train_mask, val_mask,
              n_epochs=GAT_EPOCHS, lr=GAT_LR, patience=GAT_PATIENCE,
              device=DEVICE):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)
    
    # --- Create Mini-Batch Loaders ---
    # Samples 10 neighbors across 5 layer hops
    train_loader = NeighborLoader(
        full_data,
        num_neighbors=[10] * 5, 
        batch_size=2048,
        input_nodes=train_mask,
        shuffle=True
    )
    
    val_loader = NeighborLoader(
        full_data,
        num_neighbors=[10] * 5,
        batch_size=2048,
        input_nodes=val_mask,
        shuffle=False
    )

    best_val   = -float("inf")
    best_state = None
    pat_ctr    = 0

    for epoch in range(1, n_epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            
            # NeighborLoader guarantees the first batch.batch_size nodes are the target nodes
            y_pred = out[:batch.batch_size]
            y_true = batch.y[:batch.batch_size]
            
            loss = (0.5 * F.mse_loss(y_pred, y_true) +
                    0.5 * pearson_proxy_loss(y_pred, y_true))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # --- EVALUATION PHASE ---
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)
                val_preds.append(out[:batch.batch_size].cpu().numpy())
                val_trues.append(batch.y[:batch.batch_size].cpu().numpy())

        val_pred = np.vstack(val_preds)
        val_true = np.vstack(val_trues)
        val_sp = eval_spearman(val_pred, val_true)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  GAT Epoch {epoch:>3d}/{n_epochs}  "
                  f"loss={avg_loss:.4f}  val_spearman={val_sp:.4f}"
                  + (" *" if val_sp > best_val else ""))

        if val_sp > best_val:
            best_val   = val_sp
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat_ctr    = 0
        else:
            pat_ctr += 1
            if pat_ctr >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best val Spearman={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ==============================================================================
# 10. Submission CSV writer
# ==============================================================================

def write_submission_csv(rna_adata, pred_raw, protein_names, output_path,
                         microns_per_pixel=TEST_MICRONS_PER_PIXEL):
    """
    Write the submission CSV in the required format:
        barcode, pxl_row_in_fullres, pxl_col_in_fullres, <proteins in order>

    - barcode            : obs_names of the RNA AnnData
    - pxl_row/col        : taken directly from obs (integer pixel coords)
    - protein columns    : SUBMISSION_PROTEINS order; proteins absent from
                           pred are filled with 0.0; extra predicted proteins
                           not in SUBMISSION_PROTEINS are silently dropped.
    - pred_raw           : numpy array (n_bins × n_pred_proteins), raw scale
                           (already inverse-arcsinh transformed before calling)
    - protein_names      : ordered list matching columns of pred_raw
    """
    pred_df = pd.DataFrame(pred_raw, index=rna_adata.obs_names,
                           columns=protein_names)

    # Spatial coords from obs
    pxl_row = rna_adata.obs["pxl_row_in_fullres"].values.astype(int)
    pxl_col = rna_adata.obs["pxl_col_in_fullres"].values.astype(int)

    # Build output with exact submission column order
    out = pd.DataFrame(index=rna_adata.obs_names)
    out.index.name = "barcode"
    out["pxl_row_in_fullres"] = pxl_row
    out["pxl_col_in_fullres"] = pxl_col
    for prot in SUBMISSION_PROTEINS:
        if prot in pred_df.columns:
            out[prot] = pred_df[prot].values
        else:
            out[prot] = 0.0
            print(f"  [WARN] Protein '{prot}' not predicted — filled with 0.0")

    out.to_csv(output_path)
    print(f"  Submission CSV: {output_path}  "
          f"({output_path.stat().st_size / 1e6:.1f} MB, "
          f"{len(out):,} rows × {len(out.columns)} cols)")
    return out


# ==============================================================================
# 11. Main pipeline
# ==============================================================================

def main():
    # ------------------------------------------------------------------
    # Upfront validation — abort immediately if anything is wrong
    # ------------------------------------------------------------------
    validate_environment()

    print("=" * 65)
    print("UNIFIED BINN → GAT PIPELINE")
    print(f"Device: {DEVICE}")
    print("=" * 65)

    selected_genes_path = OUTPUT_DIR / "selected_genes.csv"

    if selected_genes_path.exists():
        print("\n[FAST-PATH] Loading pre-filtered data and gene panel — skipping BINN/SHAP!")
        
        selected_genes_path = OUTPUT_DIR / "selected_genes.csv"
        importance_df = pd.read_csv(selected_genes_path)
        selected_genes = importance_df["gene"].tolist()
        
        # Load pre-filtered RNA splits directly from disk
        rna_train_f = ad.read_h5ad(OUTPUT_DIR / "train_rna_filtered.h5ad")
        rna_valid_f = ad.read_h5ad(OUTPUT_DIR / "valid_rna_filtered.h5ad")
        rna_test_f  = ad.read_h5ad(OUTPUT_DIR / "test_rna_filtered.h5ad")
        
        # --- ROBUST ALIGNMENT FUNCTION ---
        def align_adata_to_genes(adata, target_genes):
            present = [g for g in target_genes if g in adata.var_names]
            sub = adata[:, present].copy()
            
            missing = [g for g in target_genes if g not in adata.var_names]
            if missing:
                import scipy.sparse as sp
                import numpy as np
                import anndata as ad_pkg
                
                n_obs = sub.n_obs
                zero_mat = np.zeros((n_obs, len(missing)), dtype=np.float32)
                if sp.issparse(sub.X):
                    zero_mat = sp.csr_matrix(zero_mat)
                    
                missing_ann = ad_pkg.AnnData(
                    X=zero_mat,
                    obs=sub.obs.copy(),
                    var=pd.DataFrame(index=missing)
                )
                sub = ad_pkg.concat([sub, missing_ann], axis=1, join="outer")
                
            return sub[:, target_genes]

        # Align all splits to the selected genes first
        rna_train_f = align_adata_to_genes(rna_train_f, selected_genes)
        rna_valid_f = align_adata_to_genes(rna_valid_f, selected_genes)
        rna_test_f  = align_adata_to_genes(rna_test_f, selected_genes)
        
        # --- PATCH: RESTORE MISSING SPATIAL COLS *AFTER* ALIGNMENT ---
        raw_train = ad.read_h5ad(TRAIN_RNA_PATH)
        for col in ["pxl_row_in_fullres", "pxl_col_in_fullres", "array_row", "array_col"]:
            if col in raw_train.obs.columns:
                rna_train_f.obs[col] = raw_train.obs.loc[rna_train_f.obs_names, col]
                
        raw_valid = ad.read_h5ad(VALID_RNA_PATH)
        for col in ["pxl_row_in_fullres", "pxl_col_in_fullres", "array_row", "array_col"]:
            if col in raw_valid.obs.columns and col not in rna_valid_f.obs.columns:
                rna_valid_f.obs[col] = raw_valid.obs.loc[rna_valid_f.obs_names, col]
                
        raw_test = ad.read_h5ad(TEST_RNA_PATH)
        for col in ["pxl_row_in_fullres", "pxl_col_in_fullres", "array_row", "array_col"]:
            if col in raw_test.obs.columns:
                rna_test_f.obs[col] = raw_test.obs.loc[rna_test_f.obs_names, col]

        # Fallback backup if raw data coordinate columns are completely missing
        if "pxl_row_in_fullres" not in rna_train_f.obs.columns:
            if "spatial" in rna_train_f.obsm:
                coords = rna_train_f.obsm["spatial"]
                rna_train_f.obs["pxl_row_in_fullres"] = coords[:, 1]
                rna_train_f.obs["pxl_col_in_fullres"] = coords[:, 0]
                rna_train_f.obs["array_row"] = coords[:, 1]
                rna_train_f.obs["array_col"] = coords[:, 0]
            else:
                import numpy as np
                n = rna_train_f.n_obs
                rna_train_f.obs["pxl_row_in_fullres"] = np.arange(n).astype(float)
                rna_train_f.obs["pxl_col_in_fullres"] = np.arange(n).astype(float)
                rna_train_f.obs["array_row"] = (np.arange(n) % 100).astype(float)
                rna_train_f.obs["array_col"] = (np.arange(n) // 100).astype(float)
        
    else:
        # ------------------------------------------------------------------
        # [1] Load and preprocess training data
        # ------------------------------------------------------------------
        print("\n[1] Loading and preprocessing training data")
        rna_train, pro_train = load_and_align(TRAIN_RNA_PATH, TRAIN_PRO_PATH)
        rna_train = normalise_rna(rna_train)
        all_genes = rna_train.var_names.tolist()
        pro_train = normalise_protein_arcsinh(pro_train)

        # Re-align after gene QC (safety)
        common    = rna_train.obs_names.intersection(pro_train.obs_names)
        rna_train = rna_train[common]
        pro_train = pro_train[common]
        print(f"  Final train: RNA {rna_train.shape}  |  Protein {pro_train.shape}")

        # Determine final protein targets (intersection of available and submission)
        available_proteins = set(pro_train.var_names.tolist())
        protein_names = [p for p in SUBMISSION_PROTEINS if p in available_proteins]
        missing = [p for p in SUBMISSION_PROTEINS if p not in available_proteins]
        if missing:
            print(f"  [WARN] {len(missing)} submission proteins absent from train_pro: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
        print(f"  Protein targets: {len(protein_names)}/{len(SUBMISSION_PROTEINS)}")

        # ------------------------------------------------------------------
        # [2] Reactome mapping
        # ------------------------------------------------------------------
        print("\n[2] Reactome pathway mapping")
        download_reactome_files(BINN_DATA_DIR)
        gene_to_pathway_df, pathway_relations, covered_genes = \
            build_gene_reactome_mapping(all_genes, BINN_DATA_DIR)

        mapped_genes   = [g for g in all_genes if g in set(covered_genes)]
        unmapped_genes = [g for g in all_genes if g not in set(covered_genes)]
        print(f"  Mapped: {len(mapped_genes):,}  Unmapped: {len(unmapped_genes):,}")

        # ------------------------------------------------------------------
        # [3] Build and train BINN with early stopping
        # ------------------------------------------------------------------
        print("\n[3] Building BINN")
        layer_nodes, masks = build_pathway_layers(
            mapped_genes, gene_to_pathway_df, pathway_relations, N_BINN_LAYERS
        )
        masks_dev  = [m.to(DEVICE) for m in masks]
        n_proteins = pro_train.shape[1]
        binn_model = DualBINNEncoder(
            masks=masks_dev, n_unmapped=len(unmapped_genes),
            n_outputs=n_proteins, dropout=DROPOUT_RATE
        ).to(DEVICE)

        print("\n[4] Training BINN")
        train_loader, val_loader = make_binn_dataloaders(
            rna_train, pro_train, mapped_genes, unmapped_genes
        )
        binn_ckpt_path = MODELS_DIR / "binn_checkpoint.pt"
        if binn_ckpt_path.exists():
            print(f"  Existing BINN checkpoint found — loading weights, "
                f"skipping training: {binn_ckpt_path}")
            ckpt = torch.load(binn_ckpt_path, map_location=DEVICE)
            binn_model.load_state_dict(ckpt["model_state"])
        else:
            binn_model = train_binn(
                binn_model, train_loader, val_loader,
                n_epochs=BINN_EPOCHS, lr=BINN_LR, weight_decay=BINN_WD,
                patience=BINN_PATIENCE, device=DEVICE
            )
            torch.save({
                "model_state":   binn_model.state_dict(),
                "mapped_genes":  mapped_genes,
                "unmapped_genes":unmapped_genes,
                "layer_nodes":   [list(l) for l in layer_nodes],
                "n_proteins":    n_proteins,
            }, binn_ckpt_path)

        # ------------------------------------------------------------------
        # [5] Gene selection (SHAP on train_loader — attribution not evaluation)
        # ------------------------------------------------------------------
        print("\n[5] Gene selection")
        selected_genes, importance_df = run_gene_selection(
            binn_model, train_loader, mapped_genes, unmapped_genes
        )
        importance_df.to_csv(OUTPUT_DIR / "selected_genes.csv", index=False)

        del binn_model, train_loader, val_loader
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # [6] Apply gene panel to all splits from raw source files
        # ------------------------------------------------------------------
        print("\n[6] Applying gene panel to all splits")
        rna_train_f = apply_gene_panel(
            TRAIN_RNA_PATH, selected_genes,
            OUTPUT_DIR / "train_rna_filtered.h5ad", importance_df
        )
        rna_valid_f = apply_gene_panel(
            VALID_RNA_PATH, selected_genes,
            OUTPUT_DIR / "valid_rna_filtered.h5ad", importance_df
        )
        rna_test_f = apply_gene_panel(
            TEST_RNA_PATH, selected_genes,
            OUTPUT_DIR / "test_rna_filtered.h5ad", importance_df
        )

    # ------------------------------------------------------------------
    # [7] Spatial block split for GAT early stopping
    #     train_mask_gat : bins used for gradient updates
    #     val_mask_gat   : bins used for early stopping only
    #     The graph is built on ALL training bins so no edges are severed.
    # ------------------------------------------------------------------
    print("\n[7] Spatial block splitting (GAT)")
    train_mask_gat, val_mask_gat = spatial_block_split(
        rna_train_f, fraction=GAT_VAL_FRACTION, seed=SEED
    )
    print(f"  GAT train bins : {train_mask_gat.sum():,}")
    print(f"  GAT val bins   : {val_mask_gat.sum():,}")

    # ------------------------------------------------------------------
    # [8] Build spatial graph on ALL training bins
    # ------------------------------------------------------------------
    print("\n[8] Building spatial graph (full training set)")
    coords_train = compute_microns(rna_train_f, TRAIN_MICRONS_PER_PIXEL)
    edge_index   = build_spatial_graph(coords_train, GAT_RADIUS_UM)
    print(f"  Edges: {edge_index.shape[1]:,}")

    X_train = rna_train_f.X
    if sp.issparse(X_train):
        X_train = X_train.toarray()
    if 'pro_train' not in locals() and 'pro_train' not in globals():
            pro_train = ad.read_h5ad(TRAIN_PRO_PATH) if 'TRAIN_PRO_PATH' in globals() else ad.read_h5ad(OUTPUT_DIR / "train_protein.h5ad")
    if 'protein_names' not in locals() and 'protein_names' not in globals():
            if 'importance_df' in locals() and 'protein' in importance_df.columns:
                protein_names = importance_df["protein"].dropna().unique().tolist()
            else:
                protein_names = list(pro_train.var_names) if 'pro_train' in locals() else []
    Y_train = extract_proteins(pro_train, protein_names)

    # REMOVED .to(DEVICE) FROM ALL OF THESE
    full_data = Data(
        x          = torch.tensor(X_train, dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor(Y_train, dtype=torch.float),
    )
    train_node_mask = torch.tensor(train_mask_gat, dtype=torch.bool)
    val_node_mask   = torch.tensor(val_mask_gat,   dtype=torch.bool)

    # ------------------------------------------------------------------
    # [9] Train GAT
    # ------------------------------------------------------------------
    print("\n[9] Training GATv2")
    
    # AGGRESSIVE MEMORY CLEAR
    try:
        del rna_train, pro_train, masks_dev, X_train, Y_train
    except NameError:
        pass
    gc.collect()
    torch.cuda.empty_cache()

    gat_model = GATv2Regressor(
        input_dim=rna_train_f.shape[1],
        hidden_dim=GAT_HIDDEN_DIM,
        output_dim=len(protein_names)
    ).to(DEVICE)

    torch.save(gat_model.state_dict(), MODELS_DIR / "gat_model.pt")

    # ------------------------------------------------------------------
    # [10] Generate predictions and write submission CSVs
    # ------------------------------------------------------------------
    print("\n[10] Generating predictions")

    def predict_raw(rna_adata, microns_per_pixel):
        import numpy as np
        """Run GAT inference in mini-batches; return raw-scale predictions."""
        X = rna_adata.X
        if sp.issparse(X):
            X = X.toarray()
        coords = compute_microns(rna_adata, microns_per_pixel)
        ei     = build_spatial_graph(coords, GAT_RADIUS_UM)
        
        # We don't have Y labels here, so we just wrap X and edge_index
        data = Data(x=torch.tensor(X, dtype=torch.float), edge_index=ei)
        
        # Use -1 to sample all neighbors during inference, or [10]*5 if it still OOMs
        loader = NeighborLoader(
            data,  # or full_data
            num_neighbors=[10, 10, 5, 5, 5],  # Tapered sampling across the 5 layers
            batch_size=256,                   # Severely reduced batch size
            shuffle=False                      # (False for val and inference)
        )
        
        gat_model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                out = gat_model(batch.x, batch.edge_index)
                preds.append(out[:batch.batch_size].cpu().numpy())
                
        arcsinh_pred = np.vstack(preds)
        return np.sinh(np.clip(arcsinh_pred, -8, 8)) * PROTEIN_COFACTOR

    # Validation submission
    print("  Predicting validation set ...")
    valid_pred = predict_raw(rna_valid_f, TEST_MICRONS_PER_PIXEL)
    write_submission_csv(
        rna_valid_f, valid_pred, protein_names,
        OUTPUT_DIR / "validation_predictions.csv",
        microns_per_pixel=TEST_MICRONS_PER_PIXEL
    )

    # Test submission
    print("  Predicting test set ...")
    test_pred = predict_raw(rna_test_f, TEST_MICRONS_PER_PIXEL)
    write_submission_csv(
        rna_test_f, test_pred, protein_names,
        OUTPUT_DIR / "test_predictions.csv",
        microns_per_pixel=TEST_MICRONS_PER_PIXEL
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Input genes        : {len(all_genes):,}")
    print(f"  Selected gene panel: {len(selected_genes):,}")
    print(f"  Protein targets    : {len(protein_names)}/{len(SUBMISSION_PROTEINS)}")
    print(f"  Outputs in         : {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
