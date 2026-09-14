"""
BINN Pipeline — Feature Extraction + Embedding Dump
====================================================
Stage 1 — Preprocessing
    Raw counts → normalize_total(1e4) → log1p. Applied once and flagged
    in .uns['is_preprocessed'] to prevent double-normalisation.

Stage 2 — Reactome pathway mapping
    Genes mapped to Reactome hierarchy via UniProt. Cached to disk so
    subsequent runs skip the network calls entirely.

Stage 3 — BINN training
    Dual-stream Biology-Informed Neural Network trained on ALL training
    bins with a 10% held-aside val fraction for early stopping only.
    Best checkpoint is restored after stopping.
    Skipped automatically if binn_checkpoint.pt already exists.

Stage 4 — Gene importance & selection
    Gradient×Input attribution for mapped genes (SHAP proxy).
    Gradient magnitude for unmapped genes.
    Elbow algorithm per stream finds the cutoff.
    Skipped automatically if selected_genes.csv already exists.

Stage 5 — BINN embedding dump  ← NEW
    Runs encode() (forward pass up to but not including the prediction
    head) over every bin in train, valid, and test sets.
    Saves:
        binn_embeddings_train.npy   shape (n_train_bins, embed_dim)
        binn_embeddings_valid.npy   shape (n_valid_bins, embed_dim)
        binn_embeddings_test.npy    shape (n_test_bins,  embed_dim)
        binn_embed_barcodes_train.pkl  obs_names in exact row order
        binn_embed_barcodes_valid.pkl
        binn_embed_barcodes_test.pkl
    These files are consumed directly by GAT_fulldata.py as node
    features, replacing the old SVD pipeline entirely.

Upfront validation
    ALL file paths and required obs columns are checked before any
    computation begins. Aborts immediately with a clear error message
    if anything is missing.
"""

# ==============================================================================
# 0. Imports
# ==============================================================================
import gc
import os
import pickle
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

# RNA preprocessing
NORMALIZE_TARGET = 1e4
MIN_COUNTS_GENE  = 5

# BINN
N_BINN_LAYERS   = 4
DROPOUT_RATE    = 0.2
BINN_BATCH_SIZE = 4096
BINN_EPOCHS     = 100
BINN_LR         = 1e-3
BINN_WD         = 1e-4
BINN_VAL_FRAC   = 0.10
BINN_PATIENCE   = 8

# SHAP proxy (gradient × input)
N_SHAP_BACKGROUND = 256   # background samples for GradientExplainer
N_SHAP_EXPLAIN    = 512   # instances to explain
SHAP_NSAMPLES     = 50    # samples per explanation (memory driver)

# Protein normalisation
PROTEIN_COFACTOR = 150.0

# Required obs columns in each RNA file
_REQUIRED_OBS = ["pxl_row_in_fullres", "pxl_col_in_fullres",
                 "array_row", "array_col"]

SUBMISSION_PROTEINS = [
    "synd", "FOXP3", "CD16", "CD31", "CXCL13", "Ki67", "OLIG2", "CXCR5",
    "HLA-A", "PD-L1", "PSD95", "CD20", "CD68", "CD44", "SMA", "MSH6",
    "CD23", "GFAP", "SYNA", "Podoplanin", "Vimentin", "CD47", "CD74",
    "SIRP", "Granzyme B", "IDH1", "MPO", "CD45", "CD21", "FIBR", "C-KIT",
    "CD3e", "TOX", "PD-1", "PDGFR", "CD4", "MAP2", "CD8", "MGMT", "CD38",
    "HLA-DR", "CD14", "ICOS", "Granzyme K",
]


# ==============================================================================
# 2. Upfront validation
# ==============================================================================
def validate_environment():
    errors = []

    required_files = {
        "Train RNA"    : TRAIN_RNA_PATH,
        "Train protein": TRAIN_PRO_PATH,
        "Valid RNA"    : VALID_RNA_PATH,
        "Test RNA"     : TEST_RNA_PATH,
    }
    for label, path in required_files.items():
        if not path.exists():
            errors.append(f"Missing {label}: {path}")

    for d in [OUTPUT_DIR, MODELS_DIR, BINN_DATA_DIR]:
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_test"
            probe.touch(); probe.unlink()
        except Exception as e:
            errors.append(f"Cannot write to {d}: {e}")

    for label, path in [("Train RNA", TRAIN_RNA_PATH),
                         ("Valid RNA", VALID_RNA_PATH),
                         ("Test RNA",  TEST_RNA_PATH)]:
        if not path.exists():
            continue
        try:
            adata = ad.read_h5ad(path, backed="r")
            missing = [c for c in _REQUIRED_OBS if c not in adata.obs.columns]
            if missing:
                errors.append(f"{label} missing obs columns: {missing}")
            adata.file.close()
        except Exception as e:
            errors.append(f"Cannot read {label} ({path}): {e}")

    if not torch.cuda.is_available():
        print("  [WARN] No CUDA GPU detected — will be very slow on CPU.")

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
# 3. Preprocessing utilities
# ==============================================================================
def load_and_align(rna_path, pro_path=None, in_tissue_only=True):
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
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    adata.X = np.arcsinh(X.astype(np.float32) / cofactor)
    adata.uns["protein_normalised"] = "arcsinh"
    adata.uns["protein_cofactor"]   = cofactor
    return adata


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
        print("  Loading cached Reactome mapping.")
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
        self.embed_dim = merged_dim
        print(f"  DualBINNEncoder: Stream1={binn_bn}  Stream2=128  "
              f"→ embed_dim={merged_dim}  → head → {n_outputs}")

    def encode_pathway(self, x_mapped):
        x = x_mapped
        for linear, bn, drop in zip(self.pathway_layers,
                                     self.batch_norms, self.dropouts):
            x = drop(torch.tanh(bn(linear(x))))
        return x

    def encode(self, x_mapped, x_unmapped):
        """Bottleneck embedding — same as forward() but stops before the head.
        This is the vector fed to the GAT as node features."""
        return torch.cat([
            self.encode_pathway(x_mapped),
            self.dense_stream(x_unmapped),
        ], dim=1)

    def forward(self, x_mapped, x_unmapped):
        return self.head(self.encode(x_mapped, x_unmapped))


# ==============================================================================
# 6. Dataset and data loaders
# ==============================================================================
class DualSparseDataset(Dataset):
    """Sparse RNA → dense gene-subset batches via __getitems__."""
    def __init__(self, X_sparse, Y_dense, gene_list, mapped_genes, unmapped_genes):
        var_idx   = {g: i for i, g in enumerate(gene_list)}
        m_present = [g for g in mapped_genes   if g in var_idx]
        u_present = [g for g in unmapped_genes if g in var_idx]
        self.m_src = np.array([var_idx[g] for g in m_present], dtype=np.int64)
        self.m_dst = np.array([i for i, g in enumerate(mapped_genes)
                                if g in var_idx], dtype=np.int64)
        self.u_src = np.array([var_idx[g] for g in u_present], dtype=np.int64)
        self.u_dst = np.array([i for i, g in enumerate(unmapped_genes)
                                if g in var_idx], dtype=np.int64)
        self.n_mapped   = len(mapped_genes)
        self.n_unmapped = len(unmapped_genes)
        self.X = X_sparse.tocsr()
        self.Y = torch.tensor(Y_dense.astype(np.float32)) if Y_dense is not None \
                 else torch.zeros(X_sparse.shape[0], 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitems__(self, idx_list):
        idx     = np.asarray(idx_list)
        block   = self.X[idx]
        block_m = block[:, self.m_src].toarray()
        block_u = block[:, self.u_src].toarray()
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
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=gen)

    pin      = DEVICE.type == "cuda"
    n_workers = 0
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=n_workers,
                              pin_memory=pin,
                              persistent_workers=(n_workers > 0))
    val_loader   = DataLoader(val_ds, batch_size=batch_size * 2,
                              shuffle=False, num_workers=n_workers,
                              pin_memory=pin,
                              persistent_workers=(n_workers > 0))
    print(f"  BINN train: {n_train:,} bins  |  val: {n_val:,} bins")
    return train_loader, val_loader


def make_embed_loader(rna_adata, mapped_genes, unmapped_genes,
                      batch_size=BINN_BATCH_SIZE * 2):
    """Loader for embedding extraction — no labels needed, no shuffle."""
    X = rna_adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    ds = DualSparseDataset(X, None, rna_adata.var_names.tolist(),
                           mapped_genes, unmapped_genes)
    pin      = DEVICE.type == "cuda"
    n_workers = 0
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=n_workers, pin_memory=pin,
                      persistent_workers=(n_workers > 0))


# ==============================================================================
# 7. BINN training
# ==============================================================================
def train_binn(model, train_loader, val_loader, n_epochs, lr, weight_decay,
               patience, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)
    criterion = nn.MSELoss()

    best_val   = float("inf")
    best_state = None
    pat_ctr    = 0

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
# 8. Gene importance & selection
# ==============================================================================
def compute_shap_importance(model, loader, mapped_genes, device=DEVICE,
                             n_explain=N_SHAP_EXPLAIN):
    """
    True SHAP via shap.GradientExplainer on the pathway stream only.
    num_workers=0 on the loader avoids the OOM that triggered the proxy.
    """
    import shap

    model.eval()

    # Collect background and explain batches from the loader
    background_xm, background_xu = [], []
    explain_xm = []
    n_bg   = N_SHAP_BACKGROUND   # add this constant to config: e.g. 256
    n_seen = 0

    for Xm, Xu, _ in loader:
        if n_seen < n_bg:
            take = min(n_bg - n_seen, Xm.shape[0])
            background_xm.append(Xm[:take])
            background_xu.append(Xu[:take])
            n_seen += take
        else:
            explain_xm.append(Xm)
            if sum(x.shape[0] for x in explain_xm) >= n_explain:
                break

    bg_xm = torch.cat(background_xm).to(device)
    bg_xu = torch.cat(background_xu).to(device)
    ex_xm = torch.cat(explain_xm)[:n_explain].to(device)

    # GradientExplainer wraps the pathway stream only.
    # We freeze xu at background mean so the explainer sees a
    # single-input function over xm.
    bg_xu_mean = bg_xu.mean(dim=0, keepdim=True).expand(bg_xm.shape[0], -1)
    ex_xu_mean = bg_xu.mean(dim=0, keepdim=True).expand(ex_xm.shape[0], -1)

    def pathway_only(xm):
        # xu held constant — SHAP varies xm only
        xu_const = bg_xu.mean(dim=0, keepdim=True).expand(xm.shape[0], -1)
        return model(xm, xu_const)

    explainer   = shap.GradientExplainer(pathway_only, bg_xm)
    shap_values = explainer.shap_values(ex_xm, nsamples=SHAP_NSAMPLES)
    # shap_values: list of (n_explain, n_mapped) — one per output protein
    importance  = np.abs(np.stack(shap_values, axis=0)).mean(axis=(0, 1))

    assert importance.shape[0] == len(mapped_genes)
    return pd.DataFrame({
        "gene":            mapped_genes,
        "shap_importance": importance,
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)


def compute_gradient_importance(model, loader, unmapped_genes, device=DEVICE,
                                 n_explain=N_SHAP_EXPLAIN):
    model.eval()
    accum   = torch.zeros(len(unmapped_genes), device=device)
    n_total = 0

    for Xm, Xu, Yb in loader:
        Xm = Xm.to(device)
        Xu = Xu.to(device).requires_grad_(True)
        Yb = Yb.to(device)
        loss = F.mse_loss(model(Xm, Xu), Yb)
        loss.backward()
        with torch.no_grad():
            accum   += Xu.grad.abs().sum(dim=0)
            n_total += Xu.shape[0]
        Xu.grad = None
        if n_total >= n_explain:
            break

    importance = (accum / n_total).cpu().numpy()
    return pd.DataFrame({
        "gene":            unmapped_genes,
        "grad_importance": importance,
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
    print("  Running gradient×input importance on mapped genes ...")
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
# 9. BINN embedding dump  ← NEW STAGE
# ==============================================================================
def dump_embeddings(model, rna_adata, mapped_genes, unmapped_genes,
                    out_npy, out_barcodes_pkl, split_label):
    """
    Run model.encode() over every bin in rna_adata and save the result.

    Outputs
    -------
    out_npy             numpy array (n_bins, embed_dim) — float32
    out_barcodes_pkl    list of obs_names in exact row order

    The barcode pickle is critical: GAT_fulldata.py loads it and asserts
    that the embedding row order matches the h5ad obs_names before building
    the graph, preventing silent misalignment.
    """
    print(f"  Dumping embeddings for {split_label}: "
          f"{rna_adata.n_obs:,} bins ...")

    loader = make_embed_loader(rna_adata, mapped_genes, unmapped_genes)

    model.eval()
    parts = []
    with torch.no_grad():
        for Xm, Xu, _ in tqdm(loader, desc=f"  {split_label} embed"):
            Xm = Xm.to(DEVICE)
            Xu = Xu.to(DEVICE)
            z  = model.encode(Xm, Xu)          # (batch, embed_dim)
            parts.append(z.cpu().numpy())

    embeddings = np.concatenate(parts, axis=0).astype(np.float32)
    np.save(out_npy, embeddings)

    barcodes = rna_adata.obs_names.tolist()
    with open(out_barcodes_pkl, "wb") as f:
        pickle.dump(barcodes, f)

    print(f"  Saved {out_npy.name}  shape={embeddings.shape}")
    print(f"  Saved {out_barcodes_pkl.name}  ({len(barcodes):,} barcodes)")
    return embeddings


# ==============================================================================
# 10. Main
# ==============================================================================
def main():
    validate_environment()

    print("=" * 65)
    print("BINN PIPELINE  (training + gene selection + embedding dump)")
    print(f"Device: {DEVICE}")
    print("=" * 65)

    binn_ckpt_path    = MODELS_DIR / "binn_checkpoint.pt"
    selected_csv_path = OUTPUT_DIR / "selected_genes.csv"

    # ------------------------------------------------------------------
    # [1] Load and preprocess training data
    # ------------------------------------------------------------------
    print("\n[1] Loading and preprocessing training data")
    rna_train, pro_train = load_and_align(TRAIN_RNA_PATH, TRAIN_PRO_PATH)
    rna_train = normalise_rna(rna_train)
    all_genes = rna_train.var_names.tolist()
    pro_train = normalise_protein_arcsinh(pro_train)

    common    = rna_train.obs_names.intersection(pro_train.obs_names)
    rna_train = rna_train[common].copy()
    pro_train = pro_train[common].copy()
    print(f"  Final train: RNA {rna_train.shape}  |  Protein {pro_train.shape}")

    available_proteins = set(pro_train.var_names.tolist())
    protein_names      = [p for p in SUBMISSION_PROTEINS if p in available_proteins]
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
    # [3] Build BINN architecture
    # ------------------------------------------------------------------
    print("\n[3] Building BINN architecture")
    layer_nodes, masks = build_pathway_layers(
        mapped_genes, gene_to_pathway_df, pathway_relations, N_BINN_LAYERS
    )
    masks_dev  = [m.to(DEVICE) for m in masks]
    n_proteins = pro_train.shape[1]
    binn_model = DualBINNEncoder(
        masks=masks_dev, n_unmapped=len(unmapped_genes),
        n_outputs=n_proteins, dropout=DROPOUT_RATE
    ).to(DEVICE)

    # ------------------------------------------------------------------
    # [4] Train BINN (or load existing checkpoint)
    # ------------------------------------------------------------------
    print("\n[4] BINN training")
    if binn_ckpt_path.exists():
        print(f"  Existing checkpoint found — loading: {binn_ckpt_path}")
        ckpt = torch.load(binn_ckpt_path, map_location=DEVICE)
        binn_model.load_state_dict(ckpt["model_state"])
        print("  Checkpoint loaded. Skipping training.")
    else:
        train_loader, val_loader = make_binn_dataloaders(
            rna_train, pro_train, mapped_genes, unmapped_genes
        )
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
            "embed_dim":     binn_model.embed_dim,
        }, binn_ckpt_path)
        print(f"  Checkpoint saved: {binn_ckpt_path}")
        # clean up loaders — not needed beyond this point
        del train_loader, val_loader
        gc.collect()

    # ------------------------------------------------------------------
    # [5] Gene selection (SHAP proxy) — skipped if CSV already exists
    # ------------------------------------------------------------------
    print("\n[5] Gene selection")
    if selected_csv_path.exists():
        print(f"  Cached — loading: {selected_csv_path}")
        importance_df  = pd.read_csv(selected_csv_path)
        selected_genes = importance_df["gene"].tolist()
        print(f"  {len(selected_genes):,} genes loaded from cache.")
    else:
        # Need a fresh loader for attribution (must iterate from scratch)
        attr_loader, _ = make_binn_dataloaders(
            rna_train, pro_train, mapped_genes, unmapped_genes
        )
        selected_genes, importance_df = run_gene_selection(
            binn_model, attr_loader, mapped_genes, unmapped_genes
        )
        importance_df.to_csv(selected_csv_path, index=False)
        print(f"  Selected genes written: {selected_csv_path}")
        del attr_loader
        gc.collect()

    # ------------------------------------------------------------------
    # [6] Embedding dump for train / valid / test
    # Note: embeddings use ALL 18k genes (the full BINN inputs), NOT
    # the elbow-selected panel. The BINN was trained on all genes and
    # encode() reflects that. Gene selection above is kept for reference
    # and backward compatibility but is NOT used as input to the GAT.
    # ------------------------------------------------------------------
    print("\n[6] Dumping BINN bottleneck embeddings")
    binn_model.eval()

    # Free protein data — not needed for embedding pass
    del pro_train
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # --- Train embeddings ---
    train_npy = OUTPUT_DIR / "binn_embeddings_train.npy"
    train_pkl = OUTPUT_DIR / "binn_embed_barcodes_train.pkl"
    if train_npy.exists() and train_pkl.exists():
        print(f"  Train embeddings cached — skipping.")
    else:
        dump_embeddings(
            binn_model, rna_train, mapped_genes, unmapped_genes,
            train_npy, train_pkl, "train"
        )

    del rna_train
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # --- Valid embeddings ---
    valid_npy = OUTPUT_DIR / "binn_embeddings_valid.npy"
    valid_pkl = OUTPUT_DIR / "binn_embed_barcodes_valid.pkl"
    if valid_npy.exists() and valid_pkl.exists():
        print(f"  Valid embeddings cached — skipping.")
    else:
        print("\n  Loading valid RNA ...")
        rna_valid, _ = load_and_align(VALID_RNA_PATH, in_tissue_only=True)
        rna_valid    = normalise_rna(rna_valid)
        dump_embeddings(
            binn_model, rna_valid, mapped_genes, unmapped_genes,
            valid_npy, valid_pkl, "valid"
        )
        del rna_valid
        gc.collect()

    # --- Test embeddings ---
    test_npy = OUTPUT_DIR / "binn_embeddings_test.npy"
    test_pkl = OUTPUT_DIR / "binn_embed_barcodes_test.pkl"
    if test_npy.exists() and test_pkl.exists():
        print(f"  Test embeddings cached — skipping.")
    else:
        print("\n  Loading test RNA ...")
        rna_test, _ = load_and_align(TEST_RNA_PATH, in_tissue_only=True)
        rna_test    = normalise_rna(rna_test)
        dump_embeddings(
            binn_model, rna_test, mapped_genes, unmapped_genes,
            test_npy, test_pkl, "test"
        )
        del rna_test
        gc.collect()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    embed_dim = np.load(train_npy).shape[1]
    print("\n" + "=" * 65)
    print("BINN PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Mapped genes (Stream 1)  : {len(mapped_genes):,}")
    print(f"  Unmapped genes (Stream 2): {len(unmapped_genes):,}")
    print(f"  Selected gene panel      : {len(selected_genes):,}")
    print(f"  Embedding dimension      : {embed_dim}")
    print(f"  Outputs in               : {OUTPUT_DIR}")
    print(f"\n  Files ready for GAT_fulldata.py:")
    for f in [train_npy, train_pkl, valid_npy, valid_pkl, test_npy, test_pkl]:
        size = f.stat().st_size / 1e6
        print(f"    {f.name:<45s} {size:.1f} MB")
    print("=" * 65)


if __name__ == "__main__":
    main()
