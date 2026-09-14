"""
binn_gene_extraction.py
=======================
Load a pretrained Dual-Stream BINN checkpoint and extract the most important
genes from both streams (SHAP for the Reactome pathway stream, gradient
importance for the dense MLP stream). Optionally filter one or more .h5ad
files to retain only the selected genes.

Usage
-----
    python binn_gene_extraction.py \
        --checkpoint    binn_checkpoint.pt \
        --train_rna     /data/train_rna.h5ad \
        --train_pro     /data/train_pro.h5ad \
        --binn_data_dir /data/binn_data \
        --output_dir    /data/outputs \
        [--h5ad_to_filter /data/train_rna.h5ad /data/valid_rna.h5ad ...] \
        [--cumulative_frac 0.90] \
        [--n_genes_fixed 3000] \
        [--n_shap_background 256] \
        [--n_shap_explain    512] \
        [--batch_size 4096] \
        [--n_binn_layers 4] \
        [--dropout 0.2] \
        [--seed 42]

Checkpoint format
-----------------
The script accepts two checkpoint formats:

  1. Full metadata dict (saved by Section 10 of the notebook):
       {'model_state', 'binn_input_genes', 'unmapped_genes', 'selected_genes',
        'layer_node_names', 'n_proteins', 'dropout', 'training_config'}

  2. Raw state_dict only (saved by Section 6 of the notebook).
     In this case --train_rna / --train_pro / --binn_data_dir are required
     so the script can rebuild the architecture from scratch.

Outputs (written to --output_dir)
----------------------------------
  gene_shap_importance.csv        ranked mapped-gene SHAP importance
  gene_dense_importance.csv       ranked unmapped-gene gradient importance
  selected_genes.csv              final selected gene list with scores
  <stem>_filtered.h5ad            filtered copy of each --h5ad_to_filter
  gene_importance.png             SHAP importance bar-chart + cumulative curve
"""

# ── Standard library ────────────────────────────────────────────────────────
import argparse
import gc
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Optional heavy imports (fail early with a clear message) ─────────────────
try:
    import anndata as ad
    import scanpy as sc
except ImportError:
    sys.exit("anndata / scanpy not found.  Run: pip install anndata scanpy")

try:
    import shap
except ImportError:
    sys.exit("shap not found.  Run: pip install shap")

try:
    from kneed import KneeLocator
except ImportError:
    sys.exit("kneed not found.  Run: pip install kneed")

try:
    import networkx as nx
except ImportError:
    sys.exit("networkx not found.  Run: pip install networkx")

try:
    import requests
    from tqdm.auto import tqdm
except ImportError:
    sys.exit("requests / tqdm not found.  Run: pip install requests tqdm")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ══════════════════════════════════════════════════════════════════════════════
# 1. Model architecture  (identical to the notebook)
# ══════════════════════════════════════════════════════════════════════════════

class MaskedLinear(nn.Module):
    """Linear layer with a fixed binary pathway-connectivity mask."""

    def __init__(self, in_features: int, out_features: int,
                 mask: torch.Tensor, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.register_buffer("mask", mask.float())
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


class DualBINNEncoder(nn.Module):
    """
    Dual-stream BINN encoder.

    Stream 1 — Pathway BINN : Reactome-mapped genes → sparse MaskedLinear layers.
    Stream 2 — Dense MLP    : unmapped genes → two-layer MLP.

    Both streams are concatenated and passed through a shared prediction head.
    """

    def __init__(self, masks: list, n_unmapped: int, n_outputs: int,
                 dropout: float = 0.2):
        super().__init__()

        # Stream 1 ─────────────────────────────────────────────────────────
        self.pathway_layers = nn.ModuleList()
        self.batch_norms    = nn.ModuleList()
        self.dropouts       = nn.ModuleList()

        for mask in masks:
            out_dim, in_dim = mask.shape
            self.pathway_layers.append(MaskedLinear(in_dim, out_dim, mask))
            self.batch_norms.append(nn.BatchNorm1d(out_dim))
            self.dropouts.append(nn.Dropout(p=dropout))

        binn_bottleneck = masks[-1].shape[0]

        # Stream 2 ─────────────────────────────────────────────────────────
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
        dense_bottleneck = 128

        # Shared head ──────────────────────────────────────────────────────
        merged_dim = binn_bottleneck + dense_bottleneck
        self.head = nn.Sequential(
            nn.Linear(merged_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_outputs),
        )

    def encode_pathway(self, x_mapped: torch.Tensor) -> torch.Tensor:
        x = x_mapped
        for linear, bn, drop in zip(self.pathway_layers,
                                     self.batch_norms, self.dropouts):
            x = drop(torch.tanh(bn(linear(x))))
        return x

    def encode(self, x_mapped: torch.Tensor,
               x_unmapped: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.encode_pathway(x_mapped),
                          self.dense_stream(x_unmapped)], dim=1)

    def forward(self, x_mapped: torch.Tensor,
                x_unmapped: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x_mapped, x_unmapped))


# ══════════════════════════════════════════════════════════════════════════════
# 2. Data loading / preprocessing helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_spatial_data(rna_path: Path, pro_path: Path = None,
                      in_tissue_only: bool = True):
    print(f"Loading RNA from {rna_path} ...")
    rna = ad.read_h5ad(rna_path, backed="r")
    print(f"  Shape: {rna.shape}  (bins × genes)")

    rna_obs = (rna.obs_names[rna.obs["in_tissue"] == 1]
               if in_tissue_only and "in_tissue" in rna.obs.columns
               else rna.obs_names)

    pro = None
    if pro_path is not None:
        print(f"Loading protein from {pro_path} ...")
        pro = ad.read_h5ad(pro_path, backed="r")
        print(f"  Shape: {pro.shape}  (bins × markers)")
        pro_obs = (pro.obs_names[pro.obs["in_tissue"] == 1]
                   if in_tissue_only and "in_tissue" in pro.obs.columns
                   else pro.obs_names)
        common  = rna_obs.intersection(pro_obs)
        print(f"  Common in-tissue bins: {len(common)}")
        rna = rna[common].to_memory()
        pro = pro[common].to_memory()
    else:
        rna = rna[rna_obs].to_memory()

    if not sp.issparse(rna.X):
        rna.X = sp.csr_matrix(rna.X)
    if pro is not None and not sp.issparse(pro.X):
        pro.X = sp.csr_matrix(pro.X)

    print(f"  RNA loaded: {rna.shape}")
    return rna, pro


def preprocess_rna(adata, normalize_target: float = 1e4,
                   min_counts_gene: int = 5):
    print("Preprocessing RNA ...")
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    sc.pp.filter_genes(adata, min_counts=min_counts_gene)
    print(f"  After gene QC filter: {adata.shape}")
    sc.pp.normalize_total(adata, target_sum=normalize_target, inplace=True)
    sc.pp.log1p(adata, chunked=True, chunk_size=10_000)
    adata.X = adata.X.tocsr()
    all_genes = adata.var_names.tolist()
    print(f"  Preprocessing complete: {adata.shape}")
    return adata, all_genes


def preprocess_protein(adata, chunk_size: int = 500):
    print("Preprocessing protein (CLR normalisation) ...")
    n_bins, n_markers = adata.shape
    X_clr = np.empty((n_bins, n_markers), dtype=np.float32)
    for start in range(0, n_bins, chunk_size):
        end   = min(start + chunk_size, n_bins)
        chunk = adata.X[start:end]
        if sp.issparse(chunk):
            chunk = chunk.toarray()
        chunk = chunk.astype(np.float32) + 1.0
        geom  = np.exp(np.mean(np.log(chunk), axis=1, keepdims=True))
        X_clr[start:end] = np.log(chunk / geom)
    mean_ = X_clr.mean(axis=0, keepdims=True)
    std_  = X_clr.std(axis=0,  keepdims=True) + 1e-8
    X_clr = (X_clr - mean_) / std_
    adata      = adata.copy()
    adata.X    = X_clr
    adata.uns["protein_clr_mean"] = mean_
    adata.uns["protein_clr_std"]  = std_
    print(f"  Protein matrix shape: {adata.X.shape}")
    return adata


# ══════════════════════════════════════════════════════════════════════════════
# 3. Reactome pathway mapping
# ══════════════════════════════════════════════════════════════════════════════

def fetch_gene_to_uniprot_batch(gene_symbols: list,
                                batch_size: int = 200) -> dict:
    print(f"Querying MyGene.info for {len(gene_symbols)} gene symbols...")
    symbol_to_uniprot = {}
    url     = "https://mygene.info/v3/query"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    for i in tqdm(range(0, len(gene_symbols), batch_size),
                  desc="Gene→UniProt"):
        batch   = gene_symbols[i : i + batch_size]
        payload = {"q": ",".join(batch), "scopes": "symbol",
                   "fields": "uniprot", "species": "human",
                   "size": batch_size}
        try:
            resp = requests.post(url, data=payload, headers=headers,
                                 timeout=30)
            resp.raise_for_status()
            for hit in resp.json():
                sym = hit.get("query", "")
                if "uniprot" in hit and "Swiss-Prot" in hit["uniprot"]:
                    acc = hit["uniprot"]["Swiss-Prot"]
                    symbol_to_uniprot[sym] = (
                        [acc] if isinstance(acc, str) else acc)
        except Exception as e:
            print(f"  Warning: batch {i // batch_size} failed: {e}")

    print(f"  Mapped {len(symbol_to_uniprot)}/{len(gene_symbols)} genes.")
    return symbol_to_uniprot


def build_gene_reactome_mapping(gene_symbols: list, binn_data_dir: Path,
                                cache_path: Path = None):
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached mapping from {cache_path}")
        gene_to_pathway = pd.read_csv(cache_path)
    else:
        up2r = pd.read_csv(
            binn_data_dir / "uniprot_2_reactome_2025_01_14.txt",
            sep="\t", header=None,
            names=["input", "translation", "url", "name", "evidence",
                   "species"],
        )
        up2r_human = up2r[up2r["species"] == "Homo sapiens"]
        sym2up     = fetch_gene_to_uniprot_batch(gene_symbols)

        rows = []
        for sym, uids in sym2up.items():
            for uid in uids:
                matches = up2r_human[up2r_human["input"] == uid].copy()
                if len(matches):
                    matches["input"] = sym
                    rows.append(matches)

        if not rows:
            raise RuntimeError(
                "No genes mapped to Reactome. Check network connectivity.")

        gene_to_pathway = (pd.concat(rows, ignore_index=True)
                           .drop_duplicates())
        if cache_path:
            gene_to_pathway.to_csv(cache_path, index=False)
            print(f"Mapping saved to {cache_path}")

    p_rel = pd.read_csv(
        binn_data_dir / "reactome_pathways_relation_2025_01_14.txt",
        sep="\t", header=None, names=["target", "source"],
    )
    p_rel_human = p_rel[
        p_rel["target"].str.startswith("R-HSA") &
        p_rel["source"].str.startswith("R-HSA")
    ]
    pathway_relations = list(
        p_rel_human.itertuples(index=False, name=None))
    covered_genes = gene_to_pathway["input"].unique().tolist()

    print(f"  Genes with Reactome mapping : {len(covered_genes)}")
    print(f"  Human pathway relations     : {len(pathway_relations)}")
    return gene_to_pathway, pathway_relations, covered_genes


def build_pathway_layers(gene_symbols_subset: list,
                         gene_to_pathway_df: pd.DataFrame,
                         pathway_relations: list, n_layers: int):
    G = nx.DiGraph()
    for child, parent in pathway_relations:
        G.add_edge(child, parent)

    g2p        = gene_to_pathway_df[
        gene_to_pathway_df["input"].isin(gene_symbols_subset)]
    gene_to_l1 = g2p.groupby("input")["translation"].apply(list).to_dict()

    layer_nodes = [gene_symbols_subset]
    layer_sets  = [set(gene_symbols_subset)]

    current = set()
    for g in gene_symbols_subset:
        current.update(gene_to_l1.get(g, []))

    for depth in range(n_layers):
        layer_nodes.append(sorted(current))
        layer_sets.append(current)
        nxt = set()
        for p in current:
            nxt.update(G.successors(p))
        nxt -= layer_sets[1]
        current = nxt
        if not current:
            print(f"  Hierarchy exhausted at layer {depth + 1}")
            break

    print(f"Layer sizes: {[len(l) for l in layer_nodes]}")

    masks    = []
    gene_idx = {g: i for i, g in enumerate(layer_nodes[0])}
    l1_idx   = {p: i for i, p in enumerate(layer_nodes[1])}

    mask0 = torch.zeros(len(layer_nodes[1]), len(layer_nodes[0]),
                        dtype=torch.bool)
    for g, pathways in gene_to_l1.items():
        if g in gene_idx:
            for p in pathways:
                if p in l1_idx:
                    mask0[l1_idx[p], gene_idx[g]] = True
    masks.append(mask0)

    for d in range(1, len(layer_nodes) - 1):
        src = {n: i for i, n in enumerate(layer_nodes[d])}
        dst = {n: i for i, n in enumerate(layer_nodes[d + 1])}
        m   = torch.zeros(len(layer_nodes[d + 1]), len(layer_nodes[d]),
                          dtype=torch.bool)
        for child, parent in pathway_relations:
            if child in src and parent in dst:
                m[dst[parent], src[child]] = True
        masks.append(m)

    for i, m in enumerate(masks):
        density = m.float().mean().item() * 100
        print(f"  Mask {i}: shape={tuple(m.shape)}  "
              f"sparsity={100 - density:.1f}%")

    return layer_nodes, masks


# ══════════════════════════════════════════════════════════════════════════════
# 4. DataLoader (inference-only — no target needed for importance computation)
# ══════════════════════════════════════════════════════════════════════════════

class DualSparseDataset(torch.utils.data.Dataset):
    """Sparse RNA dataset for inference — no protein target required."""

    def __init__(self, X_sparse, gene_list: list,
                 mapped_genes: list, unmapped_genes: list):
        var_idx = {g: i for i, g in enumerate(gene_list)}

        m_present       = [g for g in mapped_genes  if g in var_idx]
        self.m_src_cols = [var_idx[g] for g in m_present]
        self.m_dst_cols = [i for i, g in enumerate(mapped_genes)
                           if g in var_idx]
        self.n_mapped   = len(mapped_genes)

        u_present       = [g for g in unmapped_genes if g in var_idx]
        self.u_src_cols = [var_idx[g] for g in u_present]
        self.u_dst_cols = [i for i, g in enumerate(unmapped_genes)
                           if g in var_idx]
        self.n_unmapped = len(unmapped_genes)

        self.X = X_sparse.tocsr()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx]
        if sp.issparse(row):
            row = row.toarray().squeeze()
        x_m = np.zeros(self.n_mapped,   dtype=np.float32)
        x_u = np.zeros(self.n_unmapped, dtype=np.float32)
        x_m[self.m_dst_cols] = row[self.m_src_cols]
        x_u[self.u_dst_cols] = row[self.u_src_cols]
        return torch.tensor(x_m), torch.tensor(x_u)


def make_inference_loader(rna_adata, mapped_genes: list,
                          unmapped_genes: list, batch_size: int = 32768,
                          num_workers: int = 4):
    X = rna_adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    gene_list = rna_adata.var_names.tolist()
    ds        = DualSparseDataset(X, gene_list, mapped_genes, unmapped_genes)
    pin       = torch.cuda.is_available()
    loader    = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=False)  # Set to False for single-pass loaders
    return loader


# ══════════════════════════════════════════════════════════════════════════════
# 5. Importance computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap_importance_stream1(model: nn.Module,
                                    loader: torch.utils.data.DataLoader,
                                    n_background: int = 128,
                                    n_explain:    int = 256,
                                    device: torch.device = None,
                                    shap_batch_size: int = 16) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Collect exact required slices to minimize CPU RAM usage
    Xm_list, Xu_list = [], []
    collected = 0
    target_count = n_background + n_explain

    for Xm, Xu in loader:
        Xm_list.append(Xm)
        Xu_list.append(Xu)
        collected += Xm.shape[0]
        if collected >= target_count:
            break

    all_Xm = torch.cat(Xm_list, dim=0)[:target_count]
    all_Xu = torch.cat(Xu_list, dim=0)[:target_count]
    
    del Xm_list, Xu_list
    gc.collect()

    bg_Xm  = all_Xm[:n_background].to(device)
    bg_Xu  = all_Xu[:n_background].to(device)
    ex_Xm  = all_Xm[n_background:].to(device)

    fixed_Xu_mean = bg_Xu.mean(dim=0, keepdim=True)
    del all_Xm, all_Xu, bg_Xu
    gc.collect()

    class MappedWrapper(nn.Module):
        def __init__(self, model, fixed_Xu_mean):
            super().__init__()
            self.model         = model
            self.fixed_Xu_mean = fixed_Xu_mean
            
        def forward(self, Xm):
            fixed = self.fixed_Xu_mean.expand(Xm.shape[0], -1)
            return self.model(Xm, fixed)

    wrapper   = MappedWrapper(model, fixed_Xu_mean)
    explainer = shap.GradientExplainer(wrapper, bg_Xm)

    print(f"  GradientExplainer (Stream 1): {ex_Xm.shape[0]} samples, "
          f"{bg_Xm.shape[0]} background, {ex_Xm.shape[1]} mapped genes ...")

    n_mapped = ex_Xm.shape[1]
    total_shap = np.zeros(n_mapped, dtype=np.float32)

    for i in range(0, ex_Xm.shape[0], shap_batch_size):
        ex_batch = ex_Xm[i : i + shap_batch_size]
        
        # sv_batch is a list of length 44 (n_proteins), each of shape (batch_size, n_mapped)
        sv_batch = explainer.shap_values(ex_batch)
        
        if isinstance(sv_batch, list):
            # Shape: (44, batch_size, 10416)
            sv_arr = np.array(sv_batch)
            # Take abs, sum across samples (axis 1) and across protein targets (axis 0) -> (10416,)
            batch_imp = np.abs(sv_arr).sum(axis=(0, 1))
        else:
            sv_arr = np.array(sv_batch)
            if sv_arr.ndim == 3:
                # Shape: (batch_size, 10416, 44) or (44, batch_size, 10416)
                if sv_arr.shape[0] == ex_batch.shape[0]:
                    batch_imp = np.abs(sv_arr).sum(axis=(0, 2))
                else:
                    batch_imp = np.abs(sv_arr).sum(axis=(0, 1))
            else:
                batch_imp = np.abs(sv_arr).sum(axis=0)

        total_shap += batch_imp.flatten()

        del sv_batch, sv_arr, ex_batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Mean importance across all samples and targets
    importance = total_shap / (ex_Xm.shape[0] * wrapper.model.head[-1].out_features)
    print(f"  Stream 1 importance shape: {importance.shape}")
    return importance

def compute_gradient_importance_stream2(model: nn.Module,
                                        loader: torch.utils.data.DataLoader,
                                        n_samples: int = 512,
                                        device: torch.device = None
                                        ) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    accum  = None
    n_seen = 0

    # Ensure grad context is explicitly enabled
    with torch.enable_grad():
        for Xm, Xu in loader:
            if n_seen >= n_samples:
                break
            
            Xm = Xm.to(device)
            Xu = Xu.to(device).detach().requires_grad_(True)

            out  = model(Xm, Xu)
            loss = out.mean()
            
            model.zero_grad()
            loss.backward()

            if Xu.grad is None:
                raise RuntimeError("Xu.grad is None. Check if autograd is enabled.")

            grad = Xu.grad.abs().detach().cpu().numpy()
            accum  = grad.sum(axis=0) if accum is None else accum + grad.sum(axis=0)
            n_seen += Xm.shape[0]

    importance = accum / n_seen
    print(f"  Stream 2 gradient importance shape: {importance.shape}  "
          f"(from {n_seen} samples)")
    return importance


# ══════════════════════════════════════════════════════════════════════════════
# 6. Gene selection
# ══════════════════════════════════════════════════════════════════════════════

def select_genes_by_cumulative_frac(importance_df: pd.DataFrame,
                                    frac: float = 0.90,
                                    score_col: str = "shap_importance"
                                    ) -> list:
    cum = np.cumsum(importance_df[score_col].values)
    cum = cum / cum[-1]
    n   = int(np.searchsorted(cum, frac)) + 1
    print(f"  Cumulative {int(frac * 100)}% importance → top {n} genes")
    return importance_df.head(n)["gene"].tolist()


def select_genes_by_elbow(importance_df: pd.DataFrame,
                          score_col: str = "shap_importance") -> list:
    sorted_imp = importance_df[score_col].values
    knee = KneeLocator(
        x=range(1, len(sorted_imp) + 1),
        y=sorted_imp,
        curve="convex",
        direction="decreasing",
        interp_method="polynomial",
    )
    n = knee.knee or len(sorted_imp)
    print(f"  Elbow-based selection → top {n} genes")
    return importance_df.head(n)["gene"].tolist()


# ══════════════════════════════════════════════════════════════════════════════
# 7. h5ad filtering
# ══════════════════════════════════════════════════════════════════════════════

def write_filtered_h5ad(source_path: Path, selected_genes: list,
                        output_path: Path,
                        normalize_target: float = 1e4,
                        importance_df: pd.DataFrame = None):
    """
    Write a filtered .h5ad containing only `selected_genes`.

    Preserves all obs metadata. Stores raw counts in .layers['counts']
    and log-normalised values in .X. Adds SHAP scores to .var if
    importance_df is provided.
    """
    print(f"Reading {source_path} ...")
    src = ad.read_h5ad(source_path, backed="r")
    print(f"  Original shape: {src.shape}")

    present = [g for g in selected_genes if g in src.var_names]
    missing = len(selected_genes) - len(present)
    if missing:
        print(f"  Warning: {missing} selected genes absent — skipped.")

    out = src[:, present].to_memory()
    out.layers["counts"] = (out.X.copy() if sp.issparse(out.X)
                            else sp.csr_matrix(out.X))

    sc.pp.normalize_total(out, target_sum=normalize_target, inplace=True)
    sc.pp.log1p(out)

    if importance_df is not None:
        imp = importance_df.set_index("gene")
        score_col = "importance" if "importance" in imp.columns else "shap_importance"
        rank_col  = "rank" if "rank" in imp.columns else "shap_rank"

        out.var["importance"] = [
            imp.loc[g, score_col] if g in imp.index else np.nan
            for g in out.var_names
        ]
        if rank_col in imp.columns:
            out.var["importance_rank"] = [
                int(imp.loc[g, rank_col]) if g in imp.index else -1
                for g in out.var_names
            ]

    out.uns["selected_genes"]   = present
    out.uns["n_input_genes"]    = src.shape[1]
    out.uns["n_selected_genes"] = len(present)

    out.write_h5ad(output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(f"  Written: {output_path}  ({size_mb:.1f} MB)")
    print(f"  Genes: {src.shape[1]:,} → {len(present):,}  "
          f"({100 * len(present) / src.shape[1]:.1f}% retained)")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 8. Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def plot_importance(importance_df: pd.DataFrame,
                    output_path: Path,
                    score_col: str = "shap_importance",
                    title_prefix: str = "Stream 1 — SHAP"):
    cum = np.cumsum(importance_df[score_col].values)
    cum = cum / cum[-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    top40 = importance_df.head(40)
    colors = cm.viridis(np.linspace(0.2, 0.9, len(top40)))
    axes[0].barh(top40["gene"][::-1], top40[score_col][::-1], color=colors)
    axes[0].set_xlabel("Importance score")
    axes[0].set_title(f"Top {len(top40)} genes — {title_prefix}")
    axes[0].tick_params(axis="y", labelsize=7)

    axes[1].plot(range(1, len(cum) + 1), cum, lw=1.5)
    for frac in [0.5, 0.8, 0.9]:
        n = np.searchsorted(cum, frac) + 1
        axes[1].axhline(frac, color="red", ls="--", alpha=0.4)
        axes[1].axvline(n,    color="red", ls="--", alpha=0.4)
        axes[1].text(n + 2, frac - 0.04,
                     f"{n} genes\n({int(frac * 100)}%)", fontsize=8)
    axes[1].set_xlabel("Genes (ranked)")
    axes[1].set_ylabel("Cumulative importance")
    axes[1].set_title("Cumulative importance curve")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. Checkpoint loading
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """
    Load checkpoint.  Returns either a full metadata dict (format 1) or a
    raw state_dict (format 2).  Caller must handle both cases.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        print("  Detected: full metadata checkpoint.")
        return ckpt, "full"
    else:
        print("  Detected: raw state_dict checkpoint.")
        return ckpt, "state_dict"


# ══════════════════════════════════════════════════════════════════════════════
# 10. Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract important genes from a pretrained Dual BINN.")
    p.add_argument("--checkpoint",
                   default="data/outputs/models/binn_checkpoint.pt",
                   help="Path to binn_checkpoint.pt")
    p.add_argument("--train_rna",       required=True,
                   help="Path to train_rna.h5ad")
    p.add_argument("--train_pro",       required=True,
                   help="Path to train_pro.h5ad")
    p.add_argument("--binn_data_dir",   required=True,
                   help="Directory with Reactome .txt files")
    p.add_argument("--output_dir",      default="./outputs",
                   help="Directory for all output files")
    p.add_argument("--h5ad_to_filter",  nargs="*", default=[],
                   help="One or more .h5ad files to filter to selected genes")
    p.add_argument("--cumulative_frac", type=float, default=0.90,
                   help="Retain genes covering this fraction of total SHAP "
                        "importance (ignored if --n_genes_fixed is set)")
    p.add_argument("--n_genes_fixed",   type=int,   default=None,
                   help="Force fixed number of top genes (overrides "
                        "--cumulative_frac)")
    p.add_argument("--n_shap_background", type=int, default=256)
    p.add_argument("--n_shap_explain",    type=int, default=512)
    p.add_argument("--n_gradient_samples",type=int, default=512,
                   help="Samples for Stream 2 gradient importance")
    p.add_argument("--num_workers",     type=int,   default=4,
                   help="DataLoader worker processes (0 = main process only)")
    p.add_argument("--batch_size",      type=int,   default=32768)
    p.add_argument("--n_binn_layers",   type=int,   default=4)
    p.add_argument("--dropout",         type=float, default=0.2)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--min_counts_gene", type=int,   default=5)
    p.add_argument("--normalize_target",type=float, default=1e4)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    print(f"PyTorch: {torch.__version__}\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    binn_data_dir   = Path(args.binn_data_dir)
    checkpoint_path = Path(args.checkpoint)

    # ── Step 1: Load data ───────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 — Loading & preprocessing data")
    print("=" * 60)
    rna, pro = load_spatial_data(args.train_rna, args.train_pro)
    rna, all_genes = preprocess_rna(
        rna,
        normalize_target=args.normalize_target,
        min_counts_gene=args.min_counts_gene,
    )
    pro = preprocess_protein(pro)

    common_obs = rna.obs_names.intersection(pro.obs_names)
    rna = rna[common_obs].copy()
    pro = pro[common_obs].copy()
    assert (rna.obs_names == pro.obs_names).all(), "RNA and Protein observation names must match exactly!"

    print(f"\nRNA : {rna.shape}  |  Protein : {pro.shape}")

    # ── Step 2: Reactome mapping ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Reactome pathway mapping")
    print("=" * 60)
    cache_path = output_dir / "gene_reactome_mapping.csv"
    gene_to_pathway_df, pathway_relations, covered_genes = (
        build_gene_reactome_mapping(all_genes, binn_data_dir, cache_path))

    binn_input_genes = [g for g in all_genes if g in set(covered_genes)]
    unmapped_genes   = [g for g in all_genes if g not in set(binn_input_genes)]
    n_proteins       = pro.shape[1]

    print(f"\nTotal QC-passing genes     : {len(all_genes):,}")
    print(f"Reactome-mapped (Stream 1) : {len(binn_input_genes):,}")
    print(f"Unmapped        (Stream 2) : {len(unmapped_genes):,}")
    print(f"Protein targets            : {n_proteins}")

    # ── Step 3: Build masks ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — Building pathway connectivity masks")
    print("=" * 60)
    layer_nodes, masks = build_pathway_layers(
        binn_input_genes, gene_to_pathway_df,
        pathway_relations, args.n_binn_layers,
    )
    masks_device = [m.to(device) for m in masks]

    # ── Step 4: Load model ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — Loading pretrained model")
    print("=" * 60)
    ckpt, ckpt_fmt = load_checkpoint(checkpoint_path, device)

    model = DualBINNEncoder(
        masks      = masks_device,
        n_unmapped = len(unmapped_genes),
        n_outputs  = n_proteins,
        dropout    = args.dropout,
    ).to(device)

    state_dict = ckpt["model_state"] if ckpt_fmt == "full" else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("  Model weights loaded successfully.")

    # ── Step 5: Inference data loader ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — Building inference DataLoader")
    print("=" * 60)
    loader = make_inference_loader(
        rna, binn_input_genes, unmapped_genes,
        args.batch_size, args.num_workers)

    # ── Step 6: Stream 1 — SHAP importance (mapped genes) ──────────────────
    print("\n" + "=" * 60)
    print("STEP 6 — Stream 1: SHAP importance (Reactome-mapped genes)")
    print("=" * 60)
    shap_importance = compute_shap_importance_stream1(
        model, loader,
        n_background=args.n_shap_background,
        n_explain=args.n_shap_explain,
        device=device,
    )

    shap_df = pd.DataFrame({
        "gene":            binn_input_genes,
        "shap_importance": shap_importance,
        "stream":          "pathway_binn",
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)
    shap_df["shap_rank"] = range(1, len(shap_df) + 1)

    shap_csv = output_dir / "gene_shap_importance.csv"
    shap_df.to_csv(shap_csv, index=False)
    print(f"\nTop 20 SHAP-ranked genes:")
    print(shap_df.head(20)[["shap_rank", "gene", "shap_importance"]]
          .to_string(index=False))

    plot_importance(shap_df, output_dir / "gene_shap_importance.png",
                    score_col="shap_importance",
                    title_prefix="Stream 1 — SHAP (pathway BINN)")

    del loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Step 7: Stream 2 — gradient importance (unmapped genes) ─────────────
    print("\n" + "=" * 60)
    print("STEP 7 — Stream 2: Gradient importance (unmapped/dense genes)")
    print("=" * 60)
    # Rebuild loader — gradient computation needs a fresh pass
    loader2 = make_inference_loader(
        rna, binn_input_genes, unmapped_genes,
        args.batch_size, args.num_workers)

    grad_importance = compute_gradient_importance_stream2(
        model, loader2,
        n_samples=args.n_gradient_samples,
        device=device,
    )

    grad_df = pd.DataFrame({
        "gene":              unmapped_genes,
        "grad_importance":   grad_importance,
        "stream":            "dense_mlp",
    }).sort_values("grad_importance", ascending=False).reset_index(drop=True)
    grad_df["grad_rank"] = range(1, len(grad_df) + 1)

    grad_csv = output_dir / "gene_dense_importance.csv"
    grad_df.to_csv(grad_csv, index=False)
    print(f"\nTop 20 gradient-ranked genes (unmapped stream):")
    print(grad_df.head(20)[["grad_rank", "gene", "grad_importance"]]
          .to_string(index=False))

    plot_importance(grad_df, output_dir / "gene_dense_importance.png",
                    score_col="grad_importance",
                    title_prefix="Stream 2 — Gradient (dense MLP)")

    # ── Step 8: Select final gene set ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 8 — Selecting final gene set")
    print("=" * 60)

    # Stream 1 selection
    if args.n_genes_fixed is not None:
        selected_mapped = shap_df.head(args.n_genes_fixed)["gene"].tolist()
        print(f"  Fixed N={args.n_genes_fixed} mapped genes selected.")
    else:
        selected_mapped = select_genes_by_cumulative_frac(
            shap_df, args.cumulative_frac, "shap_importance")

    # Stream 2 selection — use same strategy, same N or same fraction
    if args.n_genes_fixed is not None:
        n_dense = args.n_genes_fixed
        selected_unmapped = grad_df.head(n_dense)["gene"].tolist()
        print(f"  Fixed N={n_dense} unmapped genes selected.")
    else:
        selected_unmapped = select_genes_by_cumulative_frac(
            grad_df, args.cumulative_frac, "grad_importance")

    selected_all = selected_mapped + selected_unmapped
    print(f"\nSelected genes  (Stream 1, pathway) : {len(selected_mapped):,}")
    print(f"Selected genes  (Stream 2, dense)   : {len(selected_unmapped):,}")
    print(f"Total selected genes                 : {len(selected_all):,}")

    # Build a unified importance table for the selected genes
    shap_sub = shap_df[shap_df["gene"].isin(selected_mapped)][
        ["gene", "shap_importance", "shap_rank", "stream"]].copy()
    shap_sub.rename(columns={"shap_importance": "importance",
                              "shap_rank":       "rank"}, inplace=True)

    grad_sub = grad_df[grad_df["gene"].isin(selected_unmapped)][
        ["gene", "grad_importance", "grad_rank", "stream"]].copy()
    grad_sub.rename(columns={"grad_importance": "importance",
                              "grad_rank":       "rank"}, inplace=True)

    selected_df = pd.concat([shap_sub, grad_sub], ignore_index=True)
    selected_csv = output_dir / "selected_genes.csv"
    selected_df.to_csv(selected_csv, index=False)
    print(f"\nSelected genes written to: {selected_csv}")

    # ── Step 9: Filter h5ad files ───────────────────────────────────────────
    if args.h5ad_to_filter:
        print("\n" + "=" * 60)
        print("STEP 9 — Filtering h5ad files")
        print("=" * 60)
        for h5ad_path in args.h5ad_to_filter:
            h5ad_path = Path(h5ad_path)
            out_name  = h5ad_path.stem + "_filtered_v2.h5ad"
            out_path  = output_dir / out_name
            write_filtered_h5ad(
                source_path    = h5ad_path,
                selected_genes = selected_all,
                output_path    = out_path,
                normalize_target = args.normalize_target,
                importance_df  = selected_df,   # SHAP scores for mapped genes
            )

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Input genes (QC-passing)           : {len(all_genes):,}")
    print(f"Stream 1 — Reactome-mapped         : {len(binn_input_genes):,}")
    print(f"Stream 2 — Unmapped (dense)        : {len(unmapped_genes):,}")
    print(f"Selected genes (Stream 1, SHAP)    : {len(selected_mapped):,}")
    print(f"Selected genes (Stream 2, gradient): {len(selected_unmapped):,}")
    print(f"Total selected                     : {len(selected_all):,}")
    print()
    print("Output files:")
    for f in [
        "gene_shap_importance.csv",
        "gene_shap_importance.png",
        "gene_dense_importance.csv",
        "gene_dense_importance.png",
        "selected_genes.csv",
    ]:
        p = output_dir / f
        if p.exists():
            print(f"  {f}  ({p.stat().st_size / 1e6:.2f} MB)")
    if args.h5ad_to_filter:
        for h5ad_path in args.h5ad_to_filter:
            out_name = Path(h5ad_path).stem + "_filtered_v2.h5ad"
            p = output_dir / out_name
            if p.exists():
                print(f"  {out_name}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
