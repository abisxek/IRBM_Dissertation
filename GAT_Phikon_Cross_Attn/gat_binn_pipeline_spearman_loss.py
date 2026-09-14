"""
hybrid_phikon_binn_gat.py
================================================================================
Unified End-to-End Pipeline: Phikon Cross-Attention + Live BINN Pathway Prior + 
Direct Target mRNA + Global RNA SVD + Differentiable Spearman Loss

Usage:
------
1. TRAIN:
    python hybrid_phikon_binn_gat.py train \
        --rna_input data/train_rna_processed.h5ad \
        --pro_input data/train_pro_processed_clipped.h5ad \
        --image_embeddings outputs_histology/embeddings.npy \
        --image_index outputs_histology/embedding_index.csv \
        --gmt_path data/msigdb_hallmark_c8_c7_c6.gmt \
        --output_dir outputs_hybrid_model \
        --epochs 30 --batch_size 512 --lr 3e-4

2. PREDICT:
    python hybrid_phikon_binn_gat.py predict \
        --model_path outputs_hybrid_model/model_weights.pt \
        --valid_rna_input data/valid_rna.h5ad \
        --train_rna_reference data/train_rna_processed.h5ad \
        --valid_image_embeddings outputs_histology_valid/embeddings.npy \
        --valid_image_index outputs_histology_valid/embedding_index.csv \
        --output_path outputs_hybrid_model/predictions.csv \
        --inverse_transform \
        --train_pro_raw data/train_pro.h5ad
================================================================================
"""

import argparse
import gc
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATv2Conv


# ============================================================================
# 1. GMT Parsing & BINN Modules (from Hallmark_C8_C7_C6_gene_db_v2 & binn_gene_extraction)
# ============================================================================

def parse_gmt(gmt_path: str) -> Dict[str, List[str]]:
    """Parse a MSigDB GMT file into a dictionary mapping pathway_name -> gene_list."""
    pathways = {}
    if not os.path.exists(gmt_path):
        log(f"WARNING: GMT file not found at {gmt_path}. BINN will fallback to identity linear layer.")
        return pathways

    with open(gmt_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 2:
                pathway_name = parts[0]
                genes = parts[2:]
                pathways[pathway_name] = genes
    return pathways


def build_binn_mask(gene_list: List[str], gmt_pathways: Dict[str, List[str]]) -> torch.Tensor:
    """Constructs a binary adjacency mask matrix of shape (n_pathways, n_genes)."""
    if not gmt_pathways:
        return None

    pathway_names = list(gmt_pathways.keys())
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    mask = torch.zeros((len(pathway_names), len(gene_list)), dtype=torch.float32)
    for p_idx, p_name in enumerate(pathway_names):
        for g in gmt_pathways[p_name]:
            if g in gene_to_idx:
                mask[p_idx, gene_to_idx[g]] = 1.0

    # Prune pathways with 0 overlapping genes
    valid_pathways = mask.sum(dim=1) > 0
    mask = mask[valid_pathways]
    return mask


class MaskedLinear(nn.Module):
    """Linear layer constrained by a binary biological connectivity mask."""
    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
        self.weight = nn.Parameter(torch.Tensor(mask.shape[0], mask.shape[1]))
        self.bias = nn.Parameter(torch.Tensor(mask.shape[0]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_weight = self.weight * self.mask
        return F.linear(x, masked_weight, self.bias)


class LiveBINNEncoder(nn.Module):
    """Live BINN Encoder that processes raw mapped RNA through GMT pathway masks."""
    def __init__(self, mask: torch.Tensor, n_genes: int, out_dim: int, dropout: float = 0.15):
        super().__init__()
        if mask is not None and mask.shape[0] > 0:
            self.use_binn = True
            self.masked_layer = MaskedLinear(mask)
            self.bn1 = nn.BatchNorm1d(mask.shape[0])
            self.proj = nn.Sequential(
                nn.Linear(mask.shape[0], out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.use_binn = False
            self.proj = nn.Sequential(
                nn.Linear(n_genes, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_binn:
            x = F.gelu(self.bn1(self.masked_layer(x)))
        return self.proj(x)


# ============================================================================
# 2. Differentiable Spearman Loss (from multimodal_GAT)
# ============================================================================

def differentiable_proxy_loss_raw(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor, 
    scale: float = 10.0, 
    reg_weight: float = 0.01
) -> torch.Tensor:
    """Soft-Rank Differentiable Proxy Spearman Loss to optimize protein rank order."""
    batch_size, n_targets = y_pred.shape
    if batch_size < 2:
        return F.mse_loss(y_pred, y_true)

    # Compute pairwise difference matrices
    pred_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)  # [B, B, T]
    true_diff = y_true.unsqueeze(1) - y_true.unsqueeze(0)  # [B, B, T]

    # Sigmoid soft-ranking approximation
    soft_rank_pred = torch.sigmoid(scale * pred_diff).sum(dim=1)  # [B, T]
    soft_rank_true = torch.sigmoid(scale * true_diff).sum(dim=1)  # [B, T]

    # Center rankings
    pred_centered = soft_rank_pred - soft_rank_pred.mean(dim=0, keepdim=True)
    true_centered = soft_rank_true - soft_rank_true.mean(dim=0, keepdim=True)

    # Cosine similarity over soft ranks
    cov = (pred_centered * true_centered).sum(dim=0)
    var_p = (pred_centered ** 2).sum(dim=0) + 1e-8
    var_t = (true_centered ** 2).sum(dim=0) + 1e-8

    spearman_proxy = cov / (torch.sqrt(var_p) * torch.sqrt(var_t) + 1e-8)
    loss_spearman = 1.0 - spearman_proxy.mean()

    # Auxiliary MSE loss for stability
    loss_mse = F.mse_loss(y_pred, y_true)
    return loss_spearman + reg_weight * loss_mse


# ============================================================================
# 3. Phikon Cross-Attention Architecture (from gat_crossattn_pipeline & File 4)
# ============================================================================

class CrossAttentionFusion(nn.Module):
    """RNA embedding (Query) attends over Phikon Histology embedding (Key/Value)."""
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, rna_emb: torch.Tensor, img_emb: torch.Tensor) -> torch.Tensor:
        q = rna_emb.unsqueeze(1)
        kv = img_emb.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        fused = self.norm(attn_out.squeeze(1) + rna_emb)  # Residual connection
        return fused


class HybridPhikonBINNGAT(nn.Module):
    """Complete Unified Architecture combining BINN, Target mRNA, SVD, GAT, and Cross-Attention."""
    def __init__(
        self,
        binn_mask: torch.Tensor,
        n_genes: int,
        target_mrna_dim: int,
        svd_dim: int,
        img_dim: int = 768,       # Phikon default = 768
        gat_hidden_dim: int = 256,
        fusion_dim: int = 256,
        n_proteins: int = 44,
        dropout: float = 0.15
    ):
        super().__init__()

        # Stream A: BINN Pathway Prior Layer
        self.binn_stream = LiveBINNEncoder(binn_mask, n_genes, out_dim=128, dropout=dropout)

        # Stream B: Direct Target mRNA Skip Layer
        self.mrna_stream = nn.Sequential(
            nn.Linear(target_mrna_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Stream C: Global SVD Residual Stream
        self.svd_stream = nn.Sequential(
            nn.Linear(svd_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Total Balanced RNA Latent = 128 (BINN) + 64 (mRNA) + 128 (SVD) = 320
        rna_fused_dim = 128 + 64 + 128
        self.rna_input_proj = nn.Linear(rna_fused_dim, gat_hidden_dim)

        # Spatial GAT Message Passing
        self.gat1 = GATv2Conv(gat_hidden_dim, gat_hidden_dim, heads=4, concat=False, dropout=dropout)
        self.gat2 = GATv2Conv(gat_hidden_dim, fusion_dim, heads=1, concat=False, dropout=dropout)

        # Stream D: Phikon Histology Projection Head
        self.img_stream = nn.Sequential(
            nn.Linear(img_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Cross-Attention Fusion
        self.cross_attn = CrossAttentionFusion(fusion_dim, n_heads=4, dropout=dropout)

        # Final Predictor with Direct Target mRNA Residual Link
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim + 64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_proteins)
        )

    def forward(self, x_rna: torch.Tensor, x_target_mrna: torch.Tensor, x_svd: torch.Tensor, x_img: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 1. Process RNA Streams
        h_binn = self.binn_stream(x_rna)
        h_mrna = self.mrna_stream(x_target_mrna)
        h_svd  = self.svd_stream(x_svd)

        # 2. Fuse RNA representations equally
        rna_fused = torch.cat([h_binn, h_mrna, h_svd], dim=-1)
        x_gat = F.gelu(self.rna_input_proj(rna_fused))

        # 3. Spatial GAT Message Passing over tissue neighborhood
        x_gat = F.gelu(self.gat1(x_gat, edge_index))
        rna_spatial_emb = self.gat2(x_gat, edge_index)

        # 4. Phikon Visual Projection & Cross-Attention
        img_emb = self.img_stream(x_img)
        fused_emb = self.cross_attn(rna_spatial_emb, img_emb)

        # 5. Concatenate Direct Target mRNA Residual and Predict Proteins
        final_repr = torch.cat([fused_emb, h_mrna], dim=-1)
        return self.predictor(final_repr)


# ============================================================================
# 4. Helpers & Graph Builders
# ============================================================================

def log(msg: str, log_file: str = None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


def build_spatial_knn_graph(coords: np.ndarray, k: int = 15) -> torch.Tensor:
    """Build a k-NN spatial graph from physical pixel coordinates."""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    n_nodes = coords.shape[0]
    src = np.repeat(np.arange(n_nodes), k)
    dst = indices[:, 1:].flatten()
    edge_index = np.stack([src, dst], axis=0)
    edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)  # Undirected
    return torch.tensor(edge_index, dtype=torch.long)


# ============================================================================
# 5. Training Routine
# ============================================================================

def run_train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "training_log.txt")
    open(log_file, "w").close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[TRAIN] Device: {device}", log_file)

    # Load Data
    rna = sc.read_h5ad(args.rna_input)
    pro = sc.read_h5ad(args.pro_input)
    if not (rna.obs_names == pro.obs_names).all():
        pro = pro[rna.obs_names].copy()
    protein_names = pro.var_names.tolist()

    # Load Histology Embeddings
    img_embeddings = np.load(args.image_embeddings)
    img_index = pd.read_csv(args.image_index)
    if not (img_index["barcode"].values == rna.obs_names.values).all():
        img_df = pd.DataFrame(img_embeddings, index=img_index["barcode"].values)
        img_embeddings = img_df.reindex(rna.obs_names).values

    # Extract Target mRNA Signal
    target_genes = [p for p in protein_names if p in rna.var_names]
    target_mrna_indices = [rna.var_names.get_loc(g) for g in target_genes]
    
    X_rna_sparse = rna.X if sp.issparse(rna.X) else sp.csr_matrix(rna.X)
    X_target_mrna = X_rna_sparse[:, target_mrna_indices].toarray().astype(np.float32)

    # Build Global SVD Residual Features
    log("Computing Global SVD Residual (256 components)...", log_file)
    svd = TruncatedSVD(n_components=256, random_state=42)
    X_svd = svd.fit_transform(X_rna_sparse).astype(np.float32)

    # Parse GMT Pathways & Construct BINN Mask
    log("Parsing GMT Pathway File for BINN...", log_file)
    gmt_pathways = parse_gmt(args.gmt_path)
    binn_mask = build_binn_mask(rna.var_names.tolist(), gmt_pathways)

    # Build Spatial Graph
    coords = rna.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values
    edge_index = build_spatial_knn_graph(coords, k=args.k_neighbors)

    # PyG Data Packaging
    X_rna_dense = np.asarray(X_rna_sparse.todense()).astype(np.float32)
    Y_pro = np.asarray(pro.X.todense() if sp.issparse(pro.X) else pro.X).astype(np.float32)

    graph_data = Data(
        x=torch.from_numpy(X_rna_dense),
        x_target_mrna=torch.from_numpy(X_target_mrna),
        x_svd=torch.from_numpy(X_svd),
        x_img=torch.from_numpy(img_embeddings.astype(np.float32)),
        edge_index=edge_index,
        y=torch.from_numpy(Y_pro)
    )

    loader = NeighborLoader(
        graph_data,
        num_neighbors=[15, 10],
        batch_size=args.batch_size,
        shuffle=True
    )

    # Initialize Model
    model = HybridPhikonBINNGAT(
        binn_mask=binn_mask,
        n_genes=X_rna_dense.shape[1],
        target_mrna_dim=X_target_mrna.shape[1],
        svd_dim=256,
        img_dim=img_embeddings.shape[1],
        gat_hidden_dim=256,
        fusion_dim=256,
        n_proteins=Y_pro.shape[1],
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    log(f"Starting Training for {args.epochs} epochs...", log_file)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(batch.x, batch.x_target_mrna, batch.x_svd, batch.x_img, batch.edge_index)
            seed_pred = pred[:batch.batch_size]
            seed_y = batch.y[:batch.batch_size]

            loss = differentiable_proxy_loss_raw(seed_pred, seed_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        log(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Spearman Proxy Loss: {avg_loss:.4f}", log_file)

    # Save Checkpoint
    model_path = os.path.join(args.output_dir, "model_weights.pt")
    torch.save({
        "model_state": model.state_dict(),
        "binn_mask": binn_mask,
        "n_genes": X_rna_dense.shape[1],
        "target_genes": target_genes,
        "protein_names": protein_names,
        "svd_model": svd
    }, model_path)
    log(f"Model successfully saved to {model_path}", log_file)


# ============================================================================
# 6. Prediction Routine
# ============================================================================

def run_predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[PREDICT] Device: {device}")

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    protein_names = checkpoint["protein_names"]
    target_genes = checkpoint["target_genes"]
    svd = checkpoint["svd_model"]

    # Load Validation RNA
    valid_rna = sc.read_h5ad(args.valid_rna_input)
    train_ref = sc.read_h5ad(args.train_rna_reference, backed="r")
    train_gene_list = train_ref.var_names.tolist()
    train_ref.file.close()

    # Align Genes
    aligned_X = sp.lil_matrix((valid_rna.n_obs, len(train_gene_list)), dtype=np.float32)
    present_genes = [g for g in train_gene_list if g in valid_rna.var_names]
    present_idx = [i for i, g in enumerate(train_gene_list) if g in valid_rna.var_names]
    valid_sub = valid_rna[:, present_genes].X
    aligned_X[:, present_idx] = valid_sub if sp.issparse(valid_sub) else sp.csr_matrix(valid_sub)
    aligned_X = aligned_X.tocsr()

    # Extract Streams
    target_mrna_idx = [train_gene_list.index(g) for g in target_genes if g in train_gene_list]
    X_target_mrna = aligned_X[:, target_mrna_idx].toarray().astype(np.float32)
    X_svd = svd.transform(aligned_X).astype(np.float32)
    X_rna_dense = aligned_X.toarray().astype(np.float32)

    # Load Images & Build Validation Graph
    img_embeddings = np.load(args.valid_image_embeddings)
    coords = valid_rna.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values
    edge_index = build_spatial_knn_graph(coords, k=args.k_neighbors).to(device)

    # Initialize and Load Weights
    model = HybridPhikonBINNGAT(
        binn_mask=checkpoint["binn_mask"],
        n_genes=len(train_gene_list),
        target_mrna_dim=X_target_mrna.shape[1],
        svd_dim=256,
        img_dim=img_embeddings.shape[1],
        n_proteins=len(protein_names)
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        x_rna_t = torch.tensor(X_rna_dense, dtype=torch.float32).to(device)
        x_mrna_t = torch.tensor(X_target_mrna, dtype=torch.float32).to(device)
        x_svd_t = torch.tensor(X_svd, dtype=torch.float32).to(device)
        x_img_t = torch.tensor(img_embeddings, dtype=torch.float32).to(device)

        preds = model(x_rna_t, x_mrna_t, x_svd_t, x_img_t, edge_index).cpu().numpy()

    # Inverse Transform
    if args.inverse_transform and args.train_pro_raw:
        log("Applying inverse arcsinh + z-score transform...")
        pro_raw = sc.read_h5ad(args.train_pro_raw)
        X_raw = np.asarray(pro_raw.X.todense() if sp.issparse(pro_raw.X) else pro_raw.X)
        
        cofactor = args.arcsinh_cofactor
        X_arcsinh = np.arcsinh(X_raw / cofactor)
        marker_mean = X_arcsinh.mean(axis=0)
        marker_std = X_arcsinh.std(axis=0)

        preds = np.sinh(preds * marker_std + marker_mean) * cofactor

    # Export CSV
    out_df = pd.DataFrame({
        "barcode": valid_rna.obs_names,
        "pxl_row_in_fullres": valid_rna.obs["pxl_row_in_fullres"].values,
        "pxl_col_in_fullres": valid_rna.obs["pxl_col_in_fullres"].values,
    })
    pred_df = pd.DataFrame(preds, columns=protein_names, index=valid_rna.obs_names)
    out_df = pd.concat([out_df.set_index("barcode"), pred_df], axis=1).reset_index()
    out_df.to_csv(args.output_path, index=False)
    log(f"Predictions saved to {args.output_path}")


# ============================================================================
# 7. CLI Command Parsing
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Phikon + BINN + GAT Cross-Attention Pipeline")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Train CLI
    p_train = subparsers.add_parser("train")
    p_train.add_argument("--rna_input", type=str, required=True)
    p_train.add_argument("--pro_input", type=str, required=True)
    p_train.add_argument("--image_embeddings", type=str, required=True)
    p_train.add_argument("--image_index", type=str, required=True)
    p_train.add_argument("--gmt_path", type=str, required=True)
    p_train.add_argument("--output_dir", type=str, required=True)
    p_train.add_argument("--k_neighbors", type=int, default=15)
    p_train.add_argument("--batch_size", type=int, default=512)
    p_train.add_argument("--dropout", type=float, default=0.15)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--epochs", type=int, default=30)

    # Predict CLI
    p_pred = subparsers.add_parser("predict")
    p_pred.add_argument("--model_path", type=str, required=True)
    p_pred.add_argument("--valid_rna_input", type=str, required=True)
    p_pred.add_argument("--train_rna_reference", type=str, required=True)
    p_pred.add_argument("--valid_image_embeddings", type=str, required=True)
    p_pred.add_argument("--valid_image_index", type=str, required=True)
    p_pred.add_argument("--output_path", type=str, required=True)
    p_pred.add_argument("--k_neighbors", type=int, default=15)
    p_pred.add_argument("--inverse_transform", action="store_true")
    p_pred.add_argument("--train_pro_raw", type=str, default=None)
    p_pred.add_argument("--arcsinh_cofactor", type=float, default=5.0)

    args = parser.parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "predict":
        run_predict(args)

if __name__ == "__main__":
    main()