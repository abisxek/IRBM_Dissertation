"""
Multi-Modal RNA (GAT) + Histology (Phikon) Cross-Attention Pipeline
========================================================================
Single-file pipeline covering both TRAINING and PREDICTION for the
multi-modal protein prediction model used in this dissertation project
(STP Open Challenge, glioma spatial transcriptomics -> proteomics).

Architecture:
    RNA branch:   RNA (n_genes) -> Linear projection -> GATv2 layer 1
                  -> GATv2 layer 2 -> RNA embedding (fusion_dim)
    Image branch: Precomputed Phikon histology embedding (768-dim, frozen/
                  pretrained, extracted separately via
                  extract_histology_embeddings.py) -> Linear projection
                  -> Image embedding (fusion_dim)
    Fusion:       Single-direction cross-attention - RNA embedding (query)
                  attends over Image embedding (key/value), residual +
                  layernorm -> Fused representation
    Prediction:   Fused representation -> MLP -> 44 protein values

Training uses mini-batch neighbor sampling (PyTorch Geometric
NeighborLoader) so memory usage is independent of total dataset size -
only small sampled subgraphs are moved to GPU per step, not the full
graph/RNA matrix. Trains on the FULL dataset with no held-out split
(GAT message-passing would leak information across a random train/val
split boundary); real evaluation is done externally via the STP Open
Challenge.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------

1) TRAIN a model:

    python gat_crossattn_pipeline.py train \
        --rna_input data/train_rna_processed.h5ad \
        --pro_input data/train_pro_processed_clipped.h5ad \
        --image_embeddings outputs_histology/embeddings.npy \
        --image_index outputs_histology/embedding_index.csv \
        --output_dir outputs_gat_crossattn \
        --k_neighbors 15 --num_neighbors 15 10 --batch_size 512 \
        --gat_proj_dim 512 --gat_hidden_dim 256 --gat_heads 4 \
        --n_epochs 30

2) PREDICT on new RNA-only data (e.g. valid_rna.h5ad), producing a
   submission-ready CSV (barcode, pxl_row_in_fullres, pxl_col_in_fullres,
   then 44 protein columns):

    python gat_crossattn_pipeline.py predict \
        --model_path outputs_gat_crossattn/model_weights.pt \
        --valid_rna_input data/valid_rna.h5ad \
        --train_rna_reference data/train_rna_processed.h5ad \
        --valid_image_embeddings outputs_histology_valid/embeddings.npy \
        --valid_image_index outputs_histology_valid/embedding_index.csv \
        --output_path outputs_gat_crossattn/valid_predictions_raw.csv \
        --inverse_transform \
        --train_pro_raw data/train_pro.h5ad

   Omit --inverse_transform (and --train_pro_raw) to get predictions in
   the model's native arcsinh + z-scored + clipped space instead of raw
   protein intensity units.

--------------------------------------------------------------------------
PREREQUISITE
--------------------------------------------------------------------------
Histology embeddings must already be extracted separately (for both the
training and any prediction RNA files) via extract_histology_embeddings.py
before running this script - this pipeline consumes precomputed
embeddings, it does not run Phikon itself.

Requires torch_geometric (with torch_sparse or pyg_lib installed for
NeighborLoader support during training - not needed for prediction):
    pip install torch_geometric
    pip install torch_sparse -f https://data.pyg.org/whl/torch-<VERSION>.html
"""

import argparse
import gc
import os
import time

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data


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


def build_spatial_knn_graph(coords, k):
    """Build a k-NN graph from physical (row, col) pixel coordinates.
    Returns edge_index in the format torch_geometric expects: shape (2, num_edges).
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    n_nodes = coords.shape[0]
    src = np.repeat(np.arange(n_nodes), k)
    dst = indices[:, 1:].flatten()
    edge_index = np.stack([src, dst], axis=0)
    edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)  # undirected
    return torch.tensor(edge_index, dtype=torch.long)


# ============================================================================
# Model architecture
# ============================================================================

class RNAGATEncoder(nn.Module):
    """RNA branch: linear projection (memory-critical - keeps GAT's edge-wise
    attention computation tractable) followed by two GATv2 layers using the
    spatial neighbourhood graph."""
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
    """Single-direction cross-attention: RNA embeddings (query) attend over
    image embeddings (key/value), per bin. Residual connection + layernorm
    so the model can fall back toward the RNA signal alone if the image
    signal isn't useful for a given bin."""
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
    """Full model: RNA (GAT) branch + Image (Phikon, precomputed) branch,
    fused via cross-attention, predicting all protein markers."""
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
# TRAIN
# ============================================================================

def run_train(args):
    from torch_geometric.loader import NeighborLoader

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "training_log.txt")
    open(log_file, "w").close()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    log(f"[TRAIN] Using device: {device}", log_file)

    log(f"Loading RNA data from {args.rna_input}", log_file)
    rna = sc.read_h5ad(args.rna_input)
    log(f"RNA shape: {rna.shape}", log_file)

    log(f"Loading protein data from {args.pro_input}", log_file)
    pro = sc.read_h5ad(args.pro_input)
    if not (rna.obs_names == pro.obs_names).all():
        log("Reindexing protein data to RNA bin order", log_file)
        pro = pro[rna.obs_names].copy()
    protein_names = pro.var_names.tolist()

    log(f"Loading image embeddings from {args.image_embeddings}", log_file)
    img_embeddings = np.load(args.image_embeddings)
    img_index = pd.read_csv(args.image_index)
    if not (img_index["barcode"].values == rna.obs_names.values).all():
        log("Reindexing image embeddings to RNA bin order", log_file)
        img_df = pd.DataFrame(img_embeddings, index=img_index["barcode"].values)
        img_df = img_df.reindex(rna.obs_names)
        if img_df.isna().any().any():
            raise ValueError("Some RNA bins have no matching image embedding")
        img_embeddings = img_df.values
    log(f"Image embeddings shape (aligned): {img_embeddings.shape}", log_file)

    log(f"Building spatial k-NN graph (k={args.k_neighbors})", log_file)
    coords = rna.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values
    edge_index = build_spatial_knn_graph(coords, args.k_neighbors)
    log(f"Graph built: {edge_index.shape[1]} directed edges", log_file)

    X_rna = rna.X
    X_rna = np.asarray(X_rna.todense()) if hasattr(X_rna, "todense") else np.asarray(X_rna)
    X_rna = X_rna.astype(np.float32, copy=False)
    Y = pro.X
    Y = np.asarray(Y.todense()) if hasattr(Y, "todense") else np.asarray(Y)
    Y = Y.astype(np.float32, copy=False)
    img_embeddings = img_embeddings.astype(np.float32, copy=False)

    log(f"X_rna: {X_rna.shape}, X_img: {img_embeddings.shape}, Y: {Y.shape}", log_file)
    log("Data stays on CPU; NeighborLoader moves only sampled subgraphs to "
        "GPU per step. Using torch.from_numpy (no-copy) to avoid doubling "
        "RAM usage on the large RNA array.", log_file)

    # from_numpy shares memory rather than copying - important since X_rna
    # can be ~12GB+ at full gene count; torch.tensor() would double that.
    X_rna_cpu = torch.from_numpy(X_rna)
    X_img_cpu = torch.from_numpy(img_embeddings)
    Y_cpu = torch.from_numpy(Y)
    del rna, pro
    gc.collect()

    graph_data = Data(x=X_rna_cpu, edge_index=edge_index, y=Y_cpu)
    graph_data.img_x = X_img_cpu  # node-level attribute, auto-subset by NeighborLoader

    log("NOTE: training on FULL dataset (no held-out split) - GAT message-"
        "passing would leak information across a random split boundary. "
        "Evaluate externally (STP Open Challenge).", log_file)

    loader = NeighborLoader(
        graph_data,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        shuffle=True,
    )
    n_batches = (graph_data.num_nodes + args.batch_size - 1) // args.batch_size
    log(f"NeighborLoader: num_neighbors={args.num_neighbors}, "
        f"batch_size={args.batch_size}, ~{n_batches} batches/epoch", log_file)

    model = GATCrossAttnPredictor(
        n_genes=X_rna.shape[1], img_dim=img_embeddings.shape[1],
        gat_proj_dim=args.gat_proj_dim, gat_hidden_dim=args.gat_hidden_dim,
        gat_heads=args.gat_heads, fusion_dim=args.fusion_dim,
        cross_attn_heads=args.cross_attn_heads, n_proteins=Y.shape[1],
        dropout=args.dropout
    ).to(device)
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", log_file)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    log(f"Starting mini-batch training for {args.n_epochs} epochs", log_file)
    start = time.time()
    avg_loss = None

    for epoch in range(args.n_epochs):
        model.train()
        epoch_loss, n_batches_done = 0.0, 0
        for batch in loader:
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
        log(f"Epoch {epoch + 1}/{args.n_epochs} - Train Loss: {avg_loss:.4f} "
            f"({n_batches_done} batches)", log_file)

    elapsed = time.time() - start
    log(f"Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)", log_file)

    model_path = os.path.join(args.output_dir, "model_weights.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "n_genes": X_rna.shape[1],
        "img_dim": img_embeddings.shape[1],
        "n_proteins": Y.shape[1],
        "protein_names": protein_names,
        "final_train_loss": avg_loss,
    }, model_path)
    log(f"Saved model weights to {model_path}", log_file)
    log("Done.", log_file)


# ============================================================================
# PREDICT
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
        f"final_train_loss={checkpoint.get('final_train_loss')}")

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
    edge_index = build_spatial_knn_graph(coords, args.k_neighbors).to(device)
    log(f"Validation graph built: {edge_index.shape[1]} directed edges")

    log(f"Running inference on {valid_rna.n_obs} bins")
    X_valid_t = torch.tensor(X_valid, dtype=torch.float32).to(device)
    X_img_t = torch.tensor(img_embeddings, dtype=torch.float32).to(device)

    start = time.time()
    with torch.no_grad():
        preds = model(X_valid_t, edge_index, X_img_t)
        preds = preds.cpu().numpy()
    log(f"Inference complete in {time.time() - start:.1f}s")

    # ------------------------------------------------------------------
    # Optional inverse transform: arcsinh + z-score + clip -> raw protein scale
    # ------------------------------------------------------------------
    if args.inverse_transform:
        if not args.train_pro_raw:
            raise ValueError("--inverse_transform requires --train_pro_raw "
                              "(path to the ORIGINAL, untransformed train_pro.h5ad)")
        log(f"Applying inverse transform using raw reference {args.train_pro_raw}")
        pro_raw = sc.read_h5ad(args.train_pro_raw)
        X_raw = pro_raw.X
        X_raw = np.asarray(X_raw.todense()) if hasattr(X_raw, "todense") else np.asarray(X_raw)

        cofactor = args.arcsinh_cofactor
        X_arcsinh = np.arcsinh(X_raw / cofactor)
        marker_mean = X_arcsinh.mean(axis=0)
        marker_std = X_arcsinh.std(axis=0)

        # Reorder raw reference proteins to match checkpoint's protein order,
        # in case they differ.
        raw_protein_names = pro_raw.var_names.tolist()
        name_to_idx = {n: i for i, n in enumerate(raw_protein_names)}
        reorder = [name_to_idx[n] for n in protein_names]
        marker_mean = marker_mean[reorder]
        marker_std = marker_std[reorder]

        arcsinh_recovered = preds * marker_std + marker_mean
        preds = np.sinh(arcsinh_recovered) * cofactor
        log("Inverse transform applied - predictions now in raw protein "
            "intensity units (note: values originally clipped to [-5, 5] "
            "during preprocessing cannot be perfectly recovered)")
    else:
        log("NOTE: predictions are in arcsinh + z-scored + clipped space, "
            "NOT raw protein intensity units. Pass --inverse_transform "
            "with --train_pro_raw to convert to raw scale.")

    log("Building output dataframe")
    out_df = pd.DataFrame({
        "barcode": valid_rna.obs_names,
        "pxl_row_in_fullres": valid_rna.obs["pxl_row_in_fullres"].values,
        "pxl_col_in_fullres": valid_rna.obs["pxl_col_in_fullres"].values,
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
    p_train = subparsers.add_parser("train", help="Train the model on full training data")
    p_train.add_argument("--rna_input", type=str, required=True)
    p_train.add_argument("--pro_input", type=str, required=True)
    p_train.add_argument("--image_embeddings", type=str, required=True)
    p_train.add_argument("--image_index", type=str, required=True)
    p_train.add_argument("--output_dir", type=str, required=True)
    p_train.add_argument("--k_neighbors", type=int, default=15)
    p_train.add_argument("--num_neighbors", type=int, nargs="+", default=[15, 10],
                          help="Fan-out per GAT layer for mini-batch sampling")
    p_train.add_argument("--batch_size", type=int, default=512)
    p_train.add_argument("--gat_proj_dim", type=int, default=512)
    p_train.add_argument("--gat_hidden_dim", type=int, default=256)
    p_train.add_argument("--gat_heads", type=int, default=4)
    p_train.add_argument("--fusion_dim", type=int, default=256)
    p_train.add_argument("--cross_attn_heads", type=int, default=4)
    p_train.add_argument("--dropout", type=float, default=0.4)
    p_train.add_argument("--weight_decay", type=float, default=1e-5)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--n_epochs", type=int, default=30)
    p_train.add_argument("--seed", type=int, default=42)

    # ---- predict subcommand ----
    p_pred = subparsers.add_parser("predict", help="Generate predictions on new RNA-only data")
    p_pred.add_argument("--model_path", type=str, required=True)
    p_pred.add_argument("--valid_rna_input", type=str, required=True)
    p_pred.add_argument("--train_rna_reference", type=str, required=True)
    p_pred.add_argument("--valid_image_embeddings", type=str, required=True)
    p_pred.add_argument("--valid_image_index", type=str, required=True)
    p_pred.add_argument("--output_path", type=str, required=True)
    p_pred.add_argument("--k_neighbors", type=int, default=15,
                         help="Should match the k_neighbors used during training")
    p_pred.add_argument("--inverse_transform", action="store_true",
                         help="Convert predictions from arcsinh+z-scored+clipped space "
                              "back to raw protein intensity units")
    p_pred.add_argument("--train_pro_raw", type=str, default=None,
                         help="Path to the ORIGINAL, untransformed train_pro.h5ad - "
                              "required if --inverse_transform is set")
    p_pred.add_argument("--arcsinh_cofactor", type=float, default=5.0,
                         help="Cofactor used in the original arcsinh transform (default: 5.0)")

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
