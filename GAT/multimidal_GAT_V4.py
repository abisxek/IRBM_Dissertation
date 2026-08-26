# ── gat_train_ultimate.py ───────────────────────
import os
import time
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp

import torch
import torch._dynamo  # <-- Global scope import
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import tifffile
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATv2Conv
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION & MAPPINGS
# ==========================================
DATA_DIR      = "/home/ubuntu/data"
FEATURES_DIR  = os.path.join(DATA_DIR, "features")
OUTPUTS_DIR   = os.path.join(DATA_DIR, "outputs")
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

RNA_PATH      = os.path.join(DATA_DIR, "train_rna.h5ad")
PRO_PATH      = os.path.join(DATA_DIR, "train_pro.h5ad")
VALID_RNA     = os.path.join(DATA_DIR, "valid_rna.h5ad")
VALID_GT      = os.path.join(DATA_DIR, "valid_uniform_range.csv")
TEST_RNA      = os.path.join(DATA_DIR, "test_rna.h5ad")

IMG_SRC_TRAIN = os.path.join(DATA_DIR, 'train_HE_image_full_resolution.tif')
IMG_SRC_VALID = os.path.join(DATA_DIR, 'valid_HE_image_full_resolution.tif')
IMG_SRC_TEST  = os.path.join(DATA_DIR, 'test_HE_image_full_resolution.tif')

IMG_TRAIN_CACHE = os.path.join(FEATURES_DIR, 'train_image_features.npy')
IMG_VALID_CACHE = os.path.join(FEATURES_DIR, 'valid_image_features.npy')
IMG_TEST_CACHE  = os.path.join(FEATURES_DIR, 'test_image_features.npy')

PRO_COFACTOR  = 150.0
RANDOM_SEED   = 42

# ── Architecture Config ──
N_SVD_COMPONENTS = 256  
IMG_COMPONENTS   = 2048 

N_LAYERS        = 5   
HIDDEN_CHANNELS = 512
HEADS           = 5
DROPOUT         = 0.3
BATCH_SIZE      = 512
N_EPOCHS        = 550
PATIENCE        = 20
LEARNING_RATE   = 0.001
NUM_WORKERS     = 4

# 30um Radius Graph
RADIUS_MICRONS = 30.0
TRAIN_MICRONS_PER_PIXEL = 0.8820219467631594
TEST_MICRONS_PER_PIXEL  = 0.883043249671293

NEIGHBORS_PER_HOP = [10, 5, 5, 5, 5]  
NUM_NEIGHBORS = NEIGHBORS_PER_HOP[:N_LAYERS] if N_LAYERS <= len(NEIGHBORS_PER_HOP) else NEIGHBORS_PER_HOP + [NEIGHBORS_PER_HOP[-1]] * (N_LAYERS - len(NEIGHBORS_PER_HOP))

PROTEIN_COLS = [
    "synd", "FOXP3", "CD16", "CD31", "CXCL13", "Ki67", "OLIG2", "CXCR5", "HLA-A", "PD-L1",
    "PSD95", "CD20", "CD68", "CD44", "SMA", "MSH6", "CD23", "GFAP", "SYNA", "Podoplanin",
    "Vimentin", "CD47", "CD74", "SIRP", "Granzyme B", "IDH1", "MPO", "CD45", "CD21", "FIBR",
    "C-KIT", "CD3e", "TOX", "PD-1", "PDGFR", "CD4", "MAP2", "CD8", "MGMT", "CD38",
    "HLA-DR", "CD14", "ICOS", "Granzyme K"
]

PROTEIN_TO_GENE = {
    'synd': 'SYP', 'SYNA': 'SYP', 'PSD95': 'DLG4', 'Vimentin': 'VIM', 'SMA': 'ACTA2', 'Podoplanin': 'PDPN',
    'FIBR': 'FN1', 'PDGFR': 'PDGFRA', 'C-KIT': 'KIT', 'Ki67': 'MKI67', 'PD-1': 'PDCD1', 'PD-L1': 'CD274',
    'CD3e': 'CD3E', 'CD8': 'CD8A', 'Granzyme B': 'GZMB', 'Granzyme K': 'GZMK', 'CD16': 'FCGR3A',
    'HLA-DR': 'HLA-DRA', 'SIRP': 'SIRPA', 'CD20': 'MS4A1', 'CD21': 'CR2', 'CD23': 'FCER2',
    'CD45': 'PTPRC', 'CD31': 'PECAM1'
}

# The anchor scores to aggressively penalize the loss on poor-performing immune markers
BASELINE_SCORES = {
    "synd":0.72, "FOXP3":0.58, "CD16":0.54, "CD31":0.45, "CXCL13":0.45, "Ki67":0.49, "OLIG2":0.48, "CXCR5":0.42, 
    "HLA-A":0.38, "PD-L1":0.41, "PSD95":0.59, "CD20":0.33, "CD68":0.31, "CD44":0.60, "SMA":0.37, "MSH6":0.45, 
    "CD23":0.69, "GFAP":0.78, "SYNA":0.62, "Podoplanin":0.57, "Vimentin":0.52, "CD47":0.40, "CD74":0.54, 
    "SIRP":0.52, "Granzyme B":0.69, "IDH1":0.58, "MPO":0.57, "CD45":0.30, "CD21":0.59, "FIBR":0.45,
    "C-KIT":0.57, "CD3e":0.41, "TOX":0.57, "PD-1":0.48, "PDGFR":0.61, "CD4":0.37, "MAP2":0.64, "CD8":0.53, 
    "MGMT":0.49, "CD38":0.42, "HLA-DR":0.42, "CD14":0.42, "ICOS":0.62, "Granzyme K":0.70
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. UTILITY & EXTRACTION FUNCTIONS
# ==========================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def compute_physical_microns(adata, microns_per_pixel, fallback_df=None):
    if "pxl_row_in_fullres" in adata.obs.columns:
        rows = adata.obs["pxl_row_in_fullres"].values.astype(np.float64)
        cols = adata.obs["pxl_col_in_fullres"].values.astype(np.float64)
    elif fallback_df is not None:
        sub = fallback_df.loc[adata.obs_names]
        rows = sub["pxl_row_in_fullres"].values.astype(np.float64)
        cols = sub["pxl_col_in_fullres"].values.astype(np.float64)
    else:
        raise ValueError("Missing pixel coordinates!")
    return np.vstack([rows * microns_per_pixel, cols * microns_per_pixel]).T

def build_spatial_graph(coords, radius=RADIUS_MICRONS):
    """Builds a physical distance graph with Gaussian decay weights."""
    nbrs = NearestNeighbors(radius=radius, metric='euclidean').fit(coords)
    distances, indices = nbrs.radius_neighbors(coords)
    
    edge_list, edge_weights = [], []
    sigma = radius / 2.0 
    
    for i, (neighbors, dists) in enumerate(zip(indices, distances)):
        for j, d in zip(neighbors, dists):
            if i != j:
                edge_list.append([i, j])
                edge_weights.append([np.exp(-(d**2) / (2 * sigma**2))])
                
    if not edge_list: 
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float32)
        
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous(), torch.tensor(edge_weights, dtype=torch.float32)

def extract_markers(adata, genes):
    """Extracts explicit raw gene counts to bypass the SVD bottleneck."""
    mat = np.zeros((adata.n_obs, len(genes)), dtype=np.float32)
    for j, g in enumerate(genes):
        if g in adata.var_names:
            col = adata[:, g].X
            mat[:, j] = col.toarray().flatten() if sp.issparse(col) else np.asarray(col).flatten()
    return mat

class VisiumPatchDataset(Dataset):
    def __init__(self, adata, img_path, fallback_df=None, patch_size=224, transform=None):
        if "pxl_row_in_fullres" in adata.obs.columns:
            self.rows = adata.obs['pxl_row_in_fullres'].values.astype(int)
            self.cols = adata.obs['pxl_col_in_fullres'].values.astype(int)
        else:
            sub = fallback_df.loc[adata.obs_names]
            self.rows = sub["pxl_row_in_fullres"].values.astype(int)
            self.cols = sub["pxl_col_in_fullres"].values.astype(int)
            
        self.img_map = tifffile.memmap(img_path)
        if self.img_map.ndim == 3 and self.img_map.shape[0] in (3, 4) and self.img_map.shape[0] < self.img_map.shape[-1]:
            self.max_h, self.max_w = self.img_map.shape[1], self.img_map.shape[2]
            self.chw = True
        else:
            self.max_h, self.max_w = self.img_map.shape[:2]
            self.chw = False
            
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r, c = self.rows[idx], self.cols[idx]
        half = self.patch_size // 2
        r_start, r_end = max(0, r - half), min(self.max_h, r + half)
        c_start, c_end = max(0, c - half), min(self.max_w, c + half)
        
        if self.chw:
            patch = self.img_map[:3, r_start:r_end, c_start:c_end]
            patch = np.transpose(patch, (1, 2, 0)) 
        else:
            patch = self.img_map[r_start:r_end, c_start:c_end, :3]
            
        pad_r_top, pad_r_bot = max(0, half - r), max(0, (r + half) - self.max_h)
        pad_c_left, pad_c_right = max(0, half - c), max(0, (c + half) - self.max_w)
        
        if pad_r_top > 0 or pad_r_bot > 0 or pad_c_left > 0 or pad_c_right > 0:
            patch = np.pad(patch, ((pad_r_top, pad_r_bot), (pad_c_left, pad_c_right), (0,0)), mode='reflect')

        return self.transform(Image.fromarray(patch)) if self.transform else Image.fromarray(patch)

def get_image_features(adata, img_path, cache_path, device, fallback_df=None):
    if os.path.exists(cache_path):
        log(f"  -> Loading cached image features: {cache_path}")
        return np.load(cache_path)
    if not os.path.exists(img_path):
        log(f"  -> Warning: {img_path} missing (Expected for Validation). Returning zeros.")
        return np.zeros((adata.n_obs, IMG_COMPONENTS), dtype=np.float32)
        
    log(f"  -> Extracting features dynamically from {img_path}...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataloader = DataLoader(VisiumPatchDataset(adata, img_path, fallback_df=fallback_df, patch_size=224, transform=transform), batch_size=256, num_workers=4)
    
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model = model.to(device).eval()
    
    features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="ResNet50"):
            features.append(model(batch.to(device)).cpu().numpy())
            
    final_features = np.vstack(features).astype(np.float32)
    np.save(cache_path, final_features)
    return final_features

# ==========================================
# 3. GATv2 MULTIMODAL ARCHITECTURE
# ==========================================
class MultimodalGATRegressor(nn.Module):
    def __init__(self, rna_dim, img_dim, hidden_channels, out_channels, n_layers=4, heads=4, dropout=0.3):
        super().__init__()
        self.dropout  = dropout
        self.n_layers = n_layers
        self.rna_dim = rna_dim

        # Independent Projections
        self.rna_proj = nn.Sequential(nn.Linear(rna_dim, hidden_channels), nn.ELU(), nn.BatchNorm1d(hidden_channels))
        self.img_proj = nn.Sequential(nn.Linear(img_dim, hidden_channels), nn.ELU(), nn.BatchNorm1d(hidden_channels))
        
        in_channels = hidden_channels * 2

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True, edge_dim=1))
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        for _ in range(n_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, concat=True, edge_dim=1))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout, concat=False, edge_dim=1))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.output = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_rna, x_img, edge_index, edge_attr):
        h = torch.cat([self.rna_proj(x_rna), self.img_proj(x_img)], dim=1)
        for conv, bn in zip(self.convs, self.bns):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = bn(h)
            h = F.elu(h)
        return self.output(h)

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    log("=== The Ultimate Multimodal Pipeline (Radial Graph + Marker Bypass) ===")
    
    # --- 1. Load Data ---
    log("Loading Data...")
    rna = ad.read_h5ad(RNA_PATH)
    pro = ad.read_h5ad(PRO_PATH)
    
    shared_bins = rna.obs_names.intersection(pro.obs_names)
    rna = rna[shared_bins].copy()
    pro = pro[shared_bins].copy()

    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)

    rna_gene_list = list(rna.var_names)
    PROTEIN_NAMES = list(pro.var_names)
    
    pro_dense = pro.X.toarray() if sp.issparse(pro.X) else np.array(pro.X, dtype=np.float32)
    Y_train = np.arcsinh(pro_dense / PRO_COFACTOR).astype(np.float32)
    del pro_dense

    # --- 2. Marker Gene Bypass & SVD ---
    log("Extracting Direct Marker Genes...")
    marker_genes = [PROTEIN_TO_GENE.get(p, p) for p in PROTEIN_NAMES]
    X_train_markers = extract_markers(rna, marker_genes)

    log(f"Fitting TruncatedSVD ({N_SVD_COMPONENTS} components)...")
    X_rna_sparse = rna.X if sp.issparse(rna.X) else sp.csr_matrix(rna.X)
    svd = TruncatedSVD(n_components=N_SVD_COMPONENTS, random_state=RANDOM_SEED)
    X_train_svd = svd.fit_transform(X_rna_sparse).astype(np.float32)

    svd_scaler = StandardScaler()
    X_train_svd_scaled = svd_scaler.fit_transform(X_train_svd).astype(np.float32)
    
    marker_scaler = StandardScaler()
    X_train_markers_scaled = marker_scaler.fit_transform(X_train_markers).astype(np.float32)

    X_train_rna_fused = np.hstack([X_train_svd_scaled, X_train_markers_scaled])
    
    # --- 3. H&E Image Extraction ---
    log("Extracting/Loading Train Image Features...")
    X_train_img = get_image_features(rna, IMG_SRC_TRAIN, IMG_TRAIN_CACHE, DEVICE)
    img_scaler = StandardScaler()
    X_train_img_scaled = img_scaler.fit_transform(X_train_img).astype(np.float32)

    # --- 4. Build Radial Tissue Graph ---
    log("Building 30um Radial Tissue Graph...")
    train_coords = compute_physical_microns(rna, TRAIN_MICRONS_PER_PIXEL)
    edge_index, edge_attr = build_spatial_graph(train_coords, radius=RADIUS_MICRONS)
    
    # --- 5. PyG Data Object & Loader ---
    data = Data(
        x_rna      = torch.tensor(X_train_rna_fused, dtype=torch.float32),
        x_img      = torch.tensor(X_train_img_scaled, dtype=torch.float32),
        edge_index = edge_index,
        edge_attr  = edge_attr,
        y          = torch.tensor(Y_train, dtype=torch.float32),
        train_mask = torch.ones(rna.n_obs, dtype=torch.bool),
    )

    train_loader = NeighborLoader(data, num_neighbors=NUM_NEIGHBORS, batch_size=BATCH_SIZE, input_nodes=data.train_mask, shuffle=True, num_workers=NUM_WORKERS,pin_memory=True, persistent_workers=True)

    # --- 6. Initialize Model & Weighted Loss ---
    weights = np.zeros(len(PROTEIN_NAMES), dtype=np.float32)
    for i, p in enumerate(PROTEIN_NAMES):
        weights[i] = 1.0 - float(BASELINE_SCORES.get(p, 0.5))
    weights = weights / weights.mean()
    loss_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    def weighted_mse_loss(pred, target, weights):
        return (((pred - target) ** 2) * weights).mean()

    RNA_DIM = X_train_rna_fused.shape[1]
    
    # FIXED initialization: No `fusion_dim` parameter!
    model_gat = MultimodalGATRegressor(
        rna_dim=RNA_DIM, 
        img_dim=IMG_COMPONENTS, 
        hidden_channels=HIDDEN_CHANNELS, 
        out_channels=len(PROTEIN_NAMES),
        n_layers=N_LAYERS, 
        heads=HEADS, 
        dropout=DROPOUT
    ).to(DEVICE)

    model_gat = torch.compile(model_gat, mode="reduce-overhead")

    torch._dynamo.config.suppress_errors = True
    
    optimiser = torch.optim.Adam(model_gat.parameters(), lr=LEARNING_RATE, weight_decay=1e-4, fused=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=5)

    # --- 7. Training Loop ---
    log("Starting Multimodal Radial GAT Training...")
    best_loss, patience_count = float("inf"), 0

    for epoch in range(1, N_EPOCHS + 1):
        model_gat.train()
        total_loss, n_batches = 0, 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimiser.zero_grad(set_to_none=True)
            pred = model_gat(batch.x_rna, batch.x_img, batch.edge_index, batch.edge_attr)
            loss = weighted_mse_loss(pred[:batch.batch_size], batch.y[:batch.batch_size], loss_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_gat.parameters(), max_norm=1.0)
            optimiser.step()
            total_loss += loss.item()
            n_batches += 1
            
        train_loss = total_loss / n_batches
        scheduler.step(train_loss)

        if train_loss < best_loss:
            best_loss, patience_count, flag = train_loss, 0, " ← best"
            torch.save(model_gat.state_dict(), os.path.join(OUTPUTS_DIR, "gat_best_model.pt"))
        else:
            patience_count += 1
            flag = ""

        log(f"Epoch {epoch:03d}/{N_EPOCHS}  train_loss={train_loss:.4f}  best={best_loss:.4f}{flag}")
        if patience_count >= PATIENCE:
            log("Early stopping triggered.")
            break

    # --- 8. Validation / Test Inference & Spatial Smoothing ---
    log("=" * 55)
    log("Predicting on valid_rna.h5ad and test_rna.h5ad")
    log("=" * 55)

    valid_df = pd.read_csv(VALID_GT).set_index("Unnamed: 0")
    
    for split_name, h5ad_path, img_path, cache_path, mpp, is_valid in [
        ("Valid", VALID_RNA, IMG_SRC_VALID, IMG_VALID_CACHE, TEST_MICRONS_PER_PIXEL, True),
        ("Test", TEST_RNA, IMG_SRC_TEST, IMG_TEST_CACHE, TEST_MICRONS_PER_PIXEL, False)
    ]:
        log(f"Processing {split_name} Set...")
        adata = ad.read_h5ad(h5ad_path)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # Align genes
        adata_idx = {g: i for i, g in enumerate(adata.var_names)}
        c_tr_pos, c_ad_pos = [], []
        for pos, g in enumerate(rna_gene_list):
            if g in adata_idx:
                c_tr_pos.append(pos)
                c_ad_pos.append(adata_idx[g])
                
        X_full = sp.lil_matrix((adata.n_obs, len(rna_gene_list)), dtype=np.float32)
        X_src = adata.X.tocsc() if sp.issparse(adata.X) else sp.csc_matrix(adata.X)
        X_full[:, c_tr_pos] = X_src[:, c_ad_pos]
        X_full = X_full.tocsr()
        
        # Transform RNA
        X_svd = svd.transform(X_full).astype(np.float32)
        X_markers = extract_markers(adata, marker_genes)
        
        X_svd_scaled = svd_scaler.transform(X_svd).astype(np.float32)
        X_markers_scaled = marker_scaler.transform(X_markers).astype(np.float32)
        X_rna_fused = np.hstack([X_svd_scaled, X_markers_scaled])
        
        # Extract & Transform Image
        X_img = get_image_features(adata, img_path, cache_path, DEVICE, fallback_df=valid_df if is_valid else None)
        X_img_scaled = img_scaler.transform(X_img).astype(np.float32)
        
        # Build Radius Graph
        coords = compute_physical_microns(adata, mpp, fallback_df=valid_df if is_valid else None)
        ei, ea = build_spatial_graph(coords, radius=RADIUS_MICRONS)
        
        # Predict
        model_gat.load_state_dict(torch.load(os.path.join(OUTPUTS_DIR, "gat_best_model.pt"), map_location=DEVICE))
        model_gat.eval()
        
        with torch.no_grad():
            preds = model_gat(
                torch.tensor(X_rna_fused, dtype=torch.float32).to(DEVICE), 
                torch.tensor(X_img_scaled, dtype=torch.float32).to(DEVICE), 
                ei.to(DEVICE), ea.to(DEVICE)
            ).cpu().numpy()
            
        # Strategy 4: Test-Time Spatial Smoothing (90% Self, 10% Radial Neighbors)
        log("Applying Test-Time Radial Smoothing...")
        src_np, dst_np = ei[0].numpy(), ei[1].numpy()
        adj = sp.coo_matrix((np.ones_like(src_np), (dst_np, src_np)), shape=(adata.n_obs, adata.n_obs))
        row_sums = np.array(adj.sum(axis=1)).flatten()
        has_neighbors = row_sums > 0
        row_sums[row_sums == 0] = 1.0  
        adj_norm = sp.diags(1.0 / row_sums).dot(adj)
        
        neighbor_means = adj_norm.dot(preds)
        smoothed_preds = np.copy(preds)
        smoothed_preds[has_neighbors] = 0.90 * preds[has_neighbors] + 0.10 * neighbor_means[has_neighbors]
        
        preds_raw = np.clip(np.sinh(smoothed_preds) * PRO_COFACTOR, 0, None)
        
        # Save submission
        out_df = pd.DataFrame(0.0, index=valid_df.index if is_valid else adata.obs_names, columns=valid_df.columns if is_valid else PROTEIN_COLS)
        if is_valid:
            out_df["pxl_row_in_fullres"] = valid_df["pxl_row_in_fullres"]
            out_df["pxl_col_in_fullres"] = valid_df["pxl_col_in_fullres"]
            
        pred_df = pd.DataFrame(preds_raw, index=adata.obs_names, columns=PROTEIN_NAMES)
        
        for p in PROTEIN_COLS:
            if p in PROTEIN_NAMES:
                out_df.loc[adata.obs_names, p] = pred_df[p]
                
        out_csv = os.path.join(OUTPUTS_DIR, f"{split_name.lower()}_submission.csv")
        out_df.reset_index().rename(columns={"index": "barcode"}).to_csv(out_csv, index=False)
        log(f"Saved {split_name} predictions to {out_csv}")

if __name__ == "__main__":
    main()
