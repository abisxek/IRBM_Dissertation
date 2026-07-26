import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr



# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = '/home/ubuntu/data'
FEATURES_DIR = '/home/ubuntu/data'
OUTPUT_ROOT = '/home/ubuntu/GAT'

TRAIN_RNA_PATH = os.path.join(DATA_DIR, 'train_rna.h5ad')
TRAIN_PRO_PATH = os.path.join(DATA_DIR, 'train_pro.h5ad')
VALID_RNA_PATH = os.path.join(DATA_DIR, 'valid_rna.h5ad')
VALID_MOCK_PATH = os.path.join(DATA_DIR, 'valid_uniform_range.csv')
TEST_RNA_PATH = os.path.join(DATA_DIR, 'test_rna.h5ad')

# Extracted H&E Visual Features from ResNet50
IMG_TRAIN_PATH = os.path.join(FEATURES_DIR, 'train_image_features.npy')
IMG_VALID_PATH = os.path.join(FEATURES_DIR, 'valid_image_features.npy')
IMG_TEST_PATH = os.path.join(FEATURES_DIR, 'test_image_features.npy')

PCA_COMPONENTS = 512
IMG_COMPONENTS = 2048 # ResNet50 output size
FUSED_DIM = PCA_COMPONENTS + IMG_COMPONENTS

HIDDEN_DIM = 256 # Increased to handle the larger 2176-dim multimodal input
NUM_LAYERS = 3
RADIUS_MICRONS = 30

EPOCHS = 500
EARLY_STOPPING_PATIENCE = 20
LEARNING_RATE = 0.0001
DROPOUT = 0.2

MODEL_PATH = os.path.join(OUTPUT_ROOT, 'models', 'gatv2_multimodal_model')
PREDICTIONS_DIR = os.path.join(OUTPUT_ROOT, 'predictions')

TRAIN_MICRONS_PER_PIXEL = 0.8820219467631594
TEST_MICRONS_PER_PIXEL = 0.883043249671293

os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, 'models'), exist_ok=True)

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================
def compute_physical_microns(adata, microns_per_pixel):
    pixel_rows = adata.obs['pxl_row_in_fullres'].values.astype(np.float64)
    pixel_cols = adata.obs['pxl_col_in_fullres'].values.astype(np.float64)
    micron_matrix = np.zeros((len(pixel_rows), 2))
    micron_matrix[:, 0] = pixel_rows * microns_per_pixel
    micron_matrix[:, 1] = pixel_cols * microns_per_pixel
    return micron_matrix

def split_train_val_spatial(adata_rna, val_fraction=0.05):
    rows = adata_rna.obs['array_row'].values
    cols = adata_rna.obs['array_col'].values
    row_thresh = np.quantile(rows, 1.0 - np.sqrt(val_fraction))
    col_thresh = np.quantile(cols, 1.0 - np.sqrt(val_fraction))
    val_mask = (rows >= row_thresh) & (cols >= col_thresh)
    train_mask = ~val_mask
    return train_mask, val_mask

def build_spatial_graph(coords, radius=RADIUS_MICRONS):
    n_spots = coords.shape[0]
    nbrs = NearestNeighbors(radius=radius, metric='euclidean').fit(coords)
    _, indices = nbrs.radius_neighbors(coords)
    edge_set = set()
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i != j:
                edge_set.add((i, j))
                edge_set.add((j, i))
    edge_list = [list(e) for e in edge_set]
    if not edge_list: return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

def extract_protein_targets(adata_pro_subset, dynamic_protein_names):
    columns = []
    for protein in dynamic_protein_names:
        if protein in adata_pro_subset.var_names:
            col = adata_pro_subset[:, protein].X
        else:
            col = np.zeros((adata_pro_subset.n_obs, 1))
        col = col.toarray() if sp.issparse(col) else np.asarray(col)
        columns.append(col.reshape(adata_pro_subset.n_obs, 1))
    return np.hstack(columns)

# ==========================================
# 3. MULTIMODAL GATv2 & DIFFERENTIABLE SURROGATE LOSS
# ==========================================
class GATv2Regressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.1):
        super(GATv2Regressor, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_lin = nn.Linear(input_dim, hidden_dim)
        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout))
        self.output_lin = nn.Linear(hidden_dim, output_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.residuals = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

    def forward(self, x, edge_index):
        x = self.input_lin(x)
        x = F.relu(x)
        for i in range(self.num_layers):
            residual = x
            if edge_index.numel() > 0:
                x = self.gat_layers[i](x, edge_index)
            x = self.layer_norms[i](x + self.residuals[i](residual))
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output_lin(x)

def differentiable_proxy_loss(y_pred, y_true):
    y_pred_centered = y_pred - y_pred.mean(dim=0)
    y_true_centered = y_true - y_true.mean(dim=0)
    covariance = (y_pred_centered * y_true_centered).sum(dim=0)
    pred_std = torch.sqrt((y_pred_centered ** 2).sum(dim=0) + 1e-8)
    true_std = torch.sqrt((y_true_centered ** 2).sum(dim=0) + 1e-8)
    return 1 - (covariance / (pred_std * true_std + 1e-8)).mean()

def differentiable_proxy_loss_raw(y_pred_arcsinh, y_true_arcsinh, cofactor=150.0):
    y_pred_raw = torch.sinh(torch.clamp(y_pred_arcsinh, -8.0, 8.0)) * cofactor
    y_true_raw = torch.sinh(torch.clamp(y_true_arcsinh, -8.0, 8.0)) * cofactor
    return differentiable_proxy_loss(y_pred_raw, y_true_raw)

# ==========================================
# 4. MAIN MULTIMODAL PIPELINE
# ==========================================
def main():
    print("=== Spearman Eval Multimodal (RNA + H&E) GATv2 Pipeline ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n--- 1. Loading Datasets and Visual Embeddings ---")
    adata_train_raw = sc.read_h5ad(TRAIN_RNA_PATH)
    adata_valid_raw = sc.read_h5ad(VALID_RNA_PATH)
    adata_test_raw = sc.read_h5ad(TEST_RNA_PATH)
    adata_pro_raw = sc.read_h5ad(TRAIN_PRO_PATH)

    # SECURE ALIGNMENT: Attach image features immediately before any filtering
    print("Binding ResNet50 visual embeddings to AnnData objects...")
    adata_train_raw.obsm['image_features'] = np.load(IMG_TRAIN_PATH)
    if os.path.exists(IMG_VALID_PATH):
        adata_valid_raw.obsm['image_features'] = np.load(IMG_VALID_PATH)
    else:
        print(f"Warning: {IMG_VALID_PATH} not found. Using zero array for valid_image_features.")
        num_valid_obs = adata_valid_raw.n_obs
        adata_valid_raw.obsm['image_features'] = np.zeros((num_valid_obs, IMG_COMPONENTS))
    adata_test_raw.obsm['image_features']  = np.load(IMG_TEST_PATH)

    print("\n--- 2. Dynamic Target & Spatial Filtering ---")
    mock_valid_df = pd.read_csv(VALID_MOCK_PATH, index_col=0)
    dynamic_protein_names = sorted(list(set(mock_valid_df.columns).intersection(set(adata_pro_raw.var_names))))
    
    # Align training RNA with training Proteins
    common_train_obs = adata_train_raw.obs_names.intersection(adata_pro_raw.obs_names)
    adata_train_raw = adata_train_raw[common_train_obs].copy()
    adata_pro_raw = adata_pro_raw[common_train_obs].copy()

    # Apply Quality Control masks (image features safely follow due to .obsm binding)
    if 'in_tissue' in adata_train_raw.obs:
        adata_train_raw = adata_train_raw[adata_train_raw.obs['in_tissue'] == 1].copy()
        adata_pro_raw = adata_pro_raw[adata_train_raw.obs['in_tissue'] == 1].copy()
    if 'in_tissue' in adata_valid_raw.obs:
        adata_valid_raw = adata_valid_raw[adata_valid_raw.obs['in_tissue'] == 1].copy()
    if 'in_tissue' in adata_test_raw.obs:
        adata_test_raw = adata_test_raw[adata_test_raw.obs['in_tissue'] == 1].copy()

    common_valid_obs = adata_valid_raw.obs_names.intersection(mock_valid_df.index)
    adata_valid_raw = adata_valid_raw[common_valid_obs].copy()

    print("\n--- 3. Transcriptomic Normalization ---")
    for ad in [adata_train_raw, adata_valid_raw, adata_test_raw]:
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)

    pro_X = adata_pro_raw.X
    if sp.issparse(pro_X): pro_X = pro_X.toarray()
    adata_pro_raw.X = np.arcsinh(np.asarray(pro_X) / 150.0)

    shared_genes = sorted(list(set(adata_train_raw.var_names)
                               .intersection(set(adata_valid_raw.var_names))
                               .intersection(set(adata_test_raw.var_names))))
    
    adata_train_raw = adata_train_raw[:, shared_genes].copy()
    adata_valid_raw = adata_valid_raw[:, shared_genes].copy()
    adata_test_raw  = adata_test_raw[:, shared_genes].copy()

    print("\n--- 4. Joint RNA Latent Space Construction ---")
    def ensure_sparse(adata): return adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    
    X_all_stacked = sp.vstack([ensure_sparse(adata_train_raw), ensure_sparse(adata_valid_raw), ensure_sparse(adata_test_raw)])
    joint_scaler = StandardScaler(with_mean=False)
    X_all_scaled = joint_scaler.fit_transform(X_all_stacked)

    joint_svd = TruncatedSVD(n_components=PCA_COMPONENTS, random_state=42)
    X_all_reduced = joint_svd.fit_transform(X_all_scaled)

    n_train = adata_train_raw.n_obs
    n_valid = adata_valid_raw.n_obs
    
    # Store RNA PCA components
    adata_train_raw.obsm['rna_features'] = X_all_reduced[:n_train]
    adata_valid_raw.obsm['rna_features'] = X_all_reduced[n_train:n_train+n_valid]
    adata_test_raw.obsm['rna_features']  = X_all_reduced[n_train+n_valid:]

    print("\n--- 5. Late Multimodal Fusion (RNA + H&E) ---")
    # Concatenate 128 RNA features with 2048 Image features
    fused_train = np.hstack([adata_train_raw.obsm['rna_features'], adata_train_raw.obsm['image_features']])
    fused_valid = np.hstack([adata_valid_raw.obsm['rna_features'], adata_valid_raw.obsm['image_features']])
    fused_test  = np.hstack([adata_test_raw.obsm['rna_features'],  adata_test_raw.obsm['image_features']])
    print(f"Fused node dimension per cell: {fused_train.shape[1]}")

    train_mask, val_mask = split_train_val_spatial(adata_train_raw, val_fraction=0.05)

    print(f"\n--- 6. Radius-Based Graph Topology ({RADIUS_MICRONS} microns) ---")
    coords_train_full = compute_physical_microns(adata_train_raw, TRAIN_MICRONS_PER_PIXEL)
    coords_valid      = compute_physical_microns(adata_valid_raw, TEST_MICRONS_PER_PIXEL)
    coords_test       = compute_physical_microns(adata_test_raw, TEST_MICRONS_PER_PIXEL)

    train_data = Data(
        x=torch.tensor(fused_train[train_mask], dtype=torch.float),
        edge_index=build_spatial_graph(coords_train_full[train_mask], radius=RADIUS_MICRONS),
        y=torch.tensor(extract_protein_targets(adata_pro_raw[train_mask], dynamic_protein_names), dtype=torch.float)
    ).to(device)

    val_data = Data(
        x=torch.tensor(fused_train[val_mask], dtype=torch.float),
        edge_index=build_spatial_graph(coords_train_full[val_mask], radius=RADIUS_MICRONS),
        y=torch.tensor(extract_protein_targets(adata_pro_raw[val_mask], dynamic_protein_names), dtype=torch.float)
    ).to(device)

    valid_data = Data(
        x=torch.tensor(fused_valid, dtype=torch.float),
        edge_index=build_spatial_graph(coords_valid, radius=RADIUS_MICRONS)
    ).to(device)

    test_data = Data(
        x=torch.tensor(fused_test, dtype=torch.float),
        edge_index=build_spatial_graph(coords_test, radius=RADIUS_MICRONS)
    ).to(device)

    print("\n--- 7. Multimodal GATv2 Optimization ---")
    model = GATv2Regressor(input_dim=FUSED_DIM, hidden_dim=HIDDEN_DIM, output_dim=len(dynamic_protein_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_spearman = -float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        if epoch == 30:
            print("  [PHASE TRANSITION] Resetting early stopping for Fine-Tuning Phase...")
            best_val_spearman = -float('inf')
            patience_counter = 0

        model.train()
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index)
        
        if epoch < 30:
            current_alpha_mse, current_alpha_arcsinh, current_alpha_raw = 0.5, 0.5, 0.0  
        else:
            current_alpha_mse, current_alpha_arcsinh, current_alpha_raw = 0.2, 0.3, 0.5  
            
        loss_mse = F.mse_loss(out, train_data.y)
        loss_proxy_arcsinh = differentiable_proxy_loss(out, train_data.y)
        loss_proxy_raw = differentiable_proxy_loss_raw(out, train_data.y)
        
        loss = (current_alpha_mse * loss_mse + 
                current_alpha_arcsinh * loss_proxy_arcsinh + 
                current_alpha_raw * loss_proxy_raw)
                
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(val_data.x, val_data.edge_index)
            
            val_pred_raw = np.sinh(torch.clamp(val_out, -8.0, 8.0).cpu().numpy()) * 150.0
            val_true_raw = np.sinh(torch.clamp(val_data.y, -8.0, 8.0).cpu().numpy()) * 150.0
            
            val_scores = [spearmanr(val_true_raw[:, i], val_pred_raw[:, i])[0] 
                          if (np.std(val_true_raw[:, i]) > 0 and np.std(val_pred_raw[:, i]) > 0) else 0.0 
                          for i in range(val_true_raw.shape[1])]
            val_spearman = np.mean(val_scores)

        scheduler.step(val_spearman)
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | Proxy Grad Loss: {loss.item():.4f} | TRUE SPEARMAN SCORE: {val_spearman:.4f}")

        if val_spearman > best_val_spearman:
            best_val_spearman = val_spearman
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at Epoch {epoch+1}.")
                break

    if best_state is not None: model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH + '.pt')

    print("\n--- 8. Generating Final Multimodal Predictions ---")
    model.eval()
    with torch.no_grad():
        valid_pred = np.sinh(model(valid_data.x, valid_data.edge_index).cpu().numpy()) * 150.0
        test_pred = np.sinh(model(test_data.x, test_data.edge_index).cpu().numpy()) * 150.0

    valid_pred_df = pd.DataFrame(valid_pred, index=adata_valid_raw.obs_names, columns=dynamic_protein_names)
    valid_output_df = pd.DataFrame(0.0, index=mock_valid_df.index, columns=mock_valid_df.columns)
    valid_output_df.update(valid_pred_df)
    valid_output_df.to_csv(os.path.join(PREDICTIONS_DIR, 'validation_predictions.csv'))
    
    test_pred_df = pd.DataFrame(test_pred, index=adata_test_raw.obs_names, columns=dynamic_protein_names)
    test_pred_df.to_csv(os.path.join(PREDICTIONS_DIR, 'test_predictions.csv'))
    
    print("Files successfully generated using Multimodal RNA + ResNet50 features.")

if __name__ == "__main__":
    main()