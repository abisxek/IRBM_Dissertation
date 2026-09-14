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
import joblib
import json

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
DATA_DIR = '/home/ubuntu/data'
TRAIN_RNA_PATH = os.path.join(DATA_DIR, 'train_rna.h5ad')
TRAIN_PRO_PATH = os.path.join(DATA_DIR, 'train_pro.h5ad')
VALID_RNA_PATH = os.path.join(DATA_DIR, 'valid_rna.h5ad')
VALID_MOCK_PATH = os.path.join(DATA_DIR, 'valid_uniform_range.csv')
TEST_RNA_PATH = os.path.join(DATA_DIR, 'test_rna.h5ad')

OUTPUT_ROOT = '/home/ubuntu/GAT'

PCA_COMPONENTS = 512
HIDDEN_DIM = 512
NUM_LAYERS = 5

RADIUS_MICRONS = 50

EPOCHS = 400
EARLY_STOPPING_PATIENCE = 20

LEARNING_RATE = 0.001
DROPOUT = 0.2

MODEL_PATH = os.path.join(OUTPUT_ROOT, 'models', 'gatv2_spearman_model')
PREDICTIONS_DIR = os.path.join(OUTPUT_ROOT, 'predictions')

TRAIN_MICRONS_PER_PIXEL = 0.8820219467631594
TEST_MICRONS_PER_PIXEL = 0.883043249671293

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

def spatial_block_split(adata_rna, val_fraction=0.1, n_grid_x=10, n_grid_y=10, random_state=42):
    """
    Split spatial data into training and validation sets using spatial blocking.
    This avoids data leakage due to spatial autocorrelation by ensuring
    spatially separated training and validation sets.

    Parameters:
    adata_rna: AnnData object with spatial coordinates in obs['array_row'] and obs['array_col']
    val_fraction: fraction of data to use for validation
    n_grid_x, n_grid_y: number of grid cells in x and y directions
    random_state: random seed for reproducibility

    Returns:
    train_mask, val_mask: boolean arrays for training and validation indices
    """
    # Get spatial coordinates
    rows = adata_rna.obs['array_row'].values
    cols = adata_rna.obs['array_col'].values

    # Create grid boundaries
    row_bins = np.linspace(np.min(rows), np.max(rows) + 1, n_grid_x + 1)
    col_bins = np.linspace(np.min(cols), np.max(cols) + 1, n_grid_y + 1)

    # Assign each spot to a grid cell
    row_indices = np.digitize(rows, row_bins) - 1
    col_indices = np.digitize(cols, col_bins) - 1
    # Handle edge cases where value equals max
    row_indices = np.clip(row_indices, 0, n_grid_x - 1)
    col_indices = np.clip(col_indices, 0, n_grid_y - 1)

    # Create unique grid identifiers
    grid_ids = row_indices * n_grid_y + col_indices
    n_grids = n_grid_x * n_grid_y

    # Determine how many grids to hold out for validation
    n_validation_grids = max(1, int(np.ceil(n_grids * val_fraction)))

    # Randomly select grid IDs for validation
    rng = np.random.default_rng(seed=random_state)
    all_grid_ids = np.arange(n_grids)
    val_grid_ids = rng.choice(all_grid_ids, size=n_validation_grids, replace=False)

    # Create masks
    val_mask = np.isin(grid_ids, val_grid_ids)
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
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long)
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
# 3. GATv2 & DIFFERENTIABLE SURROGATE LOSS
# ==========================================
class GATv2Regressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.1):
        super(GATv2Regressor, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_lin = nn.Linear(input_dim, hidden_dim)
        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False, dropout=dropout))
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
    """
    Because True Spearman involves non-differentiable sorting, this acts as the
    smooth proxy engine to calculate gradients and update network weights.
    """
    y_pred_centered = y_pred - y_pred.mean(dim=0)
    y_true_centered = y_true - y_true.mean(dim=0)
    covariance = (y_pred_centered * y_true_centered).sum(dim=0)
    pred_std = torch.sqrt((y_pred_centered ** 2).sum(dim=0) + 1e-8)
    true_std = torch.sqrt((y_true_centered ** 2).sum(dim=0) + 1e-8)
    return 1 - (covariance / (pred_std * true_std + 1e-8)).mean()

def differentiable_proxy_loss_raw(y_pred_arcsinh, y_true_arcsinh, cofactor=150.0):
    """
    The raw-space differentiable proxy engine.
    """
    y_pred_raw = torch.sinh(torch.clamp(y_pred_arcsinh, -8.0, 8.0)) * cofactor
    y_true_raw = torch.sinh(torch.clamp(y_true_arcsinh, -8.0, 8.0)) * cofactor
    return differentiable_proxy_loss(y_pred_raw, y_true_raw)

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    print("=== Spearman Eval Spatial Multi-Omics Pipeline ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, 'models'), exist_ok=True)

    print("\n--- 1. Loading Datasets ---")
    adata_train_raw = sc.read_h5ad(TRAIN_RNA_PATH)
    adata_pro_raw = sc.read_h5ad(TRAIN_PRO_PATH)
    adata_valid_raw = sc.read_h5ad(VALID_RNA_PATH)
    adata_test_raw = sc.read_h5ad(TEST_RNA_PATH)

    print("\n--- 2. Dynamic Target Alignment ---")
    mock_valid_df = pd.read_csv(VALID_MOCK_PATH, index_col=0)
    dynamic_protein_names = sorted(list(set(mock_valid_df.columns).intersection(set(adata_pro_raw.var_names))))
    print(f"Dynamically aligned targets: {len(dynamic_protein_names)} proteins.")

    common_train_obs = adata_train_raw.obs_names.intersection(adata_pro_raw.obs_names)
    adata_train_raw = adata_train_raw[common_train_obs].copy()
    adata_pro_raw = adata_pro_raw[common_train_obs].copy()

    if 'in_tissue' in adata_train_raw.obs:
        adata_train_raw = adata_train_raw[adata_train_raw.obs['in_tissue'] == 1].copy()
        adata_pro_raw = adata_pro_raw[adata_train_raw.obs['in_tissue'] == 1].copy()
    if 'in_tissue' in adata_valid_raw.obs:
        adata_valid_raw = adata_valid_raw[adata_valid_raw.obs['in_tissue'] == 1].copy()
    if 'in_tissue' in adata_test_raw.obs:
        adata_test_raw = adata_test_raw[adata_test_raw.obs['in_tissue'] == 1].copy()

    common_valid_obs = adata_valid_raw.obs_names.intersection(mock_valid_df.index)
    adata_valid_raw = adata_valid_raw[common_valid_obs].copy()

    print("\n--- 3. Library Depth Normalization ---")
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

    print("\n--- 4. Joint Latent Space Construction (Fixed for no data leakage) ---")
    # Define paths for caching preprocessing models
    scaler_path = os.path.join(OUTPUT_ROOT, 'models', 'joint_scaler.save')
    svd_path = os.path.join(OUTPUT_ROOT, 'models', 'joint_svd.save')
    cache_info_path = os.path.join(OUTPUT_ROOT, 'models', 'cache_info.json')

    def ensure_sparse(adata):
        return adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)

    # Try to load cached preprocessing models, or compute and cache them
    # Include PCA_COMPONENTS in the cache validation to detect when dimensions change
    cache_valid = False
    if os.path.exists(scaler_path) and os.path.exists(svd_path) and os.path.exists(cache_info_path):
        try:
            with open(cache_info_path, 'r') as f:
                cache_info = json.load(f)
            if cache_info.get('pca_components') == PCA_COMPONENTS:
                cache_valid = True
        except:
            cache_valid = False

    if cache_valid:
        print("Loading cached preprocessing models...")
        joint_scaler = joblib.load(scaler_path)
        joint_svd = joblib.load(svd_path)

        # Transform data using loaded models
        X_train = ensure_sparse(adata_train_raw)
        X_train_scaled = joint_scaler.transform(X_train)

        X_valid = ensure_sparse(adata_valid_raw)
        X_test = ensure_sparse(adata_test_raw)
        X_valid_scaled = joint_scaler.transform(X_valid)
        X_test_scaled = joint_scaler.transform(X_test)

        X_train_reduced = joint_svd.transform(X_train_scaled)
        X_valid_reduced = joint_svd.transform(X_valid_scaled)
        X_test_reduced = joint_svd.transform(X_test_scaled)
    else:
        print("Computing and caching preprocessing models...")
        # Fit preprocessing on TRAINING DATA ONLY to prevent data leakage
        X_train = ensure_sparse(adata_train_raw)
        joint_scaler = StandardScaler(with_mean=False)
        X_train_scaled = joint_scaler.fit_transform(X_train)

        # Transform validation and test data using training parameters
        X_valid = ensure_sparse(adata_valid_raw)
        X_test = ensure_sparse(adata_test_raw)
        X_valid_scaled = joint_scaler.transform(X_valid)
        X_test_scaled = joint_scaler.transform(X_test)

        # Fit SVD on training data only
        joint_svd = TruncatedSVD(n_components=PCA_COMPONENTS, random_state=42)
        X_train_reduced = joint_svd.fit_transform(X_train_scaled)
        X_valid_reduced = joint_svd.transform(X_valid_scaled)
        X_test_reduced = joint_svd.transform(X_test_scaled)

        # Cache the models for future runs
        joblib.dump(joint_scaler, scaler_path)
        joblib.dump(joint_svd, svd_path)

        # Save cache info
        cache_info = {'pca_components': PCA_COMPONENTS}
        with open(cache_info_path, 'w') as f:
            json.dump(cache_info, f)

    # Store RNA PCA components
    adata_train_raw.obsm['rna_features'] = X_train_reduced
    adata_valid_raw.obsm['rna_features'] = X_valid_reduced
    adata_test_raw.obsm['rna_features']  = X_test_reduced

    # Extract features for data objects
    train_features = X_train_reduced
    valid_features = X_valid_reduced
    test_features  = X_test_reduced

    print(f"\n--- 5. Radius-Based Graph Topology ({RADIUS_MICRONS} microns) ---")
    coords_train = compute_physical_microns(adata_train_raw, TRAIN_MICRONS_PER_PIXEL)
    coords_valid = compute_physical_microns(adata_valid_raw, TEST_MICRONS_PER_PIXEL)
    coords_test  = compute_physical_microns(adata_test_raw, TEST_MICRONS_PER_PIXEL)

    print("\n--- 5. Building Spatial Graphs ---")
    # For training and validation, we split the TRAINING data (which has protein labels)
    # into training and validation sets using a spatial block split to avoid leakage due to
    # spatial autocorrelation. This ensures spatially separated train/validation sets.
    train_mask, val_mask = spatial_block_split(adata_train_raw, val_fraction=0.05)

    # Training data: use the training split of the training data
    train_data = Data(
        x=torch.tensor(train_features[train_mask], dtype=torch.float),
        edge_index=build_spatial_graph(coords_train[train_mask], radius=RADIUS_MICRONS),
        y=torch.tensor(extract_protein_targets(adata_pro_raw[train_mask], dynamic_protein_names), dtype=torch.float)
    ).to(device)

    # Validation data: use the validation split of the training data
    val_data = Data(
        x=torch.tensor(train_features[val_mask], dtype=torch.float),
        edge_index=build_spatial_graph(coords_train[val_mask], radius=RADIUS_MICRONS),
        y=torch.tensor(extract_protein_targets(adata_pro_raw[val_mask], dynamic_protein_names), dtype=torch.float)
    ).to(device)

    # Test data: use the holdout test set (no labels, for prediction only)
    test_data = Data(
        x=torch.tensor(test_features, dtype=torch.float),
        edge_index=build_spatial_graph(coords_test, radius=RADIUS_MICRONS)
    ).to(device)

    print("\n--- 6. GATv2 Optimization ---")
    model = GATv2Regressor(input_dim=PCA_COMPONENTS, hidden_dim=HIDDEN_DIM, output_dim=len(dynamic_protein_names)).to(device)
    # PyTorch 2.0 compilation for optimized execution
    model = torch.compile(model)
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
            current_alpha_mse = 0.5
            current_alpha_arcsinh = 0.5
            current_alpha_raw = 0.0
        else:
            current_alpha_mse = 0.2
            current_alpha_arcsinh = 0.3
            current_alpha_raw = 0.5

        # Using the differentiable proxy engines to compute mathematically valid gradients
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

            # --- TRUE SPEARMAN EVALUATION ---
            val_pred_raw = np.sinh(torch.clamp(val_out, -8.0, 8.0).cpu().numpy()) * 150.0
            val_true_raw = np.sinh(torch.clamp(val_data.y, -8.0, 8.0).cpu().numpy()) * 150.0

            val_scores = []
            for i in range(val_true_raw.shape[1]):
                if np.std(val_true_raw[:, i]) > 0 and np.std(val_pred_raw[:, i]) > 0:
                    val_scores.append(spearmanr(val_true_raw[:, i], val_pred_raw[:, i])[0])
                else:
                    val_scores.append(0.0)

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

    print("\n--- 7. Generating Validation Predictions ---")
    # Create data object for holdout validation set (no labels)
    coords_valid_holdout = compute_physical_microns(adata_valid_raw, TEST_MICRONS_PER_PIXEL)
    valid_holdout_data = Data(
        x=torch.tensor(valid_features, dtype=torch.float),
        edge_index=build_spatial_graph(coords_valid_holdout, radius=RADIUS_MICRONS)
    ).to(device)

    model.eval()
    with torch.no_grad():
        valid_pred = np.sinh(model(valid_holdout_data.x, valid_holdout_data.edge_index).cpu().numpy()) * 150.0

    valid_pred_df = pd.DataFrame(valid_pred, index=adata_valid_raw.obs_names, columns=dynamic_protein_names)
    valid_output_df = pd.DataFrame(0.0, index=mock_valid_df.index, columns=mock_valid_df.columns)
    valid_output_df.update(valid_pred_df)

    valid_csv_path = os.path.join(PREDICTIONS_DIR, 'validation_predictions.csv')
    valid_output_df.to_csv(valid_csv_path)
    print(f"Validation phase predictions ready for upload: {valid_csv_path}")

    print("\n--- 8. Generating Test Predictions ---")
    with torch.no_grad():
        test_pred = np.sinh(model(test_data.x, test_data.edge_index).cpu().numpy()) * 150.0

    test_pred_df = pd.DataFrame(test_pred, index=adata_test_raw.obs_names, columns=dynamic_protein_names)
    test_csv_path = os.path.join(PREDICTIONS_DIR, 'test_predictions.csv')
    test_pred_df.to_csv(test_csv_path)
    print(f"Final testing phase submission file generated successfully: {test_csv_path}")

if __name__ == "__main__":
    main()