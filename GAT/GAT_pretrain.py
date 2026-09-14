import os
import time
import json
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────
DATA_DIR        = "/home/ubuntu/data"
VALID_RNA       = os.path.join(DATA_DIR, "outputs/valid_rna_tuned_elbow.h5ad")
VALID_GT        = os.path.join(DATA_DIR, "valid_uniform_range.csv")
OUTPUT_CSV      = os.path.join(DATA_DIR, "submission.csv")

N_SVD_COMPONENTS = 256  
N_LAYERS        = 5     
HIDDEN_CHANNELS = 256
HEADS           = 4
DROPOUT         = 0.3

NEIGHBORS_PER_HOP = [10, 5, 5, 5, 5]

PROTEIN_COLS = [
    "synd", "FOXP3", "CD16", "CD31", "CXCL13",
    "Ki67", "OLIG2", "CXCR5", "HLA-A", "PD-L1",
    "PSD95", "CD20", "CD68", "CD44", "SMA",
    "MSH6", "CD23", "GFAP", "SYNA", "Podoplanin",
    "Vimentin", "CD47", "CD74", "SIRP", "Granzyme B",
    "IDH1", "MPO", "CD45", "CD21", "FIBR",
    "C-KIT", "CD3e", "TOX", "PD-1", "PDGFR",
    "CD4", "MAP2", "CD8", "MGMT", "CD38",
    "HLA-DR", "CD14", "ICOS", "Granzyme K"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ── Load Saved Artifacts & Pre-fitted Models ───
log("Loading saved SVD model, gene list, and training score matrix...")
with open(os.path.join(DATA_DIR, "svd_model.pkl"), "rb") as f:
    svd = pickle.load(f)

with open(os.path.join(DATA_DIR, "rna_gene_list.pkl"), "rb") as f:
    rna_gene_list = pickle.load(f)

# Re-fit the StandardScaler using the saved training score matrix
score_matrix_final = np.load(os.path.join(DATA_DIR, "score_matrix_final.npy"))
scaler = StandardScaler()
scaler.fit(score_matrix_final)
log("✓ Artifacts loaded successfully")


# ── Define GAT Architecture (Same as training) ─
class GATProteinPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2, heads=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        if n_layers == 1:
            self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        else:
            self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

            for _ in range(n_layers - 2):
                self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, concat=True))
                self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout, concat=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.output = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
        return self.output(x)

# Initialize model and load trained weights
model_gat = GATProteinPredictor(
    in_channels     = N_SVD_COMPONENTS,
    hidden_channels = HIDDEN_CHANNELS,
    out_channels    = len(PROTEIN_COLS),
    n_layers        = N_LAYERS,
    heads           = HEADS,
    dropout         = DROPOUT,
).to(DEVICE)

# ── Load trained GAT weights (handling torch.compile prefix) ─
log("Loading trained GAT weights...")
checkpoint = torch.load(
    os.path.join(DATA_DIR, "gat_best_model.pt"), 
    map_location=DEVICE
)

# Clean up '_orig_mod.' prefix added by torch.compile() during training
cleaned_checkpoint = {}
for key, value in checkpoint.items():
    new_key = key.replace("_orig_mod.", "") if key.startswith("_orig_mod.") else key
    cleaned_checkpoint[new_key] = value

model_gat.load_state_dict(cleaned_checkpoint)
model_gat.eval()
log("✓ Model weights loaded and set to eval mode")


# ── Predict on New/Validation Data ─────────────
log("Loading validation RNA data...")
rna_v = ad.read_h5ad(VALID_RNA)
valid_df = pd.read_csv(VALID_GT).set_index(pd.read_csv(VALID_GT).columns[0])

log("Normalising validation RNA using precomputed (full-transcriptome) size factors...")
assert "size_factor" in rna_v.obs.columns, \
    "size_factor missing — re-run gene selection tuning with the size-factor fix"

size_factors_v = rna_v.obs["size_factor"].values.astype(np.float32)
size_factors_v = np.where(size_factors_v == 0, 1.0, size_factors_v)

target_sum = 1e4
if sp.issparse(rna_v.X):
    inv_sf_v = sp.diags(target_sum / size_factors_v)
    rna_v.X = (inv_sf_v @ rna_v.X).tocsr()
else:
    rna_v.X = rna_v.X * (target_sum / size_factors_v[:, None])

sc.pp.log1p(rna_v)

# Align validation genes to training gene order
valid_gene_idx = {g: i for i, g in enumerate(rna_v.var_names)}
common_train_pos = []
common_valid_pos = []
for pos, g in enumerate(rna_gene_list):
    if g in valid_gene_idx:
        common_train_pos.append(pos)
        common_valid_pos.append(valid_gene_idx[g])

X_valid_full = sp.lil_matrix((rna_v.n_obs, len(rna_gene_list)), dtype=np.float32)
X_valid_source = rna_v.X.tocsc() if sp.issparse(rna_v.X) else sp.csc_matrix(rna_v.X)
X_valid_full[:, common_train_pos] = X_valid_source[:, common_valid_pos]
X_valid_full = X_valid_full.tocsr()

# Apply SVD & Scaling
log("Applying SVD transform and scaling...")
score_matrix_v = svd.transform(X_valid_full).astype(np.float32)
X_valid_scaled = scaler.transform(score_matrix_v).astype(np.float32)

# Build validation graph structure
log("Building validation graph...")
rows_v = rna_v.obs["array_row"].values.astype(int)
cols_v = rna_v.obs["array_col"].values.astype(int)
coord_to_idx_v = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows_v, cols_v))}

directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
src_v, dst_v = [], []
for i, (r, c) in enumerate(zip(rows_v, cols_v)):
    for dr, dc in directions:
        nb = coord_to_idx_v.get((int(r)+dr, int(c)+dc))
        if nb is not None:
            src_v.append(i)
            dst_v.append(nb)

edge_index_v = torch.tensor([src_v, dst_v], dtype=torch.long).to(DEVICE)
x_v = torch.tensor(X_valid_scaled, dtype=torch.float32).to(DEVICE)

# Run Inference
log("Running model inference...")
with torch.no_grad():
    preds = model_gat(x_v, edge_index_v).cpu().numpy()

# Inverse transform predictions (arcsinh back to original scale)
PRO_COFACTOR = 150.0
preds_raw = np.sinh(preds) * PRO_COFACTOR
preds_raw = np.clip(preds_raw, 0, None)

# Save Submission
log("Saving submission file...")
barcodes = rna_v.obs_names.tolist()
pxl_rows, pxl_cols = [], []
for bc in barcodes:
    if bc in valid_df.index:
        pxl_rows.append(int(valid_df.loc[bc, "pxl_row_in_fullres"]))
        pxl_cols.append(int(valid_df.loc[bc, "pxl_col_in_fullres"]))
    else:
        pxl_rows.append(0)
        pxl_cols.append(0)

submission = pd.DataFrame({"barcode": barcodes, "pxl_row_in_fullres": pxl_rows, "pxl_col_in_fullres": pxl_cols})
for j, protein in enumerate(PROTEIN_COLS):
    submission[protein] = preds_raw[:, j]

submission.to_csv(OUTPUT_CSV, index=False)
log(f"✓ Submission successfully saved to {OUTPUT_CSV}")