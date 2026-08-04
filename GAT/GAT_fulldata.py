# ── gat_train_svd.py ──────────────────────────────
# Same GAT pipeline as gat_train.py, but dimensionality
# reduction is done with TruncatedSVD on log-normalised
# gene expression instead of hallmark/C8 pathway scores.
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATv2Conv
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import warnings
import os
import time
import json
import pickle
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────
DATA_DIR      = "/home/ubuntu/data"
RNA_PATH      = os.path.join(DATA_DIR, "train_rna.h5ad")
PRO_PATH      = os.path.join(DATA_DIR, "train_pro.h5ad")
PRO_COFACTOR  = 150.0
CHUNK_SIZE    = 5000
RANDOM_SEED   = 42
N_BINS_SUBSET = None

# SVD config (replaces GMT / pathway config)
N_SVD_COMPONENTS = 256  

# GAT config
N_LAYERS        = 5     
HIDDEN_CHANNELS = 256
HEADS           = 4
DROPOUT         = 0.3
BATCH_SIZE      = 512
N_EPOCHS        = 80
PATIENCE        = 20
LEARNING_RATE   = 0.001
NUM_WORKERS     = 4

# NeighborLoader hop-sampling fan-out. Must have exactly one entry
# per GAT layer (the number of hops = the number of message-passing
# steps = N_LAYERS), otherwise the sampled receptive field won't
# match what the model actually needs. Define a base fan-out per
# hop and repeat/truncate it to N_LAYERS automatically.
NEIGHBORS_PER_HOP = [10, 5, 5, 5, 5]  # extend if you use N_LAYERS > 5
if N_LAYERS <= len(NEIGHBORS_PER_HOP):
    NUM_NEIGHBORS = NEIGHBORS_PER_HOP[:N_LAYERS]
else:
    NUM_NEIGHBORS = NEIGHBORS_PER_HOP + \
        [NEIGHBORS_PER_HOP[-1]] * (N_LAYERS - len(NEIGHBORS_PER_HOP))

# Submission protein column order
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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM  : "
          f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")


# ── Helper functions ───────────────────────────
def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Load data ──────────────────────────────────
log("Loading RNA data...")
t   = time.time()
rna = ad.read_h5ad(RNA_PATH)
log(f"RNA: {rna.n_obs:,} bins × "
    f"{rna.n_vars:,} genes "
    f"({time.time()-t:.0f}s)")

log("Loading Protein data...")
pro = ad.read_h5ad(PRO_PATH)
log(f"Protein: {pro.n_obs:,} bins × "
    f"{pro.n_vars} proteins")

# Align bins
shared_bins = rna.obs_names.intersection(
    pro.obs_names
)
rna = rna[shared_bins].copy()
pro = pro[shared_bins].copy()
log(f"Aligned: {len(shared_bins):,} bins")

# Optional subset
if N_BINS_SUBSET is not None:
    rows     = rna.obs["array_row"].values.astype(int)
    cols     = rna.obs["array_col"].values.astype(int)
    row_mid  = rows.mean()
    col_mid  = cols.mean()
    distance = np.sqrt(
        (rows - row_mid)**2 + (cols - col_mid)**2
    )
    subset_idx = np.argsort(distance)[:N_BINS_SUBSET]
    subset_idx = np.sort(subset_idx)
    rna = rna[subset_idx].copy()
    pro = pro[subset_idx].copy()
    log(f"Subset: {N_BINS_SUBSET:,} bins")


# ── Normalise ──────────────────────────────────
log("Normalising RNA...")
sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)

# Fix the training gene vocabulary/order — this is what the
# validation set will later be aligned to before SVD.transform()
rna_gene_list = list(rna.var_names)
rna_gene_idx  = {g: i for i, g in enumerate(rna_gene_list)}

log("Normalising protein...")
pro_dense = pro.X.toarray() if sp.issparse(pro.X) \
            else np.array(pro.X, dtype=np.float32)
Y_all = np.arcsinh(
    pro_dense / PRO_COFACTOR
).astype(np.float32)
del pro_dense
PROTEIN_NAMES = list(pro.var_names)
log(f"Proteins: {PROTEIN_NAMES}")


# ── Use ALL bins for training ──────────────────
n_total   = rna.n_obs
train_idx = np.arange(n_total)
log(f"Training on ALL {n_total:,} bins")


# ── Truncated SVD dimensionality reduction ─────
# Replaces hallmark/C8 pathway scoring. Fit directly on the
# sparse log-normalised gene expression matrix — TruncatedSVD
# handles sparse input natively via randomized SVD, so no need
# to densify or chunk like the pathway-score loop did.
n_bins = rna.n_obs
log(f"Fitting TruncatedSVD "
    f"(n_components={N_SVD_COMPONENTS}) on "
    f"{n_bins:,} bins × {len(rna_gene_list):,} genes...")

t_start = time.time()

X_rna_sparse = rna.X if sp.issparse(rna.X) else sp.csr_matrix(rna.X)

svd = TruncatedSVD(
    n_components = N_SVD_COMPONENTS,
    random_state = RANDOM_SEED,
)
score_matrix_final = svd.fit_transform(X_rna_sparse).astype(np.float32)

log(f"SVD done in {(time.time()-t_start)/60:.1f} min")
log(f"Explained variance ratio (sum): "
    f"{svd.explained_variance_ratio_.sum():.4f}")
log(f"Score matrix: {score_matrix_final.shape}")

# names_final now refers to SVD component indices rather than
# pathway names, kept for symmetry with the rest of the pipeline
names_final = [f"SVD_{i}" for i in range(N_SVD_COMPONENTS)]

# Save scores + SVD model (SVD model must be reused, unrefit,
# on the validation gene matrix — analogous to the scaler)
np.save(os.path.join(DATA_DIR, "score_matrix_final.npy"),
        score_matrix_final)
np.save(os.path.join(DATA_DIR, "names_final.npy"),
        np.array(names_final))
np.save(os.path.join(DATA_DIR, "train_idx.npy"),
        train_idx)
with open(os.path.join(DATA_DIR, "svd_model.pkl"), "wb") as f:
    pickle.dump(svd, f)
with open(os.path.join(DATA_DIR, "rna_gene_list.pkl"), "wb") as f:
    pickle.dump(rna_gene_list, f)
log("Scores + SVD model saved")


# ── Scale features — ALL bins ──────────────────
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(
    score_matrix_final
).astype(np.float32)
Y_train        = Y_all

log(f"X_train: {X_train_scaled.shape}")
log(f"Y_train: {Y_train.shape}")


# ── Build tissue graph ─────────────────────────
log("Building tissue graph...")
rows_arr = rna.obs["array_row"].values.astype(int)
cols_arr = rna.obs["array_col"].values.astype(int)

coord_to_idx = {
    (int(r), int(c)): i
    for i, (r, c) in enumerate(
        zip(rows_arr, cols_arr)
    )
}

directions = [
    (-1,-1), (-1,0), (-1,1),
    ( 0,-1),          ( 0,1),
    ( 1,-1), ( 1,0), ( 1,1),
]

src_list = []
dst_list = []
for i, (r, c) in enumerate(
    zip(rows_arr, cols_arr)
):
    for dr, dc in directions:
        nb = coord_to_idx.get(
            (int(r)+dr, int(c)+dc)
        )
        if nb is not None:
            src_list.append(i)
            dst_list.append(nb)

edge_index = torch.tensor(
    [src_list, dst_list], dtype=torch.long
)
log(f"Graph: {n_bins:,} nodes, "
    f"{edge_index.shape[1]:,} edges")


# ── PyG data object — ALL bins as train ────────
x = torch.tensor(
    X_train_scaled, dtype=torch.float32
)
y = torch.tensor(Y_train, dtype=torch.float32)

train_mask = torch.ones(n_bins, dtype=torch.bool)

data = Data(
    x          = x,
    edge_index = edge_index,
    y          = y,
    train_mask = train_mask,
)
log(f"Data: {data}")


# ── GAT model ──────────────────────────────────
class GATProteinPredictor(nn.Module):
    """
    Dynamic-depth GAT: n_layers GATConv blocks followed by a
    linear output head. All intermediate layers use multi-head
    attention with concatenation; the final GATConv layer
    always averages heads (concat=False) so hidden_channels is
    the width feeding the output head regardless of depth.

    n_layers = 1 is a single GATConv straight to hidden width
    (heads averaged) followed by the linear head — no
    intermediate concat layers in that case.
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        n_layers = 2,
        heads    = 4,
        dropout  = 0.2,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        self.dropout  = dropout
        self.n_layers = n_layers

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        if n_layers == 1:
            # single layer: straight to hidden_channels, heads averaged
            self.convs.append(
                GATv2Conv(
                    in_channels,
                    hidden_channels,
                    heads   = heads,
                    dropout = dropout,
                    concat  = False,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        else:
            # first layer: in_channels -> hidden_channels * heads (concat)
            self.convs.append(
                GATv2Conv(
                    in_channels,
                    hidden_channels,
                    heads   = heads,
                    dropout = dropout,
                    concat  = True,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

            # middle layers (if any): hidden*heads -> hidden*heads (concat)
            for _ in range(n_layers - 2):
                self.convs.append(
                    GATv2Conv(
                        hidden_channels * heads,
                        hidden_channels,
                        heads   = heads,
                        dropout = dropout,
                        concat  = True,
                    )
                )
                self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

            # final layer: hidden*heads -> hidden_channels, heads averaged
            self.convs.append(
                GATv2Conv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads   = 1,
                    dropout = dropout,
                    concat  = False,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.output = nn.Linear(
            hidden_channels, out_channels
        )

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = F.dropout(
                x, p=self.dropout,
                training=self.training
            )
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)

        return self.output(x)


IN_CHANNELS  = score_matrix_final.shape[1]
OUT_CHANNELS = len(PROTEIN_NAMES)

model_gat = GATProteinPredictor(
    in_channels     = IN_CHANNELS,
    hidden_channels = HIDDEN_CHANNELS,
    out_channels    = OUT_CHANNELS,
    n_layers        = N_LAYERS,
    heads           = HEADS,
    dropout         = DROPOUT,
).to(DEVICE)

# PyTorch 2.0 compilation for optimized execution
model_gat = torch.compile(model_gat)

n_params = sum(
    p.numel() for p in model_gat.parameters()
)
log(f"GAT: {n_params:,} parameters")
log(f"  Layers: {N_LAYERS}")
log(f"  Input : {IN_CHANNELS} (SVD components)")
log(f"  Hidden: {HIDDEN_CHANNELS} × {HEADS} heads")
log(f"  Output: {OUT_CHANNELS}")


# ── Data loader — train only ───────────────────
train_loader = NeighborLoader(
    data,
    num_neighbors = NUM_NEIGHBORS,
    batch_size    = BATCH_SIZE,
    input_nodes   = data.train_mask,
    shuffle       = True,
    num_workers   = NUM_WORKERS,
)
log(f"Train batches: {len(train_loader)}")


# ── Training setup ─────────────────────────────
optimiser = torch.optim.Adam(
    model_gat.parameters(),
    lr           = LEARNING_RATE,
    weight_decay = 1e-4,
)
scheduler = torch.optim.lr_scheduler\
    .ReduceLROnPlateau(
        optimiser,
        mode     = "min",
        factor   = 0.5,
        patience = 5,
    )
criterion = nn.MSELoss()


def train_epoch(model, loader,
                optimiser, criterion, device):
    model.train()
    total_loss = 0
    n_batches  = 0
    for batch in loader:
        batch = batch.to(device)
        optimiser.zero_grad()
        pred  = model(batch.x, batch.edge_index)
        seed  = batch.batch_size
        loss  = criterion(
            pred[:seed], batch.y[:seed]
        )
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / n_batches


# ── Training loop ──────────────────────────────
log("=" * 55)
log("Starting GAT training on FULL dataset")
log("=" * 55)

best_loss      = float("inf")
patience_count = 0
history        = []
t_start        = time.time()

for epoch in range(1, N_EPOCHS + 1):
    t_ep = time.time()

    train_loss = train_epoch(
        model_gat, train_loader,
        optimiser, criterion, DEVICE
    )

    scheduler.step(train_loss)

    history.append({
        "epoch"      : epoch,
        "train_loss" : train_loss,
        "time_s"     : time.time() - t_ep,
    })

    if train_loss < best_loss:
        best_loss      = train_loss
        patience_count = 0
        torch.save(
            model_gat.state_dict(),
            os.path.join(DATA_DIR,
                         "gat_best_model.pt")
        )
        flag = " ← best"
    else:
        patience_count += 1
        flag = ""

    ep_time = time.time() - t_ep
    log(f"Epoch {epoch:03d}/{N_EPOCHS}  "
        f"train={train_loss:.4f}  "
        f"best={best_loss:.4f}  "
        f"[{ep_time:.0f}s]{flag}")

    if patience_count >= PATIENCE:
        log(f"Early stopping at epoch {epoch}")
        break

total_time = time.time() - t_start
log(f"Training done in {total_time/60:.1f} min")
log(f"Best train loss: {best_loss:.4f}")


# ── Save training history ──────────────────────
pd.DataFrame(history).to_csv(
    os.path.join(DATA_DIR, "gat_history.csv"),
    index=False
)

config = {
    "in_channels"      : IN_CHANNELS,
    "hidden_channels"  : HIDDEN_CHANNELS,
    "out_channels"     : OUT_CHANNELS,
    "n_layers"         : N_LAYERS,
    "heads"            : HEADS,
    "dropout"          : DROPOUT,
    "batch_size"       : BATCH_SIZE,
    "n_epochs"         : N_EPOCHS,
    "patience"         : PATIENCE,
    "learning_rate"    : LEARNING_RATE,
    "num_neighbors"    : NUM_NEIGHBORS,
    "n_svd_components" : N_SVD_COMPONENTS,
    "svd_explained_var": float(svd.explained_variance_ratio_.sum()),
    "best_train_loss"  : best_loss,
    "total_time_min"   : total_time / 60,
}

with open(
    os.path.join(DATA_DIR, "gat_config.json"),
    "w"
) as f:
    json.dump(config, f, indent=2)

log("Saved: gat_history.csv, gat_config.json")


# ══════════════════════════════════════════════
# PREDICT ON valid_rna.h5ad
# ══════════════════════════════════════════════

VALID_RNA  = os.path.join(DATA_DIR, "valid_rna.h5ad")
VALID_GT   = os.path.join(DATA_DIR, "valid_uniform_range.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "submission.csv")

log("=" * 55)
log("Predicting on valid_rna.h5ad")
log("=" * 55)

# Load validation RNA
log("Loading valid_rna.h5ad...")
t     = time.time()
rna_v = ad.read_h5ad(VALID_RNA)
log(f"Valid RNA: {rna_v.n_obs:,} bins × "
    f"{rna_v.n_vars:,} genes "
    f"({time.time()-t:.0f}s)")

# Load pixel coordinates
log("Loading valid_uniform_range.csv...")
valid_df  = pd.read_csv(VALID_GT)
first_col = valid_df.columns[0]
valid_df  = valid_df.set_index(first_col)
log(f"Valid CSV: {valid_df.shape}")

# Normalise validation RNA
log("Normalising validation RNA...")
sc.pp.normalize_total(rna_v, target_sum=1e4)
sc.pp.log1p(rna_v)

# ── Align validation genes to the training gene vocabulary ──
# TruncatedSVD.transform() requires the exact same feature
# (gene) columns, in the exact same order, that svd.fit() saw.
# Any training gene missing from the validation set is filled
# with zeros; any validation-only gene is dropped.
log("Aligning validation genes to training gene order...")
n_bins_v = rna_v.n_obs

valid_gene_idx = {g: i for i, g in enumerate(rna_v.var_names)}

# indices into rna_v for genes that exist in both, in the
# ORDER of rna_gene_list (the training gene order)
common_train_pos = []  # position in rna_gene_list
common_valid_pos = []  # position in rna_v.var_names
for pos, g in enumerate(rna_gene_list):
    if g in valid_gene_idx:
        common_train_pos.append(pos)
        common_valid_pos.append(valid_gene_idx[g])

log(f"  {len(common_train_pos):,}/{len(rna_gene_list):,} "
    f"training genes found in validation set")

X_valid_full = sp.lil_matrix(
    (n_bins_v, len(rna_gene_list)), dtype=np.float32
)
X_valid_source = rna_v.X
if sp.issparse(X_valid_source):
    X_valid_source = X_valid_source.tocsc()
else:
    X_valid_source = sp.csc_matrix(X_valid_source)

X_valid_full[:, common_train_pos] = X_valid_source[:, common_valid_pos]
X_valid_full = X_valid_full.tocsr()

# Apply the SAME (already-fit) SVD model
log("Applying trained SVD transform to validation data...")
t_start = time.time()
score_matrix_v = svd.transform(X_valid_full).astype(np.float32)
log(f"SVD transform done in {(time.time()-t_start):.0f}s")
log(f"Valid score matrix: {score_matrix_v.shape}")

# Scale using training scaler
log("Scaling validation scores...")
X_valid_scaled = scaler.transform(
    score_matrix_v
).astype(np.float32)

# Build validation graph
log("Building validation graph...")
rows_v = rna_v.obs["array_row"].values.astype(int)
cols_v = rna_v.obs["array_col"].values.astype(int)

coord_to_idx_v = {
    (int(r), int(c)): i
    for i, (r, c) in enumerate(
        zip(rows_v, cols_v)
    )
}

src_v = []
dst_v = []
for i, (r, c) in enumerate(zip(rows_v, cols_v)):
    for dr, dc in directions:
        nb = coord_to_idx_v.get(
            (int(r)+dr, int(c)+dc)
        )
        if nb is not None:
            src_v.append(i)
            dst_v.append(nb)

edge_index_v = torch.tensor(
    [src_v, dst_v], dtype=torch.long
)
log(f"Valid graph: {n_bins_v:,} nodes, "
    f"{edge_index_v.shape[1]:,} edges")

# Load best model
log("Loading best model...")
model_gat.load_state_dict(
    torch.load(
        os.path.join(DATA_DIR,
                     "gat_best_model.pt"),
        map_location=DEVICE
    )
)
model_gat.eval()
log("✓ Best model loaded")

# Run predictions
log("Running predictions...")
x_v          = torch.tensor(
    X_valid_scaled, dtype=torch.float32
).to(DEVICE)
edge_index_v = edge_index_v.to(DEVICE)

with torch.no_grad():
    preds = model_gat(
        x_v, edge_index_v
    ).cpu().numpy()

log(f"Predictions shape: {preds.shape}")

# Convert from arcsinh to original scale
preds_raw = np.sinh(preds) * PRO_COFACTOR
preds_raw = np.clip(preds_raw, 0, None)
log(f"Prediction range: "
    f"[{preds_raw.min():.3f}, "
    f"{preds_raw.max():.3f}]")

# Build submission dataframe
log("Building submission CSV...")
barcodes = rna_v.obs_names.tolist()

pxl_rows = []
pxl_cols = []
for bc in barcodes:
    if bc in valid_df.index:
        pxl_rows.append(
            int(valid_df.loc[
                bc, "pxl_row_in_fullres"
            ])
        )
        pxl_cols.append(
            int(valid_df.loc[
                bc, "pxl_col_in_fullres"
            ])
        )
    else:
        pxl_rows.append(0)
        pxl_cols.append(0)

submission = pd.DataFrame()
submission["barcode"]            = barcodes
submission["pxl_row_in_fullres"] = pxl_rows
submission["pxl_col_in_fullres"] = pxl_cols

for j, protein in enumerate(PROTEIN_COLS):
    if protein in PROTEIN_NAMES:
        idx = PROTEIN_NAMES.index(protein)
        submission[protein] = preds_raw[:, idx]
    else:
        submission[protein] = 0.0
        log(f"WARNING: {protein} not in training")

# Save submission
submission.to_csv(OUTPUT_CSV, index=False)

log(f"\n✓ Submission saved: {OUTPUT_CSV}")
log(f"  Rows: {len(submission)}")
log(f"  Cols: {len(submission.columns)}")
log(f"\nFirst 3 rows:")
print(submission.head(3).to_string())
log("\nDone! Ready to upload to Codabench")