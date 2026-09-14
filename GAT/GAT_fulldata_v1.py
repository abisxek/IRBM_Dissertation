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
from sklearn.preprocessing import StandardScaler
import warnings
import os
import time
import json
import pickle
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────
DATA_DIR      = "/home/ubuntu/data"
PRO_PATH      = os.path.join(DATA_DIR, "train_pro.h5ad")
PRO_COFACTOR  = 150.0
RANDOM_SEED   = 42

# BINN embedding files produced by pipeline.py Stage 6.
# These replace the old RNA h5ad + SVD pipeline entirely.
EMBED_DIR              = os.path.join(DATA_DIR, "outputs")
TRAIN_EMBED_NPY        = os.path.join(EMBED_DIR, "binn_embeddings_train.npy")
TRAIN_EMBED_BARCODES   = os.path.join(EMBED_DIR, "binn_embed_barcodes_train.pkl")
VALID_EMBED_NPY        = os.path.join(EMBED_DIR, "binn_embeddings_valid.npy")
VALID_EMBED_BARCODES   = os.path.join(EMBED_DIR, "binn_embed_barcodes_valid.pkl")

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


def load_gene_importance_weights(gene_list, shap_csv, grad_csv,
                                  w_min=0.2, w_max=1.0):
    """
    Build a per-gene weight vector, aligned to `gene_list` order, from the
    BINN SHAP (mapped-gene) and gradient (unmapped-gene) importance CSVs.

    Each stream is min-max normalised to [w_min, w_max] SEPARATELY before
    merging, since SHAP and gradient scores live on different scales and
    are not directly comparable. Genes present in `gene_list` but absent
    from both CSVs (shouldn't normally happen, since gene_list is exactly
    the BINN-selected panel) fall back to the midpoint weight.
    """
    shap_df = pd.read_csv(shap_csv)
    grad_df = pd.read_csv(grad_csv)

    def _minmax_weight_map(df, score_col):
        s = df[score_col].values.astype(np.float64)
        lo, hi = s.min(), s.max()
        if hi - lo < 1e-12:
            w = np.full_like(s, (w_min + w_max) / 2.0)
        else:
            w = w_min + (s - lo) / (hi - lo) * (w_max - w_min)
        return dict(zip(df["gene"], w))

    weight_map = {}
    weight_map.update(_minmax_weight_map(shap_df, "shap_importance"))
    weight_map.update(_minmax_weight_map(grad_df, "grad_importance"))

    default_w = (w_min + w_max) / 2.0
    n_missing = sum(1 for g in gene_list if g not in weight_map)
    if n_missing:
        print(f"  Warning: {n_missing}/{len(gene_list)} genes in panel "
              f"have no importance score — using default weight "
              f"{default_w:.3f}")

    weights = np.array(
        [weight_map.get(g, default_w) for g in gene_list],
        dtype=np.float32,
    )
    return weights


# ── Load data ──────────────────────────────────
# ── Load BINN embeddings (replaces RNA load + SVD) ─────
log("Loading BINN bottleneck embeddings (train)...")
assert os.path.exists(TRAIN_EMBED_NPY), \
    f"Missing: {TRAIN_EMBED_NPY} — run pipeline.py first to dump embeddings"
assert os.path.exists(TRAIN_EMBED_BARCODES), \
    f"Missing: {TRAIN_EMBED_BARCODES} — run pipeline.py first to dump embeddings"

score_matrix_final = np.load(TRAIN_EMBED_NPY)          # (n_bins, embed_dim)
with open(TRAIN_EMBED_BARCODES, "rb") as f:
    train_barcodes = pickle.load(f)

log(f"Embeddings: {score_matrix_final.shape[0]:,} bins × "
    f"{score_matrix_final.shape[1]} dims")

# ── Load protein labels, aligned to embedding barcode order ────
log("Loading Protein data...")
pro = ad.read_h5ad(PRO_PATH)
log(f"Protein: {pro.n_obs:,} bins × {pro.n_vars} proteins")

# Align protein to the exact barcode order from the embedding file.
# train_barcodes is the ground truth row order — never reorder embeddings.
shared_bins = [b for b in train_barcodes if b in pro.obs_names]
assert len(shared_bins) == len(train_barcodes), \
    (f"Barcode mismatch: {len(train_barcodes)} embed barcodes but only "
     f"{len(shared_bins)} found in protein file. "
     f"Re-run pipeline.py to regenerate embeddings.")

pro = pro[shared_bins]
pro_dense = pro.X.toarray() if sp.issparse(pro.X) \
            else np.array(pro.X, dtype=np.float32)
Y_all = np.arcsinh(pro_dense / PRO_COFACTOR).astype(np.float32)
del pro_dense
PROTEIN_NAMES = list(pro.var_names)
log(f"Proteins: {PROTEIN_NAMES}")

n_total   = score_matrix_final.shape[0]
train_idx = np.arange(n_total)
log(f"Training on ALL {n_total:,} bins")

# ── Scale embeddings ────────────────────────────
# StandardScaler is still needed: the pathway bottleneck and dense
# bottleneck outputs live on different numeric scales.
log("Scaling BINN embeddings...")
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(
    score_matrix_final
).astype(np.float32)
Y_train = Y_all

log(f"X_train: {X_train_scaled.shape}")
log(f"Y_train: {Y_train.shape}")

# Save scaler for validation inference
with open(os.path.join(DATA_DIR, "embed_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
log("Scaler saved → embed_scaler.pkl")


# ── Build tissue graph ─────────────────────────
# Spatial coordinates come from the RNA AnnData
log("Building tissue graph...")

# Load RNA just for spatial coordinates
TRAIN_RNA = os.path.join(DATA_DIR, "train_rna.h5ad")
rna_t = ad.read_h5ad(TRAIN_RNA, backed="r")

# Align to the exact barcode order used for embeddings and proteins
rna_t = rna_t[train_barcodes].to_memory()

rows_arr = rna_t.obs["array_row"].values.astype(int)
cols_arr = rna_t.obs["array_col"].values.astype(int)

# Define n_bins to ensure the graph and mask build correctly
n_bins = len(train_barcodes)

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
for i, (r, c) in enumerate(zip(rows_arr, cols_arr)):
    for dr, dc in directions:
        nb = coord_to_idx.get((int(r)+dr, int(c)+dc))
        if nb is not None:
            src_list.append(i)
            dst_list.append(nb)

edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
log(f"Graph: {n_bins:,} nodes, {edge_index.shape[1]:,} edges")


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
    "embed_dim"        : IN_CHANNELS,
    "embed_source"     : "binn_bottleneck",
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

OUTPUT_CSV = os.path.join(DATA_DIR, "submission.csv")

log("=" * 55)
log("Predicting on validation set")
log("=" * 55)

# ── Load pre-computed validation embeddings ─────────────────
# No RNA loading, no normalisation, no SVD — pipeline.py already
# ran encode() over every validation bin and saved the result.
log("Loading BINN embeddings (valid)...")
assert os.path.exists(VALID_EMBED_NPY), \
    f"Missing: {VALID_EMBED_NPY} — run pipeline.py first"
assert os.path.exists(VALID_EMBED_BARCODES), \
    f"Missing: {VALID_EMBED_BARCODES} — run pipeline.py first"

score_matrix_v = np.load(VALID_EMBED_NPY)
with open(VALID_EMBED_BARCODES, "rb") as f:
    valid_barcodes = pickle.load(f)

log(f"Valid embeddings: {score_matrix_v.shape[0]:,} bins × "
    f"{score_matrix_v.shape[1]} dims")

# Load valid h5ad only for spatial coordinates (array_row / array_col)
VALID_RNA = os.path.join(DATA_DIR, "valid_rna.h5ad")
log("Loading valid_rna.h5ad for spatial coordinates only...")
rna_v = ad.read_h5ad(VALID_RNA, backed="r")
# Align to embedding barcode order
rna_v = rna_v[valid_barcodes].to_memory()
assert list(rna_v.obs_names) == valid_barcodes, \
    "Valid barcode order mismatch — re-run pipeline.py"

# Scale using the training scaler
log("Scaling validation embeddings...")
X_valid_scaled = scaler.transform(score_matrix_v).astype(np.float32)

n_bins_v = score_matrix_v.shape[0]

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
# Pixel coords come directly from rna_v.obs (already aligned to embedding order)
log("Building submission CSV...")
barcodes = rna_v.obs_names.tolist()

submission = pd.DataFrame()
submission["barcode"]            = barcodes
submission["pxl_row_in_fullres"] = rna_v.obs["pxl_row_in_fullres"].values.astype(int)
submission["pxl_col_in_fullres"] = rna_v.obs["pxl_col_in_fullres"].values.astype(int)

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