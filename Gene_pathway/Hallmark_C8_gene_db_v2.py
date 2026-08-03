# ── gat_train.py ──────────────────────────────
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
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
import warnings
import os
import time
import json
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────
DATA_DIR      = "/home/ubuntu/data"
RNA_PATH      = os.path.join(DATA_DIR, "train_rna.h5ad")
PRO_PATH      = os.path.join(DATA_DIR, "train_pro.h5ad")
GMT_H         = os.path.join(DATA_DIR, "hallmark_pathways.gmt")
GMT_C8        = os.path.join(DATA_DIR, "c8_pathways.gmt")
PRO_COFACTOR  = 150.0
CHUNK_SIZE    = 5000
RANDOM_SEED   = 42
N_BINS_SUBSET = None

# GAT config
HIDDEN_CHANNELS = 256
HEADS           = 4
N_LAYERS        = 4             
DROPOUT         = 0.2
BATCH_SIZE      = 512
N_EPOCHS        = 50
PATIENCE        = 20
LEARNING_RATE   = 0.001
NUM_NEIGHBORS   = [10, 5, 5, 5]  # must match N_LAYERS    
NUM_WORKERS     = 4

assert len(NUM_NEIGHBORS) == N_LAYERS, (
    f"NUM_NEIGHBORS length ({len(NUM_NEIGHBORS)}) "
    f"must match N_LAYERS ({N_LAYERS})"
)

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
def parse_gmt(filepath):
    pathways = {}
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            pathways[parts[0]] = parts[2:]
    return pathways


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

rna_gene_set  = set(rna.var_names)
rna_gene_list = list(rna.var_names)
rna_gene_idx  = {g: i for i, g
                 in enumerate(rna_gene_list)}

log("Normalising protein...")
pro_dense = pro.X.toarray() if sp.issparse(pro.X) \
            else np.array(pro.X, dtype=np.float32)
Y_all = np.arcsinh(
    pro_dense / PRO_COFACTOR
).astype(np.float32)
del pro_dense
PROTEIN_NAMES = list(pro.var_names)
log(f"Proteins: {PROTEIN_NAMES}")


# ── CHANGE 1: Use ALL bins for training ────────
n_total   = rna.n_obs
train_idx = np.arange(n_total)
log(f"Training on ALL {n_total:,} bins")


# ── Pathway scores ─────────────────────────────
log("Parsing GMT files...")
hallmark = parse_gmt(GMT_H)
c8       = parse_gmt(GMT_C8)

MIN_GENES    = 10
combined_raw = {**hallmark, **c8}
matched_sets = {}
for name, genes in combined_raw.items():
    matched = [g for g in genes
               if g in rna_gene_set]
    if len(matched) >= MIN_GENES:
        matched_sets[name] = matched

log(f"Matched gene sets: {len(matched_sets)}")

all_names = list(matched_sets.keys())
n_sets    = len(all_names)
n_bins    = rna.n_obs

col_idx = {
    name: [rna_gene_idx[g] for g in genes]
    for name, genes in matched_sets.items()
}

score_matrix = np.zeros(
    (n_bins, n_sets), dtype=np.float32
)

log(f"Computing {n_sets} pathway scores "
    f"for {n_bins:,} bins...")
t_start = time.time()

for chunk_start in range(0, n_bins, CHUNK_SIZE):
    chunk_end = min(
        chunk_start + CHUNK_SIZE, n_bins
    )
    chunk = rna.X[chunk_start:chunk_end]
    if sp.issparse(chunk):
        chunk = chunk.toarray()
    chunk = np.array(chunk, dtype=np.float32)

    for j, name in enumerate(all_names):
        cols = col_idx[name]
        score_matrix[chunk_start:chunk_end, j] = \
            chunk[:, cols].mean(axis=1)

    if chunk_start % 25000 == 0 or \
            chunk_end == n_bins:
        progress = chunk_end / n_bins * 100
        log(f"  {progress:.0f}% — "
            f"{chunk_end:,}/{n_bins:,}")

log(f"Scores done in "
    f"{(time.time()-t_start)/60:.1f} min")


# ── Filter scores ──────────────────────────────
log("Filtering pathway scores...")

set_stds      = score_matrix.std(axis=0)
var_threshold = np.percentile(set_stds, 25)
var_mask      = set_stds > var_threshold

MIN_R     = 0.05
corr_mask = np.zeros(n_sets, dtype=bool)

for j in range(n_sets):
    if not var_mask[j]:
        continue
    path_vals = score_matrix[:, j]
    if path_vals.std() < 1e-9:
        continue
    for i in range(Y_all.shape[1]):
        pro_vals = Y_all[:, i]
        if pro_vals.std() < 1e-9:
            continue
        r, _ = pearsonr(path_vals, pro_vals)
        if abs(r) >= MIN_R:
            corr_mask[j] = True
            break

final_mask         = var_mask & corr_mask
kept_idx           = np.where(final_mask)[0]
score_matrix_final = score_matrix[:, kept_idx]
names_final        = [all_names[i]
                      for i in kept_idx]

log(f"Features after filter: {len(kept_idx)}")
log(f"Score matrix: {score_matrix_final.shape}")

# Save scores
np.save(os.path.join(DATA_DIR,
                     "score_matrix_final.npy"),
        score_matrix_final)
np.save(os.path.join(DATA_DIR,
                     "names_final.npy"),
        np.array(names_final))
np.save(os.path.join(DATA_DIR,
                     "train_idx.npy"),
        train_idx)
log("Scores saved")


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


# ── GATv2 model (dynamic depth) ────────────────
class GATProteinPredictor(nn.Module):
    """
    Dynamic-depth GATv2-based regressor.

    n_layers == 1:
        single GATv2Conv, in_channels -> out_channels (1 head, no concat)
    n_layers >= 2:
        first layer : in_channels -> hidden * heads (concat)
        middle layer(s): hidden*heads -> hidden*heads (concat)
        last layer  : hidden*heads -> hidden (1 head, no concat)
        followed by a Linear(hidden, out_channels)
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
        assert n_layers >= 1, "n_layers must be >= 1"
        self.dropout  = dropout
        self.n_layers = n_layers

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        if n_layers == 1:
            self.convs.append(
                GATv2Conv(
                    in_channels,
                    out_channels,
                    heads   = 1,
                    dropout = dropout,
                    concat  = False,
                )
            )
            self.output = None
        else:
            # First layer: in_channels -> hidden * heads
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

            # Middle layers: hidden*heads -> hidden*heads
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

            # Last layer: hidden*heads -> hidden (single head, no concat)
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

            self.output = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.dropout(
                x, p=self.dropout,
                training=self.training
            )
            x = conv(x, edge_index)
            if i < len(self.bns):
                x = self.bns[i](x)
                x = F.elu(x)

        if self.output is not None:
            x = self.output(x)
        return x


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
log(f"GATv2: {n_params:,} parameters")
log(f"  Layers: {N_LAYERS}")
log(f"  Input : {IN_CHANNELS}")
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
log("Starting GATv2 training on FULL dataset")
log("=" * 55)

# CHANGE 2: Track train loss not val r
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
    "in_channels"     : IN_CHANNELS,
    "hidden_channels" : HIDDEN_CHANNELS,
    "n_layers"        : N_LAYERS,
    "out_channels"    : OUT_CHANNELS,
    "heads"           : HEADS,
    "dropout"         : DROPOUT,
    "batch_size"      : BATCH_SIZE,
    "n_epochs"        : N_EPOCHS,
    "patience"        : PATIENCE,
    "learning_rate"   : LEARNING_RATE,
    "num_neighbors"   : NUM_NEIGHBORS,
    "best_train_loss" : best_loss,
    "total_time_min"  : total_time / 60,
}

with open(
    os.path.join(DATA_DIR, "gat_config.json"),
    "w"
) as f:
    json.dump(config, f, indent=2)

log("Saved: gat_history.csv, gat_config.json")


# ══════════════════════════════════════════════
# CHANGE 2: PREDICT ON valid_rna.h5ad
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

valid_gene_idx = {
    g: i for i, g
    in enumerate(rna_v.var_names)
}

# Compute pathway scores for validation
n_bins_v  = rna_v.n_obs
n_sets_f  = len(names_final)
col_idx_v = {}

for name in names_final:
    genes   = matched_sets.get(name, [])
    matched = [g for g in genes
               if g in valid_gene_idx]
    col_idx_v[name] = [
        valid_gene_idx[g] for g in matched
    ] if matched else []

score_matrix_v = np.zeros(
    (n_bins_v, n_sets_f), dtype=np.float32
)

log(f"Computing pathway scores for "
    f"{n_bins_v:,} validation bins...")
t_start = time.time()

for chunk_start in range(
    0, n_bins_v, CHUNK_SIZE
):
    chunk_end = min(
        chunk_start + CHUNK_SIZE, n_bins_v
    )
    chunk = rna_v.X[chunk_start:chunk_end]
    if sp.issparse(chunk):
        chunk = chunk.toarray()
    chunk = np.array(chunk, dtype=np.float32)

    for j, name in enumerate(names_final):
        cols = col_idx_v[name]
        if len(cols) > 0:
            score_matrix_v[
                chunk_start:chunk_end, j
            ] = chunk[:, cols].mean(axis=1)

    if chunk_start % 10000 == 0 or \
            chunk_end == n_bins_v:
        progress = chunk_end / n_bins_v * 100
        log(f"  {progress:.0f}% — "
            f"{chunk_end:,}/{n_bins_v:,}")

log(f"Scores done in "
    f"{(time.time()-t_start):.0f}s")

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
