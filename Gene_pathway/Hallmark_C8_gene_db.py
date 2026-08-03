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
from torch_geometric.nn import GATConv
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
import warnings
import os
import time
import json
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────
DATA_DIR      = "/opt/dlami/nvme/data"
RNA_PATH      = os.path.join(DATA_DIR, "train_rna.h5ad")
PRO_PATH      = os.path.join(DATA_DIR, "train_pro.h5ad")
GMT_H         = os.path.join(DATA_DIR, "hallmark_pathways.gmt")
GMT_C8        = os.path.join(DATA_DIR, "c8_pathways.gmt")
PRO_COFACTOR  = 150.0
CHUNK_SIZE    = 5000
RANDOM_SEED   = 42
N_BINS_SUBSET = None   # None = full dataset

# GAT config
HIDDEN_CHANNELS = 128
HEADS           = 4
DROPOUT         = 0.2
BATCH_SIZE      = 512
N_EPOCHS        = 100
PATIENCE        = 15
LEARNING_RATE   = 0.001
NUM_NEIGHBORS   = [10, 5]
NUM_WORKERS     = 4

# Device
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


# ── Train/val split ────────────────────────────
np.random.seed(RANDOM_SEED)
n_total = rna.n_obs
n_val   = int(n_total * 0.05)
val_idx   = np.random.choice(
    n_total, size=n_val, replace=False
)
train_idx = np.setdiff1d(
    np.arange(n_total), val_idx
)
log(f"Train: {len(train_idx):,} | "
    f"Val: {len(val_idx):,}")


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
np.save(os.path.join(DATA_DIR,
                     "val_idx.npy"),
        val_idx)
log("Scores saved")


# ── Scale features ─────────────────────────────
X_train = score_matrix_final[train_idx].astype(
    np.float32
)
X_val   = score_matrix_final[val_idx].astype(
    np.float32
)
Y_train = Y_all[train_idx]
Y_val   = Y_all[val_idx]

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

log(f"X_train: {X_train_scaled.shape}")
log(f"X_val  : {X_val_scaled.shape}")
log(f"Y_train: {Y_train.shape}")
log(f"Y_val  : {Y_val.shape}")


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


# ── PyG data object ────────────────────────────
x = torch.tensor(
    score_matrix_final, dtype=torch.float32
)
y = torch.tensor(Y_all, dtype=torch.float32)

train_mask = torch.zeros(n_bins, dtype=torch.bool)
val_mask   = torch.zeros(n_bins, dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx]     = True

data = Data(
    x          = x,
    edge_index = edge_index,
    y          = y,
    train_mask = train_mask,
    val_mask   = val_mask,
)
log(f"Data: {data}")


# ── GAT model ──────────────────────────────────
class GATProteinPredictor(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        heads   = 4,
        dropout = 0.2,
    ):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads   = heads,
            dropout = dropout,
            concat  = True,
        )
        self.conv2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads   = 1,
            dropout = dropout,
            concat  = False,
        )
        self.bn1 = nn.BatchNorm1d(
            hidden_channels * heads
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.output = nn.Linear(
            hidden_channels, out_channels
        )

    def forward(self, x, edge_index):
        x = F.dropout(
            x, p=self.dropout,
            training=self.training
        )
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)

        x = F.dropout(
            x, p=self.dropout,
            training=self.training
        )
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)

        return self.output(x)


IN_CHANNELS  = score_matrix_final.shape[1]
OUT_CHANNELS = len(PROTEIN_NAMES)

model_gat = GATProteinPredictor(
    in_channels     = IN_CHANNELS,
    hidden_channels = HIDDEN_CHANNELS,
    out_channels    = OUT_CHANNELS,
    heads           = HEADS,
    dropout         = DROPOUT,
).to(DEVICE)

n_params = sum(
    p.numel() for p in model_gat.parameters()
)
log(f"GAT: {n_params:,} parameters")
log(f"  Input : {IN_CHANNELS}")
log(f"  Hidden: {HIDDEN_CHANNELS} × {HEADS} heads")
log(f"  Output: {OUT_CHANNELS}")


# ── Data loaders ───────────────────────────────
train_loader = NeighborLoader(
    data,
    num_neighbors = NUM_NEIGHBORS,
    batch_size    = BATCH_SIZE,
    input_nodes   = data.train_mask,
    shuffle       = True,
    num_workers   = NUM_WORKERS,
)
val_loader = NeighborLoader(
    data,
    num_neighbors = NUM_NEIGHBORS,
    batch_size    = BATCH_SIZE,
    input_nodes   = data.val_mask,
    shuffle       = False,
    num_workers   = NUM_WORKERS,
)
log(f"Train batches: {len(train_loader)} | "
    f"Val batches: {len(val_loader)}")


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


def eval_epoch(model, loader,
               criterion, device):
    model.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0
    n_batches  = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred  = model(
                batch.x, batch.edge_index
            )
            seed  = batch.batch_size
            loss  = criterion(
                pred[:seed], batch.y[:seed]
            )
            total_loss  += loss.item()
            n_batches   += 1
            all_preds.append(
                pred[:seed].cpu().numpy()
            )
            all_labels.append(
                batch.y[:seed].cpu().numpy()
            )
    return (
        total_loss / n_batches,
        np.vstack(all_preds),
        np.vstack(all_labels),
    )


# ── Training loop ──────────────────────────────
log("=" * 55)
log("Starting GAT training")
log("=" * 55)

best_val_r     = -999
patience_count = 0
history        = []
t_start        = time.time()

for epoch in range(1, N_EPOCHS + 1):
    t_ep = time.time()

    train_loss = train_epoch(
        model_gat, train_loader,
        optimiser, criterion, DEVICE
    )

    val_loss, val_preds, val_labels = eval_epoch(
        model_gat, val_loader,
        criterion, DEVICE
    )

    rs = [
        pearsonr(val_preds[:, i],
                 val_labels[:, i])[0]
        for i in range(val_preds.shape[1])
        if val_preds[:, i].std() > 1e-9
        and val_labels[:, i].std() > 1e-9
    ]
    mean_r = float(np.mean(rs)) if rs else 0.0

    scheduler.step(val_loss)

    history.append({
        "epoch"      : epoch,
        "train_loss" : train_loss,
        "val_loss"   : val_loss,
        "val_pearson": mean_r,
        "time_s"     : time.time() - t_ep,
    })

    if mean_r > best_val_r:
        best_val_r     = mean_r
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
        f"val={val_loss:.4f}  "
        f"r={mean_r:.4f}  "
        f"best={best_val_r:.4f}  "
        f"[{ep_time:.0f}s]{flag}")

    if patience_count >= PATIENCE:
        log(f"Early stopping at epoch {epoch}")
        break

total_time = time.time() - t_start
log(f"Training done in {total_time/60:.1f} min")
log(f"Best val Pearson r: {best_val_r:.4f}")


# ── Final evaluation ───────────────────────────
log("Loading best model...")
model_gat.load_state_dict(
    torch.load(
        os.path.join(DATA_DIR,
                     "gat_best_model.pt"),
        map_location=DEVICE
    )
)

_, val_preds, val_labels = eval_epoch(
    model_gat, val_loader, criterion, DEVICE
)
_, train_preds, train_labels = eval_epoch(
    model_gat, train_loader, criterion, DEVICE
)

records = []
for i, protein in enumerate(PROTEIN_NAMES):

    if (train_preds[:, i].std() > 1e-9
            and train_labels[:, i].std() > 1e-9):
        tr_p = pearsonr(
            train_preds[:, i], train_labels[:, i]
        )[0]
        tr_s = spearmanr(
            train_preds[:, i], train_labels[:, i]
        )[0]
    else:
        tr_p = tr_s = 0.0

    if (val_preds[:, i].std() > 1e-9
            and val_labels[:, i].std() > 1e-9):
        vr_p = pearsonr(
            val_preds[:, i], val_labels[:, i]
        )[0]
        vr_s = spearmanr(
            val_preds[:, i], val_labels[:, i]
        )[0]
    else:
        vr_p = vr_s = 0.0

    records.append({
        "protein"       : protein,
        "val_pearson"   : float(vr_p),
        "val_spearman"  : float(vr_s),
        "train_pearson" : float(tr_p),
        "train_spearman": float(tr_s),
        "gap"           : float(tr_p - vr_p),
    })

results_df = pd.DataFrame(records).sort_values(
    "val_pearson", ascending=False
)
mean_gat = results_df["val_pearson"].mean()


# ── Print results ──────────────────────────────
log("=" * 55)
log("FINAL RESULTS — GAT")
log("=" * 55)
log(f"\n{'Protein':<22} {'Val P':>8}  "
    f"{'Val S':>8}  {'Train P':>8}  {'Gap':>7}")
log("-" * 58)

for _, row in results_df.iterrows():
    bar = "█" * max(
        0, int(row["val_pearson"] * 20)
    )
    log(f"{row['protein']:<22} "
        f"{row['val_pearson']:>8.4f}  "
        f"{row['val_spearman']:>8.4f}  "
        f"{row['train_pearson']:>8.4f}  "
        f"{row['gap']:>7.4f}  {bar}")

log("-" * 58)
log(f"Mean Val Pearson  : {mean_gat:.4f}")
log(f"Mean Val Spearman : "
    f"{results_df['val_spearman'].mean():.4f}")
log(f"\nHigh     (r >= 0.5): "
    f"{(results_df['val_pearson'] >= 0.5).sum()}")
log(f"Moderate (r >= 0.3): "
    f"{((results_df['val_pearson'] >= 0.3) & (results_df['val_pearson'] < 0.5)).sum()}")
log(f"Low      (r <  0.3): "
    f"{(results_df['val_pearson'] < 0.3).sum()}")


# ── Save results ───────────────────────────────
results_df.to_csv(
    os.path.join(DATA_DIR, "gat_results.csv"),
    index=False
)
pd.DataFrame(history).to_csv(
    os.path.join(DATA_DIR, "gat_history.csv"),
    index=False
)

config = {
    "in_channels"      : IN_CHANNELS,
    "hidden_channels"  : HIDDEN_CHANNELS,
    "out_channels"     : OUT_CHANNELS,
    "heads"            : HEADS,
    "dropout"          : DROPOUT,
    "batch_size"       : BATCH_SIZE,
    "n_epochs"         : N_EPOCHS,
    "patience"         : PATIENCE,
    "learning_rate"    : LEARNING_RATE,
    "num_neighbors"    : NUM_NEIGHBORS,
    "best_val_r"       : best_val_r,
    "mean_val_pearson" : mean_gat,
    "total_time_min"   : total_time / 60,
}

with open(
    os.path.join(DATA_DIR, "gat_config.json"),
    "w"
) as f:
    json.dump(config, f, indent=2)

log("Saved:")
log("  gat_results.csv")
log("  gat_history.csv")
log("  gat_best_model.pt")
log("  gat_config.json")
log("Done!")