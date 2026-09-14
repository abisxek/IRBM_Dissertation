import numpy as np
import pickle
import scipy.sparse as sp
import anndata as ad
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD
from scipy.stats import spearmanr
import torch
from torch.utils.data import DataLoader, TensorDataset

# ── Paths ───────────────────────────────────────────────────────────
DATA_DIR        = "/home/ubuntu/data"
EMBED_DIR       = f"{DATA_DIR}/outputs"

TRAIN_EMBED_NPY      = f"{EMBED_DIR}/binn_embeddings_train.npy"
TRAIN_EMBED_BARCODES = f"{EMBED_DIR}/binn_embed_barcodes_train.pkl"
VALID_EMBED_NPY      = f"{EMBED_DIR}/binn_embeddings_valid.npy"
VALID_EMBED_BARCODES = f"{EMBED_DIR}/binn_embed_barcodes_valid.pkl"
TRAIN_PRO_PATH       = f"{DATA_DIR}/train_pro.h5ad"
TRAIN_RNA_PATH       = f"{DATA_DIR}/train_rna.h5ad"
CKPT_PATH            = f"{DATA_DIR}/outputs/models/binn_checkpoint.pt"

PRO_COFACTOR = 150.0
N_FOLDS      = 5
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 256

# ── Load embeddings ─────────────────────────────────────────────────
print("Loading embeddings...")
X_train = np.load(TRAIN_EMBED_NPY)
X_valid = np.load(VALID_EMBED_NPY)

with open(TRAIN_EMBED_BARCODES, "rb") as f:
    train_barcodes = pickle.load(f)
with open(VALID_EMBED_BARCODES, "rb") as f:
    valid_barcodes = pickle.load(f)

print(f"Train embeddings : {X_train.shape}")
print(f"Valid embeddings : {X_valid.shape}")

# ── Load train protein labels ───────────────────────────────────────
print("\nLoading train protein labels...")
pro_train    = ad.read_h5ad(TRAIN_PRO_PATH)
shared_train = [b for b in train_barcodes if b in pro_train.obs_names]
assert len(shared_train) == len(train_barcodes), \
    f"Barcode mismatch: {len(train_barcodes)} embed vs {len(shared_train)} in protein file"

pro_train     = pro_train[shared_train]
Y_train_raw   = pro_train.X.toarray() if sp.issparse(pro_train.X) \
                else np.array(pro_train.X, dtype=np.float32)
Y_train       = np.arcsinh(Y_train_raw / PRO_COFACTOR).astype(np.float32)
protein_names = list(pro_train.var_names)
del Y_train_raw, pro_train
print(f"Y_train : {Y_train.shape}  |  proteins: {len(protein_names)}")

# ── Scale ───────────────────────────────────────────────────────────
print("\nScaling embeddings...")
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_valid_scaled = scaler.transform(X_valid).astype(np.float32)

# ── Cross-validated Ridge on training data ──────────────────────────
print(f"\nRunning {N_FOLDS}-fold cross-validation...")
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_r2        = []
fold_spearman  = []
protein_scores = np.zeros((N_FOLDS, len(protein_names)))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_tr, X_va = X_train_scaled[tr_idx], X_train_scaled[val_idx]
    Y_tr, Y_va = Y_train[tr_idx],        Y_train[val_idx]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, Y_tr)
    preds = ridge.predict(X_va)

    r2 = r2_score(Y_va, preds, multioutput="variance_weighted")
    sp_scores = [
        spearmanr(Y_va[:, j], preds[:, j]).correlation
        for j in range(Y_va.shape[1])
    ]
    mean_sp = np.nanmean(sp_scores)

    fold_r2.append(r2)
    fold_spearman.append(mean_sp)
    protein_scores[fold] = sp_scores
    print(f"  Fold {fold+1}: R²={r2:.4f}  mean Spearman={mean_sp:.4f}")

mean_r2          = np.mean(fold_r2)
std_r2           = np.std(fold_r2)
mean_spearman    = np.mean(fold_spearman)
std_spearman     = np.std(fold_spearman)
per_protein_mean = protein_scores.mean(axis=0)

# ── Distribution shift check ────────────────────────────────────────
train_mean = X_train_scaled.mean(axis=0)
valid_mean = X_valid_scaled.mean(axis=0)
train_std  = X_train_scaled.std(axis=0)
valid_std  = X_valid_scaled.std(axis=0)

mean_shift = np.abs(valid_mean - train_mean).mean()
std_shift  = np.abs(valid_std  - train_std).mean()

# ── Report ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("RIDGE CROSS-VALIDATION DIAGNOSTIC")
print("=" * 55)
print(f"\nEmbedding quality (no valid labels needed):")
print(f"  CV R² (variance-weighted) : {mean_r2:.4f} ± {std_r2:.4f}")
print(f"  CV mean Spearman          : {mean_spearman:.4f} ± {std_spearman:.4f}")

print(f"\nDistribution shift (train → valid embeddings):")
print(f"  Mean feature mean shift   : {mean_shift:.4f}")
print(f"  Mean feature std  shift   : {std_shift:.4f}")
if mean_shift > 0.3:
    print("  ⚠ Large mean shift — train/valid may come from different tissue regions")
    print("    This would cause GAT predictions to degrade silently on submission")
else:
    print("  ✓ Distribution looks stable across train/valid")

print(f"\nPer-protein Spearman (mean across {N_FOLDS} folds):")
for name, score in sorted(zip(protein_names, per_protein_mean),
                           key=lambda x: -x[1]):
    bar = "█" * int(max(0, score) * 20)
    print(f"  {name:<20s}  {score:.4f}  {bar}")

print("\n" + "=" * 55)
print("INTERPRETATION:")
print(f"  Your GAT leaderboard score : 0.58")
print(f"  Ridge CV R²               : {mean_r2:.4f}")
if mean_r2 < 0.45:
    print("\n  → Embeddings are weak even on training data.")
    print("    The BINN bottleneck is losing too much information.")
    print("    NEXT RUN: concatenate BINN embeddings + SVD(64) on full 18k genes.")
elif mean_r2 > 0.60:
    print("\n  → Embeddings carry strong signal — GAT is underperforming its features.")
    print("    The spatial reasoning layer is the bottleneck, not the features.")
    print("    NEXT RUN: increase GAT depth (layers 5→7) and heads (4→8).")
else:
    print("\n  → Moderate signal in embeddings. Both feature quality and")
    print("    GAT capacity are contributing to the gap.")
    print("    NEXT RUN: concatenate BINN embeddings + SVD(64) on full 18k genes.")
print("=" * 55)

# ── SVD(64) baseline on full 18k genes ─────────────────────────────
print("\nRunning SVD(64) baseline on full 18k genes for comparison...")
rna_train = ad.read_h5ad(TRAIN_RNA_PATH)
rna_train = rna_train[shared_train]

X_rna = rna_train.X
if not sp.issparse(X_rna):
    X_rna = sp.csr_matrix(X_rna)

svd = TruncatedSVD(n_components=64, random_state=42)
X_svd = svd.fit_transform(X_rna).astype(np.float32)
print(f"SVD explained variance: {svd.explained_variance_ratio_.sum():.4f}")

scaler_svd   = StandardScaler()
X_svd_scaled = scaler_svd.fit_transform(X_svd)

fold_r2_svd = []
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_svd_scaled)):
    ridge_svd = Ridge(alpha=1.0)
    ridge_svd.fit(X_svd_scaled[tr_idx], Y_train[tr_idx])
    preds_svd = ridge_svd.predict(X_svd_scaled[val_idx])
    fold_r2_svd.append(r2_score(Y_train[val_idx], preds_svd,
                                multioutput="variance_weighted"))

print(f"SVD(64) CV R²   : {np.mean(fold_r2_svd):.4f}")
print(f"BINN embed CV R²: {mean_r2:.4f}")

# Also test concatenation
# Re-fit scaler_svd on X_svd so it's clean (fit_transform above already did this,
# but we re-fit here to be explicit and avoid data-leakage confusion)
X_svd_for_concat = scaler_svd.fit_transform(X_svd)
X_concat = np.concatenate([X_train_scaled, X_svd_for_concat], axis=1)

fold_r2_concat = []
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_concat)):
    ridge_c = Ridge(alpha=1.0)
    ridge_c.fit(X_concat[tr_idx], Y_train[tr_idx])
    preds_c = ridge_c.predict(X_concat[val_idx])
    fold_r2_concat.append(r2_score(Y_train[val_idx], preds_c,
                                   multioutput="variance_weighted"))

print(f"BINN + SVD(64) CV R²: {np.mean(fold_r2_concat):.4f}")

# ── Inspect BINN checkpoint architecture ───────────────────────────
print("\nBINN pathway layer sizes:")
ckpt  = torch.load(CKPT_PATH, map_location="cpu")
state = ckpt["model_state"]
for key, val in state.items():
    if "pathway_layers" in key and "weight" in key:
        layer_num = key.split(".")[1]
        out_dim, in_dim = val.shape
        print(f"  Layer {layer_num}: {in_dim} → {out_dim} activations")

print(f"\nDense stream output: 128")
print(f"Current bottleneck : {ckpt.get('embed_dim', 'check manually')}")

# ── Load BINN model and build DataLoader ────────────────────────────
# Attempt to import your BINN model class. Adjust the import path to
# wherever BINN is defined in your project (e.g. "models.binn", "binn", etc.)
try:
    from models.binn import BINN  # ← update this import if your path differs
    MODEL_IMPORTABLE = True
except ImportError:
    print("\n⚠ Could not import BINN model class — skipping layer-0 extraction.")
    print("  Update the import on the line above to match your project layout.")
    MODEL_IMPORTABLE = False

if MODEL_IMPORTABLE:
    # ── Reconstruct model from checkpoint ──────────────────────────
    # Most BINN checkpoints store constructor kwargs alongside state_dict.
    # Fall back to inferring dims from state if they're not stored.
    model_kwargs = ckpt.get("model_kwargs", None)
    if model_kwargs is not None:
        binn_model = BINN(**model_kwargs)
    else:
        # Infer input dims from the first pathway layer weight
        first_pw_key = next(
            k for k in state if "pathway_layers.0.weight" in k
        )
        in_dim_mask  = state[first_pw_key].shape[1]

        first_dense_key = next(
            (k for k in state if "dense_stream" in k and "weight" in k), None
        )
        in_dim_unmasked = state[first_dense_key].shape[1] if first_dense_key else 0

        print(f"\nInferred BINN input dims — masked: {in_dim_mask}, "
              f"unmasked: {in_dim_unmasked}")
        print("If these look wrong, pass model_kwargs explicitly in the checkpoint "
              "or construct BINN manually here.")
        binn_model = BINN(in_dim_mask=in_dim_mask, in_dim_unmasked=in_dim_unmasked)

    binn_model.load_state_dict(state, strict=False)
    binn_model = binn_model.to(DEVICE)
    binn_model.eval()
    print("BINN model loaded successfully.")

    # ── Build train DataLoader from RNA data ───────────────────────
    # We need both the masked (pathway) input Xm and the unmasked
    # dense input Xu.  Adjust slicing to match how your training
    # DataLoader was originally built.
    rna_arr = rna_train.X
    if sp.issparse(rna_arr):
        rna_arr = rna_arr.toarray()
    rna_arr = rna_arr.astype(np.float32)

    # If your BINN splits features into masked / unmasked streams,
    # set mask_indices to the appropriate column indices; otherwise
    # the whole matrix is used for both (adjust as needed).
    mask_indices = ckpt.get("mask_indices", None)
    if mask_indices is not None:
        Xm_all = rna_arr[:, mask_indices]
        unmasked_indices = [i for i in range(rna_arr.shape[1])
                            if i not in set(mask_indices)]
        Xu_all = rna_arr[:, unmasked_indices] if unmasked_indices else rna_arr
    else:
        # Fallback: treat all features as masked input, empty unmasked
        Xm_all = rna_arr
        Xu_all = np.zeros((rna_arr.shape[0], 0), dtype=np.float32)

    # Dummy labels tensor so TensorDataset has three elements (Xm, Xu, y)
    dummy_y = torch.zeros(len(Xm_all), dtype=torch.float32)

    dataset = TensorDataset(
        torch.from_numpy(Xm_all),
        torch.from_numpy(Xu_all),
        dummy_y,
    )
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # ── Extract layer-0 activations ────────────────────────────────
    def extract_layer0(model, loader, device):
        model.eval()
        parts = []
        with torch.no_grad():
            for Xm, Xu, _ in loader:
                Xm = Xm.to(device)
                Xu = Xu.to(device)
                l0 = model.dropouts[0](
                    torch.tanh(
                        model.batch_norms[0](
                            model.pathway_layers[0](Xm)
                        )
                    )
                )
                # Only concatenate dense stream if it has features
                if Xu.shape[1] > 0:
                    dense = model.dense_stream(Xu)
                    z = torch.cat([l0, dense], dim=1)
                else:
                    z = l0
                parts.append(z.cpu().numpy())
        return np.concatenate(parts, axis=0).astype(np.float32)

    X_l0_train = extract_layer0(binn_model, train_loader, DEVICE)
    print(f"Layer 0 activations: {X_l0_train.shape}")

    # Apply SVD to compress to manageable size
    svd_bio  = TruncatedSVD(n_components=256, random_state=42)
    X_l0_svd = svd_bio.fit_transform(X_l0_train)
    print(f"After SVD: {X_l0_svd.shape}")
    print(f"Explained variance: {svd_bio.explained_variance_ratio_.sum():.4f}")

    scaler_l0   = StandardScaler()
    X_l0_scaled = scaler_l0.fit_transform(X_l0_svd)

    fold_r2_l0 = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_l0_scaled)):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_l0_scaled[tr_idx], Y_train[tr_idx])
        preds = ridge.predict(X_l0_scaled[val_idx])
        fold_r2_l0.append(
            r2_score(Y_train[val_idx], preds, multioutput="variance_weighted")
        )

    print(f"\nRidge CV R² comparison:")
    print(f"  Raw gene SVD(64)          : {np.mean(fold_r2_svd):.4f}")
    print(f"  BINN bottleneck           : {mean_r2:.4f}")
    print(f"  Layer 0 + SVD(256)        : {np.mean(fold_r2_l0):.4f}")