"""
STP Open Challenge — Elastic Net Baseline
==========================================
Purpose:
    Establish a non-spatial regression baseline to compare against GAT.
    Elastic Net uses the SAME 3000 HVG input features as the GAT model
    but has NO spatial context — each bin is predicted independently
    from its own RNA expression alone.

Why this comparison matters:
    GAT result - ElasticNet result = the value of spatial context
    If GAT >> ElasticNet: neighbourhood aggregation is essential
    If GAT ≈ ElasticNet: local RNA signal is sufficient (unlikely given r=0.031)

Design choices:
    • Same HVGs as GAT (3000 genes, seurat flavour) — fair comparison
    • MultiOutputRegressor wraps ElasticNet to predict all 44 proteins
    • Cross-validated alpha + l1_ratio via ElasticNetCV
    • Per-marker Pearson r reported and compared to Step 4 concordance baseline
    • Saves predictions in same format as GAT for direct comparison

CPU only — no GPU needed. RAM ~4-6 GB.

Install:
    pip install scikit-learn anndata scanpy scipy numpy pandas tqdm joblib
"""

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────
import os, time, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
from scipy.stats import pearsonr

from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.multioutput  import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
CFG = {
    # Files — same as GAT script
    "train_rna"   : "train_rna.h5ad",
    "train_pro"   : "train_pro.h5ad",
    "valid_rna"   : "valid_rna.h5ad",
    "valid_gt"    : "valid_uniform_range.csv",
    "out_dir"     : "elasticnet_run",

    # Preprocessing — MUST match GAT exactly for fair comparison
    "n_hvg"       : 3000,
    "pro_cofactor": 150.0,

    # ElasticNet hyperparameters
    "use_cv"      : True,
    "alpha"       : 0.01,
    "l1_ratio"    : 0.5,
    "cv_alphas"   : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    "cv_folds"    : 5,
    "max_iter"    : 10000,
    "n_jobs"      : 2,         # sequential — avoids OOM on machines with limited RAM
                                # change to 2 if you have >16 GB free RAM
    "tol"         : 1e-4,
}

os.makedirs(CFG["out_dir"], exist_ok=True)

print("=" * 65)
print("STP Challenge — Elastic Net Baseline")
print("=" * 65)
print(f"  Mode    : {'ElasticNetCV (auto alpha)' if CFG['use_cv'] else 'ElasticNet (fixed alpha)'}")
print(f"  HVGs    : {CFG['n_hvg']}")
print(f"  l1_ratio: {CFG['l1_ratio']}")
print(f"  n_jobs  : {CFG['n_jobs']} (all cores)")


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD & NORMALISE DATA
# ─────────────────────────────────────────────────────────────
print("\n[1/5] Loading and normalising data ...")
t0 = time.time()

# Load RNA
rna = ad.read_h5ad(CFG["train_rna"])
if "in_tissue" in rna.obs.columns:
    rna = rna[rna.obs["in_tissue"].astype(bool)].copy()
print(f"  Train RNA  : {rna.n_obs:,} bins × {rna.n_vars:,} genes")

# Load protein
pro = ad.read_h5ad(CFG["train_pro"])
PROTEIN_NAMES = list(pro.var_names)
N_PROTEINS    = len(PROTEIN_NAMES)
print(f"  Train PRO  : {pro.n_obs:,} bins × {N_PROTEINS} proteins")

# Align
shared = rna.obs_names.intersection(pro.obs_names)
rna    = rna[shared].copy()
pro    = pro[shared].copy()
print(f"  Aligned    : {len(shared):,} bins")

# ── Detect train/val split ────────────────────────────────
def make_fallback_split(obs_names, frac=0.05, min_val=1000):
    """Carve out the last `frac` of bins as validation."""
    n      = len(obs_names)
    n_val  = max(min_val, int(n * frac))
    flags  = np.zeros(n, dtype=bool)
    flags[n - n_val:] = True
    return (
        pd.Series(~flags, index=obs_names),
        pd.Series( flags, index=obs_names),
    )

TRAIN_LABELS = {"train", "1", "1.0", "true", "trn"}
VAL_LABELS   = {"val", "valid", "validation", "test",
                "0", "0.0", "false", "test_val", "2", "2.0"}

if "split" in rna.obs.columns:
    raw_vals  = rna.obs["split"].unique().tolist()
    split_str = rna.obs["split"].astype(str).str.strip().str.lower()
    val_counts = split_str.value_counts()

    print(f"  Split column unique values : {raw_vals}")
    print(f"  Split value counts:\n{val_counts.to_string()}")

    train_flag = split_str.isin(TRAIN_LABELS)
    val_flag   = split_str.isin(VAL_LABELS)

    # Fallback 1: unrecognised labels → majority=train, minority=val
    if train_flag.sum() == 0 or val_flag.sum() == 0:
        print("  ⚠ Split labels unrecognised — trying majority/minority split")
        majority   = val_counts.index[0]
        minority   = val_counts.index[-1] if len(val_counts) > 1 else None
        if minority is not None and minority != majority:
            train_flag = split_str == majority
            val_flag   = split_str == minority
        else:
            print("  ⚠ Only one split value found — carving 5% from tail as val")
            train_flag, val_flag = make_fallback_split(rna.obs_names)

    # Fallback 2: val still empty after majority/minority
    if val_flag.sum() == 0:
        print("  ⚠ Val set still empty — carving 5% from tail as val")
        train_flag, val_flag = make_fallback_split(rna.obs_names)

else:
    print("  No 'split' column found — carving 5% from tail as val")
    train_flag, val_flag = make_fallback_split(rna.obs_names)

print(f"  Train bins : {train_flag.sum():,}")
print(f"  Val   bins : {val_flag.sum():,}")

assert val_flag.sum() > 0,   "Val set is empty after all fallbacks — check your data!"
assert train_flag.sum() > 0, "Train set is empty — check your data!"

# ── Normalise RNA: library-size → log1p → HVG ─────────────
sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)
sc.pp.highly_variable_genes(
    rna, n_top_genes=CFG["n_hvg"],
    flavor="seurat", subset=False
)
rna_hvg   = rna[:, rna.var["highly_variable"]].copy()
HVG_NAMES = list(rna_hvg.var_names)
print(f"  HVG genes  : {len(HVG_NAMES):,}")

# Save HVG names for comparison with GAT
pd.Series(HVG_NAMES).to_csv(
    os.path.join(CFG["out_dir"], "hvg_names.csv"), index=False
)

# ── Convert RNA to dense numpy ─────────────────────────────
print("  Converting RNA to dense ...")
X_all = rna_hvg.X.toarray().astype(np.float32) \
        if sp.issparse(rna_hvg.X) \
        else np.array(rna_hvg.X, dtype=np.float32)
print(f"  RNA dense: {X_all.shape}  ({X_all.nbytes/1024**3:.2f} GB)")

# ── Normalise protein: arcsinh ─────────────────────────────
pro_dense = pro.X.toarray() if sp.issparse(pro.X) \
            else np.array(pro.X, dtype=np.float32)
Y_all = np.arcsinh(pro_dense / CFG["pro_cofactor"]).astype(np.float32)
del pro_dense
print(f"  Protein arcsinh: [{Y_all.min():.2f}, {Y_all.max():.2f}]")

# ── Split arrays ───────────────────────────────────────────
train_idx = np.where(train_flag.values)[0]
val_idx   = np.where(val_flag.values)[0]

X_train = X_all[train_idx]    # (N_train, 3000)
Y_train = Y_all[train_idx]    # (N_train, 44)
X_val   = X_all[val_idx]      # (N_val,   3000)
Y_val   = Y_all[val_idx]      # (N_val,   44)

print(f"\n  X_train : {X_train.shape}  ({X_train.nbytes/1024**3:.2f} GB)")
print(f"  Y_train : {Y_train.shape}")
print(f"  X_val   : {X_val.shape}")
print(f"  Y_val   : {Y_val.shape}")
print(f"  Loaded in {time.time()-t0:.1f}s")

# Free full arrays — only need train/val splits from here on
del X_all, Y_all
import gc; gc.collect()


# ─────────────────────────────────────────────────────────────
# STEP 2 — SCALE FEATURES
# ─────────────────────────────────────────────────────────────
print("\n[2/5] Scaling features ...")

# StandardScaler: zero mean, unit variance per gene
# Important for ElasticNet — regularisation is scale-sensitive
scaler         = StandardScaler(with_mean=True, with_std=True, copy=False)
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

print(f"  Feature mean range  : [{X_train_scaled.mean(0).min():.4f}, "
      f"{X_train_scaled.mean(0).max():.4f}]")
print(f"  Feature std  range  : [{X_train_scaled.std(0).min():.4f}, "
      f"{X_train_scaled.std(0).max():.4f}]")


# ─────────────────────────────────────────────────────────────
# STEP 3 — FIT ELASTIC NET (one model per protein)
# ─────────────────────────────────────────────────────────────
print(f"\n[3/5] Fitting Elastic Net "
      f"({'CV' if CFG['use_cv'] else 'fixed'} alpha) ...")
print(f"  Fitting {N_PROTEINS} protein models "
      f"({'parallel' if CFG['n_jobs'] != 1 else 'sequential'}) ...")

t_fit = time.time()

def fit_one_protein(i, protein_name, y_train, y_val, x_tr, x_va, cfg):
    """
    Fit one ElasticNet model for a single protein.
    Returns dict with protein name, model, train_r, val_r, best_alpha.
    """
    if cfg["use_cv"]:
        model = ElasticNetCV(
            alphas      = cfg["cv_alphas"],
            l1_ratio    = cfg["l1_ratio"],
            cv          = cfg["cv_folds"],
            max_iter    = cfg["max_iter"],
            tol         = cfg["tol"],
            n_jobs      = 1,
            verbose     = False,
        )
    else:
        model = ElasticNet(
            alpha    = cfg["alpha"],
            l1_ratio = cfg["l1_ratio"],
            max_iter = cfg["max_iter"],
            tol      = cfg["tol"],
        )

    model.fit(x_tr, y_train)

    pred_train = model.predict(x_tr)
    pred_val   = model.predict(x_va)

    tr_r = pearsonr(pred_train, y_train)[0] \
           if pred_train.std() > 1e-9 and y_train.std() > 1e-9 else 0.0
    va_r = pearsonr(pred_val,   y_val)[0]   \
           if pred_val.std()   > 1e-9 and y_val.std()   > 1e-9 else 0.0

    best_alpha = model.alpha_ if cfg["use_cv"] else cfg["alpha"]
    n_nonzero  = np.sum(model.coef_ != 0)

    return {
        "protein"   : protein_name,
        "model"     : model,
        "train_r"   : float(tr_r),
        "val_r"     : float(va_r),
        "best_alpha": float(best_alpha),
        "n_nonzero" : int(n_nonzero),
        "idx"       : i,
    }

results_list = Parallel(n_jobs=CFG["n_jobs"], verbose=10)(
    delayed(fit_one_protein)(
        i, protein_name,
        Y_train[:, i], Y_val[:, i],
        X_train_scaled, X_val_scaled,
        CFG
    )
    for i, protein_name in enumerate(PROTEIN_NAMES)
)

results_list = sorted(results_list, key=lambda x: x["idx"])
fit_time = time.time() - t_fit
print(f"\n  Fitting complete in {fit_time:.1f}s "
      f"({fit_time/N_PROTEINS:.1f}s per protein)")

# Free training matrix — no longer needed
del X_train, X_train_scaled
gc.collect()


# ─────────────────────────────────────────────────────────────
# STEP 4 — EVALUATE AND REPORT
# ─────────────────────────────────────────────────────────────
print(f"\n[4/5] Evaluation results ...")

records = []
for res in results_list:
    records.append({
        "protein"   : res["protein"],
        "val_r"     : res["val_r"],
        "train_r"   : res["train_r"],
        "best_alpha": res["best_alpha"],
        "n_nonzero" : res["n_nonzero"],
    })

results_df = pd.DataFrame(records).sort_values("val_r", ascending=False)

CONCORDANCE_BASELINE = {
    "CD74":0.2100,"SMA":0.1151,"MAP2":0.1121,"OLIG2":0.1055,
    "CD31":0.1033,"CD68":0.0867,"Ki67":0.0696,"CD14":0.0690,
    "CD44":0.0671,"Vimentin":0.0614,"MSH6":0.0546,"CD4":0.0537,
    "IDH1":0.0446,"SYNA":0.0401,"TOX":0.0399,"CD3e":0.0310,
    "PSD95":0.0291,"MGMT":0.0247,"CD45":0.0238,"CD38":0.0234,
    "SIRP":0.0223,"Podoplanin":0.0187,"CD16":0.0184,"synd":0.0121,
    "CD47":0.0114,"CD20":0.0051,"CD8":0.0046,"MPO":0.0043,
    "CXCR5":0.0035,"CD21":0.0013,"Granzyme B":-0.0017,
    "CD23":-0.0024,"PD-L1":-0.0027,"ICOS":-0.0041,"GFAP":-0.0065,
    "CXCL13":-0.0067,"PD-1":-0.0096,"FOXP3":-0.0121,
    "Granzyme K":-0.0165,"C-KIT":-0.0184,"FIBR":-0.0302,
    "PDGFR":-0.0331,"HLA-A":np.nan,"HLA-DR":np.nan,
}

results_df["concordance_r"]    = results_df["protein"].map(CONCORDANCE_BASELINE)
results_df["vs_concordance"]   = results_df["val_r"] - results_df["concordance_r"]
results_df["concordance_class"] = results_df["val_r"].apply(
    lambda r: "High" if r >= 0.5 else ("Moderate" if r >= 0.3 else "Low")
)

results_df.to_csv(
    os.path.join(CFG["out_dir"], "elasticnet_results.csv"), index=False
)

mean_val_r  = results_df["val_r"].mean()
mean_base_r = results_df["concordance_r"].mean()

print(f"\n  ══════ Elastic Net Results ══════")
print(f"  Mean val Pearson r     : {mean_val_r:.4f}")
print(f"  Concordance baseline r : {mean_base_r:.4f}")
print(f"  ElasticNet improvement : {mean_val_r - mean_base_r:+.4f}")
print()
print(f"  {'Protein':<22} {'EN val_r':>9}  {'Train_r':>8}  {'Baseline':>9}  "
      f"{'Δ baseline':>10}  {'α':>7}  {'Nonzero':>8}")
print(f"  {'─'*90}")

for _, row in results_df.iterrows():
    base   = row["concordance_r"]
    diff   = row["val_r"] - base if not np.isnan(base) else np.nan
    bar    = "█" * max(0, int(row["val_r"] * 25))
    diff_s = f"{diff:+.4f}" if not np.isnan(diff) else "   n/a"
    base_s = f"{base:.4f}"  if not np.isnan(base) else "   n/a"
    print(f"  {row['protein']:<22} {row['val_r']:>9.4f}  "
          f"{row['train_r']:>8.4f}  {base_s:>9}  "
          f"{diff_s:>10}  {row['best_alpha']:>7.4f}  "
          f"{row['n_nonzero']:>8}  {bar}")

print(f"\n  Concordance class distribution:")
print(f"    High     (r ≥ 0.5) : {(results_df['val_r'] >= 0.5).sum()}")
print(f"    Moderate (r ≥ 0.3) : {(results_df['val_r'] >= 0.3).sum()}")
print(f"    Low      (r < 0.3) : {(results_df['val_r'] < 0.3).sum()}")

overfit_df = results_df[results_df["train_r"] - results_df["val_r"] > 0.2]
if len(overfit_df) > 0:
    print(f"\n  ⚠ Potentially overfit markers (train_r - val_r > 0.2):")
    for _, row in overfit_df.iterrows():
        print(f"    {row['protein']:<22}  "
              f"train={row['train_r']:.4f}  val={row['val_r']:.4f}  "
              f"gap={row['train_r']-row['val_r']:.4f}")

mean_nonzero = results_df["n_nonzero"].mean()
print(f"\n  Mean non-zero coefficients: {mean_nonzero:.0f} / {len(HVG_NAMES)} genes")
print(f"  (ElasticNet selected {mean_nonzero/len(HVG_NAMES)*100:.1f}% of genes on average)")


# ─────────────────────────────────────────────────────────────
# STEP 5 — PREDICTIONS & SUBMISSION
# ─────────────────────────────────────────────────────────────
print(f"\n[5/5] Generating predictions ...")

val_preds = np.column_stack([
    res["model"].predict(X_val_scaled)
    for res in results_list
])
print(f"  Val predictions shape : {val_preds.shape}")

# ── Local score vs valid_uniform_range.csv ────────────────
try:
    import zipfile
    gt_path = CFG["valid_gt"]
    if zipfile.is_zipfile(gt_path):
        with zipfile.ZipFile(gt_path) as z:
            fname = [f for f in z.namelist() if f.endswith(".csv")][0]
            with z.open(fname) as f:
                val_gt = pd.read_csv(f, index_col="barcode")
    else:
        val_gt = pd.read_csv(gt_path, index_col="barcode")

    gt_cols      = [c for c in val_gt.columns
                    if c not in ["pxl_row_in_fullres","pxl_col_in_fullres"]]
    val_barcodes = rna.obs_names[val_flag.values].tolist()
    matched_bc   = [b for b in val_barcodes if b in val_gt.index]
    gt_aligned   = val_gt.loc[matched_bc][gt_cols]

    print(f"\n  Scoring vs valid_uniform_range.csv "
          f"({len(gt_aligned):,} matched barcodes):")
    gt_r_list = []
    for j, pname in enumerate(PROTEIN_NAMES):
        if pname not in gt_aligned.columns:
            continue
        pred_col = val_preds[
            [val_barcodes.index(b) for b in matched_bc], j
        ]
        true_col = gt_aligned[pname].values
        if true_col.std() > 1e-9 and pred_col.std() > 1e-9:
            r, _ = pearsonr(pred_col, true_col)
            gt_r_list.append(r)
    if gt_r_list:
        print(f"  Mean Pearson r (vs GT) : {np.mean(gt_r_list):.4f}")
except Exception as e:
    print(f"  Could not score vs GT: {e}")

# ── Test set predictions ──────────────────────────────────
print("\n  Generating test set predictions ...")
test_rna = ad.read_h5ad(CFG["valid_rna"])
sc.pp.normalize_total(test_rna, target_sum=1e4)
sc.pp.log1p(test_rna)

test_hvg_genes = [g for g in HVG_NAMES if g in test_rna.var_names]
missing_ct     = len(HVG_NAMES) - len(test_hvg_genes)
if missing_ct > 0:
    print(f"  ⚠ {missing_ct} HVGs missing in test — zero-padded")

test_hvg = test_rna[:, test_hvg_genes].copy()
test_mat = test_hvg.X.toarray().astype(np.float32) \
           if sp.issparse(test_hvg.X) \
           else np.array(test_hvg.X, dtype=np.float32)

if missing_ct > 0 or test_hvg_genes != HVG_NAMES:
    full_mat = np.zeros((test_rna.n_obs, len(HVG_NAMES)), dtype=np.float32)
    col_map  = {g: i for i, g in enumerate(HVG_NAMES)}
    for j, g in enumerate(test_hvg_genes):
        if g in col_map:
            full_mat[:, col_map[g]] = test_mat[:, j]
    test_mat = full_mat

test_mat_scaled = scaler.transform(test_mat)

test_preds = np.column_stack([
    res["model"].predict(test_mat_scaled)
    for res in results_list
])

sub_df = pd.DataFrame(
    test_preds,
    columns   = PROTEIN_NAMES,
    index     = test_rna.obs_names,
)
sub_df.index.name = "barcode"
sub_df["pxl_row_in_fullres"] = test_rna.obs["pxl_row_in_fullres"].values
sub_df["pxl_col_in_fullres"] = test_rna.obs["pxl_col_in_fullres"].values

sub_path = os.path.join(CFG["out_dir"], "submission.csv")
sub_df.to_csv(sub_path)
print(f"  Submission saved : {sub_path}  shape={sub_df.shape}")

# ── Top genes per protein (interpretability) ──────────────
print("\n  Extracting top gene coefficients per protein ...")
top_genes_records = []
for res in results_list:
    coefs     = res["model"].coef_
    top_idx   = np.argsort(np.abs(coefs))[::-1][:20]
    top_genes = [(HVG_NAMES[i], float(coefs[i])) for i in top_idx]
    top_genes_records.append({
        "protein"  : res["protein"],
        "top_genes": json.dumps(top_genes),
    })
pd.DataFrame(top_genes_records).to_csv(
    os.path.join(CFG["out_dir"], "top_genes_per_protein.csv"),
    index=False
)
print("  Top genes saved → top_genes_per_protein.csv")

# ── Summary ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("ELASTIC NET BASELINE COMPLETE")
print("=" * 65)
print(f"  Mean val Pearson r     : {mean_val_r:.4f}")
print(f"  Concordance baseline   : {mean_base_r:.4f}")
print(f"  Improvement            : {mean_val_r - mean_base_r:+.4f}")
print(f"  Fit time               : {fit_time:.0f}s")
print(f"  Mean non-zero genes    : {mean_nonzero:.0f} / {len(HVG_NAMES)}")
print(f"\n  Outputs in: {CFG['out_dir']}/")
print(f"    elasticnet_results.csv     per-marker Pearson r")
print(f"    submission.csv             test predictions")
print(f"    top_genes_per_protein.csv  top-20 predictive genes per protein")
print(f"    hvg_names.csv              HVG gene list")
print(f"\nNext: compare elasticnet_results.csv vs GAT final_results.csv")
print(f"  GAT mean r - EN mean r = value of spatial context")