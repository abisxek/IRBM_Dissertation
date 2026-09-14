import pandas as pd
import numpy as np
import anndata as ad
from pathlib import Path
from kneed import KneeLocator # Uses the same library as your main script
import scipy.sparse as sp

# --- 1. Set your new tuning parameters here ---
output_dir = Path("./data/outputs")
h5ad_to_filter = ["data/train_rna.h5ad", "data/valid_rna.h5ad"]

# Choose your method: 'elbow' or 'relative'
selection_method = "elbow"  

# If using 'relative', what percentage of the max score is your minimum? (e.g., 0.05 for 5%)
relative_threshold = 0.05  

# --- 2. Load pre-calculated scores ---
shap_df = pd.read_csv(output_dir / "gene_shap_importance.csv")
grad_df = pd.read_csv(output_dir / "gene_dense_importance.csv")

def get_genes_by_threshold(df, score_col, method="elbow", rel_thresh=0.05):
    scores = df[score_col].values
    
    if method == "elbow":
        # Finds the natural cliff in the descending scores
        knee = KneeLocator(
            x=range(1, len(scores) + 1),
            y=scores,
            curve="convex",
            direction="decreasing",
            interp_method="polynomial",
        )
        # Fallback to all if no knee is found
        n = knee.knee or len(scores) 
        print(f"Elbow found at rank {n}")
        return df.head(n)["gene"].tolist()
        
    elif method == "relative":
        # Keep genes with a score >= (rel_thresh * max_score)
        max_score = scores[0]
        cutoff_value = max_score * rel_thresh
        selected_df = df[df[score_col] >= cutoff_value]
        print(f"Kept {len(selected_df)} genes with score >= {cutoff_value:.6f}")
        return selected_df["gene"].tolist()
    
    else:
        raise ValueError("Method must be 'elbow' or 'relative'")

# --- 3. Run the selection ---
print("--- Stream 1 (SHAP) ---")
selected_mapped = get_genes_by_threshold(shap_df, "shap_importance", method=selection_method, rel_thresh=relative_threshold)

print("\n--- Stream 2 (Gradient) ---")
selected_unmapped = get_genes_by_threshold(grad_df, "grad_importance", method=selection_method, rel_thresh=relative_threshold)

selected_all = selected_mapped + selected_unmapped
print(f"\nTotal selected genes: {len(selected_all)}")

# --- 4. Re-filter the h5ad file ---
for h5ad_path in h5ad_to_filter:
    h5ad_path = Path(h5ad_path)
    src = ad.read_h5ad(h5ad_path, backed="r")

    # Compute size factors from the FULL, unfiltered gene matrix —
    # this must happen BEFORE column subsetting, or the size factor
    # reflects only the ~2k selected genes instead of true library depth.
    full_X = src.X[:]  # materializes full matrix (same cost as to_memory() below)
    if sp.issparse(full_X):
        size_factors = np.asarray(full_X.sum(axis=1)).ravel().astype(np.float32)
    else:
        size_factors = np.asarray(full_X).sum(axis=1).astype(np.float32)
    del full_X

    present = [g for g in selected_all if g in src.var_names]
    out = src[:, present].to_memory()

    # Row order of `out` matches `src` (no reordering happens on column
    # subsetting), so this aligns directly with out.obs.
    out.obs["size_factor"] = size_factors

    out_name = h5ad_path.stem + f"_tuned_{selection_method}.h5ad"
    out.write_h5ad(output_dir / out_name)
    print(f"Saved to {out_name} (with size_factor from full {src.shape[1]:,}-gene matrix)")