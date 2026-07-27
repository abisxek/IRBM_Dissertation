import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

# ── Load filtered h5ad ────────────────────────────────────────────────────────
print('Loading filtered RNA...')
rna = ad.read_h5ad(OUTPUT_DIR / 'train_rna_filtered.h5ad')
pro = ad.read_h5ad(TRAIN_PRO_PATH)

print(f'  RNA: {rna.shape}')
print(f'  Protein: {pro.shape}')

# ── Align barcodes ────────────────────────────────────────────────────────────
common = rna.obs_names.intersection(pro.obs_names)
rna    = rna[common]
pro    = pro[common]
print(f'  Aligned bins: {len(common)}')

# ── Extract matrices ──────────────────────────────────────────────────────────
X = rna.X
if sp.issparse(X):
    X = X.toarray()
X = X.astype(np.float32)

Y = pro.X
if sp.issparse(Y):
    Y = Y.toarray()
Y = Y.astype(np.float32)

# Re-apply CLR normalisation to protein (same as training)
Y = Y + 1.0
geom  = np.exp(np.mean(np.log(Y), axis=1, keepdims=True))
Y_clr = np.log(Y / geom)
mean_ = Y_clr.mean(axis=0, keepdims=True)
std_  = Y_clr.std(axis=0,  keepdims=True) + 1e-8
Y     = (Y_clr - mean_) / std_

protein_names = pro.var_names.tolist()
print(f'  Genes (features): {X.shape[1]}')
print(f'  Proteins (targets): {Y.shape[1]}')

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
print(f'  Train: {X_train.shape[0]} bins | Test: {X_test.shape[0]} bins')

# ── Scale features ────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# ── Helper: per-protein Pearson correlation ───────────────────────────────────
def pearson_per_protein(Y_true, Y_pred, protein_names):
    """Compute per-protein Pearson r and return as DataFrame."""
    pearsons = {}
    for j, prot in enumerate(protein_names):
        r = np.corrcoef(Y_pred[:, j], Y_true[:, j])[0, 1]
        pearsons[prot] = r
    return pd.DataFrame({
        'protein': list(pearsons.keys()),
        'pearson_r': list(pearsons.values())
    }).sort_values('pearson_r', ascending=False).reset_index(drop=True)


# ── Model 1: Ridge Regression ─────────────────────────────────────────────────
print('\nTraining Ridge Regression...')
ridge      = Ridge(alpha=1.0)
ridge.fit(X_train, Y_train)
Y_pred_ridge = ridge.predict(X_test)

ridge_df         = pearson_per_protein(Y_test, Y_pred_ridge, protein_names)
ridge_mean       = ridge_df['pearson_r'].mean()
ridge_df['model'] = 'Ridge'
print(f'  Ridge mean Pearson r: {ridge_mean:.4f}')
print(ridge_df.to_string(index=False))


# ── Model 2: Random Forest ────────────────────────────────────────────────────
print('\nTraining Random Forest (this may take a few minutes)...')
rf = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators = 200,
        max_depth    = 10,
        n_jobs       = -1,
        random_state = 42,
    ),
    n_jobs = -1,
)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)

rf_df         = pearson_per_protein(Y_test, Y_pred_rf, protein_names)
rf_mean       = rf_df['pearson_r'].mean()
rf_df['model'] = 'Random Forest'
print(f'  Random Forest mean Pearson r: {rf_mean:.4f}')
print(rf_df.to_string(index=False))


# ── Comparison ────────────────────────────────────────────────────────────────
print('\n' + '='*50)
print('MODEL COMPARISON')
print('='*50)
print(f'  Ridge Regression  mean Pearson r: {ridge_mean:.4f}')
print(f'  Random Forest     mean Pearson r: {rf_mean:.4f}')
winner = 'Ridge' if ridge_mean > rf_mean else 'Random Forest'
print(f'  Winner: {winner}')


# ── Visualise: side-by-side per-protein comparison ────────────────────────────
combined_df = pd.merge(
    ridge_df[['protein', 'pearson_r']].rename(columns={'pearson_r': 'Ridge'}),
    rf_df[['protein', 'pearson_r']].rename(columns={'pearson_r': 'Random Forest'}),
    on='protein'
).sort_values('Ridge', ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: Ridge per-protein
colors_ridge = ['#2196F3' if r > 0.3 else '#FF9800' if r > 0.1 else '#F44336'
                for r in combined_df['Ridge']]
axes[0].barh(combined_df['protein'], combined_df['Ridge'], color=colors_ridge)
axes[0].axvline(ridge_mean, color='black', ls='--', label=f'Mean={ridge_mean:.3f}')
axes[0].set_xlabel('Pearson r')
axes[0].set_title('Ridge Regression — per protein')
axes[0].legend(); axes[0].grid(alpha=0.3)

# Plot 2: Random Forest per-protein
colors_rf = ['#2196F3' if r > 0.3 else '#FF9800' if r > 0.1 else '#F44336'
             for r in combined_df['Random Forest']]
axes[1].barh(combined_df['protein'], combined_df['Random Forest'], color=colors_rf)
axes[1].axvline(rf_mean, color='black', ls='--', label=f'Mean={rf_mean:.3f}')
axes[1].set_xlabel('Pearson r')
axes[1].set_title('Random Forest — per protein')
axes[1].legend(); axes[1].grid(alpha=0.3)

# Plot 3: Head-to-head scatter
axes[2].scatter(combined_df['Ridge'], combined_df['Random Forest'],
                c=range(len(combined_df)), cmap='viridis', s=80, zorder=3)
for _, row in combined_df.iterrows():
    axes[2].annotate(row['protein'],
                     (row['Ridge'], row['Random Forest']),
                     fontsize=6, alpha=0.7)
lims = [min(combined_df[['Ridge', 'Random Forest']].min()) - 0.05,
        max(combined_df[['Ridge', 'Random Forest']].max()) + 0.05]
axes[2].plot(lims, lims, 'k--', alpha=0.4, label='Equal performance')
axes[2].set_xlabel('Ridge Pearson r')
axes[2].set_ylabel('Random Forest Pearson r')
axes[2].set_title('Head-to-head: Ridge vs Random Forest')
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Save results ──────────────────────────────────────────────────────────────
combined_df.to_csv(OUTPUT_DIR / 'model_comparison.csv', index=False)
print(f'\nResults saved to {OUTPUT_DIR / "model_comparison.csv"}')