import scanpy as sc
import os

DATA_DIR      = "/home/ubuntu/data"
TRAIN_RNA = os.path.join(DATA_DIR, "outputs/train_rna_tuned_elbow.h5ad")
VALID_RNA = os.path.join(DATA_DIR, "outputs/valid_rna_tuned_elbow.h5ad")

# 1. Load the two files
adata1 = sc.read_h5ad(TRAIN_RNA)
adata2 = sc.read_h5ad(VALID_RNA)

# 2. Check if the genes are identical AND in the exact same order
exact_match = adata1.var_names.equals(adata2.var_names)
print(f"Perfectly aligned (same genes, same order): {exact_match}")

# 3. Check if they contain the same set of genes (ignoring order)
same_genes = set(adata1.var_names) == set(adata2.var_names)
print(f"Contain the exact same gene set: {same_genes}")