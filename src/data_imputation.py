import os
import numpy as np
import pandas as pd
import anndata as ad

AIDA_PATH = "../data/AIDA.h5ad"
OUTPUT_DIR = "../data_for_imputation_nonlinear/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log_norm(X):
    row_sums = np.array(X.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1e-12
    X = X.multiply(1 / row_sums[:, None]) * 1e4
    return X.log1p()

adata = ad.read_h5ad(AIDA_PATH)
adata.X = log_norm(adata.X.tocsr()).tocsr()

cell_types = adata.obs["cell_type"].unique()
for cell_type in cell_types:
    print(f"Processing cell type: {cell_type}")

    mask = adata.obs["cell_type"] == cell_type
    if np.sum(mask) == 0:
        print(f"  Skipping {cell_type}: no cells found")
        continue

    subset = adata[mask].copy()
    subset.X = subset.X.tocsr()
    X_dense = subset.X.toarray()

    mean_vals = X_dense.mean(axis=0)
    std_vals = X_dense.std(axis=0, ddof=0) + 1e-8

    gene_names = subset.var_names

    df = pd.DataFrame([mean_vals], columns=gene_names)
    df_std = pd.DataFrame([std_vals], columns=gene_names)

    df_combined = pd.concat([df, df_std], keys=["mean", "std"])

    out_path = os.path.join(OUTPUT_DIR, f"Impute_avg_{cell_type}.csv")
    df_combined.to_csv(out_path)
    print(f"  Saved: {out_path}")
