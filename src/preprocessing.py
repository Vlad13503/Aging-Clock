import anndata as ad
import pandas as pd
import numpy as np
import scipy.sparse as sp
import os

def load_celltype_data(filepath, cell_type):
    """
    Load expression data for a specific cell type from a .h5ad file,
    normalize it, and extract donor and age metadata.

    Parameters:
        filepath (str): Path to the .h5ad file.
        cell_type (str): Name of the cell type to extract.

    Returns:
        pd.DataFrame: Normalized gene expression with 'age' and 'donor_id' columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    adata = ad.read_h5ad(filepath, backed="r")

    selected = adata.obs["cell_type"] == cell_type
    indices = np.where(selected)[0]

    if len(indices) == 0:
        raise ValueError(f"No cells found for type: {cell_type}")

    print(f"Found {len(indices)} cells of type '{cell_type}'")

    sub_X = adata.X[indices]
    sub_obs = adata.obs.iloc[indices]

    row_sums = np.array(sub_X.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1e-12

    normalizer = sp.diags(1 / row_sums)
    X_norm = normalizer @ sub_X
    X_norm = X_norm.multiply(10000)
    X_log = X_norm.copy()
    X_log.data = np.log1p(X_log.data)

    df = pd.DataFrame.sparse.from_spmatrix(
        X_log,
        index=sub_obs.index,
        columns=adata.var_names
    )

    df["age"] = sub_obs["development_stage"].str.split("-", expand=True)[0].astype(int)
    df["donor_id"] = sub_obs["donor_id"].values
    df["cell_id"] = sub_obs.index

    return df
