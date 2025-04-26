import anndata as ad
import pandas as pd
import numpy as np
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

    adata = ad.read_h5ad(filepath, backed='r')
    cell_data = adata[adata.obs["cell_type"] == cell_type]

    df = pd.DataFrame.sparse.from_spmatrix(cell_data.X, index=cell_data.obs_names, columns=cell_data.var_names)

    df = df.div(df.sum(axis=1), axis=0) * 10000
    df = df.map(np.log1p)

    age_series = cell_data.obs["development_stage"].str.split("-", expand=True)[0].astype(int)
    df["age"] = age_series
    df["donor_id"] = cell_data.obs["donor_id"].values

    return df
