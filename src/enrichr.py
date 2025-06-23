import mygene
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
import anndata as ad

input_dir = "../enrichr"
output_dir = "../enrichr"
output_enrichr_dir = "../enrichr"
os.makedirs(output_enrichr_dir, exist_ok=True)

def extract_gene_symbols(var_names):
    return [name.split("_")[0] for name in var_names]

def save_top_variable_genes_sparse(h5ad_path, top_k, output_txt_path):
    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(f"File not found: {h5ad_path}")

    adata = ad.read_h5ad(h5ad_path)
    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)

    print(f"Loaded {adata.shape[0]} cells and {adata.shape[1]} genes")

    row_sums = np.array(X.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1e-12
    row_scale = sp.diags(1 / row_sums)
    X_norm = row_scale @ X
    X_norm = X_norm.multiply(10000)
    X_log = X_norm.copy()
    X_log.data = np.log1p(X_log.data)

    mean = np.array(X_log.mean(axis=0)).ravel()
    mean_sq = np.array(X_log.multiply(X_log).mean(axis=0)).ravel()
    variances = mean_sq - mean ** 2

    top_gene_indices = np.argsort(variances)[-top_k:]
    top_genes = adata.var_names[top_gene_indices]
    gene_symbols = extract_gene_symbols(top_genes)

    with open(output_txt_path, "w") as f:
        for symbol in gene_symbols:
            f.write(f"{symbol}\n")
    print(f"Saved top {top_k} gene symbols to {output_txt_path}")

def save_top_shap_genes(shap_dir, top_n, output_txt_path):
    shap_sum = {}

    for file in os.listdir(shap_dir):
        if file.endswith("_shap_summary.csv"):
            df = pd.read_csv(os.path.join(shap_dir, file), usecols=["gene", "mean_abs_shap"])
            grouped = df.groupby("gene")["mean_abs_shap"].mean()
            for gene, value in grouped.items():
                shap_sum[gene] = shap_sum.get(gene, []) + [value]

    gene_avg_shap = {gene: np.mean(vals) for gene, vals in shap_sum.items()}
    top_genes = sorted(gene_avg_shap, key=gene_avg_shap.get, reverse=True)[:top_n]
    gene_symbols = extract_gene_symbols(top_genes)

    with open(output_txt_path, "w") as f:
        for symbol in gene_symbols:
            f.write(f"{symbol}\n")
    print(f"Saved top {top_n} SHAP gene symbols to {output_txt_path}")

save_top_variable_genes_sparse(
    "../data/AIDA.h5ad",
    5000,
    os.path.join(output_enrichr_dir, "top5000_all_cells_symbols.txt")
)

save_top_variable_genes_sparse(
    "../data/AIDA.h5ad",
    3000,
    os.path.join(output_enrichr_dir, "top3000_all_cells_symbols_nonlinear.txt")
)

save_top_shap_genes(
    "../predictions_enhanced/",
    30,
    os.path.join(output_enrichr_dir, "top30_shap_genes_symbols_linear.txt")
)

save_top_shap_genes(
    "../predictions_nonlinear_enhanced/",
    30,
    os.path.join(output_enrichr_dir, "top30_shap_genes_symbols_nonlinear.txt")
)

files_to_convert = {
    "top5000_all_cells_symbols.txt": "top5000_all_cells_entrez.txt",
    "top3000_all_cells_symbols_nonlinear.txt": "top3000_all_cells_entrez_nonlinear.txt",
    "top30_shap_genes_symbols_linear.txt": "top30_shap_genes_entrez.txt",
    "top30_shap_genes_symbols_nonlinear.txt": "top30_shap_genes_entrez_nonlinear.txt"
}

mg = mygene.MyGeneInfo()

def convert_ensembl_to_entrez(ensembl_list):
    response = mg.querymany(
        ensembl_list,
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        as_dataframe=True
    )
    clean = response[~response.index.duplicated(keep="first")]
    clean = clean.dropna(subset=["symbol"])
    return clean["symbol"].astype(str).tolist()

for infile, outfile in files_to_convert.items():
    in_path = os.path.join(input_dir, infile)
    out_path = os.path.join(output_dir, outfile)

    with open(in_path, "r") as f:
        genes = [line.strip() for line in f.readlines() if line.strip()]

    entrez_ids = convert_ensembl_to_entrez(genes)

    with open(out_path, "w") as f:
        for gene in entrez_ids:
            f.write(f"{gene}\n")

    print(f"Converted {infile} → {outfile} ({len(entrez_ids)} Entrez IDs)")
