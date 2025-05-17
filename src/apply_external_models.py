import os
import pandas as pd
import numpy as np
import anndata as ad
from celltype_mappings import CELLTYPE_MAPPINGS

DATA_FOLDER = "../data/"
MODEL_FOLDER = "../models/"
IMPUTE_FOLDER = "../data_for_imputation/"
OUTPUT_FOLDER = "../predictions_external/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def log_norm(X):
    row_sums = np.array(X.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1e-12
    X = X.multiply(1 / row_sums[:, None]) * 10000
    return X.log1p()

def map_yoshida_age(dev_stage):
    if pd.isnull(dev_stage):
        return np.nan
    val = str(dev_stage).lower().strip()
    if "newborn" in val:
        return "newborn human stage"
    if "infant" in val:
        return "infant stage"
    if "child" in val and ("2-5" in val or "1-4" in val):
        return "2-5 year-old child stage"
    if "child" in val and ("6-12" in val or "5-14" in val or "juvenile" in val):
        return "6-12 year-old child stage"
    if "child" in val or "pediatric" in val:
        return "6-12 year-old child stage"
    if "adolescent" in val:
        return "adolescent stage"
    if "adult" in val:
        return "human adult stage"
    if "late adult" in val or "aged" in val or "elderly" in val:
        return "human aged stage"
    try:
        num = float(val.split("-")[0])
        if num < 65:
            return "human adult stage"
        else:
            return "human aged stage"
    except Exception:
        return val


def map_stephenson_age(stage):
    if pd.isnull(stage):
        return np.nan
    s = str(stage).lower().strip()

    if "decade stage" in s:
        return s

    import re
    m = re.match(r"(\d+)-year-old stage", s)
    if m:
        age = int(m.group(1))
        decade = (age // 10) + 1
        if decade > 10:
            decade = 10
        ordinals = {3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh",
                    8: "eighth", 9: "ninth", 10: "tenth"}
        decade_str = ordinals.get(decade, f"{decade}th")
        return f"{decade_str} decade stage"

    return stage


def transform_age_column(df, dataset_name):
    if dataset_name.lower().startswith("yoshida"):
        df["age"] = df["age"].map(map_yoshida_age)
    elif dataset_name.lower().startswith("stephenson"):
        df["age"] = df["age"].map(map_stephenson_age)
    return df

def get_expression_df(adata, cell_type, healthy_only, dataset_name):
    mask = adata.obs["cell_type"] == cell_type
    if healthy_only and "disease" in adata.obs.columns:
        mask &= adata.obs["disease"] == "normal"
    if not np.any(mask):
        return None

    subset = adata[mask]
    X = log_norm(subset.X.tocsr())
    df = pd.DataFrame(X.toarray(), index=subset.obs_names, columns=subset.var_names)

    df["age"] = subset.obs["development_stage"].values
    df["donor_id"] = subset.obs["donor_id"].values
    return df

def apply_clock(df, model_df, impute_df):
    preds = []
    for i in range(len(model_df)):
        print(f"  Applying model {i + 1}/{len(model_df)}...")
        model = model_df.iloc[i].dropna()
        intercept = model["intercept"]
        weights = model.drop("intercept")
        weights = weights[weights != 0]

        genes_to_use = weights.index
        present_genes = genes_to_use.intersection(df.columns)
        missing_genes = genes_to_use.difference(df.columns)

        expr = df[present_genes].copy()
        print(f"    Using {len(present_genes)} present genes, imputing {len(missing_genes)}")

        if not missing_genes.empty:
            impute_vals = {
                gene: impute_df[gene].values[0] if gene in impute_df.columns else 0.0
                for gene in missing_genes
            }
            impute_df_full = pd.DataFrame(
                np.tile(list(impute_vals.values()), (expr.shape[0], 1)),
                index=expr.index,
                columns=missing_genes
            )
            expr = pd.concat([expr, impute_df_full], axis=1)

        expr = expr[genes_to_use]
        pred = expr @ weights.values + intercept
        preds.append(pred)

    avg_pred = np.mean(preds, axis=0)
    result = df[["donor_id", "age"]].copy()
    result["predicted_age"] = avg_pred
    result["cell_name"] = df.index
    return result

def process_dataset(dataset_name):
    print(f"\nProcessing {dataset_name}")
    file_path = os.path.join(DATA_FOLDER, f"{dataset_name}.h5ad")
    adata = ad.read_h5ad(file_path, backed="r")
    healthy_only = dataset_name.lower() != "eqtl"
    mapping = CELLTYPE_MAPPINGS.get(dataset_name, {})
    output_dir = os.path.join(OUTPUT_FOLDER, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    for cell_type in adata.obs["cell_type"].unique():
        print(f"Processing cell type: {cell_type}")
        clock_targets = mapping.get(cell_type, cell_type)
        if isinstance(clock_targets, str):
            clock_targets = [clock_targets]

        df = get_expression_df(adata, cell_type, healthy_only, dataset_name)
        if df is None or df.empty:
            continue

        for clock in clock_targets:
            model_path = os.path.join(MODEL_FOLDER, f"{clock}_models5.csv")
            impute_path = os.path.join(IMPUTE_FOLDER, f"Impute_avg_{clock}.csv")
            if not (os.path.exists(model_path) and os.path.exists(impute_path)):
                print(f"Skipping {cell_type} → {clock}: missing model or impute")
                continue

            model_df = pd.read_csv(model_path)
            impute_df = pd.read_csv(impute_path)
            predictions = apply_clock(df.copy(), model_df, impute_df)
            predictions = transform_age_column(predictions, dataset_name)

            out_file = f"{cell_type.replace(' ', '_')}.csv"
            predictions.to_csv(os.path.join(output_dir, out_file), index=False)

if __name__ == "__main__":
    for dataset in ["Yoshida", "Liu", "eQTL", "Stephenson"]:
        process_dataset(dataset)
