import os
import pandas as pd
import numpy as np
import anndata as ad
import lightgbm as lgb
from celltype_mappings import CELLTYPE_MAPPINGS

DATA_FOLDER = "../data/"
MODEL_FOLDER = "../nonlinear_models_enhanced/"
IMPUTE_FOLDER = "../data_for_imputation_nonlinear/"
OUTPUT_FOLDER = "../predictions_external_nonlinear_enhanced/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def log_norm(X):
    row_sums = np.array(X.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1e-12
    X = X.multiply(1 / row_sums[:, None]) * 10000
    return X.log1p()

def map_yoshida_age(dev_stage):
    if pd.isnull(dev_stage): return np.nan
    val = str(dev_stage).lower().strip()
    if "newborn" in val: return "newborn human stage"
    if "infant" in val: return "infant stage"
    if "2-5" in val or "1-4" in val: return "2-5 year-old child stage"
    if "6-12" in val or "5-14" in val or "juvenile" in val: return "6-12 year-old child stage"
    if "child" in val: return "6-12 year-old child stage"
    if "adolescent" in val: return "adolescent stage"
    if "adult" in val: return "human adult stage"
    if "aged" in val or "elderly" in val: return "human aged stage"
    try:
        num = float(val.split("-")[0])
        return "human aged stage" if num >= 65 else "human adult stage"
    except:
        return val

def map_stephenson_age(stage):
    if pd.isnull(stage): return np.nan
    s = str(stage).lower().strip()
    if "decade stage" in s: return s
    import re
    m = re.match(r"(\d+)-year-old stage", s)
    if m:
        age = int(m.group(1))
        decade = min((age // 10) + 1, 10)
        ordinals = {3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh",
                    8: "eighth", 9: "ninth", 10: "tenth"}
        return f"{ordinals.get(decade, str(decade)+'th')} decade stage"
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

def process_dataset(dataset_name):
    print(f"\nProcessing {dataset_name}")
    file_path = os.path.join(DATA_FOLDER, f"{dataset_name}.h5ad")
    adata = ad.read_h5ad(file_path, backed="r")
    healthy_only = dataset_name.lower() != "eqtl"
    mapping = CELLTYPE_MAPPINGS.get(dataset_name, {})
    output_dir = os.path.join(OUTPUT_FOLDER, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    for cell_type in adata.obs["cell_type"].unique():
        print(f"Cell type: {cell_type}")
        clock_targets = mapping.get(cell_type, cell_type)
        if isinstance(clock_targets, str):
            clock_targets = [clock_targets]

        df = get_expression_df(adata, cell_type, healthy_only, dataset_name)
        if df is None or df.empty:
            continue
        if dataset_name.lower() in ["liu", "eqtl"]:
            df["age"] = pd.to_numeric(df["age"].str.extract(r"(\d+)")[0], errors="coerce")

        for target in clock_targets:
            model_files = sorted([
                f for f in os.listdir(MODEL_FOLDER)
                if f.startswith(f"{target}_model_fold") and f.endswith(".txt")
            ])

            if not model_files:
                print(f"  Skipping: no models found for {target}")
                continue

            impute_file = os.path.join(IMPUTE_FOLDER, f"Impute_avg_{target}.csv")
            if not os.path.exists(impute_file):
                print(f"  Skipping: missing impute file for {target}")
                continue

            impute_df = pd.read_csv(impute_file, index_col=0)
            mean_row = impute_df.loc["mean"]
            std_row = impute_df.loc["std"]

            all_preds = []
            for model_file in model_files:
                model_path = os.path.join(MODEL_FOLDER, model_file)
                booster = lgb.Booster(model_file=model_path)

                model_genes = booster.feature_name()
                present_genes = df.columns.intersection(model_genes)
                missing_genes = set(model_genes).difference(df.columns)

                print(f"    Using {len(present_genes)} present genes, imputing {len(missing_genes)}")

                expr = df[present_genes].copy()

                if missing_genes:
                    np.random.seed(42)
                    imputed_values = {}
                    for gene in missing_genes:
                        mean = mean_row.get(gene, 0.0)
                        std = std_row.get(gene, 1e-8)
                        imputed_values[gene] = np.random.normal(mean, std, size=len(expr))

                    imputed_df = pd.DataFrame(imputed_values, index=expr.index)
                    expr = pd.concat([expr, imputed_df], axis=1)

                expr = expr[model_genes]
                pred = booster.predict(expr.values)
                all_preds.append(pred)

            if not all_preds:
                continue

            avg_pred = np.mean(all_preds, axis=0)
            out = df[["donor_id", "age"]].copy()
            out["predicted_age"] = avg_pred
            out["cell_name"] = df.index
            out = transform_age_column(out, dataset_name)
            out.to_csv(os.path.join(output_dir, f"{cell_type.replace(' ', '_')}.csv"), index=False)

if __name__ == "__main__":
    for dataset in ["Yoshida", "Liu", "eQTL", "Stephenson"]:
        process_dataset(dataset)
