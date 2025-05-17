import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_ind
import glob

PREDICTIONS_FOLDER = "../predictions_external/"
OUTPUT_FOLDER = "../results"
DATASETS = ["Yoshida", "Liu", "eQTL", "Stephenson"]
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def group_ages_eQTL_Liu(age):
    try:
        age = float(age)
    except Exception:
        return np.nan
    if age < 18:
        return "young"
    elif age < 65:
        return "adult"
    else:
        return "aged adult"

def get_files(dataset):
    folder = os.path.join(PREDICTIONS_FOLDER, dataset)
    return sorted(glob.glob(os.path.join(folder, "*.csv")))

def evaluate_csv(df, dataset):
    df = df.dropna(subset=["age", "predicted_age"])
    results = {}

    if dataset in ["eQTL", "Liu"]:
        pear, pval = pearsonr(df["age"], df["predicted_age"])
        mae = np.mean(np.abs(df["predicted_age"] - df["age"]))
        results.update({
            "pearson_corr": pear,
            "pearson_pval": pval,
            "mae": mae,
            "n_cells": len(df)
        })

        df["age_group"] = df["age"].map(group_ages_eQTL_Liu)
        group1 = df[df["age_group"] == "adult"]["predicted_age"]
        group2 = df[df["age_group"] == "aged adult"]["predicted_age"]
        if not group1.empty and not group2.empty:
            tstat, t_pval = ttest_ind(group1, group2, equal_var=False, nan_policy="omit")
        else:
            tstat, t_pval = np.nan, np.nan
        results.update({
            "tstat_adult_vs_aged": tstat,
            "t_pval_adult_vs_aged": t_pval,
        })

    elif dataset in ["Yoshida", "Stephenson"]:
        categories = pd.Categorical(df["age"])
        df = df.assign(age_code=categories.codes)
        spear, spval = spearmanr(df["age_code"], df["predicted_age"])
        results.update({
            "spearman_corr": spear,
            "spearman_pval": spval,
            "n_age_groups": df["age"].nunique(),
            "n_cells": len(df)
        })

    return results

if __name__ == "__main__":
    for dataset in DATASETS:
        print(f"== Evaluating: {dataset} ==")
        dataset_results = []

        for filepath in get_files(dataset):
            cell_type = os.path.splitext(os.path.basename(filepath))[0].replace("_", " ")
            df = pd.read_csv(filepath)
            metrics = evaluate_csv(df, dataset)
            row = {"cell_type": cell_type, **metrics}
            dataset_results.append(row)

        df_summary = pd.DataFrame(dataset_results)
        summary_path = os.path.join(OUTPUT_FOLDER, f"{dataset}_summary.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}\n")
