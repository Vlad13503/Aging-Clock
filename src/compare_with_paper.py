import os
import pandas as pd
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

paper_root = "../paper_results"
your_enhanced_aida = "../predictions_enhanced"
your_external_root = "../predictions_external_enhanced"
output_csv = "../results_enhanced/comparison_per_celltype.csv"

results = []

def normalize(name):
    return name.lower().replace(" ", "_").replace(",", "").replace("-", "").replace(".", "").replace("(", "").replace(")", "")

external_datasets = ["eQTL", "liu", "stephenson", "yoshida"]

for dataset in os.listdir(paper_root):
    dataset_path = os.path.join(paper_root, dataset)
    if not os.path.isdir(dataset_path):
        continue

    print(f"\nComparing: {dataset}")
    is_external = dataset.lower() in [d.lower() for d in external_datasets]

    paper_files = {
        normalize(os.path.splitext(f)[0]): os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path) if f.endswith(".csv")
    }

    your_path = os.path.join(your_external_root, dataset) if is_external else your_enhanced_aida
    if not os.path.exists(your_path):
        print(f"Your predictions missing for dataset: {dataset}")
        continue

    for your_file in os.listdir(your_path):
        if not your_file.endswith("_predictions.csv"):
            continue

        cell_type_raw = your_file.replace("_predictions.csv", "")
        cell_type_key = normalize(cell_type_raw)

        if cell_type_key not in paper_files:
            continue

        try:
            df_paper = pd.read_csv(paper_files[cell_type_key])
            df_yours = pd.read_csv(os.path.join(your_path, your_file))
        except Exception as e:
            print(f"[ERROR] {dataset}/{cell_type_raw}: Failed to load files — {e}")
            continue

        df_paper = df_paper.rename(columns={
            "index": "cell_id",
            "cell_name": "cell_id",
            "donor": "donor_id",
            "age": "true_age",
            "predicted_age": "pred_paper"
        })

        df_yours = df_yours.rename(columns={
            "cell_name": "cell_id",
            "predicted_age": "pred_yours"
        })

        if "true_age" not in df_yours.columns:
            if "age" in df_yours.columns:
                df_yours["true_age"] = df_yours["age"]
            else:
                print(f"[ERROR] {dataset}/{cell_type_raw}: 'true_age' missing after merge")
                continue

        df_paper["cell_id"] = df_paper["cell_id"].astype(str)
        df_yours["cell_id"] = df_yours["cell_id"].astype(str)

        if "true_age" not in df_yours.columns:
            print(f"[ERROR] {dataset}/{cell_type_raw}: 'true_age' missing in your predictions")
            continue
        merged = pd.merge(df_yours, df_paper[["cell_id", "pred_paper"]], on="cell_id")
        if merged.empty:
            print(f"[EMPTY] No overlap for {dataset}/{cell_type_raw}")
            continue

        try:
            mae_yours = mean_absolute_error(merged["true_age"], merged["pred_yours"])
            mae_paper = mean_absolute_error(merged["true_age"], merged["pred_paper"])
            r_yours, _ = pearsonr(merged["true_age"], merged["pred_yours"])
            r_paper, _ = pearsonr(merged["true_age"], merged["pred_paper"])
        except Exception as e:
            print(f"[ERROR] {dataset}/{cell_type_raw}: could not compute metrics — {e}")
            continue

        results.append({
            "dataset": dataset,
            "cell_type": cell_type_raw,
            "n_cells": len(merged),
            "MAE_yours": mae_yours,
            "MAE_paper": mae_paper,
            "r_yours": r_yours,
            "r_paper": r_paper
        })

if results:
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by=["dataset", "MAE_yours"])
    df_results.to_csv(output_csv, index=False)
    print(f"\nSaved comparison to: {output_csv}")
else:
    print("\nNo valid comparisons were made. Check for matching cell types and column consistency.")
