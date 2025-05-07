import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

PREDICTIONS_FOLDER = "../predictions/"
OUTPUT_PATH = "../results/summary_performance.csv"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

summary = []

for fname in os.listdir(PREDICTIONS_FOLDER):
    if not fname.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(PREDICTIONS_FOLDER, fname))

    mae = mean_absolute_error(df["true_age"], df["predicted_age"])
    pearson, _ = pearsonr(df["true_age"], df["predicted_age"])
    spearman, _ = spearmanr(df["true_age"], df["predicted_age"])
    r2 = r2_score(df["true_age"], df["predicted_age"])

    summary.append({
        "cell_type": fname.replace("predictions_", "").replace(".csv", ""),
        "MAE": mae,
        "Pearson": pearson,
        "Spearman": spearman,
        "R2": r2,
        "n_cells": len(df)
    })

summary_df = pd.DataFrame(summary).sort_values("MAE")
summary_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved summary to {OUTPUT_PATH}")
