import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SUMMARY_PATH = "../results/summary_performance.csv"
PREDICTIONS_FOLDER = "../predictions/"
OUTPUT_FOLDER = "../figures/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

summary_df = pd.read_csv(SUMMARY_PATH)

# === Plot b: Cell counts per cell type ===
plt.figure(figsize=(10, 12))
sorted_df = summary_df.sort_values("n_cells", ascending=True)
ax = sns.barplot(
    data=summary_df.sort_values("n_cells", ascending=True),
    x="n_cells", y="cell_type", palette="Blues_d", hue="cell_type", legend=False
)
for i, count in enumerate(sorted_df["n_cells"]):
    ax.text(count + 1000, i, f"{count:,}", va='center', ha='left', fontsize=8)
plt.title("Number of Cells per Cell Type")
plt.xlabel("Cell Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "cell_counts.png"))
plt.close()

# === Plot d: Pearson Correlation per Cell Type ===
plt.figure(figsize=(10, 12))
sorted_df = summary_df.sort_values("Pearson", ascending=False)
ax = sns.barplot(
    data=sorted_df,
    x="Pearson", y="cell_type", palette="Blues_d", hue="cell_type", legend=False
)
ax.set_xlim(-0.2, 0.65)
for i, (pearson, mae) in enumerate(zip(sorted_df["Pearson"], sorted_df["MAE"])):
    ax.text(pearson + 0.005, i, f"{pearson:.2f}", va='center', ha='left', fontsize=8, color="black")
    ax.text(0.62, i, f"MAE = {mae:.2f}", va='center', ha='left', fontsize=8, color="gray")

plt.title("Pearson Correlation per Cell Type")
plt.xlabel("Pearson r")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "pearson_per_cell_type.png"))
plt.close()

# === Plot 2: MAE per Cell Type ===
plt.figure(figsize=(10, 12))
sorted_df = summary_df.sort_values("MAE", ascending=True)
ax = sns.barplot(
    data=summary_df.sort_values("MAE", ascending=True),
    x="MAE", y="cell_type", palette="Reds_r", hue="cell_type", legend=False
)
for i, mae in enumerate(sorted_df["MAE"]):
    ax.text(mae + 0.05, i, f"{mae:.2f}", va='center', ha='left', fontsize=8)
plt.title("Mean Absolute Error (MAE) per Cell Type")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "mae_per_cell_type.png"))
plt.close()

# === Plot e: Predicted vs. True Age for Top 9 cell types ===
top_n = summary_df.sort_values("Pearson", ascending=False).head(9)
fig, axs = plt.subplots(3, 3, figsize=(16, 14))
axs = axs.flatten()

for i, row in enumerate(top_n.itertuples()):
    df = pd.read_csv(os.path.join(PREDICTIONS_FOLDER, f"{row.cell_type}_predictions.csv"))
    sns.regplot(ax=axs[i], data=df, x="true_age", y="predicted_age",
                scatter_kws={"alpha": 0.3, "s": 10}, line_kws={"color": "blue"})
    axs[i].set_title(
        f"{row.cell_type}\nMAE={row.MAE:.2f}, r={row.Pearson:.2f}, n={row.n_cells}"
    )
    axs[i].set_xlim(20, 65)
    axs[i].set_ylim(20, 65)
    axs[i].set_xlabel("True Age")
    axs[i].set_ylabel("Predicted Age")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "scatter_top9_celltypes.png"))
plt.close()
