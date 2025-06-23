import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

shap_dir = "../predictions_nonlinear_enhanced/"
output_fig_dir = "../figures_nonlinear_enhanced/"
os.makedirs(output_fig_dir, exist_ok=True)

shap_dfs = []
for file in os.listdir(shap_dir):
    if file.endswith("_shap_summary.csv"):
        cell_type = file.replace("_shap_summary.csv", "")
        df = pd.read_csv(os.path.join(shap_dir, file))
        df["cell_type"] = cell_type
        shap_dfs.append(df)

shap_df = pd.concat(shap_dfs, ignore_index=True)
shap_df_grouped = (
    shap_df.groupby(["cell_type", "gene"])["mean_abs_shap"]
    .mean()
    .reset_index()
)

top_genes = (
    shap_df_grouped.groupby("gene")["mean_abs_shap"]
    .mean()
    .sort_values(ascending=False)
    .head(30)
    .index
)

pivot = shap_df_grouped[shap_df_grouped["gene"].isin(top_genes)]
pivot_table = pivot.pivot(index="gene", columns="cell_type", values="mean_abs_shap")

pivot_table["all_cells"] = pivot_table.mean(axis=1)
cols = ["all_cells"] + sorted([col for col in pivot_table.columns if col != "all_cells"])
pivot_table = pivot_table[cols]

plt.figure(figsize=(16, 10))
sns.heatmap(
    pivot_table,
    cmap="viridis",
    linewidths=0.5,
    linecolor="gray",
    annot=True,
    fmt=".2f",
    annot_kws={"size": 10}
)
plt.title("SHAP Importance Heatmap (Top 30 Genes × Cell Types)", fontsize=16)
plt.xlabel("Cell Type", fontsize=14)
plt.ylabel("Gene", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_fig_dir, "shap_heatmap_top30_nonlinear.png"), dpi=300)
plt.close()

top_n = 20
top_per_celltype = shap_df_grouped.groupby("cell_type").apply(
    lambda g: g.nlargest(top_n, "mean_abs_shap")
).reset_index(drop=True)

freq = top_per_celltype["gene"].value_counts().head(20).reset_index()
freq.columns = ["gene", "appearances"]

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=freq, x="appearances", y="gene", palette="flare", hue="gene", legend=False)
ax.set_title(f"Top {top_n} SHAP Genes — Appearance Frequency Across Cell Types")
ax.set_xlabel("# of Cell Types Appeared In Top 20")
for container in ax.containers:
    ax.bar_label(container, fmt="%d", label_type="edge", padding=3)
plt.tight_layout()
plt.savefig(os.path.join(output_fig_dir, "shap_gene_frequency_nonlinear.png"), dpi=300)
plt.close()
