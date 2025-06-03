import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

shap_dir = "../predictions_enhanced/"
output_fig_dir = "../figures_enhanced/"
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

overall = (
    shap_df_grouped.groupby("gene")["mean_abs_shap"]
    .mean()
    .sort_values(ascending=False)
    .head(20)
    .reset_index()
)

plt.figure(figsize=(10, 8))
ax = sns.barplot(data=overall, x="mean_abs_shap", y="gene", palette="mako", hue="gene", legend=False)
ax.set_title("Top 20 Genes by SHAP Importance Across All Cell Types")
ax.set_xlabel("Mean |SHAP Value|")
ax.set_ylabel("Gene")
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", label_type="edge", padding=3)
plt.tight_layout()
plt.savefig(os.path.join(output_fig_dir, "shap_top20_genes_overall.png"))
plt.close()

top_genes = (
    shap_df_grouped.groupby("gene")["mean_abs_shap"]
    .mean()
    .sort_values(ascending=False)
    .head(30)
    .index
)

pivot = shap_df_grouped[shap_df_grouped["gene"].isin(top_genes)]
pivot = pivot.pivot(index="gene", columns="cell_type", values="mean_abs_shap")

plt.figure(figsize=(14, 10))
sns.heatmap(pivot, cmap="viridis", linewidths=0.5, linecolor="gray")
plt.title("SHAP Importance Heatmap (Top 30 Genes × Cell Types)")
plt.xlabel("Cell Type")
plt.ylabel("Gene")
plt.tight_layout()
plt.savefig(os.path.join(output_fig_dir, "shap_heatmap_top30.png"))
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
plt.savefig(os.path.join(output_fig_dir, "shap_gene_frequency.png"))
plt.close()
