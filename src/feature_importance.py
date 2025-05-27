import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

MODEL_DIR = "../models_enhanced/"
OUTPUT_PATH = "../results_enhanced/feature_importance_summary.csv"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

importance_summary = []

for file in os.listdir(MODEL_DIR):
    if not file.endswith("_models5.csv"):
        continue

    cell_type = file.replace("_models5.csv", "")
    df = pd.read_csv(os.path.join(MODEL_DIR, file))

    coef_df = df.drop(columns=["intercept"], errors="ignore").abs()
    mean_importance = coef_df.mean(axis=0).sort_values(ascending=False)

    top_genes = mean_importance.head(20)

    for gene, importance in top_genes.items():
        importance_summary.append({
            "cell_type": cell_type,
            "gene": gene,
            "mean_abs_coefficient": importance
        })

importance_df = pd.DataFrame(importance_summary)
importance_df.to_csv(OUTPUT_PATH, index=False)

overall_importance = (
    importance_df.groupby("gene")["mean_abs_coefficient"]
    .mean()
    .sort_values(ascending=False)
    .head(20)
    .reset_index()
)

plt.figure(figsize=(10, 8))
ax = sns.barplot(data=overall_importance, x="mean_abs_coefficient", y="gene", hue="gene", palette="mako", legend=False)
for i, v in enumerate(overall_importance["mean_abs_coefficient"]):
    ax.text(v + 0.005, i, f"{v:.2f}", va='center', ha='left', fontsize=8)
plt.title("Top 20 Most Predictive Genes Across All Cell Types")
plt.xlabel("Mean |Coefficient|")
plt.ylabel("Gene")
plt.tight_layout()
plt.savefig("../figures_enhanced/feature_importance_overall_top20.png")
plt.close()

pivot_df = importance_df.pivot(index="gene", columns="cell_type", values="mean_abs_coefficient")
top_genes = importance_df.groupby("gene")["mean_abs_coefficient"].mean().sort_values(ascending=False).head(30).index
pivot_df = pivot_df.loc[top_genes]

plt.figure(figsize=(14, 10))
sns.heatmap(pivot_df, cmap="viridis", linewidths=0.5, linecolor='gray')
plt.title("Gene Importance Heatmap (Top 30 Genes × Cell Types)")
plt.xlabel("Cell Type")
plt.ylabel("Gene")
plt.tight_layout()
plt.savefig("../figures_enhanced/feature_importance_heatmap.png")
plt.close()

top_counts = importance_df.groupby("gene")["cell_type"].count().sort_values(ascending=False).head(20).reset_index()
top_counts.columns = ["gene", "appearances"]

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=top_counts, x="appearances", y="gene", hue="gene", palette="flare", legend=False)
for i, v in enumerate(top_counts["appearances"]):
    ax.text(v + 0.3, i, str(v), va='center', ha='left', fontsize=8)
plt.title("Most Frequent Genes in Top 20 Across Cell Types")
plt.xlabel("# of Cell Types Appeared In Top 20")
plt.tight_layout()
plt.savefig("../figures_enhanced/feature_importance_gene_frequency.png")
plt.close()
