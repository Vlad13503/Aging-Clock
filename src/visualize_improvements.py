import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

results_dir = "../results"
results_enhanced_dir = "../results_enhanced"
output_dir = "../figures_external_enhanced"
os.makedirs(output_dir, exist_ok=True)

datasets = {
    "Liu": "Liu_summary.csv",
    "eQTL": "eQTL_summary.csv",
    "AIDA": "summary_performance.csv"
}

for dataset, file_name in datasets.items():
    orig_path = os.path.join(results_dir, file_name)
    enh_path = os.path.join(results_enhanced_dir, file_name)

    df_orig = pd.read_csv(orig_path)
    df_enh = pd.read_csv(enh_path)

    df_orig = df_orig[["cell_type", "MAE" if dataset == "AIDA" else "mae"]].rename(columns={"MAE" if dataset == "AIDA" else "mae": "MAE_original"})
    df_enh = df_enh[["cell_type", "MAE" if dataset == "AIDA" else "mae"]].rename(columns={"MAE" if dataset == "AIDA" else "mae": "MAE_enhanced"})

    df_merge = pd.merge(df_orig, df_enh, on="cell_type", how="inner")
    df_melt = df_merge.melt(id_vars="cell_type", value_vars=["MAE_original", "MAE_enhanced"],
                            var_name="Model", value_name="MAE")

    plt.figure(figsize=(10, max(6, len(df_merge) * 0.35)))
    ax = sns.barplot(data=df_melt, y="cell_type", x="MAE", hue="Model", palette="Set1")

    for i, row in df_melt.iterrows():
        bar = ax.patches[i]
        width = bar.get_width()
        ax.text(width + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{row['MAE']:.2f}", va='center', ha='left', fontsize=8)

    plt.title(f"{dataset}: MAE Comparison (Original vs. Enhanced)")
    plt.legend(title="Model", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset.lower()}_mae_comparison.png"), dpi=300)
    plt.close()
