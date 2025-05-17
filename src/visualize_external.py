import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

SUMMARY_DIR = "../results"
SAVE_DIR = "../figures_external"
os.makedirs(SAVE_DIR, exist_ok=True)

datasets = {
    "Yoshida": {
        "file": "Yoshida_summary.csv",
        "metric": "spearman_corr",
        "pval": "spearman_pval",
        "title": "a. Yoshida",
        "xlabel": "Spearman ρ",
        "filename": "figure2a_yoshida.png"
    },
    "Liu": {
        "file": "Liu_summary.csv",
        "metric": "pearson_corr",
        "pval": "pearson_pval",
        "mae": "mae",
        "title": "b. Liu",
        "xlabel": "Pearson r",
        "filename": "figure2b_liu.png"
    },
    "eQTL": {
        "file": "eQTL_summary.csv",
        "metric": "pearson_corr",
        "pval": "pearson_pval",
        "mae": "mae",
        "title": "c. eQTL",
        "xlabel": "Pearson r",
        "filename": "figure2c_eqtl.png"
    },
    "Stephenson": {
        "file": "Stephenson_summary.csv",
        "metric": "spearman_corr",
        "pval": "spearman_pval",
        "title": "d. Stephenson",
        "xlabel": "Spearman ρ",
        "filename": "figure2d_stephenson.png"
    }
}

for name, meta in datasets.items():
    path = os.path.join(SUMMARY_DIR, meta["file"])
    df = pd.read_csv(path)
    df = df.dropna(subset=[meta["metric"]])
    df_sorted = df.sort_values(by=meta["metric"], ascending=False)

    width = 10 if name == "Liu" else 8
    plt.figure(figsize=(width, max(6, len(df_sorted) * 0.3)))
    ax = sns.barplot(
        y="cell_type",
        x=meta["metric"],
        hue="cell_type",
        data=df_sorted,
        palette="coolwarm" if "spearman" in meta["metric"] else "crest",
        legend=False
    )

    ax.set_title(meta["title"], fontsize=14, loc="left")
    ax.set_xlabel(meta["xlabel"], fontsize=12)
    ax.set_ylabel("")

    for i, (val, cell_type) in enumerate(zip(df_sorted[meta["metric"]], df_sorted["cell_type"])):
        ax.text(val + 0.01 if val >= 0 else val - 0.01, i,
                f"{val:.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)

        if "mae" in meta:
            mae_val = df_sorted.iloc[i][meta["mae"]]
            ax.text(ax.get_xlim()[1] + 0.02, i, f"MAE = {mae_val:.1f}",
                    va="center", ha="left", fontsize=9, color="black")

    if name == "Liu":
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 0.1)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, meta["filename"])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
