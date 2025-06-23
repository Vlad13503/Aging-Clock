import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad

DATASET_DIR = "../data"
DATASETS = {
    "AIDA": "AIDA.h5ad",
    "Yoshida": "Yoshida.h5ad",
    "Liu": "Liu.h5ad",
    "eQTL": "eQTL.h5ad",
    "Stephenson": "Stephenson.h5ad"
}
OUTPUT_DIR = "../figures_dataset_stats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

summary = []

for name, file in DATASETS.items():
    path = os.path.join(DATASET_DIR, file)
    adata = ad.read_h5ad(path, backed="r")

    n_cells = adata.n_obs
    n_genes = adata.n_vars
    n_donors = adata.obs["donor_id"].nunique() if "donor_id" in adata.obs else None
    n_celltypes = adata.obs["cell_type"].nunique() if "cell_type" in adata.obs else None

    try:
        ages = pd.to_numeric(adata.obs["development_stage"].str.extract(r"(\d+)")[0], errors="coerce").dropna()
        age_min, age_max = int(ages.min()), int(ages.max())
    except:
        age_min, age_max = None, None

    summary.append({
        "Dataset": name,
        "N_Cells": n_cells,
        "N_Genes": n_genes,
        "N_Donors": n_donors,
        "N_CellTypes": n_celltypes,
        "Min_Age": age_min,
        "Max_Age": age_max
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "dataset_summary.csv"), index=False)

sns.set(style="whitegrid")
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
fig.subplots_adjust(top=0.92, wspace=0.3, hspace=0.4)
fig.suptitle("Dataset Overview", fontsize=20)

panel_labels = ['a', 'b', 'c', 'd']
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
for label, (i, j) in zip(panel_labels, positions):
    axs[i, j].text(-0.1, 1.05, f"{label}.", transform=axs[i, j].transAxes,
                   fontsize=18, fontweight='bold', va='top', ha='right')

bar_color = "#69b3a2"

axs[0, 0].bar(summary_df["Dataset"], summary_df["N_Cells"], color=bar_color)
axs[0, 0].set_title("Number of Cells", fontsize=16)
axs[0, 0].set_ylabel("N Cells", fontsize=14)
axs[0, 0].tick_params(axis='both', labelsize=13)
axs[0, 0].ticklabel_format(style='plain', axis='y')
for i, v in enumerate(summary_df["N_Cells"]):
    axs[0, 0].text(i, v + 0.01 * v, f"{v:,}", ha='center', fontsize=13)

axs[0, 1].bar(summary_df["Dataset"], summary_df["N_CellTypes"], color=bar_color)
axs[0, 1].set_title("Number of Cell Types", fontsize=16)
axs[0, 1].set_ylabel("N Cell Types", fontsize=14)
axs[0, 1].tick_params(axis='both', labelsize=13)
for i, v in enumerate(summary_df["N_CellTypes"]):
    axs[0, 1].text(i, v + 0.3, str(v), ha='center', fontsize=13)

axs[1, 0].bar(summary_df["Dataset"], summary_df["N_Donors"], color=bar_color)
axs[1, 0].set_title("Number of Donors", fontsize=16)
axs[1, 0].set_ylabel("N Donors", fontsize=14)
axs[1, 0].tick_params(axis='both', labelsize=13)
for i, v in enumerate(summary_df["N_Donors"]):
    axs[1, 0].text(i, v + 0.3, str(v), ha='center', fontsize=13)

age_ranges = summary_df.apply(
    lambda row: f'{row["Min_Age"]}-{row["Max_Age"]}' if pd.notnull(row["Min_Age"]) else "N/A", axis=1
)
axs[1, 1].bar(summary_df["Dataset"], summary_df["Max_Age"] - summary_df["Min_Age"], color=bar_color)
axs[1, 1].set_title("Age Range (Min - Max)", fontsize=16)
axs[1, 1].set_ylabel("Years", fontsize=14)
axs[1, 1].tick_params(axis='both', labelsize=13)
for i, label in enumerate(age_ranges):
    height = (summary_df["Max_Age"][i] - summary_df["Min_Age"][i]) if pd.notnull(summary_df["Min_Age"][i]) else 0
    axs[1, 1].text(i, height + 1, label, ha='center', fontsize=13)

plt.savefig(os.path.join(OUTPUT_DIR, "dataset_panel_overview.png"), dpi=300)
plt.close()
