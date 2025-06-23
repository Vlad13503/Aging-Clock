import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

results_dir = "../results"
results_enhanced_dir = "../results_enhanced"
results_nonlinear_dir = "../results_nonlinear_enhanced"
output_dir = "../figures_external_enhanced"
os.makedirs(output_dir, exist_ok=True)

sns.set(style="whitegrid")

datasets = {
    "AIDA": "summary_performance.csv",
    "eQTL": "eQTL_summary.csv",
    "Liu": "Liu_summary.csv"
}

fig, axes = plt.subplots(1, 3, figsize=(26, 12))
fig.subplots_adjust(wspace=0.5)
panel_labels = ['a. AIDA', 'b. eQTL', 'c. Liu']

for ax, (dataset, file_name), panel_label in zip(axes, datasets.items(), panel_labels):
    orig_path = os.path.join(results_dir, file_name)
    enh_path = os.path.join(results_enhanced_dir, file_name)

    df_orig = pd.read_csv(orig_path)
    df_enh = pd.read_csv(enh_path)

    mae_col = "MAE" if dataset == "AIDA" else "mae"
    df_orig = df_orig[["cell_type", mae_col]].rename(columns={mae_col: "MAE_replication"})
    df_enh = df_enh[["cell_type", mae_col]].rename(columns={mae_col: "MAE_enhanced"})

    df_merge = pd.merge(df_orig, df_enh, on="cell_type")
    df_melt = df_merge.melt(id_vars="cell_type", value_vars=["MAE_replication", "MAE_enhanced"],
                            var_name="Model", value_name="MAE")

    sns.barplot(data=df_melt, y="cell_type", x="MAE", hue="Model", palette="Set1", ax=ax)
    ax.set_title(panel_label, fontsize=17, weight='bold')
    ax.set_xlabel("MAE", fontsize=15)
    ax.set_ylabel("")
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='x', labelsize=15)
    ax.legend().remove()

    for bar, (_, row) in zip(ax.patches, df_melt.iterrows()):
        value = row["MAE"]
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}", va='center', ha='left', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mae_panel_comparison.png"), dpi=300)

fig2, axes2 = plt.subplots(1, 3, figsize=(26, 12))
fig2.subplots_adjust(wspace=0.5)
panel_labels2 = ['a. AIDA', 'b. eQTL', 'c. Liu']

for ax, (dataset, file_name), panel_label in zip(axes2, datasets.items(), panel_labels2):
    linear_path = os.path.join(results_enhanced_dir, file_name)
    nonlinear_path = os.path.join(results_nonlinear_dir, file_name)

    df_linear = pd.read_csv(linear_path)
    df_nonlinear = pd.read_csv(nonlinear_path)

    mae_col = "MAE" if dataset == "AIDA" else "mae"
    df_linear = df_linear[["cell_type", mae_col]].rename(columns={mae_col: "MAE_linear"})
    df_nonlinear = df_nonlinear[["cell_type", mae_col]].rename(columns={mae_col: "MAE_nonlinear"})

    df_merge = pd.merge(df_linear, df_nonlinear, on="cell_type")
    df_melt = df_merge.melt(id_vars="cell_type", value_vars=["MAE_linear", "MAE_nonlinear"],
                            var_name="Model", value_name="MAE")

    sns.barplot(data=df_melt, y="cell_type", x="MAE", hue="Model", palette="Set2", ax=ax)
    ax.set_title(panel_label, fontsize=17, weight='bold')
    ax.set_xlabel("MAE", fontsize=15)
    ax.set_ylabel("")
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='x', labelsize=15)
    ax.legend().remove()

    for bar, (_, row) in zip(ax.patches, df_melt.iterrows()):
        value = row["MAE"]
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}", va='center', ha='left', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mae_panel_comparison_nonlinear.png"), dpi=300)
