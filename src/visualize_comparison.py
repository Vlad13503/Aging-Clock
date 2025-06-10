import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import cm

df = pd.read_csv("../results_enhanced/comparison_per_celltype.csv")
output_dir = "../figures_comparison"
os.makedirs(output_dir, exist_ok=True)
sns.set(style="whitegrid", context="talk")

def get_combined_palette(cell_types):
    base_palette = sns.color_palette("tab20")
    if len(cell_types) <= len(base_palette):
        return dict(zip(cell_types, base_palette))
    else:
        extra_colors = cm.hsv(np.linspace(0, 1, len(cell_types) - len(base_palette)))
        combined_colors = base_palette + [tuple(color[:3]) for color in extra_colors]
        return dict(zip(cell_types, combined_colors))

cell_types = df["cell_type"].unique()
palette = get_combined_palette(cell_types)

def configure_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(), unique.keys(),
        bbox_to_anchor=(1.02, 1.0), loc="upper left",
        title="Cell Type", fontsize=15, title_fontsize=15,
        ncol=1, borderaxespad=0.5, labelspacing=0.6
    )

plt.figure(figsize=(16, 16))
ax = plt.gca()

min_mae = min(df["MAE_yours"].min(), df["MAE_paper"].min())
max_mae = max(df["MAE_yours"].max(), df["MAE_paper"].max())
ax.set_xlim(min_mae - 0.5, max_mae + 0.5)
ax.set_ylim(min_mae - 0.5, max_mae + 0.5)
ax.set_aspect('equal', adjustable='box')

ax.plot([min_mae, max_mae], [min_mae, max_mae], linestyle="--", color="gray", label="y = x")

for _, row in df.iterrows():
    color = palette[row["cell_type"]]
    ax.scatter(row["MAE_yours"], row["MAE_paper"], color=color, label=row["cell_type"], s=60)

ax.set_xlabel("MAE (Our Model)")
ax.set_ylabel("MAE (Paper Model)")
ax.set_title("MAE Comparison: Our Model vs Paper Model")
configure_legend(ax)
plt.savefig(os.path.join(output_dir, "mae_scatter_by_celltype.png"), bbox_inches="tight", dpi=300)
plt.close()

plt.figure(figsize=(16, 16))
ax = plt.gca()

min_r = min(df["r_yours"].min(), df["r_paper"].min())
max_r = max(df["r_yours"].max(), df["r_paper"].max())
ax.set_xlim(min_r - 0.05, max_r + 0.05)
ax.set_ylim(min_r - 0.05, max_r + 0.05)
ax.set_aspect('equal', adjustable='box')

ax.plot([min_r, max_r], [min_r, max_r], linestyle="--", color="gray", label="y = x")

for _, row in df.iterrows():
    color = palette[row["cell_type"]]
    ax.scatter(row["r_yours"], row["r_paper"], color=color, label=row["cell_type"], s=60)

ax.set_xlabel("Pearson r (Our Model)")
ax.set_ylabel("Pearson r (Paper Model)")
ax.set_title("Correlation Comparison: Our Model vs Paper Model")
configure_legend(ax)
plt.savefig(os.path.join(output_dir, "correlation_scatter_by_celltype.png"), bbox_inches="tight", dpi=300)
plt.close()

print(f"Final plots saved to: {output_dir}")
