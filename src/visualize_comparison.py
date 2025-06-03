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
        bbox_to_anchor=(1.0, 1.0), loc="upper left",
        title="Cell Type", fontsize=8, title_fontsize=9,
        ncol=1, borderaxespad=0
    )

plt.figure(figsize=(11, 8))
ax = plt.gca()

min_mae = min(df["MAE_yours"].min(), df["MAE_paper"].min())
max_mae = max(df["MAE_yours"].max(), df["MAE_paper"].max())
ax.set_xlim(min_mae - 0.5, max_mae + 0.5)
ax.set_ylim(min_mae - 0.5, max_mae + 0.5)

for _, row in df.iterrows():
    color = palette[row["cell_type"]]
    x, y = row["MAE_yours"], row["MAE_paper"]
    ax.scatter(x, y, color=color, label=row["cell_type"], s=40)
    ax.vlines(x=x, ymin=ax.get_ylim()[0], ymax=y, color=color, linestyle=":", linewidth=1)
    ax.hlines(y=y, xmin=ax.get_xlim()[0], xmax=x, color=color, linestyle=":", linewidth=1)

ax.plot([min_mae, max_mae], [min_mae, max_mae], linestyle="--", color="gray")
ax.set_xlabel("MAE (Our Model)")
ax.set_ylabel("MAE (Paper Model)")
ax.set_title("MAE Comparison: Our Model vs Paper Model")
configure_legend(ax)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(os.path.join(output_dir, "mae_scatter_by_celltype.png"))
plt.close()

plt.figure(figsize=(11, 8))
ax = plt.gca()

min_r = min(df["r_yours"].min(), df["r_paper"].min())
max_r = max(df["r_yours"].max(), df["r_paper"].max())
ax.set_xlim(min_r - 0.1, max_r + 0.1)
ax.set_ylim(min_r - 0.1, max_r + 0.1)

for _, row in df.iterrows():
    color = palette[row["cell_type"]]
    x, y = row["r_yours"], row["r_paper"]
    ax.scatter(x, y, color=color, label=row["cell_type"], s=40)
    ax.vlines(x=x, ymin=ax.get_ylim()[0], ymax=y, color=color, linestyle=":", linewidth=1)
    ax.hlines(y=y, xmin=ax.get_xlim()[0], xmax=x, color=color, linestyle=":", linewidth=1)

ax.plot([min_r, max_r], [min_r, max_r], linestyle="--", color="gray")
ax.set_xlabel("Pearson r (Our Model)")
ax.set_ylabel("Pearson r (Paper Model)")
ax.set_title("Correlation Comparison: Our Model vs Paper Model")
configure_legend(ax)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(os.path.join(output_dir, "correlation_scatter_by_celltype.png"))
plt.close()

print(f"Final plots saved with axis lines in: {output_dir}")
