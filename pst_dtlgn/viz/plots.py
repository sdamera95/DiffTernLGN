"""Plotting helpers shared across the experiment notebooks.

Matplotlib/seaborn are imported lazily inside each function so importing this
module does not pull in a plotting backend.
"""

import os

import numpy as np


# ------------------------------------------------------------------
# figure saving and theme
def save_fig(fig, name, plots_dir="plots"):
    """Save a figure as both .svg and .pdf into plots_dir (tight bbox)."""
    for ext in ["svg", "pdf"]:
        fig.savefig(os.path.join(plots_dir, f"{name}.{ext}"), bbox_inches="tight")
    print(f"  Saved {name}")


def setup_plot_theme(font_scale=1.1, fig_dpi=150, savefig_dpi=300):
    """Apply the seaborn whitegrid theme and matplotlib DPI used in the notebooks."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=font_scale)
    plt.rcParams["figure.dpi"] = fig_dpi
    plt.rcParams["savefig.dpi"] = savefig_dpi


# ------------------------------------------------------------------
# decision-boundary support
def make_grid(raw_data, resolution=150):
    """Create a 2D meshgrid spanning the dataset range with a 0.3 margin.

    Returns (xx, yy, grid_points) where grid_points is float32 of shape (M, 2).
    """
    x_all = np.vstack([raw_data["train_x"], raw_data["test_x"]])
    margin = 0.3
    x_min, x_max = x_all[:, 0].min() - margin, x_all[:, 0].max() + margin
    y_min, y_max = x_all[:, 1].min() - margin, x_all[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid_points = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    return xx, yy, grid_points


def decision_boundary_colormaps():
    """Return (ter_cmap, ter_norm, bin_cmap, bin_norm) for decision-boundary fills.

    Ternary: -1 red, 0 light gray, +1 blue. Binary: 0 red, 1 blue.
    """
    import matplotlib.colors as mcolors

    ter_cmap = mcolors.ListedColormap(["#d73027", "#e0e0e0", "#4575b4"])
    ter_norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], ter_cmap.N)
    bin_cmap = mcolors.ListedColormap(["#d73027", "#4575b4"])
    bin_norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], bin_cmap.N)
    return ter_cmap, ter_norm, bin_cmap, bin_norm
