"""Plotting helpers for DiffTernLGN notebooks.

Re-exports the figure utilities from ``pst_dtlgn.viz.plots`` so notebooks can do
``from pst_dtlgn.viz import save_fig, setup_plot_theme``.
"""

from pst_dtlgn.viz.plots import (
    save_fig,
    setup_plot_theme,
    make_grid,
    decision_boundary_colormaps,
)

__all__ = [
    "save_fig",
    "setup_plot_theme",
    "make_grid",
    "decision_boundary_colormaps",
]
