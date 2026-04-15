"""Shared ATLAS mplhep style for all scaling law plots.

Usage:
    from plot_style import setup_style, SCATTER_KW, FILL_ALPHA, CMAP
    setup_style()
"""

import mplhep as hep
import matplotlib.pyplot as plt


def setup_style():
    hep.style.use(hep.style.ATLAS)
    plt.rcParams.update({
        'figure.figsize': (13, 9),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'lines.linewidth': 2.5,
        'lines.markersize': 10,
        'axes.labelsize': 28,
        'legend.fontsize': 22,
        'legend.frameon': True,
        'legend.facecolor': 'white',
        'legend.framealpha': 1.0,
        'legend.edgecolor': 'black',
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'font.size': 24,
        'xaxis.labellocation': 'center',
        'yaxis.labellocation': 'center',
        'savefig.bbox': 'tight',
    })


# Reusable styling constants
SCATTER_KW = dict(edgecolors='black', linewidths=0.8)
FILL_ALPHA = 0.15
CMAP = 'magma'
