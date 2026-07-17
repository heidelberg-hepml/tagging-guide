"""Quick-and-dirty placeholder: input-feature relevance for jetclass and jetset.

x-axis = input feature choice, y-axis = a performance metric, one marker (per
network) with errorbar at each x. Only model_size=0 is used; central value =
median over seeds, errorbar = std over seeds. All pages go into a single
features.pdf, one page per dataset and metric (jetclass AUC, jetclass loss,
jetset AUC, jetset loss). Run from repo root: python -m scaling.features
"""

import glob
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}"
)
LEFT, RIGHT, TOP = 0.21, 0.95, 0.95
XTICK_FONTSIZE = 11
X_LABEL_POS, Y_LABEL_POS = -0.1, -0.2
FIGSIZE = (5, 5)
FONTSIZE = 15
PAGEWIDTH = 11
MATPLOTLIB_PARAMS = {
    "font.size": FONTSIZE,
    "axes.titlesize": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "legend.fontsize": FONTSIZE,
    "figure.titlesize": FONTSIZE,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "figure.figsize": (PAGEWIDTH / 2, PAGEWIDTH / 2),
    "lines.markeredgewidth": 0.8,
    "axes.edgecolor": "black",
    "axes.grid": False,
    "grid.color": "0.9",
    "axes.grid.which": "both",
    "xtick.bottom": True,
    "xtick.direction": "out",
    "xtick.color": "black",
    "xtick.major.bottom": True,
    "xtick.major.size": 4,
    "xtick.minor.bottom": False,
    "ytick.left": True,
    "ytick.direction": "out",
    "ytick.color": "black",
    "ytick.major.left": True,
    "ytick.major.size": 4,
    "ytick.minor.left": True,
    "ytick.minor.size": 2,
}
matplotlib.rcParams.update(MATPLOTLIB_PARAMS)

colors = {
    "tr": "#E26D5C",
    "lloca": "#8C271E",
    "slim": "#1E838C",
    "part": "#E9C46A",
}
markers = {"tr": "o", "lloca": "D", "slim": "X", "part": "s"}
labels = {"tr": "Transformer", "lloca": "LLoCa-Tr.", "slim": "L-GATr-slim", "part": "ParT"}
MODEL_ORDER = ["tr", "part", "lloca", "slim"]

# (metric key in results_*.json, y-axis label) -> one PDF page each
METRICS = [("auc_ovo", r"AUC"), ("loss", r"loss")]

# (x tick label, glob pattern for size=0 runs across seeds), in x-axis order.
# Each incremental feature group goes on its own label row, so the nesting is
# visible without brackets.
JETCLASS = [
    (r"$p$", "runs/horeka3/v18_{m}_0_jetclass-fourmomenta_*"),
    (r"$p,$" "\n" r"$\mathrm{PID}$", "runs/horeka3/v20_{m}_0_jetclass-pid_*"),
    (r"$p,$" "\n" r"$d_0,\,d_z$", "runs/horeka3/v19_{m}_0_jetclass-displacements_*"),
    (r"$\mathrm{all}$", "runs/horeka3/v9_{m}_0_jetclass_*"),
]
JETCLASS_MODELS = ["tr", "part", "lloca", "slim"]  # lloca only ran for the "all" feature set

JETSET = [
    (r"$p,$" "\n" r"$S_{d_0},\,S_{z_0}$", "runs/horeka4/v14_{m}_0_jetset-ipsig_*"),
    (
        r"$p,$" "\n" r"$S_{d_0},\,S_{z_0},$" "\n" r"$d_0,\,z_0$",
        "runs/horeka4/v14_{m}_0_jetset-ip_*",
    ),
    (
        r"$p,$"
        "\n"
        r"$S_{d_0},\,S_{z_0},$"
        "\n"
        r"$d_0,\,z_0,$"
        "\n"
        r"$p_T^{\mathrm{rel}},\,\Delta R$",
        "runs/horeka4/v14_{m}_0_jetset-ipkin_*",
    ),
    (r"$\mathrm{all}$", "runs/horeka4/v14_{m}_0_jetset-all_*"),
]
JETSET_MODELS = ["tr", "part", "lloca", "slim"]


def collect(pattern, model, metric):
    """Median, std and count of metric over seeds for one (feature, model)."""
    values = []
    for run_dir in sorted(glob.glob(pattern.format(m=model))):
        for results_file in glob.glob(f"{run_dir}/results_*.json"):
            with open(results_file) as f:
                results = json.load(f)
            if metric in results:
                values.append(results[metric])
            break  # one results file per run
    if not values:
        return np.nan, np.nan, 0
    return float(np.median(values)), float(np.std(values)), len(values)


def plot_page(pdf, features, models, metric, ylabel):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax.yaxis.set_label_coords(Y_LABEL_POS, 0.5)
    # just enough bottom margin for the tallest (multi-row) x tick label
    max_rows = max(lab.count("\n") + 1 for lab, _ in features)
    bottom = 0.07 + 0.038 * max_rows
    plt.subplots_adjust(LEFT, bottom, RIGHT, TOP)

    xpos = np.arange(len(features))
    # horizontal dodge so the per-network markers at one x don't overlap
    offsets = np.linspace(-0.18, 0.18, len(models))

    for model in [m for m in MODEL_ORDER if m in models]:
        off = offsets[models.index(model)]
        x, y_med, y_std = [], [], []
        for i, (_, pattern) in enumerate(features):
            med, std, n = collect(pattern, model, metric)
            if n == 0:
                continue
            x.append(xpos[i] + off)
            y_med.append(med)
            y_std.append(std)
        ax.errorbar(
            x,
            y_med,
            yerr=y_std,
            color=colors[model],
            marker=markers[model],
            label=labels[model],
            markersize=8,
            lw=0,
            elinewidth=1.2,
            capsize=3,
        )

    ax.set_xticks(xpos)
    ax.set_xticklabels([lab for lab, _ in features], fontsize=XTICK_FONTSIZE)
    ax.set_xlim(-0.5, len(features) - 0.5)
    ax.legend(frameon=False)
    pdf.savefig(fig)
    plt.close()


def main():
    with PdfPages("scaling/features.pdf") as pdf:
        for metric, ylabel in METRICS:
            plot_page(pdf, JETCLASS, JETCLASS_MODELS, metric, ylabel)
        for metric, ylabel in METRICS:
            plot_page(pdf, JETSET, JETSET_MODELS, metric, ylabel)


if __name__ == "__main__":
    main()
