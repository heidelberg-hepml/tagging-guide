"""Quick-and-dirty placeholder: input-feature relevance for jetclass and jetset.

x-axis = input feature choice, y-axis = a performance metric, one marker (per
network) with errorbar at each x. Only model_size=0 is used; central value =
median over seeds, errorbar = std over seeds. All pages go into a single
features.pdf, one page per dataset and metric (jetclass AUC, jetclass loss,
jetset AUC, jetset loss). Run from repo root: python -m paper.features
"""

import glob
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}"
)
LEFT, RIGHT, TOP = 0.21, 0.95, 0.95
XTICK_FONTSIZE = 15
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
    # colors
    "axes.grid": True,
    "grid.color": "0.85",
    "axes.grid.which": "major",

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
    "lgatr": "#419108",
}
markers = {"tr": "o", "lloca": "D", "slim": "X", "lgatr": "p"}
labels = {"tr": "Transformer", "lloca": "LLoCa-Tr.", "slim": "L-GATr-slim", "lgatr": "L-GATr"}
MODEL_ORDER = ["tr", "lgatr", "lloca", "slim"]

# (metric key in results_*.json, y-axis label) -> one PDF page each
METRICS = [("auc_ovo", r"AUC"), ("loss", r"loss")]

# (x tick label, glob pattern for size=0 runs across seeds), in x-axis order.
# Each incremental feature group goes on its own label row, so the nesting is
# visible without brackets.
JETCLASS = [
    (r"$p$", "runs/jetclass_fourmomenta/v*_{m}_*"),
    (r"$p,$" "\n" r"$\mathrm{PID}$", "runs/jetclass_pid/v*_{m}_*"),
    (r"$p,$" "\n" r"$d_0,\,d_z$", "runs/jetclass_displacements/v*_{m}_*"),
    (r"$\mathrm{all}$", "runs/jetclass_all/v*_{m}_*"),
]
JETCLASS_MODELS = ["tr", "lgatr", "lloca", "slim"]  # lloca only ran for the "all" feature set

JETSET = [
    (r"$p$", "runs/jetset_fourmomenta/v*_{m}_*"),
    (r"$p,$" "\n" r"$S_{d_0},\,S_{z_0}$", "runs/jetset_ipsig/v*_{m}_*"),
    (
        r"$p, S_{d_0},\,S_{z_0},$" "\n" r"$d_0,\,z_0$",
        "runs/jetset_ip/v*_{m}_*",
    ),
    (r"$\mathrm{all}$", "runs/jetset_all/v*_{m}_*"),
]
JETSET_MODELS = ["tr", "lgatr", "lloca", "slim"]


def collect(pattern, model, metric):
    """Median, std and count of metric over seeds for one (feature, model)."""
    values = []
    for run_dir in sorted(glob.glob(pattern.format(m=model))):
        for results_file in glob.glob(f"{run_dir}/*/results_*.json"):
            with open(results_file) as f:
                results = json.load(f)
            if metric in results:
                values.append(results[metric])
            break  # one results file per run
    if not values:
        return np.nan, np.nan, 0
    return float(np.median(values)), float(np.std(values)), len(values)


def plot_page(pdf, features, models, metric, ylabel, broken=False):
    xpos = np.arange(len(features))
    # horizontal dodge so the per-network markers at one x don't overlap
    offsets = np.linspace(-0.18, 0.18, len(models))

    series = {}
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
        series[model] = (x, y_med, y_std)

    def draw(ax):
        for model in [m for m in MODEL_ORDER if m in models]:
            x, y_med, y_std = series[model]
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

    # just enough bottom margin for the tallest (multi-row) x tick label
    max_rows = max(lab.count("\n") + 1 for lab, _ in features)
    bottom = 0.07 + 0.038 * max_rows

    if not broken:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set_ylabel(ylabel, fontsize=FONTSIZE)
        ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
        ax.yaxis.set_label_coords(Y_LABEL_POS, 0.5)
        plt.subplots_adjust(LEFT, bottom, RIGHT, TOP)
        draw(ax)
        ax.set_xticks(xpos)
        ax.set_xticklabels([lab for lab, _ in features], fontsize=XTICK_FONTSIZE)
        ax.set_xlim(-0.5, len(features) - 0.5)
        ax.legend(frameon=False)
        pdf.savefig(fig)
        plt.close()
        return

    spans = sorted(
        (med - std, med + std)
        for _, y_med, y_std in series.values()
        for med, std in zip(y_med, y_std)
    )
    gaps = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]
    split = max(range(len(gaps)), key=lambda i: gaps[i])
    pad_lo = 0.08 * (spans[split][1] - spans[0][0])
    pad_hi = 0.08 * (spans[-1][1] - spans[split + 1][0])

    if metric == "auc_ovo":
        fig, (ax_hi, ax_lo) = plt.subplots(
            2,
            1,
            figsize=FIGSIZE,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        )
    elif metric == "loss":
        fig, (ax_hi, ax_lo) = plt.subplots(
            2,
            1,
            figsize=FIGSIZE,
            sharex=True,
            gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08},
        )

    plt.subplots_adjust(LEFT, bottom, RIGHT, TOP)
    fig.supylabel(ylabel, fontsize=FONTSIZE)

    draw(ax_hi)
    draw(ax_lo)
    if metric == "auc_ovo":
        ax_hi.set_ylim(spans[split + 1][0] - pad_hi, spans[-1][1] + pad_hi)
        ax_lo.set_ylim(spans[0][0] - 0.004, spans[split][1] + 0.004)
        ax_lo.yaxis.set_major_locator(MaxNLocator(2))
    elif metric == "loss":
        ax_hi.set_ylim(spans[split + 1][0] - 0.015, spans[-1][1] + 0.015)
        ax_lo.set_ylim(spans[0][0] - pad_lo, spans[split][1] + pad_lo)
        ax_hi.yaxis.set_major_locator(MaxNLocator(2))
    ax_hi.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax_lo.tick_params(axis="both", which="major", labelsize=FONTSIZE)

    ax_hi.spines["bottom"].set_visible(False)
    ax_lo.spines["top"].set_visible(False)
    ax_hi.tick_params(bottom=False, labelbottom=False)

    # slanted marks on both sides of the break
    d = 0.5
    break_kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="black",
        mec="black",
        mew=1,
        clip_on=False,
    )
    ax_hi.plot([0, 1], [0, 0], transform=ax_hi.transAxes, **break_kwargs)
    ax_lo.plot([0, 1], [1, 1], transform=ax_lo.transAxes, **break_kwargs)

    ax_lo.set_xticks(xpos)
    ax_lo.set_xticklabels([lab for lab, _ in features], fontsize=XTICK_FONTSIZE)
    ax_lo.set_xlim(-0.5, len(features) - 0.5)
    ax_lo.legend(frameon=False, loc=(0.45, 0.1))
    pdf.savefig(fig)
    plt.close()


def main():
    with PdfPages("paper/features.pdf") as pdf:
        for metric, ylabel in METRICS:
            plot_page(pdf, JETCLASS, JETCLASS_MODELS, metric, ylabel)
        for metric, ylabel in METRICS:
            plot_page(pdf, JETSET, JETSET_MODELS, metric, ylabel, broken=True)


if __name__ == "__main__":
    main()
