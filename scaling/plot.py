import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .scaling_laws import fit_func

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}"
)
LEFT, BOTTOM, RIGHT, TOP = 0.26, 0.20, 0.95, 0.95
X_LABEL_POS, Y_LABEL_POS = -0.14, -0.26

FIGSIZE = (5, 5)
FONTSIZE = 20  # pt
PARETO_ALPHA = 0.075
Y_HEADROOM = 0.15
LEGEND_KWARGS = {
    "frameon": False,
    "labelspacing": 0.2,
    "handlelength": 1.0,
    "handletextpad": 0.1,
    "borderpad": 0.0,
    # matches the 0.03 axes-fraction inset of the dataset label, in font-size units
    "borderaxespad": 0.375,
}
PAGEWIDTH = 11  # inches
MATPLOTLIB_PARAMS = {
    # Font sizes
    "font.size": FONTSIZE,  # controls default text sizes
    "axes.titlesize": FONTSIZE,  # fontsize of the axes title
    "axes.labelsize": FONTSIZE,  # fontsize of the x and y labels
    "xtick.labelsize": FONTSIZE,  # fontsize of the tick labels
    "ytick.labelsize": FONTSIZE,  # fontsize of the tick labels
    "legend.fontsize": FONTSIZE,  # legend fontsize
    "figure.titlesize": FONTSIZE,  # fontsize of the figure title
    # Figure size and DPI
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "figure.figsize": (PAGEWIDTH / 2, PAGEWIDTH / 2),
    # colors
    "lines.markeredgewidth": 0.8,
    "axes.edgecolor": "black",
    "axes.grid": True,
    "grid.color": "0.85",
    "axes.grid.which": "major",
    # x-axis ticks and grid
    "xtick.bottom": True,
    "xtick.direction": "out",
    "xtick.color": "black",
    "xtick.major.bottom": True,
    "xtick.major.size": 4,
    "xtick.minor.bottom": True,
    "xtick.minor.size": 2,
    # y-axis ticks and grid
    "ytick.left": True,
    "ytick.direction": "out",
    "ytick.color": "black",
    "ytick.major.left": True,
    "ytick.major.size": 4,
    "ytick.minor.left": True,
    "ytick.minor.size": 2,
}
matplotlib.rcParams.update(MATPLOTLIB_PARAMS)

MODEL_ORDER = [
    "lgatr",
    "lgatr-sparse",
    "slim",
    "lloca",
    "part",
    "tr",
    "gn3",
    "pelicanlite",
    "lorentznet",
    "particlenet",
]

colors = {
    "tr": "#E26D5C",
    "gn3": "#4C6E91",
    "lloca": "#8C271E",
    "slim": "#1E838C",
    "lgatr": "#419108",
    "lgatr-sparse": "#A2C523",
    "part": "#E9C46A",
    "pelicanlite": "#D97706",
    "lorentznet": "#6A4C93",
    "particlenet": "#7F7F7F",
}
markers = {
    "tr": "o",
    "gn3": "^",
    "lloca": "D",
    "slim": "X",
    "lgatr": "p",
    "lgatr-sparse": "h",
    "part": "s",
    "pelicanlite": "v",
    "lorentznet": "*",
    "particlenet": "P",
}

DATASET_LABELS = {
    "jetclass": "JetClass",
    "jetset": "JetSet",
    "atlastop": "ATLASTop",
    "toptagxl": "TopTagXL",
}

# the dense/sparse distinction only matters in the cost plots
PERF_MODEL_LABELS = {"lgatr": "L-GATr"}

# axis labels for the cost metrics, shared by scan_cost.py and cost.py
COST_LABELS = {
    "params": "Network parameters",
    "flops_measured": "Inference FLOPs",
    "flops_estimated": "Inference FLOPs (est.)",
    "energy": "GPU inference energy [J]",
    "inference_cpu": "CPU inference time [ms]",
    "memory_cpu": "CPU memory usage [GB]",
    "inference_gpu_bs512": "GPU inference time [ms]",
    "memory_gpu_bs512": "GPU memory usage [GB]",
    "train_gpu_bs512": "GPU training time [ms]",
    "memory_train_gpu_bs512": "GPU training memory usage [GB]",
}

labels = {
    "tr": "Transformer",
    "gn3": "Salt/GN3",
    "lloca": "LLoCa-Tr.",
    "slim": "L-GATr-slim",
    "lgatr": r"L-GATr$_\mathrm{dense}$",
    "lgatr-sparse": r"L-GATr$_\mathrm{sparse}$",
    "part": "ParT",
    "pelicanlite": "PELICAN-lite",
    "lorentznet": "LorentzNet",
    "particlenet": "ParticleNet",
}


def dataset_label(name):
    for key, label in DATASET_LABELS.items():
        if name.startswith(key):
            return label
    return None


def place_labels(fig, ax):
    """Shift long axis labels along their axis so that they stay inside the figure."""
    fig.draw_without_rendering()
    for axis, span, along_x in ((ax.xaxis, RIGHT - LEFT, True), (ax.yaxis, TOP - BOTTOM, False)):
        bbox = axis.label.get_window_extent().transformed(fig.transFigure.inverted())
        lo, hi = (bbox.x0, bbox.x1) if along_x else (bbox.y0, bbox.y1)
        shift = max(0.0, -lo) - max(0.0, hi - 1.0)
        if shift == 0.0:
            continue
        if along_x:
            axis.set_label_coords(0.5 + shift / span, X_LABEL_POS)
        else:
            axis.set_label_coords(Y_LABEL_POS, 0.5 + shift / span)


def draw_corner_label(ax, text, sublabel=None, lower=False, right=False):
    """Short label in the corner opposite the legend, with an optional second line."""
    if text is None:
        return
    ax.text(
        0.97 if right else 0.03,
        0.03 if lower else 0.97,
        text if sublabel is None else f"{text}\n{sublabel}",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
        ha="right" if right else "left",
        va="bottom" if lower else "top",
        multialignment="right" if right else "left",
        linespacing=1.4,
    )


def add_headroom(ax, top=True):
    """Extra space on the side of the y-axis where the legend sits, to reduce overlap."""
    log = ax.get_yscale() == "log"
    lo, hi = ax.get_ylim()
    if log:
        lo, hi = np.log10(lo), np.log10(hi)
    pad = Y_HEADROOM * (hi - lo)
    lo, hi = (lo, hi + pad) if top else (lo - pad, hi)
    ax.set_ylim(10**lo if log else lo, 10**hi if log else hi)


def plot_pareto(ax, x, curves, yrange, decreasing):
    """Envelope of the fitted scaling laws, i.e. the best metric reachable at a given cost."""
    envelope = np.min(curves, axis=0) if decreasing else np.max(curves, axis=0)
    # the region away from the envelope is achievable, above it for a loss and below otherwise
    ax.fill_between(
        x,
        envelope,
        yrange[1] if decreasing else yrange[0],
        color="k",
        alpha=PARETO_ALPHA,
        lw=0,
        zorder=0.5,
    )
    ax.plot(x, envelope, color="k", alpha=0.3, lw=1, zorder=0.5)


def plot_metric(
    file,
    perf,
    cost,
    models,
    sizes,
    fit=None,
    quantile=0.1,
    dataset=None,
    sublabel=None,
    decreasing=False,
    label_lower_left=None,
    model_labels=None,
    pareto=False,
):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlabel(cost["label"], fontsize=FONTSIZE)
    ax.set_ylabel(perf["label"], fontsize=FONTSIZE)

    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax.xaxis.set_label_coords(0.5, X_LABEL_POS)
    ax.yaxis.set_label_coords(Y_LABEL_POS, 0.5)
    plt.subplots_adjust(LEFT, BOTTOM, RIGHT, TOP)

    variants = cost.get("variants", {})
    for model in models:
        variant = variants.get(model, model)
        x = np.full(len(sizes), np.nan)
        y_mean, y_std, y_pls, y_mns = x.copy(), x.copy(), x.copy(), x.copy()
        for i, size in enumerate(sizes):
            x[i] = cost[model].get(size, np.nan)
            y = perf[model].get(size, np.nan)
            if len(y) > 0:
                y_mean[i] = np.mean(y, axis=-1)
                y_std[i] = np.std(y, axis=-1)
                y_pls[i], y_mns[i] = y_mean[i] + y_std[i], y_mean[i] - y_std[i]

        ax.errorbar(
            x,
            y_mean,
            yerr=y_std,
            color=colors[variant],
            marker=markers[variant],
            label=(model_labels or {}).get(variant, labels[variant]),
            markersize=8,
            lw=0,
        )

    ax.set_xscale("log")
    lower = decreasing if label_lower_left is None else label_lower_left
    draw_corner_label(ax, dataset, sublabel=sublabel, lower=lower)
    ax.legend(loc="upper right" if decreasing else "lower right", **LEGEND_KWARGS)
    ax.relim()
    ax.autoscale_view()
    add_headroom(ax, top=decreasing)

    xrange = ax.get_xlim()
    yrange = ax.get_ylim()
    x0 = np.exp(np.linspace(*[np.log(a) for a in xrange], 1000))
    curves = []
    for model in models:
        if fit[model] is not None:
            params_best = [fit[model][key]["best"] for key in ["L_inf", "B", "beta"]]
            y_hat = fit_func(x0, *params_best)
            curves.append(y_hat)
            plt.plot(x0, y_hat, color=colors[variants.get(model, model)])

            params_all = [np.array(fit[model][key]["all"]) for key in ["L_inf", "B", "beta"]]
            y_all = fit_func(x0[:, None], *params_all)
            y_lower = np.quantile(y_all, quantile, axis=-1)
            y_upper = np.quantile(y_all, 1 - quantile, axis=-1)

            ax.fill_between(
                x0,
                y_lower,
                y_upper,
                edgecolor=colors[variants.get(model, model)],
                color=colors[variants.get(model, model)],
                alpha=0.2,
                lw=0.1,
            )

        ax.set_xlim(xrange)
        ax.set_ylim(yrange)

    if pareto and curves:
        plot_pareto(ax, x0, curves, yrange, decreasing)

    place_labels(fig, ax)
    fig.savefig(file, format="pdf")
    plt.close()
