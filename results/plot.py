import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from results.scaling_laws import fit_func

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Charter"
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage[bitstream-charter]{mathdesign} \usepackage{amsmath}"
)
LEFT, BOTTOM, RIGHT, TOP = 0.16, 0.16, 0.95, 0.95
X_LABEL_POS, Y_LABEL_POS = -0.1, -0.15

FIGSIZE = (5, 5)
FONTSIZE = 15  # pt
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
    "axes.grid": False,
    "grid.color": "0.9",
    "axes.grid.which": "both",
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

colors = {
    "tr": "#E26D5C",
    "lloca": "#8C271E",
    "slim": "#419108",
    "part": "#E9C46A",
}
markers = {
    "tr": "o",
    "lloca": "D",
    "slim": "X",
    "part": "s",
}

labels = {
    "tr": "Transformer",
    "lloca": "LLoCa-Tr.",
    "slim": "L-GATr-slim",
    "part": "ParT",
}


def plot_metric(file, perf, cost, models, sizes, fit=None, quantile=0.3):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xscale("log")
    ax.set_xlabel(cost["label"], fontsize=FONTSIZE)
    ax.set_ylabel(perf["label"], fontsize=FONTSIZE)

    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax.xaxis.set_label_coords(0.5, X_LABEL_POS)
    ax.yaxis.set_label_coords(Y_LABEL_POS, 0.5)
    plt.subplots_adjust(LEFT, BOTTOM, RIGHT, TOP)

    for model in models:
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
            color=colors[model],
            marker=markers[model],
            label=labels[model],
            markersize=8,
            lw=0,
        )

    ax.legend(frameon=False)
    ax.relim()
    ax.autoscale_view()

    xrange = ax.get_xlim()
    yrange = ax.get_ylim()
    for model in models:
        if fit[model] is not None:
            x0 = np.exp(np.linspace(*[np.log(a) for a in xrange], 1000))

            params_best = [fit[model][key]["best"] for key in ["A", "B", "alpha"]]
            y_hat = fit_func(x0, *params_best)
            plt.plot(x0, y_hat, color=colors[model])

            params_all = [np.array(fit[model][key]["all"]) for key in ["A", "B", "alpha"]]
            y_all = fit_func(x0[:, None], *params_all)
            y_lower = np.quantile(y_all, quantile, axis=-1)
            y_upper = np.quantile(y_all, 1 - quantile, axis=-1)
            
            ax.fill_between(x0, y_lower, y_upper, edgecolor=colors[model], color=colors[model], alpha=0.2, lw=0.1)


        ax.set_xlim(xrange)
        ax.set_ylim(yrange)

    fig.savefig(file, format="pdf")
    plt.close()
