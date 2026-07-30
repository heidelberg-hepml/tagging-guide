"""Per-step training tracking for the LLoCa frame-normalization study (appendix A).

Reads training_tracking_0.npz of the six runs in runs/itp4, i.e. Transformer,
LLoCa-Tr. with raw frames (preserve_variance=false) and LLoCa-Tr. with frames
divided by the boost factor (preserve_variance=true), each at size -2 and 2.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator

from paper.plot import (
    BOTTOM,
    FIGSIZE,
    FONTSIZE,
    LEFT,
    LEGEND_KWARGS,
    RIGHT,
    TOP,
    X_LABEL_POS,
    Y_LABEL_POS,
    colors,
    labels,
    place_labels,
)

matplotlib.rcParams["axes.grid"] = False

BASE = "runs/itp4"
OUT = "paper/tracking.pdf"

MODELS = [
    ("tr", labels["tr"], colors["tr"]),
    ("lloca_novar", labels["lloca"] + r" ($L$)", "#2A6F97"),
    ("lloca", labels["lloca"] + r" ($L/\gamma$)", colors["lloca"]),
]
RUN_DIRS = {
    ("lloca", -2): "v1_appendixA_lloca_-2-new_1",
    ("lloca", 2): "v1_appendixA_lloca_2-new_1",
    ("lloca_novar", -2): "v1_appendixA_lloca_-2-old_1",
    ("lloca_novar", 2): "v1_appendixA_lloca_2-old_1",
    ("tr", -2): "v1_appendixA_tr_-2_1",
    ("tr", 2): "v1_appendixA_tr_2_1",
}
SIZES = [-2, 2]

# metric, axis label, log y-axis, and a floor that cuts away the initial transient
METRICS = [
    ("loss", "Training loss", True, None),
    ("grad_norm", "Gradient norm", True, None),
    ("logits_var", "Logits variance", True, 1.0),
    ("boost_median", "Median boost factor", False, None),
]

ITERATIONS = 250000
XTICKS, XTICKS_MINOR = [0, 100000, 200000], 50000
N_BINS = 1000
LINEWIDTH = 2.0
# LEGEND_KWARGS is too tight for these labels
LEGEND_SPACING = {"handlelength": 1.6, "handletextpad": 0.5}

# axis margins in units of the axis height, and the clearance kept around legend and label
Y_BOTTOM, Y_TOP, OBSTACLE_PAD = 0.06, 0.03, 0.02

# below this many decades powers of ten alone would leave a single tick label
PLAIN_LOG_DECADES = 2.0


def load():
    data = {}
    for key, run in RUN_DIRS.items():
        with np.load(os.path.join(BASE, run, "training_tracking_0.npz")) as npz:
            data[key] = {k: npz[k] for k in npz.files}
    return data


def binmean(a, n_bins=N_BINS):
    n = len(a) // n_bins * n_bins
    x = (np.arange(n_bins) + 0.5) * (n / n_bins)
    return x, a[:n].reshape(n_bins, -1).mean(axis=1)


def kfmt(x, _pos):
    return "0" if x == 0 else f"{x / 1000:g}k"


def plain_log_fmt(v, _pos):
    """Label only the 1, 2 and 5 ticks of each decade, to keep the axis readable."""
    mantissa = round(v / 10 ** np.floor(np.log10(v) + 1e-9))
    return f"{v:g}" if mantissa in (1, 2, 5) else ""


def format_log_axis(ax):
    lo, hi = ax.get_ylim()
    if np.log10(hi / lo) >= PLAIN_LOG_DECADES:
        return
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_log_fmt))
    ax.yaxis.set_minor_formatter(FuncFormatter(plain_log_fmt))


def box_in_axes(fig, artist, ax):
    fig.draw_without_rendering()
    return artist.get_window_extent().transformed(ax.transAxes.inverted())


def in_span(xs, ys, box):
    """The parts of the curves that run through the horizontal span of a box."""
    inside = [
        y[(x >= box.x0 - OBSTACLE_PAD) & (x <= box.x1 + OBSTACLE_PAD)]
        for x, y in zip(xs, ys, strict=True)
    ]
    return [y for y in inside if y.size > 0]


def fit_ylim(xs, ys, logy, legend, corner, floor=None):
    """Smallest y-range that keeps every curve below the top margin, below the
    legend where it passes underneath it, and above the corner label."""
    lo = min(np.min(y) for y in ys)
    if floor is not None:
        lo = max(lo, floor)
    tops = [(max(np.max(y) for y in ys), 1 - Y_TOP)]
    if inside := in_span(xs, ys, legend):
        tops.append((max(np.max(y) for y in inside), legend.y0 - OBSTACLE_PAD))
    bottoms = []
    if inside := in_span(xs, ys, corner):
        # a curve cut off by the floor does not constrain the label below it
        bottoms.append((max(min(np.min(y) for y in inside), lo), corner.y1))
    if logy:
        lo = np.log10(lo)
        tops = [(np.log10(v), frac) for v, frac in tops]
        bottoms = [(np.log10(v), frac) for v, frac in bottoms]

    # with y_frac = (v - lo + bottom * span) / span, an obstacle at frac either caps
    # the curves above it, span >= (v - lo) / (frac - bottom), or pushes the axis down,
    # bottom >= frac - (v - lo) / span; the two are resolved by iterating
    bottom = Y_BOTTOM
    for _ in range(10):
        span = max((v - lo) / (frac - bottom) for v, frac in tops)
        needed = [frac + OBSTACLE_PAD - (v - lo) / span for v, frac in bottoms]
        bottom_new = max([Y_BOTTOM, *needed])
        if abs(bottom_new - bottom) < 1e-4:
            break
        bottom = bottom_new
    lo, hi = lo - bottom * span, lo + (1 - bottom) * span
    return (10**lo, 10**hi) if logy else (lo, hi)


def plot_metric(pdf, data, metric, ylabel, logy, floor, size):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlabel("Iteration", fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.xaxis.set_label_coords(0.5, X_LABEL_POS)
    ax.yaxis.set_label_coords(Y_LABEL_POS, 0.5)
    plt.subplots_adjust(LEFT, BOTTOM, RIGHT, TOP)

    xs, ys = [], []
    for model, label, color in MODELS:
        x, y = binmean(data[(model, size)][metric])
        ax.plot(x, y, color=color, lw=LINEWIDTH, label=label)
        xs.append(x / ITERATIONS)
        ys.append(y)

    if logy:
        ax.set_yscale("log")
    ax.set_xlim(0, ITERATIONS)
    ax.set_xticks(XTICKS)
    ax.xaxis.set_minor_locator(MultipleLocator(XTICKS_MINOR))
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))

    legend = ax.legend(loc="upper right", **LEGEND_KWARGS | LEGEND_SPACING)
    corner = ax.text(
        0.03,
        0.03,
        rf"$s={size}$",
        transform=ax.transAxes,
        fontsize=FONTSIZE,
        ha="left",
        va="bottom",
    )
    boxes = [box_in_axes(fig, artist, ax) for artist in (legend, corner)]
    ax.set_ylim(*fit_ylim(xs, ys, logy, *boxes, floor=floor))
    if logy:
        format_log_axis(ax)

    place_labels(fig, ax)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    data = load()
    with PdfPages(OUT) as pdf:
        for metric, ylabel, logy, floor in METRICS:
            for size in SIZES:
                plot_metric(pdf, data, metric, ylabel, logy, floor, size)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
