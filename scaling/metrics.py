import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .plot import (
    BOTTOM,
    FIGSIZE,
    FONTSIZE,
    LEFT,
    MODEL_ORDER,
    RIGHT,
    TOP,
    X_LABEL_POS,
    Y_LABEL_POS,
    colors,
    labels,
    markers,
)


def load_data(file):
    with open(file) as f:
        return json.load(f)


def merge_data(*sources):
    merged = {}
    for data in sources:
        for point, models in data.items():
            if not isinstance(models, dict):
                continue
            dst = merged.setdefault(point, {})
            for model, entry in models.items():
                if isinstance(entry, dict):
                    dst.setdefault(model, {}).update(entry)
    return merged


def available_models(data, points, archs=None):
    models = set(MODEL_ORDER if archs is None else archs)
    present = {m for p in points for m in data[str(p)] if m in models}
    return [m for m in MODEL_ORDER if m in present]


def extract_series(data, model, points, mode=None, x_key="memory_alloc", y_key="mean"):
    x, y, ye_lo, ye_hi = [], [], [], []
    for p in points:
        entry = data[str(p)].get(model)
        if entry is None:
            continue
        if mode is not None:
            entry = entry[mode]
        x.append(entry[x_key])
        y.append(entry[y_key])
        ye_lo.append(entry.get("std_minus", 0.0))
        ye_hi.append(entry.get("std_plus", 0.0))
    return (np.array(v) for v in (x, y, ye_lo, ye_hi))


def plot_metric_scatter(
    filename,
    data,
    points,
    xlabel,
    ylabel,
    archs=None,
    x_key="memory_alloc",
    y_key="mean",
    series=(("", "-"),),
    series_labels=None,
    skip=None,
    yerr=True,
    xscale="log",
    yscale="log",
):
    models = available_models(data, points, archs)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.xaxis.set_label_coords(0.5, X_LABEL_POS)
    ax.yaxis.set_label_coords(Y_LABEL_POS, 0.5)
    plt.subplots_adjust(LEFT, BOTTOM, RIGHT, TOP)

    for model in models:
        for mode, ls in series:
            if skip is not None and skip(model, mode):
                continue
            x, y, ye_lo, ye_hi = extract_series(
                data, model, points, mode=mode or None, x_key=x_key, y_key=y_key
            )
            ax.errorbar(
                x,
                y,
                yerr=np.stack([ye_lo, ye_hi]) if yerr else None,
                color=colors[model],
                marker=markers[model],
                linestyle=ls,
                markersize=8,
                lw=1.2,
            )

    model_handles = [
        Line2D(
            [0],
            [0],
            color=colors[m],
            marker=markers[m],
            linestyle="-",
            lw=1.2,
            markersize=8,
            label=labels[m],
        )
        for m in models
    ]
    leg1 = ax.legend(handles=model_handles, loc="upper left", frameon=False)
    ax.add_artist(leg1)

    if series_labels is not None:
        series_handles = [
            Line2D([0], [0], color="gray", linestyle=ls, lw=1.2, label=lbl)
            for (_, ls), lbl in zip(series, series_labels, strict=True)
        ]
        ax.legend(handles=series_handles, loc="lower right", frameon=False)

    fig.savefig(filename, format="pdf")
    plt.close()
