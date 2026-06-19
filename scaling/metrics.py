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

MIN_SPEEDUP = 1.05


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


def parse_mode(mode):
    flags = {}
    for token in mode.split(","):
        key = token[len("no-") :] if token.startswith("no-") else token
        flags[key] = not token.startswith("no-")
    return flags


def mode_key(flags):
    return ",".join(
        f"{'' if flags[k] else 'no-'}{k}" for k in ("zeropad", "amp", "compile") if k in flags
    )


def ablated_mode(best_mode, ablate):
    flags = parse_mode(best_mode)
    if ablate == "zeropad":
        return None if flags.get("zeropad", True) else mode_key({**flags, "zeropad": True})
    if ablate in ("amp", "compile"):
        return mode_key({**flags, ablate: False}) if flags.get(ablate, False) else None
    raise ValueError(ablate)


def extract_series(data, model, points, ablate=None, x_key="memory_alloc", y_key="mean"):
    x, y, ye_lo, ye_hi = [], [], [], []
    for p in points:
        entry = data[str(p)].get(model)
        sub = None
        if entry is not None:
            if ablate is None:
                sub = entry
            else:
                mode = ablated_mode(entry["best_mode"], ablate)
                sub = entry.get(mode) if mode is not None else None
                if sub is not None and sub["mean"] < MIN_SPEEDUP * entry["mean"]:
                    sub = None  # flipping this option barely changes the time
        if sub is None:
            x.append(np.nan)
            y.append(np.nan)
            ye_lo.append(np.nan)
            ye_hi.append(np.nan)
        else:
            x.append(sub.get(x_key, entry.get(x_key)))
            y.append(sub[y_key])
            ye_lo.append(sub.get("std_minus", 0.0))
            ye_hi.append(sub.get("std_plus", 0.0))
    return (np.array(v, dtype=float) for v in (x, y, ye_lo, ye_hi))


def plot_metric_scatter(
    filename,
    data,
    points,
    xlabel,
    ylabel,
    archs=None,
    x_key="memory_alloc",
    y_key="mean",
    ablate=None,
    series_labels=None,
    yerr=True,
    xscale="log",
    yscale="log",
):
    models = available_models(data, points, archs)
    if ablate is not None:
        # only keep models where flipping the option helps at some size
        def _helps(m):
            _, y, _, _ = extract_series(data, m, points, ablate=ablate, y_key=y_key)
            return np.isfinite(y).any()

        models = [m for m in models if _helps(m)]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.xaxis.set_label_coords(0.5, X_LABEL_POS)
    ax.yaxis.set_label_coords(Y_LABEL_POS, 0.5)
    plt.subplots_adjust(LEFT, BOTTOM, RIGHT, TOP)

    for model in models:
        for mode, ls in [(None, "-")] + ([(ablate, "--")] if ablate is not None else []):
            x, y, ye_lo, ye_hi = extract_series(
                data, model, points, ablate=mode, x_key=x_key, y_key=y_key
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
            for ls, lbl in zip(("-", "--"), series_labels, strict=True)
        ]
        ax.legend(handles=series_handles, loc="lower right", frameon=False)

    fig.savefig(filename, format="pdf")
    plt.close()
