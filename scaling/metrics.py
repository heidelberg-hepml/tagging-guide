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
    return ",".join(f"{'' if v else 'no-'}{k}" for k, v in flags.items())


def ablated_mode(best_mode, ablate):
    flags = parse_mode(best_mode)
    if ablate == "zeropad":
        return None if flags.get("zeropad", True) else mode_key({**flags, "zeropad": True})
    if ablate in ("amp", "compile"):
        return mode_key({**flags, ablate: False}) if flags.get(ablate, False) else None
    raise ValueError(ablate)


def extract_series(data, model, points, x_key="memory_alloc", y_key="mean"):
    x, y, ye_lo, ye_hi = [], [], [], []
    for p in points:
        entry = data[str(p)].get(model)
        if entry is None:
            x.append(np.nan)
            y.append(np.nan)
            ye_lo.append(np.nan)
            ye_hi.append(np.nan)
        else:
            x.append(entry.get(x_key))
            y.append(entry[y_key])
            ye_lo.append(entry.get("std_minus", 0.0))
            ye_hi.append(entry.get("std_plus", 0.0))
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
    yerr=True,
    xscale="log",
    yscale="log",
):
    models = available_models(data, points, archs)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.xaxis.set_label_coords(0.5, X_LABEL_POS)
    ax.yaxis.set_label_coords(Y_LABEL_POS, 0.5)
    plt.subplots_adjust(LEFT, BOTTOM, RIGHT, TOP)

    for model in models:
        x, y, ye_lo, ye_hi = extract_series(data, model, points, x_key=x_key, y_key=y_key)
        ax.errorbar(
            x,
            y,
            yerr=np.stack([ye_lo, ye_hi]) if yerr else None,
            color=colors[model],
            marker=markers[model],
            markersize=8,
            lw=1.2,
        )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

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
    ax.legend(handles=model_handles, loc="upper left", frameon=False)

    fig.savefig(filename, format="pdf")
    plt.close()


GAIN_IMPROVES_WHEN = {"compile": True, "amp": True, "zeropad": False}


def _flag_modes(best_mode, ablate):
    flags = parse_mode(best_mode)
    on = GAIN_IMPROVES_WHEN[ablate]
    return mode_key({**flags, ablate: on}), mode_key({**flags, ablate: not on})


def _technique_relevant(data, model, points, ablate):
    # is this flag ever the network's own best choice at some size?
    for p in points:
        entry = data[str(p)].get(model)
        if entry is not None and ablated_mode(entry["best_mode"], ablate) is not None:
            return True
    return False


def _has_meaningful_gain(gain, threshold=MIN_SPEEDUP):
    finite = gain[np.isfinite(gain)]
    return finite.size > 0 and (np.any(finite > threshold) or np.any(finite < 1 / threshold))


def extract_gain(data, model, points, ablate, x_key="params", y_key="mean"):
    x, gain = [], []
    for p in points:
        entry = data[str(p)].get(model)
        xv, g = np.nan, np.nan
        if entry is not None:
            better_mode, worse_mode = _flag_modes(entry["best_mode"], ablate)
            better, worse = entry.get(better_mode), entry.get(worse_mode)
            if better is not None and worse is not None:
                xv, g = entry[x_key], worse[y_key] / better[y_key]
        x.append(xv)
        gain.append(g)
    return (np.array(v, dtype=float) for v in (x, gain))


def plot_gain_scatter(
    filename,
    data,
    points,
    xlabel,
    ylabel,
    ablate,
    archs=None,
    x_key="params",
    y_key="mean",
    reduction=False,
    xscale="log",
):
    models = available_models(data, points, archs)
    models = [m for m in models if _technique_relevant(data, m, points, ablate)]
    series = {m: tuple(extract_gain(data, m, points, ablate, x_key, y_key)) for m in models}
    models = [m for m in models if _has_meaningful_gain(series[m][1])]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.xaxis.set_label_coords(0.5, X_LABEL_POS)
    ax.yaxis.set_label_coords(Y_LABEL_POS, 0.5)
    plt.subplots_adjust(LEFT, BOTTOM, RIGHT, TOP)

    ax.axhline(0.0 if reduction else 1.0, color="gray", linestyle=":", lw=0.8)
    for model in models:
        x, gain = series[model]
        y = 1 - 1 / gain if reduction else gain
        ax.plot(x, y, color=colors[model], marker=markers[model], markersize=8, lw=1.2)

    ax.set_xscale(xscale)
    ax.set_yscale("linear")
    if reduction:
        ax.set_ylim(0, 1)

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
    ax.legend(handles=model_handles, loc="upper left", frameon=False)

    fig.savefig(filename, format="pdf")
    plt.close()
