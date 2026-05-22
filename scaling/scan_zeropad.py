import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scaling.plot import (
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

ARCHS = ["tr", "lloca", "part", "slim", "lgatr"]


def main():
    with open("cost_estimate/inference_gpu_zeropad.json") as f:
        data = json.load(f)
    batchsizes = data["benchmarking"]["batchsizes"]
    models = [m for m in MODEL_ORDER if m in ARCHS]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("GPU memory [GB]", fontsize=FONTSIZE)
    ax.set_ylabel("GPU inference time [ms]", fontsize=FONTSIZE)
    ax.xaxis.set_label_coords(0.5, X_LABEL_POS)
    ax.yaxis.set_label_coords(Y_LABEL_POS, 0.5)
    plt.subplots_adjust(LEFT, BOTTOM, RIGHT, TOP)

    for model in models:
        for mode, ls in [("no-zeropad", "-"), ("zeropad", "--")]:
            if model == "part" and mode == "no-zeropad":
                continue
            x, y, ye_lo, ye_hi = [], [], [], []
            for bs in batchsizes:
                d = data[str(bs)][model][mode]
                x.append(d["memory_alloc"])
                y.append(d["mean"])
                ye_lo.append(d["std_minus"])
                ye_hi.append(d["std_plus"])
            ax.plot(
                x,
                y,
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
    mode_handles = [
        Line2D([0], [0], color="gray", linestyle="-", lw=1.2, label="no zero-pad"),
        Line2D([0], [0], color="gray", linestyle="--", lw=1.2, label="zero-pad"),
    ]
    leg1 = ax.legend(handles=model_handles, loc="lower right", frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=mode_handles, loc="lower left", frameon=False)

    fig.savefig("scaling/scan_zeropad.pdf", format="pdf")
    plt.close()


if __name__ == "__main__":
    main()
