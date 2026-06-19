import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

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

INPUT = "scaling/atlastop1ep_jun1.json"
OUTPUT = "scaling/unc_atlastop1ep_jun1.pdf"
# INPUT = "scaling/atlastop5ep_jun1.json"
# OUTPUT = "scaling/unc_atlastop5ep_jun1.pdf"
USED_MODELS = None

PERF_AXES = {
    "rej05": r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
    "auc": "AUC",
}

# Ordered as in fig. 9 of arXiv:2407.20127: total, then the 5 leading uncertainty
# groups (signal modeling, scales, bkg modeling, track, cluster), then the
# sub-uncertainties grouped by their parent (cluster, track, bkg model, scale).
UNC_VARIANTS = [
    ("total", "total"),
    ("sig_model", "signal modeling"),
    ("scale", "scales"),
    ("bkg_model", "bkg modeling"),
    ("track", "track"),
    ("cluster", "cluster"),
    ("es", "cluster energy scale"),
    ("cer", "cluster energy resolution"),
    ("cpos", "cluster position resolution"),
    ("eff", "track efficiency"),
    ("fake", "track fake rate"),
    ("bias", "track bias"),
    ("bkg_ps", "bkg parton shower"),
    ("bkg_had", "bkg hadronization"),
    ("sig_ISR", "signal ISR"),
    ("sig_FSR", "signal FSR"),
    ("bkg_ISR", "bkg ISR"),
    ("bkg_FSR", "bkg FSR"),
]

# (name, rej05, rel_unc_total [%]) digitized from fig 10b of arXiv:2407.20127
ATLAS_REF_REJ05_TOTAL = [
    ["EFN", 23, 26.2],
    ["hIDNN", 47, 31.5],
    ["DNN", 74, 31.9],
    ["PFN", 86, 36.3],
    ["ResNet50", 18, 37.8],
    ["ParticleNet", 155, 41.9],
]


def plot_unc(file, by_key, models, sizes, perf_key, unc_key, perf_label, unc_label, atlas_ref=None):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlabel(perf_label, fontsize=FONTSIZE)
    ax.set_ylabel(unc_label, fontsize=FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax.xaxis.set_label_coords(0.5, X_LABEL_POS)
    ax.yaxis.set_label_coords(Y_LABEL_POS, 0.5)
    plt.subplots_adjust(LEFT, BOTTOM, RIGHT, TOP)

    for model in models:
        xs, ys, xerrs, yerrs = [], [], [], []
        for size in sizes:
            entry = by_key.get((model, size))
            if entry is None:
                continue
            perf = entry.get(perf_key)
            unc = entry.get(unc_key)
            if not perf or not unc or perf[0] is None or unc[0] is None:
                continue
            xs.append(np.mean(perf))
            ys.append(np.mean(unc) * 100)
            xerrs.append(np.std(perf))
            yerrs.append(np.std(unc) * 100)
        ax.errorbar(
            xs,
            ys,
            xerr=xerrs,
            yerr=yerrs,
            color=colors[model],
            marker=markers[model],
            label=labels[model],
            markersize=8,
            elinewidth=1,
            capsize=3,
        )

    if atlas_ref is not None:
        for name, x, y in atlas_ref:
            ax.plot(x, y, marker="o", color="black", markersize=6, lw=0)
            ax.annotate(
                name,
                (x, y),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=FONTSIZE - 4,
            )

    ax.legend(frameon=False)
    fig.savefig(file, format="pdf")
    plt.close()


def main():
    with open(INPUT) as f:
        entries = json.load(f)
    by_key = {(e["model"], e["size"]): e for e in entries}
    models_in_file = {e["model"] for e in entries}
    if USED_MODELS is not None:
        models_in_file &= set(USED_MODELS)
    models = [m for m in MODEL_ORDER if m in models_in_file]
    sizes = sorted({e["size"] for e in entries})

    with PdfPages(OUTPUT) as file:
        for perf_key, perf_label in PERF_AXES.items():
            for variant, variant_label in UNC_VARIANTS:
                unc_key = f"{perf_key}_unc_{variant}"
                unc_label = rf"Relative {variant_label} uncertainty [\%]"
                atlas_ref = (
                    ATLAS_REF_REJ05_TOTAL if perf_key == "rej05" and variant == "total" else None
                )
                plot_unc(
                    file,
                    by_key,
                    models,
                    sizes,
                    perf_key,
                    unc_key,
                    perf_label,
                    unc_label,
                    atlas_ref=atlas_ref,
                )


if __name__ == "__main__":
    main()
