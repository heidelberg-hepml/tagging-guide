"""Top tagging performance over time and over the estimated training cost.

The performance numbers are Tab. 1 of https://arxiv.org/abs/2512.17011,
the training cost is created by cost_estimate/toptagger_cost.py.
"""

import json

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from paper.plot import FIGSIZE, PARETO_ALPHA

LEFT, BOTTOM, RIGHT, TOP = 0.16, 0.16, 0.95, 0.95
X_LABEL_POS, Y_LABEL_POS = -0.1, -0.15
FONTSIZE = 14
JOULE_PER_KWH = 3.6e6

CLASSES = ["Classical", "Lorentz", "Pretrained", "Pretrained+Lorentz"]
COLORS = ["#B1AFD4", "#a7f86d", "#726C93", "#419108"]
MARKERS = ["o", "X", "P", "*"]
MARKERSIZES = [6, 8, 8, 10]

# time of publication, 1/eps_B at eps_S=0.3, its uncertainty, and the class
TAGGERS = {
    "BDT": (2017 + 1 / 12, 90, 1, 0),
    "TopoDNN": (2017 + 2 / 12, 295, 5, 0),
    "LoLa": (2017 + 7 / 12, 722, 17, 0),
    "PFN": (2018 + 10 / 12, 888, 17, 0),
    "TreeNiN": (2019 + 12 / 12, 1025, 11, 0),
    "ParticleNet": (2019 + 2 / 12, 1615, 93, 0),
    "ParT": (2022 + 3 / 12, 1602, 81, 0),
    "MIParT": (2024 + 7 / 12, 2010, 97, 0),
    "IAFormer": (2025 + 5 / 12, 2012, 30, 0),
    "PET-v2-s": (2025 + 10 / 12, 2167, 153, 0),
    "Transformer": (2025 + 8 / 12, 1613, 118, 0),
    "LorentzNet": (2022 + 1 / 12, 2195, 173, 1),
    "PELICAN": (2022 + 11 / 12, 2250, 75, 1),
    "L-GATr": (2024 + 5 / 12, 2240, 70, 1),
    "LLoCa-Tr.": (2025 + 8 / 12, 2150, 130, 1),
    "L-GATr-slim": (2025 + 12 / 12, 2264, 93, 1),
    "ParticleNet-f.t.": (2022 + 3 / 12, 1771, 80, 2),
    "ParT-f.t.": (2022 + 3 / 12, 2766, 130, 2),
    "MIParT-f.t.": (2024 + 7 / 12, 2789, 133, 2),
    "OmniLearned-M": (2025 + 10 / 12, 3208, 176, 2),
    "OmniLearned-L": (2025 + 10 / 12, 3486, 157, 2),
    "L-GATr-f.t.": (2024 + 10 / 12, 2894, 84, 3),
    "L-GATr-slim-f.t.": (2025 + 12 / 12, 2927, 70, 3),
    "L-GATr-slim-f.t. s=2": (2025 + 12 / 12, 3062, 84, 3),
}

# label placement in data coordinates; MIParT-f.t. is left out because it overlaps with ParT-f.t.
TEXT_YEAR = {
    "BDT": {},
    "TopoDNN": {},
    "LoLa": {},
    "PFN": {},
    "TreeNiN": dict(dx=0.4),
    "ParticleNet": dict(dx=0, dy=-100, ha="center", va="top"),
    "ParT": {},
    "MIParT": dict(dx=-0.3, ha="right"),
    "IAFormer": dict(dx=0.8, dy=-70, ha="center", va="top"),
    "PET-v2-s": dict(dx=0.5, dy=-30),
    "Transformer": {},
    "LorentzNet": dict(dx=-0.3, ha="right"),
    "PELICAN": dict(dx=0, dy=50, ha="right", va="bottom", rotation=-20),
    "L-GATr": dict(dx=0, dy=50, ha="center", va="bottom"),
    "LLoCa-Tr.": dict(dy=-155),
    "L-GATr-slim": dict(shift=0.2, dy=50),
    "ParticleNet-f.t.": dict(dx=-0.3, ha="right"),
    "ParT-f.t.": dict(dx=-0.3, ha="right"),
    "OmniLearned-M": dict(shift=-0.15, dx=-0.3, dy=55, ha="right"),
    "OmniLearned-L": dict(dx=-0.3, ha="right"),
    "L-GATr-f.t.": dict(dx=-0.7, dy=-80, ha="center", va="top"),
    "L-GATr-slim-f.t.": dict(dx=-0.4, dy=-120),
    "L-GATr-slim-f.t. s=2": dict(shift=0.2, dx=-0.45, ha="right"),
}
# same for the models with a cost estimate, where dx is a factor on the logarithmic axis;
# the two cost pages are kept separate so that they can be tuned independently
TEXT_FLOPS = {
    "Transformer": dict(dx=1.15, dy=-190),
    "ParT": dict(dx=1.3),
    "LLoCa-Tr.": dict(dx=1 / 1.3, ha="right"),
    "L-GATr-slim": dict(dx=1 / 1.7, dy=170, ha="center"),
    "PET-v2-s": dict(dy=-260, ha="center"),
    "LorentzNet": dict(dx=1.3, dy=-140),
    "L-GATr": dict(dx=1.6),
    "ParT-f.t.": dict(dx=1.3),
    "L-GATr-f.t.": dict(dx=1.4),
    "L-GATr-slim-f.t.": dict(dx=1 / 1.3, ha="right"),
    "L-GATr-slim-f.t. s=2": dict(dx=1 / 1.3, ha="right"),
    "OmniLearned-M": dict(dx=1 / 2.5, ha="right"),
    "OmniLearned-L": dict(dx=1 / 2.5, ha="right"),
}
TEXT_ENERGY = {
    "Transformer": dict(dx=1.15, dy=-190),
    "ParT": dict(dx=1.3),
    "LLoCa-Tr.": dict(dx=1 / 1.3, ha="right"),
    "L-GATr-slim": dict(dx=1 / 1.7, dy=170, ha="center"),
    "PET-v2-s": dict(dy=-260, ha="center"),
    "LorentzNet": dict(dx=1.3, dy=-140),
    "L-GATr": dict(dx=1.6),
    "ParT-f.t.": dict(dx=1.3),
    "L-GATr-f.t.": dict(dx=1.4),
    "L-GATr-slim-f.t.": dict(dx=1 / 1.3, ha="right"),
    "L-GATr-slim-f.t. s=2": dict(dx=1 / 1.3, ha="right"),
    "OmniLearned-M": dict(dx=1 / 2.5, ha="right"),
    "OmniLearned-L": dict(dx=1 / 2.5, ha="right"),
}


def main():
    with open("cost_estimate/toptagger_cost.json") as file:
        cost = json.load(file)

    kwh = 1 / JOULE_PER_KWH
    with PdfPages("paper/toptagging.pdf") as pdf:
        plot_year(pdf)
        plot_cost(pdf, cost, "flops", 1, "Training FLOPs", (1.3e15, 1e22), TEXT_FLOPS)
        plot_cost(pdf, cost, "energy", kwh, "Training energy [kWh]", (9e-5, 5e2), TEXT_ENERGY)


def setup(xlabel, ylim=(0, 3700), cheaper_is_better=False):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(r"Background rejection $1/\epsilon_B\ @\ \epsilon_S=0.3$", fontsize=FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax.xaxis.set_label_coords(0.5, X_LABEL_POS)
    ax.yaxis.set_label_coords(Y_LABEL_POS, 0.5)
    ax.grid(lw=0.5)
    ax.set_ylim(*ylim)
    plt.subplots_adjust(LEFT, BOTTOM, RIGHT, TOP)

    # the arrow points to the top left if a smaller x is better, and up otherwise
    xy, xytext, text = (0.01, 0.99), (0.07, 0.93), 0.08
    if not cheaper_is_better:
        xy, xytext, text = (0.03, 0.99), (0.03, 0.92), 0.05
    ax.annotate(
        "",
        xy=xy,
        xycoords="axes fraction",
        xytext=xytext,
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="red", lw=2),
        color="red",
        ha="left",
        va="center",
        zorder=10,
        clip_on=False,
    )
    ax.text(text, 0.94, "better", transform=ax.transAxes, color="red", fontsize=FONTSIZE)

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[i],
            linestyle="None",
            marker=MARKERS[i],
            markersize=MARKERSIZES[i],
            label=CLASSES[i],
        )
        for i in range(len(CLASSES))
    ]
    ax.legend(
        handles=handles,
        loc=4,
        frameon=False,
        fontsize=FONTSIZE,
        markerfirst=False,
        handletextpad=0,
    )
    return fig, ax


def scatter(ax, label, x, text, fontsize=FONTSIZE):
    _, eps03, unc, idx = TAGGERS[label]
    ax.plot(x, eps03, marker=MARKERS[idx], color=COLORS[idx], markersize=MARKERSIZES[idx])
    ax.errorbar(x, eps03, yerr=unc, color=COLORS[idx])
    # dx shifts the label along the x-axis, multiplicatively if the axis is logarithmic
    log = ax.get_xscale() == "log"
    dx = text.get("dx", 1.2 if log else 0.3)
    ax.text(
        x * dx if log else x + dx,
        eps03 + text.get("dy", 0),
        label,
        horizontalalignment=text.get("ha", "left"),
        verticalalignment=text.get("va", "center"),
        rotation=text.get("rotation", 0),
        fontsize=fontsize,
    )


def plot_year(pdf):
    fig, ax = setup(r"time of publication")
    for label, text in TEXT_YEAR.items():
        scatter(ax, label, TAGGERS[label][0] + text.get("shift", 0), text)

    ax.set_xlim(2016, 2030)
    ax.set_xticks([2016, 2018, 2020, 2022, 2024, 2026])
    pdf.savefig(fig)
    plt.close()


def plot_cost(pdf, cost, key, scale, xlabel, xlim, texts):
    fig, ax = setup(xlabel, ylim=(1300, 3700), cheaper_is_better=True)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    # float() because the FLOPs counts in the json overflow numpy's int64
    points = {label: float(cost[label]["total"][key]) * scale for label in texts}
    plot_pareto(ax, [(x, TAGGERS[label][1]) for label, x in points.items()])
    for label, text in texts.items():
        scatter(ax, label, points[label], text, fontsize=FONTSIZE - 2)
    pdf.savefig(fig)
    plt.close()


def plot_pareto(ax, points):
    """Staircase through the models that no other model beats in both cost and rejection."""
    front = []
    for x, y in sorted(points):
        if not front or y > front[-1][1]:
            front.append((x, y))
    # the staircase drops to the x-axis at the cheapest model and extends to the right edge
    bottom = ax.get_ylim()[0]
    front = [(front[0][0], bottom)] + front + [(ax.get_xlim()[1], front[-1][1])]
    x, y = zip(*front, strict=True)

    ax.fill_between(x, bottom, y, step="post", color="k", alpha=PARETO_ALPHA, lw=0, zorder=0.5)
    ax.step(x, y, where="post", color="k", alpha=0.3, lw=1, zorder=0.5)


if __name__ == "__main__":
    main()
