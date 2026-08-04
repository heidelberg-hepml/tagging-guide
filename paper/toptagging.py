"""Top tagging performance over time and over the estimated training cost.

The performance numbers are Tab. 1 of https://arxiv.org/abs/2512.17011, cross-checked against
Tab. 1 of https://arxiv.org/abs/2411.00446, which quotes the same two working points. BDT,
TopoDNN, LoLa, TreeNiN and PELICAN have no published rejection at eps_S=0.5, so they only
appear at eps_S=0.3. The training cost is created by cost_estimate/toptagger_cost.py.
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

# time of publication, class, and 1/eps_B with its uncertainty at eps_S=0.3 and at eps_S=0.5;
# None means that no reference quotes the tagger at that working point
TAGGERS = {
    "BDT": (2017 + 1 / 12, 0, (90, 1), None),
    "TopoDNN": (2017 + 2 / 12, 0, (295, 5), None),
    "LoLa": (2017 + 7 / 12, 0, (722, 17), None),
    "PFN": (2018 + 10 / 12, 0, (888, 17), (247, 3)),
    "TreeNiN": (2019 + 12 / 12, 0, (1025, 11), None),
    "ParticleNet": (2019 + 2 / 12, 0, (1615, 93), (397, 7)),
    "ParT": (2022 + 3 / 12, 0, (1602, 81), (413, 16)),
    "MIParT": (2024 + 7 / 12, 0, (2010, 97), (505, 8)),
    "IAFormer": (2025 + 5 / 12, 0, (2012, 30), (510, 6)),
    "PET-v2-s": (2025 + 10 / 12, 0, (2167, 153), (505, 14)),
    "Transformer": (2025 + 8 / 12, 0, (1613, 118), (389, 6)),
    "LorentzNet": (2022 + 1 / 12, 1, (2195, 173), (498, 18)),
    "PELICAN": (2022 + 11 / 12, 1, (2250, 75), None),
    "L-GATr": (2024 + 5 / 12, 1, (2240, 70), (540, 20)),
    "LLoCa-Tr.": (2025 + 8 / 12, 1, (2150, 130), (492, 15)),
    "L-GATr-slim": (2025 + 12 / 12, 1, (2264, 93), (546, 7)),
    "ParticleNet-f.t.": (2022 + 3 / 12, 2, (1771, 80), (487, 9)),
    "ParT-f.t.": (2022 + 3 / 12, 2, (2766, 130), (691, 15)),
    "MIParT-f.t.": (2024 + 7 / 12, 2, (2789, 133), (640, 10)),
    "OmniLearned-M": (2025 + 10 / 12, 2, (3208, 176), (656, 12)),
    "OmniLearned-L": (2025 + 10 / 12, 2, (3486, 157), (688, 9)),
    "L-GATr-f.t.": (2024 + 10 / 12, 3, (2894, 84), (651, 11)),
    "L-GATr-slim-f.t.": (2025 + 12 / 12, 3, (2927, 70), (655, 5)),
    "L-GATr-slim-f.t. 48M": (2026 + 8 / 12, 3, (3062, 84), (693, 17)),
}
EPS_COLUMN = {0.3: 2, 0.5: 3}

# labels sit at the same distance from their marker on every page, on the side given by `side`;
# `dx`/`dy` are small extra nudges in points and `shift` moves the marker itself along the x-axis
LABEL_PAD = 6  # pt
SIDES = {
    "right": (1, 0, "left", "center"),
    "left": (-1, 0, "right", "center"),
    "top": (0, 1, "center", "bottom"),
    "bottom": (0, -1, "center", "top"),
}

# MIParT-f.t. is left out because it overlaps with ParT-f.t. at 0.3 and with L-GATr-f.t. at 0.5
TEXT_YEAR_03 = {
    "BDT": dict(side="right"),
    "TopoDNN": dict(side="right"),
    "LoLa": dict(side="right", dy=-4),
    "PFN": dict(side="right", dy=-2),
    "TreeNiN": dict(side="right", dx=5),
    "ParticleNet": dict(side="bottom"),
    "ParT": dict(side="right"),
    "MIParT": dict(side="left", dx=4),
    "IAFormer": dict(side="bottom", dx=16, dy=2),
    "PET-v2-s": dict(side="right", dx=5, dy=5),
    "Transformer": dict(side="right"),
    "LorentzNet": dict(side="left"),
    "PELICAN": dict(side="top", ha="right", dy=-8, rotation=-20),
    "L-GATr": dict(side="top"),
    "LLoCa-Tr.": dict(side="right", dy=-8),
    "L-GATr-slim": dict(side="right", dy=12, shift=0.2),
    "ParticleNet-f.t.": dict(side="left", dy=5),
    "ParT-f.t.": dict(side="left"),
    "OmniLearned-M": dict(side="left", dy=4, shift=-0.15),
    "OmniLearned-L": dict(side="left"),
    "L-GATr-f.t.": dict(side="bottom", dx=-14, dy=13),
    "L-GATr-slim-f.t.": dict(side="bottom", ha="left", dx=-6, dy=9),
    "L-GATr-slim-f.t. 48M": dict(side="left", dx=-10),
}
# IAFormer additionally drops out at 0.5, where MIParT, PET-v2-s and LLoCa-Tr. box its marker in
TEXT_YEAR_05 = {
    "PFN": dict(side="right"),
    "ParticleNet": dict(side="bottom"),
    "ParT": dict(side="right"),
    "MIParT": dict(side="bottom", dy=3),
    "PET-v2-s": dict(side="right", dy=4),
    "Transformer": dict(side="right"),
    "LorentzNet": dict(side="left"),
    "LLoCa-Tr.": dict(side="right", dy=-6),
    "L-GATr": dict(side="left"),
    "L-GATr-slim": dict(side="right"),
    "ParticleNet-f.t.": dict(side="bottom", dy=2),
    "ParT-f.t.": dict(side="left"),
    "OmniLearned-M": dict(side="left", dx=4, dy=-16, shift=-0.15),
    "OmniLearned-L": dict(side="left", dx=8, dy=10),
    "L-GATr-f.t.": dict(side="left"),
    "L-GATr-slim-f.t.": dict(side="bottom", ha="left", dx=-10),
    "L-GATr-slim-f.t. 48M": dict(side="top", ha="left", dx=-40),
}
# same for the models with a cost estimate; the cost pages are kept separate from the year pages
# and from each other so that they can be tuned independently
TEXT_FLOPS_03 = {
    "Transformer": dict(side="bottom", ha="left", dx=2),
    "ParT": dict(side="right"),
    "LLoCa-Tr.": dict(side="left", dx=4),
    "L-GATr-slim": dict(side="top", dx=-10),
    "PET-v2-s": dict(side="bottom", dx=2),
    "LorentzNet": dict(side="right", dy=-8),
    "L-GATr": dict(side="right"),
    "ParT-f.t.": dict(side="right", dy=-2),
    "L-GATr-f.t.": dict(side="right"),
    "L-GATr-slim-f.t.": dict(side="left"),
    "L-GATr-slim-f.t. 48M": dict(side="left"),
    "OmniLearned-M": dict(side="left", dx=-7),
    "OmniLearned-L": dict(side="left", dx=-7),
}
TEXT_ENERGY_03 = {
    "Transformer": dict(side="bottom", ha="left", dx=2),
    "ParT": dict(side="right"),
    "LLoCa-Tr.": dict(side="left", dx=4),
    "L-GATr-slim": dict(side="top", dx=-10),
    "PET-v2-s": dict(side="bottom", dx=2),
    "LorentzNet": dict(side="right", dy=-8),
    "L-GATr": dict(side="right"),
    "ParT-f.t.": dict(side="right", dy=-2),
    "L-GATr-f.t.": dict(side="right"),
    "L-GATr-slim-f.t.": dict(side="left"),
    "L-GATr-slim-f.t. 48M": dict(side="left"),
    "OmniLearned-M": dict(side="left"),
    "OmniLearned-L": dict(side="left"),
}
TEXT_FLOPS_05 = {
    "Transformer": dict(side="bottom", ha="left", dx=2),
    "ParT": dict(side="right"),
    "LLoCa-Tr.": dict(side="left", dx=4),
    "L-GATr-slim": dict(side="top", dx=-10),
    "PET-v2-s": dict(side="right"),
    "LorentzNet": dict(side="right", dy=-10),
    "L-GATr": dict(side="right"),
    "ParT-f.t.": dict(side="left"),
    "L-GATr-f.t.": dict(side="right"),
    "L-GATr-slim-f.t.": dict(side="left"),
    "L-GATr-slim-f.t. 48M": dict(side="top"),
    "OmniLearned-M": dict(side="bottom"),
    "OmniLearned-L": dict(side="left", dx=-7),
}
TEXT_ENERGY_05 = {
    "Transformer": dict(side="bottom", ha="left", dx=2),
    "ParT": dict(side="right"),
    "LLoCa-Tr.": dict(side="left", dx=4),
    "L-GATr-slim": dict(side="top", dx=-10),
    "PET-v2-s": dict(side="right"),
    "LorentzNet": dict(side="right", dy=-10),
    "L-GATr": dict(side="right"),
    "ParT-f.t.": dict(side="left"),
    "L-GATr-f.t.": dict(side="right"),
    "L-GATr-slim-f.t.": dict(side="left"),
    "L-GATr-slim-f.t. 48M": dict(side="top"),
    "OmniLearned-M": dict(side="bottom"),
    "OmniLearned-L": dict(side="left", dy=-4),
}

# the eps_S=0.5 year page carries the same 18 labels in a fifth of the vertical spread, so the
# tagger names have to be as small there as they are on the cost pages
PAGES = (
    dict(
        eff=0.3,
        year_ylim=(0, 3700),
        cost_ylim=(1300, 3700),
        year_fontsize=FONTSIZE,
        texts=(TEXT_YEAR_03, TEXT_FLOPS_03, TEXT_ENERGY_03),
    ),
    dict(
        eff=0.5,
        year_ylim=(200, 760),
        cost_ylim=(350, 760),
        year_fontsize=FONTSIZE - 2,
        texts=(TEXT_YEAR_05, TEXT_FLOPS_05, TEXT_ENERGY_05),
    ),
)
# key in the cost json, unit conversion, axis label and axis range of the two cost pages
COSTS = (
    ("flops", 1, "Training FLOPs", (1.3e15, 1e22)),
    ("energy", 1 / JOULE_PER_KWH, "Training energy [kWh]", (9e-5, 5e2)),
)


def main():
    with open("cost_estimate/toptagger_cost.json") as file:
        cost = json.load(file)

    with PdfPages("paper/toptagging.pdf") as pdf:
        for page in PAGES:
            eff, ylim = page["eff"], page["cost_ylim"]
            year, *cost_texts = page["texts"]
            plot_year(pdf, eff, page["year_ylim"], year, page["year_fontsize"])
            for (key, scale, xlabel, xlim), texts in zip(COSTS, cost_texts, strict=True):
                plot_cost(pdf, cost, key, scale, xlabel, xlim, ylim, eff, texts)


def setup(xlabel, eff, ylim, cheaper_is_better=False):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ylabel = rf"Background rejection $1/\epsilon_B\ @\ \epsilon_S={eff}$"
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)
    ax.xaxis.set_label_coords(0.5, X_LABEL_POS)
    ax.yaxis.set_label_coords(Y_LABEL_POS, 0.5)
    ax.grid(lw=0.5)
    ax.set_ylim(*ylim)
    plt.subplots_adjust(LEFT, BOTTOM, RIGHT, TOP)

    # the arrow points to the top left if a smaller x is better, and up otherwise
    """
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
    """

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


def rejection(label, eff):
    rej = TAGGERS[label][EPS_COLUMN[eff]]
    assert rej is not None, f"{label} has no published rejection at eps_S={eff}"
    return rej


def scatter(ax, label, x, rej, text, fontsize=FONTSIZE):
    value, unc = rej
    idx = TAGGERS[label][1]
    ax.plot(x, value, marker=MARKERS[idx], color=COLORS[idx], markersize=MARKERSIZES[idx])
    ax.errorbar(x, value, yerr=unc, color=COLORS[idx])

    # labels to the side clear the marker, labels above and below clear the error bar, so that
    # the visible gap is the same everywhere; the extra dx, dy are in points as well
    sx, sy, ha, va = SIDES[text.get("side", "right")]
    pad = LABEL_PAD + (MARKERSIZES[idx] / 2 if sy == 0 else 0)
    ax.annotate(
        label,
        xy=(x, value + sy * unc),
        xytext=(sx * pad + text.get("dx", 0), sy * pad + text.get("dy", 0)),
        textcoords="offset points",
        horizontalalignment=text.get("ha", ha),
        verticalalignment=text.get("va", va),
        rotation=text.get("rotation", 0),
        fontsize=fontsize,
        annotation_clip=False,
    )


def plot_year(pdf, eff, ylim, texts, fontsize):
    fig, ax = setup(r"time of publication", eff, ylim)
    for label, text in texts.items():
        x = TAGGERS[label][0] + text.get("shift", 0)
        scatter(ax, label, x, rejection(label, eff), text, fontsize)

    ax.set_xlim(2016, 2030)
    ax.set_xticks([2016, 2018, 2020, 2022, 2024, 2026])
    pdf.savefig(fig)
    plt.close()


def plot_cost(pdf, cost, key, scale, xlabel, xlim, ylim, eff, texts):
    fig, ax = setup(xlabel, eff, ylim, cheaper_is_better=True)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    # float() because the FLOPs counts in the json overflow numpy's int64
    points = {label: float(cost[label]["total"][key]) * scale for label in texts}
    rejections = {label: rejection(label, eff) for label in texts}
    plot_pareto(ax, [(x, rejections[label][0]) for label, x in points.items()])
    for label, text in texts.items():
        scatter(ax, label, points[label], rejections[label], text, FONTSIZE - 2)
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
