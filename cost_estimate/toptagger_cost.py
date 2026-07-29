"""Training cost of the top taggers in Tab. 1 of https://arxiv.org/abs/2512.17011.

In contrast to energy_model.py, which estimates a single forward pass in pJ, this script
covers the full training including the pretraining of the foundation models, in J.
Only the network is counted, i.e. we neglect data loading, the optimizer step, exponential
moving averages and evaluation. The JetClass-pretrained models are assumed to be pretrained
with training/jc_ParT.yaml and finetuned with their from-scratch recipe.
"""

import glob
import json

import awkward as ak
import h5py
import hydra
import numpy as np
import uproot
from omegaconf import OmegaConf

from cost_estimate.arch_kwargs import FAMILIES, get_arch_kwargs
from cost_estimate.estimate import FLOAT32_ARCHITECTURES, estimate_energy, estimate_flops
from experiments.tagging.embedding import EPS

# the '# Train' column of Tab. 1
DATASET_SIZE = dict(toptagging=1.211e6, jetclass=100e6, omnilearned=1057.7e6)
# the backward pass costs twice as much as the forward pass
BACKWARD_FACTOR = 3

# 'sparse': mean jet size, 'batch': longest jet in the batch, 'file': full width of the h5 files,
# which the OmniLearned collate function never truncates
PADDING = {"part": "batch", "pet2": "file"}

MODELS = {
    "Transformer": [dict(model="tag_transformer", training="top_transformer")],
    "ParT": [dict(model="tag_ParT", training="top_ParT")],
    "PET-v2-s": [dict(model="pet2_s", training="top_pet2_s")],
    "LorentzNet": [dict(model="tag_lorentznet", training="top_lorentznet")],
    "LLoCa-Tr.": [dict(model="tag_transformer", framesnet="learnedpd", training="top_transformer")],
    "L-GATr": [dict(model="tag_lgatr", training="top_lgatr")],
    "L-GATr-slim": [dict(model="tag_slim", training="top_slim")],
    "ParT-f.t.": [
        dict(model="tag_ParT", training="jc_ParT", dataset="jetclass"),
        dict(model="tag_ParT", training="top_ParT"),
    ],
    "L-GATr-f.t.": [
        dict(model="tag_lgatr", training="jc_ParT", dataset="jetclass"),
        dict(model="tag_lgatr", training="top_lgatr"),
    ],
    "L-GATr-slim-f.t.": [
        dict(model="tag_slim", training="jc_ParT", dataset="jetclass"),
        dict(model="tag_slim", training="top_slim"),
    ],
    "L-GATr-slim-f.t. s=2": [
        dict(model="slim", size=2, training="jc_5epoch", dataset="jetclass"),
        dict(model="slim", size=2, training="top_slim"),
    ],
    "OmniLearned-M": [
        dict(model="pet2_m", training="pretrain_pet2", dataset="omnilearned", mode="pretrain"),
        dict(model="pet2_m", training="top_pet2_ft"),
    ],
    "OmniLearned-L": [
        dict(model="pet2_l", training="pretrain_pet2", dataset="omnilearned", mode="pretrain"),
        dict(model="pet2_l", training="top_pet2_ft"),
    ],
    "L-GATr-slim-2 (OmniLearned)": [
        dict(model="slim", size=2, training="pretrain_pet2", dataset="omnilearned"),
    ],
}

OmegaConf.register_new_resolver("eval", eval, replace=True)


def main(save=True):
    counts, width = constituents()
    mean = ", ".join(f"{key} {value.mean():.2f}" for key, value in counts.items())
    print(f"mean jetsize: {mean}, padded array width: {width}")

    results = dict()
    header = (
        f"{'model':<28}{'stage':<12}{'epochs':>9}{'padding':>9}{'jetsize':>9}{'dtype':>10}"
        f"{'flops/jet':>12}{'flops':>11}{'energy [J]':>13}"
    )
    print(header)
    print("-" * len(header))
    for label, stages in MODELS.items():
        results[label] = dict()
        for stage in stages:
            pretraining = stage.get("dataset", "toptagging") != "toptagging"
            key = "pretraining" if pretraining else "training"
            results[label][key] = single_stage(**stage, counts=counts, width=width)
            print_stage(label, key, results[label][key])

        results[label]["total"] = {
            key: sum(stage[key] for stage in results[label].values()) for key in ["flops", "energy"]
        }
        if len(stages) > 1:
            print_stage(label, "total", results[label]["total"])

    if save:
        with open("cost_estimate/toptagger_cost.json", "w") as file:
            json.dump(results, file, indent=2)
    return results


def constituents():
    counts = dict()

    fourmomenta = np.load("data/toptagging_mini.npz")["kinematics_train"]
    counts["toptagging"] = (np.abs(fourmomenta) > EPS).any(axis=-1).sum(axis=-1)

    num = []
    for path in sorted(glob.glob("data/jetclass/train_100M/*.root")):
        with uproot.open(path) as file:
            tree = file[file.keys()[0]]
            num.append(np.asarray(ak.num(tree["part_px"].array(library="ak"))))
    # the miniweaver configs truncate at 128 constituents
    counts["jetclass"] = np.minimum(np.concatenate(num), 128)

    num, widths = [], set()
    for path in sorted(glob.glob("data/pretrain/*/train/*.h5")):
        with h5py.File(path, "r") as file:
            data = file["data"][:]
        widths.add(data.shape[1])
        # zero-padded constituents have a vanishing log pT
        jetsize = (data[..., 2] != 0).sum(axis=-1)
        num.append(jetsize[jetsize > 0])
    counts["omnilearned"] = np.concatenate(num)

    assert len(widths) == 1, f"OmniLearned files disagree on the array width: {widths}"
    return counts, widths.pop()


def expected_max(counts, batchsize):
    """Mean size of the longest jet in a batch of iid jets."""
    values, num = np.unique(counts, return_counts=True)
    cdf = np.cumsum(num) / len(counts)
    probs = cdf**batchsize - np.concatenate([[0.0], cdf[:-1]]) ** batchsize
    return float((values * probs).sum())


def print_stage(label, key, results):
    def entry(name, width, fmt=""):
        return f"{results[name]:>{width}{fmt}}" if name in results else " " * width

    print(
        f"{label:<28}{key:<12}{entry('epochs', 9, '.1f')}{entry('padding', 9)}"
        f"{entry('jetsize', 9, '.1f')}{entry('dtype', 10)}{entry('flops_forward', 12, '.2e')}"
        f"{results['flops']:>11.2e}{results['energy']:>13.2e}"
    )


def single_stage(
    model, training, counts, width, size=None, framesnet=None, dataset="toptagging", mode=None
):
    with hydra.initialize(config_path="../config", version_base=None):
        overrides = [
            f"model={model}",
            f"training={training}",
            "save=false",
            "data.dataset=mini",
            "gpu=false",
        ]
        if size is not None:
            overrides.append(f"model.net.size={size}")
        if framesnet is not None:
            overrides.append(f"model/framesnet={framesnet}")
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)

    padding = PADDING.get(FAMILIES[model], "sparse")
    if padding == "sparse":
        jetsize = float(counts[dataset].mean())
    elif padding == "batch":
        jetsize = expected_max(counts[dataset], cfg.training.batchsize)
    else:
        jetsize = width

    extra = dict() if mode is None else dict(mode=mode)
    architecture, kwargs = get_arch_kwargs(model, cfg, jetsize, **extra)

    samples = cfg.training.iterations * cfg.training.batchsize
    dtype = "float32" if architecture in FLOAT32_ARCHITECTURES else "bfloat16"
    flops = estimate_flops(architecture=architecture, arch_kwargs=kwargs)
    energy = estimate_energy(
        architecture=architecture,
        arch_kwargs=kwargs,
        dtype_a=dtype,
        dtype_w=dtype,
        dtype_default=dtype,
        mode="H100-estimate",
    )

    results = dict(
        samples=samples,
        epochs=samples / DATASET_SIZE[dataset],
        padding=padding,
        jetsize=jetsize,
        dtype=dtype,
        flops_forward=flops,
        flops=BACKWARD_FACTOR * flops * samples,
        energy=BACKWARD_FACTOR * energy * samples * 1e-12,  # estimate_energy returns pJ
    )
    return results


if __name__ == "__main__":
    main()
