import json

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from results.plot import plot_metric
from results.scaling_laws import fit_scaling_law

MODELS = ["slim", "lloca", "part", "tr", "gn3"]
SIZES = np.arange(-2.0, 2.1, step=1.0).astype(str)
DO_FIT = True
N_BOOTSTRAP = 100
QUANTILE = 0.1
UP = {
    "auc": True,
    "auc_ovo": True,
    "accuracy": True,
    "loss": False,
    "rej03": True,
    "rej05": True,
    "rej08": True,
    "rej05_HToBB": True,
    "rej05_HToCC": True,
    "rej05_HToGG": True,
    "rej05_HToWW4Q": True,
    "rej099_HToWW2Q1L": True,
    "rej05_TTBar": True,
    "rej0995_TTBarLep": True,
    "rej05_WToQQ": True,
    "rej05_ZToQQ": True,
}

COST_METRICS = {
    "params": {
        "label": "Network parameters",
        "file": "cost_estimate/basics.json",
        "keys": ["params"],
    },
    "flops_measured": {
        "label": "Inference FLOPs, $N=50$ (measured)",
        "file": "cost_estimate/basics.json",
        "keys": ["flops"],
    },
    "flops_estimated": {
        "label": "Inference FLOPs, $N=50$ (estimated)",
        "file": "cost_estimate/energy_model.json",
        "keys": ["flops"],
    },
    "energy": {
        "label": "Energy [pJ], $N=50$",
        "file": "cost_estimate/energy_model.json",
        "keys": ["float16"],
    },
    "inference_cpu": {
        "label": "CPU inference time [ms], $N=50$",
        "file": "cost_estimate/inference_cpu.json",
        "keys": ["mean"],
    },
    "inference_gpu_bs512": {
        "label": "GPU inference time [ms], $N=50$, BS$=512$",
        "file": "cost_estimate/inference_gpu_bs512.json",
        "keys": ["mean"],
    },
    "memory_gpu_bs512": {
        "label": "GPU memory usage [GB], $N=50$, BS$=512$",
        "file": "cost_estimate/inference_gpu_bs512.json",
        "keys": ["memory_alloc"],
    },
    "train_gpu_bs512": {
        "label": "GPU training time [ms], $N=50$, BS$=512$",
        "file": "cost_estimate/train_gpu_bs512.json",
        "keys": ["mean"],
    },
}

PERF_METRICS = {
    "jetclass_mar2": {
        "file": "results/jetclass_mar2.json",
        "labels": [
            "Loss",
            "JetClass AUC",
            "Accuracy",
            r"$t\to b q\bar q$ $\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
        ],
        "keys": ["loss", "auc_ovo", "accuracy", "rej05_TTBar"],
        "models": ["slim", "lloca", "part", "tr", "gn3"],
    },
    "toptagxl_mar2": {
        "file": "results/toptagxl_mar2.json",
        "labels": ["Loss", "AUC", "Accuracy"],
        "keys": ["loss", "auc", "accuracy"],
        "models": ["slim", "lloca", "part", "tr", "gn3"],
    },
}


def walk_dict(d, keys):
    cur = d
    for k in keys:
        cur = cur[k]
    return cur


def main(save=True):
    perf = {}
    for label, vals in PERF_METRICS.items():
        with open(vals["file"]) as file:
            entries = json.load(file)
        by_key = {(e["model"], e["size"]): e for e in entries}
        perf[label] = {}
        for metric, metric_label in zip(vals["keys"], vals["labels"], strict=True):
            perf[label][metric] = {"label": metric_label}
            for model in vals["models"]:
                perf[label][metric][model] = {
                    size: by_key.get((model, size), {}).get(metric, []) for size in SIZES
                }

    cost = {}
    for label, vals in COST_METRICS.items():
        with open(vals["file"]) as file:
            metrics = json.load(file)
        cost[label] = {"label": vals["label"]}
        for model in MODELS:
            cost[label][model] = {}
            for size in SIZES:
                cost[label][model][size] = walk_dict(metrics[size][model], vals["keys"])

    for perf_label, perf_dict in perf.items():
        models = PERF_METRICS[perf_label]["models"]
        filename_fit = f"results/{perf_label}_fit.json"
        if not DO_FIT:
            with open(filename_fit) as file:
                fits = json.load(file)
        else:
            fits = {metric_label: {} for metric_label in perf_dict.keys()}

        filename = f"results/{perf_label}.pdf"
        with PdfPages(filename) as file:
            for metric_label, metric_dict in perf_dict.items():
                up = UP[metric_label]
                for cost_label, cost_dict in cost.items():
                    if DO_FIT:
                        fit = fit_scaling_law(
                            metric_dict,
                            cost_dict,
                            models,
                            SIZES,
                            n_bootstrap=N_BOOTSTRAP,
                            quantile=QUANTILE,
                            up=up,
                        )
                        fits[metric_label][cost_label] = fit
                    else:
                        fit = fits[metric_label][cost_label]
                    plot_metric(
                        file,
                        metric_dict,
                        cost_dict,
                        models,
                        SIZES,
                        fit=fit,
                        quantile=QUANTILE,
                        up=up,
                    )

        if DO_FIT and save:
            with open(filename_fit, "w") as file:
                json.dump(fits, file, indent=2)


if __name__ == "__main__":
    main()
