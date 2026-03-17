import json

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from results.plot import plot_metric
from results.scaling_laws import fit_scaling_law

MODELS = ["slim", "lloca", "part", "tr"]
SIZES = np.arange(-2.0, 2.1, step=1.0).astype(str)
DO_FIT = True
N_BOOTSTRAP = 100
QUANTILE = 0.1
UP = {"AUC": True, "accuracy": True, "loss": False}

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
    "jetclass_mar1": {
        "file": "results/jetclass_mar1.json",
        "labels": ["Loss", "JetClass AUC", "Accuracy"],
        "keys": ["loss", "AUC", "accuracy"],
    },
    "toptagxl_mar1": {
        "file": "results/toptagxl_feb3.json",
        "labels": ["Loss", "AUC", "Accuracy"],
        "keys": ["loss", "AUC", "accuracy"],
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
            metrics = json.load(file)
        perf[label] = {}
        for metric, metric_label in zip(vals["keys"], vals["labels"], strict=True):
            perf[label][metric] = {"label": metric_label}
            for model in MODELS:
                perf[label][metric][model] = {}
                for size in SIZES:
                    perf[label][metric][model][size] = metrics[size][model][metric]

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
                            MODELS,
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
                        MODELS,
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
