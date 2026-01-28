import json

from matplotlib.backends.backend_pdf import PdfPages

from results import utils

MODELS = ["slim", "lloca", "part", "tr"]
SIZES = ["xxs", "xs", "s", "m", "l"]

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
    "flops_estimate": {
        "label": "Inference FLOPs, $N=50$ (estimated)",
        "file": "cost_estimate/energy_model.json",
        "keys": ["flops"],
    },
    "energy": {
        "label": "Energy [pJ], $N=50$",
        "file": "cost_estimate/energy_model.json",
        "keys": ["float32", "Horowitz"],
    },
    "inference_cpu": {
        "label": "CPU inference time [ms], $N=50$",
        "file": "cost_estimate/inference_cpu.json",
        "keys": ["mean"],
    },
    #"inference_gpu_bs1": {
    #    "label": "GPU inference time [ms], $N=50$, BS$=1$",
    #    "file": "cost_estimate/inference_gpu_bs1.json",
    #    "keys": ["mean"],
    #},
    "inference_gpu_bs512": {
        "label": "GPU inference time [ms], $N=50$, BS$=512$",
        "file": "cost_estimate/inference_gpu_bs512.json",
        "keys": ["mean"],
    },
    #"memory_gpu_bs1": {
    #    "label": "GPU memory usage [GB], $N=50$, BS$=1$",
    #    "file": "cost_estimate/inference_gpu_bs1.json",
    #    "keys": ["memory_alloc"],
    #},
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
    "jetclass_jan1": {
        "file": "results/jetclass_jan1.json",
        "labels": ["JetClass AUC", "JetClass Accuracy"],
        "keys": ["AUC", "accuracy"],
    },
    "toptagxl_jan1": {
        "file": "results/toptagxl_jan1.json",
        "labels": ["TopTagXL rejection rate at 0.8", "TopTagXL AUC", "TopTagXL Accuracy"],
        "keys": ["rej08", "AUC", "accuracy"],
    },
}


def walk_dict(d, keys):
    cur = d
    for k in keys:
        cur = cur[k]
    return cur


def main():
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
        filename = f"results/{perf_label}.pdf"
        with PdfPages(filename) as file:
            for metric_label, metric_dict in perf_dict.items():
                for cost_label, cost_dict in cost.items():
                    print(f"Plotting {perf_label} / {metric_label} / {cost_label}")
                    utils.plot_metric(file, metric_dict, cost_dict, MODELS, SIZES)


if __name__ == "__main__":
    main()
