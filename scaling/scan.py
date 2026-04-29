import json

from matplotlib.backends.backend_pdf import PdfPages

from .plot import plot_metric
from .scaling_laws import fit_with_uncertainty


def walk_dict(d, keys):
    cur = d
    for k in keys:
        cur = cur[k]
    return cur


def fit_scaling_law(metric_dict, cost_dict, models, sizes, n_bootstrap=100, quantile=0.1):
    fits = {"label_metric": metric_dict["label"], "label_cost": cost_dict["label"]}
    for model in models:
        costs, metrics = [], []
        for size in sizes:
            for m in metric_dict[model][size]:
                costs.append(cost_dict[model][size])
                metrics.append(m)
        fits[model] = fit_with_uncertainty(
            costs,
            metrics,
            n_bootstrap=n_bootstrap,
            quantile=quantile,
        )
    return fits


def scan_scaling_laws(
    models,
    sizes,
    perf_metrics,
    cost_metrics,
    prefix="",
    save=True,
    do_fit=True,
    n_bootstrap=100,
    quantile=0.1,
):
    perf = {}
    for label, vals in perf_metrics.items():
        with open(f"scaling/{label}.json") as file:
            entries = json.load(file)
        by_key = {(e["model"], e["size"]): e for e in entries}
        perf[label] = {}
        for metric, metric_label in zip(vals["keys"], vals["labels"], strict=True):
            perf[label][metric] = {"label": metric_label}
            for model in vals["models"]:
                perf[label][metric][model] = {
                    size: by_key.get((model, size), {}).get(metric, []) for size in sizes
                }

    cost = {}
    for label, vals in cost_metrics.items():
        with open(vals["file"]) as file:
            metrics = json.load(file)
        cost[label] = {"label": vals["label"]}
        for model in models:
            cost[label][model] = {}
            for size in sizes:
                cost[label][model][size] = walk_dict(metrics[size][model], vals["keys"])

    for perf_label, perf_dict in perf.items():
        used_models = perf_metrics[perf_label]["models"]
        filename_fit = f"scaling/{perf_label}{'' if prefix == '' else '_' + prefix}_fit.json"
        if not do_fit:
            with open(filename_fit) as file:
                fits = json.load(file)
        else:
            fits = {metric_label: {} for metric_label in perf_dict.keys()}

        filename = f"scaling/{perf_label}{'' if prefix == '' else '_' + prefix}.pdf"
        with PdfPages(filename) as file:
            for metric_label, metric_dict in perf_dict.items():
                for cost_label, cost_dict in cost.items():
                    if do_fit:
                        fit = fit_scaling_law(
                            metric_dict,
                            cost_dict,
                            used_models,
                            sizes,
                            n_bootstrap=n_bootstrap,
                            quantile=quantile,
                        )
                        fits[metric_label][cost_label] = fit
                    else:
                        fit = fits[metric_label][cost_label]
                    plot_metric(
                        file,
                        metric_dict,
                        cost_dict,
                        used_models,
                        sizes,
                        fit=fit,
                        quantile=quantile,
                    )

        if do_fit and save:
            with open(filename_fit, "w") as file:
                json.dump(fits, file, indent=2)
