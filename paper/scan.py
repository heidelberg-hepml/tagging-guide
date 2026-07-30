import json

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .plot import MODEL_ORDER, dataset_label, plot_metric
from .scaling_laws import fit_with_uncertainty

# metrics that decrease with cost: dataset label to the lower left, pareto front from min()
DECREASING_METRICS = {"loss", "auc_unc_total"}


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
        if len(np.unique(costs)) < 3:
            fits[model] = None
            continue
        fits[model] = fit_with_uncertainty(
            costs,
            metrics,
            n_bootstrap=n_bootstrap,
            quantile=quantile,
        )
    return fits


def scan_scaling_laws(
    perf_metrics,
    cost_metrics,
    prefix="",
    save=True,
    do_fit=True,
    n_bootstrap=100,
    quantile=0.1,
    used_models=None,
    pareto=False,
    label_top_left=False,
    model_labels=None,
):
    perf = {}
    models_per_label = {}
    sizes_per_label = {}
    for label, vals in perf_metrics.items():
        with open(f"paper/{label}.json") as file:
            entries = json.load(file)
        by_key = {(e["model"], e["size"]): e for e in entries}
        models_in_file = {e["model"] for e in entries}
        if used_models is not None:
            models_in_file &= set(used_models)
        models_in_file -= set(vals.get("exclude_models", []))
        models_per_label[label] = [m for m in MODEL_ORDER if m in models_in_file]
        sizes_per_label[label] = sorted({e["size"] for e in entries})
        perf[label] = {}
        sublabels = vals.get("sublabels", [None] * len(vals["keys"]))
        for metric, metric_label, sublabel in zip(
            vals["keys"], vals["labels"], sublabels, strict=True
        ):
            perf[label][metric] = {"label": metric_label, "sublabel": sublabel}
            for model in models_per_label[label]:
                perf[label][metric][model] = {
                    size: by_key.get((model, size), {}).get(metric, [])
                    for size in sizes_per_label[label]
                }

    union_models = [m for m in MODEL_ORDER if any(m in ms for ms in models_per_label.values())]
    union_sizes = sorted({s for ss in sizes_per_label.values() for s in ss})
    cost = {}
    for label, vals in cost_metrics.items():
        with open(vals["file"]) as file:
            metrics = json.load(file)
        # some cost metrics are quoted for a different implementation of the same network
        variants = vals.get("variants", {})
        cost[label] = {"label": vals["label"], "variants": variants}
        for model in union_models:
            cost[label][model] = {}
            for size in union_sizes:
                cost[label][model][size] = walk_dict(
                    metrics[str(size)][variants.get(model, model)], vals["keys"]
                ) * vals.get("scale", 1.0)

    for perf_label, perf_dict in perf.items():
        models = models_per_label[perf_label]
        sizes = sizes_per_label[perf_label]
        filename_fit = f"paper/{'' if prefix == '' else prefix + '_'}{perf_label}_fit.json"
        if not do_fit:
            with open(filename_fit) as file:
                fits = json.load(file)
        else:
            fits = {metric_label: {} for metric_label in perf_dict.keys()}

        filename = f"paper/{'' if prefix == '' else prefix + '_'}{perf_label}.pdf"
        with PdfPages(filename) as file:
            for metric_label, metric_dict in perf_dict.items():
                for cost_label, cost_dict in cost.items():
                    if do_fit:
                        fit = fit_scaling_law(
                            metric_dict,
                            cost_dict,
                            models,
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
                        models,
                        sizes,
                        fit=fit,
                        quantile=quantile,
                        dataset=dataset_label(perf_label),
                        sublabel=metric_dict["sublabel"],
                        decreasing=metric_label in DECREASING_METRICS,
                        label_lower_left=False if label_top_left else None,
                        model_labels=model_labels,
                        pareto=pareto,
                    )

        if do_fit and save:
            with open(filename_fit, "w") as file:
                json.dump(fits, file, indent=2)
