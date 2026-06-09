import json
import math

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .plot import MODEL_ORDER, labels, plot_metric
from .scaling_laws import fit_with_uncertainty

PARAM_COLUMNS = [("L_inf", r"$L_\infty$"), ("B", r"$B$"), ("beta", r"$\beta$")]
PATHOLOGICAL_REL_THRESHOLD = 10.0


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


def _q(summary, p):
    # Keys are floats when `fit` is freshly computed, strings after a JSON
    # round-trip (do_fit=False path).
    return summary[p] if p in summary else summary[str(p)]


def _is_pathological(median, l_unc, u_unc, max_rel=PATHOLOGICAL_REL_THRESHOLD):
    if not all(np.isfinite(v) for v in (median, l_unc, u_unc)):
        return True
    if median == 0:
        return False
    return max(abs(l_unc), abs(u_unc)) / abs(median) > max_rel


def _round_sig(x, sig=2):
    if x == 0 or not np.isfinite(x):
        return x
    p = math.floor(math.log10(abs(x)))
    return round(x, -(p - sig + 1))


def _decimals_for_sig(x, sig=2):
    """How many digits after the decimal point are needed to display `x` with `sig` sig figs."""
    if x == 0 or not np.isfinite(x):
        return 0
    p = math.floor(math.log10(abs(x)))
    return max(0, -(p - sig + 1))


def _fmt_decimal(median, l_unc, u_unc, decimals):
    return (
        f"${median:.{decimals}f}^{{+{_round_sig(u_unc):.{decimals}f}}}"
        f"_{{-{_round_sig(l_unc):.{decimals}f}}}$"
    )


def write_latex_table(filename, fit, perf_label, models, quantile=0.1):
    cells = {}
    for model in models:
        if fit.get(model) is None:
            continue
        for key, _ in PARAM_COLUMNS:
            summary = fit[model][key]
            med = _q(summary, 0.5)
            l_unc = abs(med - _q(summary, quantile))
            u_unc = abs(_q(summary, 1 - quantile) - med)
            cells[(model, key)] = (
                None if _is_pathological(med, l_unc, u_unc) else (med, l_unc, u_unc)
            )

    decimals_for_col = {}
    for key, _ in PARAM_COLUMNS:
        ds = [
            _decimals_for_sig(u)
            for m in models
            if cells.get((m, key)) is not None
            for u in cells[(m, key)][1:]
        ]
        decimals_for_col[key] = max(ds) if ds else 2

    rows = []
    for model in models:
        if fit.get(model) is None:
            continue
        row = [labels[model]]
        for key, _ in PARAM_COLUMNS:
            triple = cells[(model, key)]
            if triple is None:
                row.append(r"\textemdash")
            else:
                row.append(_fmt_decimal(*triple, decimals_for_col[key]))
        rows.append(" & ".join(row) + r" \\")

    header = "Architecture & " + " & ".join(label for _, label in PARAM_COLUMNS) + r" \\"
    perf_label_tex = perf_label.replace("_", r"\_")
    caption = (
        f"Power-law fit parameters for {perf_label_tex}: {fit['label_metric']} "
        f"vs.\\ {fit['label_cost']}. Entries marked \\textemdash{{}} have bootstrap "
        f"uncertainties exceeding {PATHOLOGICAL_REL_THRESHOLD:g} times the central value."
    )
    body = "\n".join(
        [
            r"\begin{table}[h]",
            r"  \centering",
            r"  \begin{tabular}{l" + "c" * len(PARAM_COLUMNS) + "}",
            r"    \toprule",
            "    " + header,
            r"    \midrule",
            *("    " + r for r in rows),
            r"    \bottomrule",
            r"  \end{tabular}",
            f"  \\caption{{{caption}}}",
            f"  \\label{{tab:{perf_label}}}",
            r"\end{table}",
            "",
        ]
    )
    with open(filename, "w") as file:
        file.write(body)


def scan_scaling_laws(
    perf_metrics,
    cost_metrics,
    prefix="",
    save=True,
    do_fit=True,
    n_bootstrap=100,
    quantile=0.1,
    export_latex=False,
    used_models=None,
):
    perf = {}
    models_per_label = {}
    sizes_per_label = {}
    for label, vals in perf_metrics.items():
        with open(f"scaling/{label}.json") as file:
            entries = json.load(file)
        by_key = {(e["model"], e["size"]): e for e in entries}
        models_in_file = {e["model"] for e in entries}
        if used_models is not None:
            models_in_file &= set(used_models)
        models_in_file -= set(vals.get("exclude_models", []))
        models_per_label[label] = [m for m in MODEL_ORDER if m in models_in_file]
        sizes_per_label[label] = sorted({e["size"] for e in entries})
        perf[label] = {}
        for metric, metric_label in zip(vals["keys"], vals["labels"], strict=True):
            perf[label][metric] = {"label": metric_label}
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
        cost[label] = {"label": vals["label"]}
        for model in union_models:
            cost[label][model] = {}
            for size in union_sizes:
                cost[label][model][size] = walk_dict(metrics[str(size)][model], vals["keys"])

    for perf_label, perf_dict in perf.items():
        models = models_per_label[perf_label]
        sizes = sizes_per_label[perf_label]
        filename_fit = f"scaling/{'' if prefix == '' else prefix + '_'}{perf_label}_fit.json"
        if not do_fit:
            with open(filename_fit) as file:
                fits = json.load(file)
        else:
            fits = {metric_label: {} for metric_label in perf_dict.keys()}

        filename = f"scaling/{'' if prefix == '' else prefix + '_'}{perf_label}.pdf"
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
                    )

        if do_fit and save:
            with open(filename_fit, "w") as file:
                json.dump(fits, file, indent=2)

        if export_latex and "loss" in fits and "params" in fits["loss"]:
            filename_tex = f"scaling/{'' if prefix == '' else prefix + '_'}{perf_label}_table.tex"
            write_latex_table(
                filename_tex,
                fits["loss"]["params"],
                perf_label,
                models,
                quantile=quantile,
            )
