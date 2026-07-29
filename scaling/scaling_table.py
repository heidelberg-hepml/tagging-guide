import json

from scaling.plot import PERF_MODEL_LABELS, dataset_label, labels

MODELS = ["tr", "part", "lloca", "lgatr", "slim"]
DATASETS = ["atlastop5ep_final", "jetclass5ep_final", "jetset5ep_final"]
QUANTILE = 0.1
OUTPUT = "scaling/scaling_table.tex"


def format_entry(summary, fmt):
    median = summary["0.5"]
    l_unc = median - summary[str(QUANTILE)]
    u_unc = summary[str(1 - QUANTILE)] - median
    return f"${fmt(median)}_{{-{fmt(l_unc)}}}^{{+{fmt(u_unc)}}}$"


def main():
    params = {}
    for dataset in DATASETS:
        with open(f"scaling/perf_{dataset}_fit.json") as file:
            params[dataset] = json.load(file)["loss"]["params"]

    formats = {"beta": lambda x: f"{x:.2f}", "B": lambda x: f"{x:.1f}"}
    rows = []
    for model in MODELS:
        label = PERF_MODEL_LABELS.get(model, labels[model])
        cells = [
            format_entry(params[dataset][model][key], formats[key])
            if params[dataset].get(model)
            else r"\textemdash"
            for key in ("beta", "B")
            for dataset in DATASETS
        ]
        rows.append("  " + " & ".join([label, *cells]) + r" \\")

    n = len(DATASETS)
    header_coeffs = (
        f"  & \\multicolumn{{{n}}}{{c}}{{Exponent $\\beta$}}"
        f" & \\multicolumn{{{n}}}{{c}}{{Prefactor $B$}} \\\\"
    )
    dataset_labels = " & ".join(dataset_label(d) for d in DATASETS)
    header_datasets = f"  & {dataset_labels} & {dataset_labels} \\\\"
    body = "\n".join(
        [
            r"\begin{tabular}{l" + "c" * 2 * n + "}",
            r"  \toprule",
            header_coeffs,
            header_datasets,
            r"  \midrule",
            *rows,
            r"  \bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    with open(OUTPUT, "w") as file:
        file.write(body)


if __name__ == "__main__":
    main()
