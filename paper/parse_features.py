"""Collect the input-feature runs into paper/features.json.

One entry per (dataset, feature set, model), with the metrics listed as one
value per seed, in the same format as the other paper/*.json files. Run from
repo root: python -m paper.parse_features
"""

import glob
import json
import os
import re
from collections import defaultdict

OUTPUT = "paper/features.json"

# dataset -> feature sets, in the order they appear on the x-axis of features.pdf
FEATURES = {
    "jetclass": ["fourmomenta", "pid", "displacements", "all"],
    "jetset": ["fourmomenta", "ipsig", "ip", "all"],
}
MODEL_ORDER = ["tr", "part", "lgatr", "lloca", "slim"]


def main():
    entries = []
    for dataset, feature_sets in FEATURES.items():
        for features in feature_sets:
            grouped = defaultdict(lambda: defaultdict(list))  # model -> {key: [values]}
            sizes, train_sizes = {}, {}
            for run_dir in sorted(glob.glob(f"runs/{dataset}_{features}/v*_*")):
                results_files = glob.glob(f"{run_dir}/*/results_*.json")
                if not results_files:
                    continue
                results_file = max(results_files, key=os.path.getmtime)
                with open(results_file) as f:
                    results = json.load(f)
                model = results.pop("model_name")
                sizes[model] = results.pop("model_size")
                train_sizes[model] = results.pop("train_size")
                for key, value in results.items():
                    grouped[model][key].append(float(f"{value:.6g}"))

            for model in [m for m in MODEL_ORDER if m in grouped]:
                entries.append(
                    {
                        "dataset": dataset,
                        "features": features,
                        "model": model,
                        "size": float(sizes[model]),
                        "train_size": train_sizes[model],
                        **dict(grouped[model]),
                    }
                )

    text = json.dumps(entries, indent=2)
    # collapse innermost (number) lists onto a single line; nested lists are skipped via [^\[\]]
    text = re.sub(
        r"\[\s+([^\[\]]+?)\s+\]",
        lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]",
        text,
    )
    with open(OUTPUT, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
