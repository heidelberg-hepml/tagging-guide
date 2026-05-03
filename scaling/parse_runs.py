import glob
import json
import re
from collections import defaultdict
from pathlib import Path

PATTERN = "runs/horeka3/v1_*_jetclass_*"
OUTPUT = "scaling/jetclass_apr1.json"
KEYS = None


def main():
    grouped = defaultdict(lambda: defaultdict(list))  # (model, size) -> {key: [values]}
    train_sizes = {}  # (model, size) -> train_size
    for run_dir in sorted(glob.glob(PATTERN)):
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            continue
        for results_file in sorted(run_dir.glob("results_*.json")):
            with open(results_file) as f:
                results = json.load(f)
            model = results.pop("model_name")
            size = results.pop("model_size")
            train_sizes[(model, size)] = results.pop("train_size")
            keys = KEYS if KEYS is not None else results
            for key in keys:
                grouped[(model, size)][key].append(float(f"{results[key]:.6g}"))

    entries = [
        {
            "model": model,
            "size": float(size),
            "train_size": train_sizes[(model, size)],
            **dict(metrics),
        }
        for (model, size), metrics in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0]))
    ]

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
