import glob
import json
import re
from collections import defaultdict
from pathlib import Path

PATTERN = "runs/horeka3/v1_*_jetclass_*"
OUTPUT = "results/jetclass_apr1.json"
KEYS = None


def match_pattern(pattern_name, name):
    prefix, middle, suffix = pattern_name.split("*")
    inner = name[len(prefix) : len(name) - len(suffix)] if suffix else name[len(prefix) :]
    return inner.split(middle, 1)


def main():
    pattern_name = Path(PATTERN).name
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for run_dir in sorted(glob.glob(PATTERN)):
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            continue
        model_scale, _ = match_pattern(pattern_name, run_dir.name)
        model, scale = model_scale.rsplit("_", 1)
        scale = f"{float(scale):.1f}"
        for results_file in sorted(run_dir.glob("results_*.json")):
            with open(results_file) as f:
                results = json.load(f)
            keys = KEYS if KEYS is not None else results
            for key in keys:
                out[scale][model][key].append(float(f"{results[key]:.6g}"))

    out = {
        scale: {model: dict(metrics) for model, metrics in models.items()}
        for scale, models in sorted(out.items(), key=lambda kv: float(kv[0]))
    }

    text = json.dumps(out, indent=2)
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
