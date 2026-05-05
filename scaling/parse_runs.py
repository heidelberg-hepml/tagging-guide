import json
import re
from collections import defaultdict
from pathlib import Path

PATTERN = r"runs/horeka3/v9_.*"
OUTPUT = "scaling/jetclass_apr1.json"
# PATTERN = r"runs/horeka3/v10_.*"
# OUTPUT = "scaling/toptagxl_apr1.json"
# PATTERN = r"runs/horeka3/v12_.*"
# OUTPUT = "scaling/atlastop_apr1.json"

REGEX_METACHARACTERS = set(".*+?[](){}|^$\\")


def find_matching_dirs(pattern):
    """Find directories whose path matches the regex pattern, component by component."""
    candidates = [Path(".")]
    for part in pattern.split("/"):
        is_regex = any(c in REGEX_METACHARACTERS for c in part)
        new_candidates = []
        if is_regex:
            regex = re.compile(part)
            for cand in candidates:
                if not cand.is_dir():
                    continue
                for child in cand.iterdir():
                    if regex.fullmatch(child.name):
                        new_candidates.append(child)
        else:
            for cand in candidates:
                p = cand / part
                if p.exists():
                    new_candidates.append(p)
        candidates = new_candidates
    return [c for c in candidates if c.is_dir()]


def main():
    grouped = defaultdict(lambda: defaultdict(list))  # (model, size) -> {key: [values]}
    train_sizes = {}  # (model, size) -> train_size
    for run_dir in sorted(find_matching_dirs(PATTERN)):
        results_files = list(run_dir.glob("results_*.json"))
        if not results_files:
            continue
        results_file = max(results_files, key=lambda p: p.stat().st_mtime)  # most recent file
        with open(results_file) as f:
            results = json.load(f)
        model = results.pop("model_name")
        size = results.pop("model_size")
        train_sizes[(model, size)] = results.pop("train_size")
        for key, value in results.items():
            grouped[(model, size)][key].append(float(f"{value:.6g}"))

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
