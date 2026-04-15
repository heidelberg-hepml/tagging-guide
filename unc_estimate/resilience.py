import json
import time

import numpy as np

ARCHS = ["tr", "lloca", "part", "slim", "gn3"]
SIZES = np.arange(-2.0, 2.1, step=1.0)
SYSTS = {
    "track fake rate": ["tfj", "nominal", "tfl"],
    "track efficiency": ["teg", "nominal", "tej"],
    "track bias": ["bias", "nominal"],
    "cluster energy scale": ["esup", "nominal", "esdown"],
    "cluster energy resolution": ["cer", "nominal"],
    "cluster position resolution": ["cpos", "nominal"],
    "sig modeling": ["ttbar_herwig", "ttbar_pythia"],
    "bkg parton shower": ["dipole", "angular"],
    "bkg hadronization": ["cluster", "string"],
}
FILE = "results/atlastop_blueprint.json"
KEYS = ["AUC", "accuracy", "br_0.5"]

with open(FILE) as file:
    METRICS = json.load(file)


def main(save=True):
    all_systs = dict()
    for syst in SYSTS.keys():
        print(f"################ syst={syst} ################")
        all_systs[syst] = single_systematic(syst, save=save)

    if save:
        with open("unc_estimate/unc_estimates.json", "w") as file:
            json.dump(all_systs, file, indent=2)


def single_systematic(syst, save=True):
    results = dict()

    t0 = time.time()
    for size in SIZES:
        print(f"################ {size} ################")
        size = str(size)
        results[size] = dict()
        for arch in ARCHS:
            results[size][arch] = dict()
            for key in KEYS:
                if SYSTS[syst][1] == "nominal":
                    reference = METRICS[size][arch][key]
                else:
                    reference = METRICS[size][arch][SYSTS[syst][1]][key]
                metric1 = METRICS[size][arch][SYSTS[syst][0]][key]
                if len(SYSTS[syst]) == 3:
                    metric2 = METRICS[size][arch][SYSTS[syst][2]][key]
                else:
                    metric2 = None
                metric_unc = calculate_uncertainty(metric1, metric2, reference)
                results[size][arch][key] = metric_unc.tolist()
                print(f"{arch:<6} {size}: {key} (rel. diff.) = {results[size][arch][key]}")

    dt = time.time() - t0
    print(f"Finished scan after {dt:.2f}s")
    return results


def calculate_uncertainty(metric_syst1, metric_syst2, reference):
    metric_syst1 = np.array(metric_syst1)
    reference = np.array(reference)
    relative_metric_syst1 = np.abs((metric_syst1 - reference) / reference)
    if metric_syst2 is not None:
        metric_syst2 = np.array(metric_syst2)
        relative_metric_syst2 = np.abs((metric_syst2 - reference) / reference)
        metric_unc = np.maximum(relative_metric_syst1, relative_metric_syst2)
        return metric_unc
    else:
        metric_unc = relative_metric_syst1
        return metric_unc


if __name__ == "__main__":
    main()
