import numpy as np

from scaling.scan import scan_scaling_laws

models = ["slim", "lloca", "part", "tr", "gn3"]
sizes = np.arange(-2.0, 2.1, step=1.0).tolist()
do_fit = True
n_bootstrap = 100
quantile = 0.1
save = True

cost_metrics = {
    "params": {
        "label": "Network parameters",
        "file": "cost_estimate/basics.json",
        "keys": ["params"],
    },
}

perf_metrics = {
    "jetclass_mar2": {
        "labels": [
            "Loss",
            "JetClass AUC",
            "Accuracy",
            r"$t\to b q\bar q$ $\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
        ],
        "keys": ["loss", "auc_ovo", "accuracy", "rej05_TTBar"],
        "models": ["slim", "lloca", "part", "tr", "gn3"],
    },
    "toptagxl_mar2": {
        "labels": ["Loss", "AUC", "Accuracy"],
        "keys": ["loss", "auc", "accuracy"],
        "models": ["slim", "lloca", "part", "tr", "gn3"],
    },
    "toptagxlall_mar2": {
        "labels": ["Loss", "AUC", "Accuracy"],
        "keys": ["loss", "auc", "accuracy"],
        "models": ["slim", "lloca", "part", "tr"],
    },
}

if __name__ == "__main__":
    scan_scaling_laws(
        models,
        sizes,
        perf_metrics,
        cost_metrics,
        do_fit=do_fit,
        n_bootstrap=n_bootstrap,
        quantile=quantile,
        save=save,
        prefix="perf",
    )
