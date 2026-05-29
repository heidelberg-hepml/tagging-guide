from scaling.scan import scan_scaling_laws

do_fit = True
n_bootstrap = 100
quantile = 0.1
save = True
export_latex = True
used_models = None

cost_metrics = {
    "params": {
        "label": "Network parameters",
        "file": "cost_estimate/basics.json",
        "keys": ["params"],
    },
}

perf_metrics = {
    "jetclass_apr1": {
        "labels": [
            "Loss",
            "Averaged AUC",
            "Accuracy",
            r"$H\to b\bar b$ $\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
            r"$H\to c\bar c$ $\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
            r"$H\to gg$ $\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
            r"$H\to 4q$ $\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
            r"$H\to l\nu q\bar q'$ $\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.99$",
            r"$t\to b q\bar q'$ $\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
            r"$t\to bl\nu$ $\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.995$",
            r"$W\to q\bar q$ $\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
            r"$Z\to q\bar q$ $\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
        ],
        "keys": [
            "loss",
            "auc_ovo",
            "accuracy",
            "rej05_HToBB",
            "rej05_HToCC",
            "rej05_HToGG",
            "rej05_HToWW4Q",
            "rej099_HToWW2Q1L",
            "rej05_TTBar",
            "rej0995_TTBarLep",
            "rej05_WToQQ",
            "rej05_ZToQQ",
        ],
    },
    "toptagxl_apr1": {
        "labels": [
            "Loss",
            "AUC",
            "Accuracy",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
        ],
        "keys": ["loss", "auc", "accuracy", "rej05"],
    },
    "atlastop_apr1": {
        "labels": [
            "Loss",
            "AUC",
            "Accuracy",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
            "AUC relative uncertainty",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$ relative uncertainty",
        ],
        "keys": [
            "loss",
            "auc",
            "accuracy",
            "rej05",
            "auc_unc_total",
            "rej05_unc_total",
        ],
    },
}

if __name__ == "__main__":
    scan_scaling_laws(
        perf_metrics,
        cost_metrics,
        do_fit=do_fit,
        n_bootstrap=n_bootstrap,
        quantile=quantile,
        save=save,
        prefix="perf",
        export_latex=export_latex,
        used_models=used_models,
    )
