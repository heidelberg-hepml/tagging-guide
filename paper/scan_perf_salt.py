from paper.plot import COST_LABELS, PERF_MODEL_LABELS
from paper.scan import scan_scaling_laws

do_fit = True
n_bootstrap = 100
quantile = 0.1
save = True
used_models = None
label_top_left = True
model_labels = PERF_MODEL_LABELS

cost_metrics = {
    "params": {
        "label": COST_LABELS["params"],
        "file": "cost_estimate/basics.json",
        "keys": ["params"],
    },
}

perf_metrics = {
    "jetclass5ep_final_wsalt": {
        "labels": [
            "Loss",
            "Averaged AUC",
            "Accuracy",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.99$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.995$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
        ],
        "sublabels": [
            None,
            None,
            None,
            r"$H\to b\bar b$",
            r"$H\to c\bar c$",
            r"$H\to gg$",
            r"$H\to 4q$",
            r"$H\to \ell\nu q\bar q'$",
            r"$t\to b q\bar q'$",
            r"$t\to b\ell\nu$",
            r"$W\to q\bar q$",
            r"$Z\to q\bar q$",
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
    "jetset5ep_final_wsalt": {
        "labels": [
            "Loss",
            "Averaged AUC",
            "Accuracy",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.99$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.995$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
        ],
        "sublabels": [
            None,
            None,
            None,
            r"$H\to b\bar b$",
            r"$H\to c\bar c$",
            r"$H\to gg$",
            r"$H\to 4q$",
            r"$H\to \ell\nu q\bar q'$",
            r"$t\to b q\bar q'$",
            r"$t\to b\ell\nu$",
            r"$W\to q\bar q$",
            r"$Z\to q\bar q$",
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
        used_models=used_models,
        label_top_left=label_top_left,
        model_labels=model_labels,
    )
