from scaling.plot import COST_LABELS, PERF_MODEL_LABELS
from scaling.scan import scan_scaling_laws

do_fit = True
n_bootstrap = 100
quantile = 0.1
save = True
export_latex = True
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
    "jetclass5ep_final": {
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
    "atlastop5ep_final": {
        "exclude_models": ["pelicanlite", "particlenet", "lorentznet"],
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
    "jetset5ep_final": {
        "exclude_models": ["part"],
        "labels": [
            "Loss",
            "AUC",
            "Accuracy",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_b=0.7$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_b=0.7$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_b=0.7$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_c=0.3$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_c=0.3$",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_c=0.3$",
        ],
        "sublabels": [
            None,
            None,
            None,
            r"$b$-tag, $c$-jets",
            r"$b$-tag, light-jets",
            r"$b$-tag, $\tau$-jets",
            r"$c$-tag, $b$-jets",
            r"$c$-tag, light-jets",
            r"$c$-tag, $\tau$-jets",
        ],
        "keys": [
            "loss",
            "auc_ovo",
            "accuracy",
            "rej07_bottomjet_cjets",
            "rej07_bottomjet_ujets",
            "rej07_bottomjet_taujets",
            "rej03_charmjet_bjets",
            "rej03_charmjet_ujets",
            "rej03_charmjet_taujets",
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
        label_top_left=label_top_left,
        model_labels=model_labels,
    )
