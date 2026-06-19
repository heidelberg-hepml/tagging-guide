from scaling.scan import scan_scaling_laws

do_fit = True
n_bootstrap = 100
quantile = 0.1
save = True
export_latex = False
used_models = None

cost_metrics = {
    "params": {
        "label": "Network parameters",
        "file": "cost_estimate/basics.json",
        "keys": ["params"],
    },
}

REJ05_LABEL = r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$"
AUC_LABEL = "AUC"

# (suffix, wording) for each ATLAS top-tagging simulation variant
# (https://opendata.cern.ch/record/80030)
SYSTEMATICS = [
    ("nominal", "nominal"),
    ("angular", "Herwig angular-ordered shower"),
    ("dipole", "Herwig dipole shower"),
    ("string", "Lund string hadronization"),
    ("cluster", "Sherpa cluster hadronization"),
    ("ttbar_pythia", r"Pythia $t\bar t$ signal"),
    ("ttbar_herwig", r"Herwig $t\bar t$ signal"),
    ("esup", "cluster energy scale up"),
    ("esdown", "cluster energy scale down"),
    ("cer", "cluster energy resolution"),
    ("cpos", "cluster position"),
    ("bias", "track bias"),
    ("teg", "track efficiency, global"),
    ("tej", "track efficiency, in jets"),
    ("tfl", "track fake rate, loose"),
    ("tfj", "track fake rate, in jets"),
    ("sig_ISRx2", r"signal ISR $\times2$"),
    ("sig_ISRxp5", r"signal ISR $\times0.5$"),
    ("bkg_ISRx2", r"background ISR $\times2$"),
    ("bkg_ISRxp5", r"background ISR $\times0.5$"),
    ("sig_FSRx2", r"signal FSR $\times2$"),
    ("sig_FSRxp5", r"signal FSR $\times0.5$"),
    ("bkg_FSRx2", r"background FSR $\times2$"),
    ("bkg_FSRxp5", r"background FSR $\times0.5$"),
]

keys = [f"rej05_{suffix}" for suffix, _ in SYSTEMATICS]
keys += [f"auc_{suffix}" for suffix, _ in SYSTEMATICS]
labels = [f"{REJ05_LABEL}, {wording}" for _, wording in SYSTEMATICS]
labels += [f"{AUC_LABEL}, {wording}" for _, wording in SYSTEMATICS]

atlastop = {
    "exclude_models": ["pelicanlite", "particlenet", "lorentznet"],
    "labels": labels,
    "keys": keys,
}

perf_metrics = {
    "atlastop1ep_jun1": atlastop,
    "atlastop5ep_jun1": atlastop,
}

if __name__ == "__main__":
    scan_scaling_laws(
        perf_metrics,
        cost_metrics,
        do_fit=do_fit,
        n_bootstrap=n_bootstrap,
        quantile=quantile,
        save=save,
        prefix="perf-syst",
        export_latex=export_latex,
        used_models=used_models,
    )
