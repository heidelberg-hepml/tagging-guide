from scaling.scan import scan_scaling_laws

do_fit = True
n_bootstrap = 100
quantile = 0.1
save = True
used_models = None

cost_metrics = {
    "params": {
        "label": "Network parameters",
        "file": "cost_estimate/basics.json",
        "keys": ["params"],
    },
    "flops_measured": {
        "label": "Inference FLOPs, $N=50$ (measured)",
        "file": "cost_estimate/basics.json",
        "keys": ["flops"],
    },
    "flops_estimated": {
        "label": "Inference FLOPs, $N=50$ (estimated)",
        "file": "cost_estimate/energy_model.json",
        "keys": ["flops"],
    },
    "energy": {
        "label": "Energy [pJ], $N=50$",
        "file": "cost_estimate/energy_model.json",
        "keys": ["float16"],
    },
    "inference_cpu": {
        "label": "CPU inference time [ms], $N=50$",
        "file": "cost_estimate/inference_cpu.json",
        "keys": ["mean"],
    },
    "inference_gpu_bs512": {
        "label": "GPU inference time [ms], $N=50$, BS$=512$",
        "file": "cost_estimate/inference_gpu_bs512.json",
        "keys": ["mean"],
    },
    "memory_gpu_bs512": {
        "label": "GPU memory usage [GB], $N=50$, BS$=512$",
        "file": "cost_estimate/inference_gpu_bs512.json",
        "keys": ["memory_alloc"],
    },
    "train_gpu_bs512": {
        "label": "GPU training time [ms], $N=50$, BS$=512$",
        "file": "cost_estimate/train_gpu_bs512.json",
        "keys": ["mean"],
    },
}

perf_metrics = {
    "jetclass_apr1": {
        "labels": [
            "Loss",
            "Averaged AUC",
        ],
        "keys": ["loss", "auc_ovo"],
    },
    "toptagxl_apr1": {
        "labels": [
            "Loss",
            "AUC",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
        ],
        "keys": ["loss", "auc", "rej05"],
    },
    "atlastop_apr1": {
        "labels": [
            "Loss",
            "AUC",
            r"$\epsilon_\mathrm{bkg}^{-1}$ @ $\epsilon_\mathrm{sig}=0.5$",
        ],
        "keys": ["loss", "auc", "rej05"],
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
        prefix="cost",
        used_models=used_models,
    )
