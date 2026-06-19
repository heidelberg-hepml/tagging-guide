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
        "label": "Energy on GPU [J], $N=50$",
        "file": "cost_estimate/energy_model.json",
        "keys": ["energy"],
        "scale": 1e-12,
    },
    "inference_cpu": {
        "label": "CPU inference time [ms], $N=50$",
        "file": "cost_estimate/inference_cpu.json",
        "keys": ["mean"],
    },
    "memory_cpu": {
        "label": "CPU memory usage [GB], $N=50$",
        "file": "cost_estimate/inference_cpu.json",
        "keys": ["memory_rss"],
    },
    "inference_gpu_bs512": {
        "label": "GPU inference time [ms], BS$=512$",
        "file": "cost_estimate/inference_gpu.json",
        "keys": ["mean"],
    },
    "memory_gpu_bs512": {
        "label": "GPU memory usage [GB], BS$=512$",
        "file": "cost_estimate/inference_gpu.json",
        "keys": ["memory_alloc"],
    },
    "train_gpu_bs512": {
        "label": "GPU training time [ms], BS$=512$",
        "file": "cost_estimate/train_gpu_bs512.json",
        "keys": ["mean"],
    },
}

perf_metrics = {
    "jetclass1ep_jun1": {
        "labels": [
            "Loss",
            "Averaged AUC",
        ],
        "keys": ["loss", "auc_ovo"],
    },
    "jetclass5ep_jun1": {
        "labels": [
            "Loss",
            "Averaged AUC",
        ],
        "keys": ["loss", "auc_ovo"],
    },
    "toptagxl1ep_jun1": {
        "labels": [
            "Loss",
            "AUC",
        ],
        "keys": ["loss", "auc"],
    },
    "atlastop1ep_jun1": {
        "exclude_models": ["pelicanlite", "particlenet", "lorentznet"],
        "labels": [
            "Loss",
            "AUC",
        ],
        "keys": ["loss", "auc"],
    },
    "atlastop5ep_jun1": {
        "exclude_models": ["pelicanlite", "particlenet", "lorentznet"],
        "labels": [
            "Loss",
            "AUC",
        ],
        "keys": ["loss", "auc"],
    },
    "jetset1ep_jun1": {
        "labels": [
            "Loss",
            "AUC",
        ],
        "keys": ["loss", "auc_ovo"],
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
