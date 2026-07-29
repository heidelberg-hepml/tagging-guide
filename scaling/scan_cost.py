from scaling.plot import COST_LABELS
from scaling.scan import scan_scaling_laws

do_fit = True
n_bootstrap = 100
quantile = 0.1
save = True
used_models = None

# FLOPs and CPU cost are quoted for the sparse L-GATr, the GPU metrics for the dense one
SPARSE_VARIANTS = {"lgatr": "lgatr-sparse"}

cost_metrics = {
    "params": {
        "label": COST_LABELS["params"],
        "file": "cost_estimate/basics.json",
        "keys": ["params"],
    },
    "flops_measured": {
        "label": COST_LABELS["flops_measured"],
        "variants": SPARSE_VARIANTS,
        "file": "cost_estimate/basics.json",
        "keys": ["flops"],
    },
    "flops_estimated": {
        "label": COST_LABELS["flops_estimated"],
        "variants": SPARSE_VARIANTS,
        "file": "cost_estimate/energy_model.json",
        "keys": ["flops"],
    },
    "energy": {
        "label": COST_LABELS["energy"],
        "file": "cost_estimate/energy_model.json",
        "keys": ["energy"],
        "scale": 1e-12,
    },
    "inference_cpu": {
        "label": COST_LABELS["inference_cpu"],
        "variants": SPARSE_VARIANTS,
        "file": "cost_estimate/inference_cpu.json",
        "keys": ["mean"],
    },
    "memory_cpu": {
        "label": COST_LABELS["memory_cpu"],
        "variants": SPARSE_VARIANTS,
        "file": "cost_estimate/inference_cpu.json",
        "keys": ["memory_rss"],
    },
    "inference_gpu_bs512": {
        "label": COST_LABELS["inference_gpu_bs512"],
        "file": "cost_estimate/inference_gpu.json",
        "keys": ["mean"],
    },
    "memory_gpu_bs512": {
        "label": COST_LABELS["memory_gpu_bs512"],
        "file": "cost_estimate/inference_gpu.json",
        "keys": ["memory_alloc"],
    },
    "train_gpu_bs512": {
        "label": COST_LABELS["train_gpu_bs512"],
        "file": "cost_estimate/train_gpu_bs512.json",
        "keys": ["mean"],
    },
}

perf_metrics = {
    "jetclass5ep_final": {
        "labels": [
            "Loss",
            "Averaged AUC",
        ],
        "keys": ["loss", "auc_ovo"],
    },
    "atlastop5ep_final": {
        "exclude_models": ["pelicanlite", "particlenet", "lorentznet"],
        "labels": [
            "Loss",
            "AUC",
        ],
        "keys": ["loss", "auc"],
    },
    "jetset5ep_final": {
        "exclude_models": ["part"],
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
        pareto=True,
    )
