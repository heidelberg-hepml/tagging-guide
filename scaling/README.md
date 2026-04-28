# Scaling Law Analysis

## Data (`scaling/data/`)

| File | Description |
|------|-------------|
| `basics.json` | Architecture specs: params and forward FLOPs per sample, per model per size — copy of `cost_estimate/basics.json` |
| `grid_runs.json` | 1-epoch grid scan. Fields: `model`, `N`, `D`, `d_label`, `val_loss` |
| `optimal_runs.json` | Training curves from close to compute-optimal runs. Fields: `exp_name`, `model`, `N`, `D`, `d_label`, `val_loss`, `elapsed_seconds`, `train_iters`, `train_losses` |
| `data_scaling.json` | Best val_loss from multi-epoch data repetition runs. Fields: `model`, `d_label`, `D`, `best_val_loss` |

`cost_estimate/inference_cpu.json` (for inference-time-constrained plots).

## Generating plots

```bash

# Data scaling with repetition
python scaling/scripts/plot_data_scaling.py
# -> results/data_scaling.pdf
# -> results/data_scaling_fits.json, data_scaling_fits.tex

# Scaling law fits: N-D surface + iso-FLOP curves + fits json/tex (run first)
python scaling/scripts/plot_scaling_laws.py
# -> results/scaling_law_ND_{model}.pdf
# -> results/scaling_law_isoflop_{model}.pdf
# -> results/scaling_law_fits.json, scaling_law_fits.tex

# Compute-optimal training curves (per-model + combined, FLOPs + GPU-hrs)
python scaling/scripts/plot_optimal_training_curves.py
# -> results/optimal_training_curves_{model}.pdf
# -> results/optimal_training_curves_gpuhrs_{model}.pdf
# -> results/optimal_training_curves_combined.pdf
# -> results/optimal_training_curves_gpuhrs_combined.pdf

# Compute-optimal trajectory plots (N vs D, N vs inference time) + budget-constrained L curves
python scaling/scripts/plot_compute_optimal_trajectories.py
# -> results/compute_optimal_trajectories_NvsD.pdf
# -> results/compute_optimal_trajectories_NvsTinference_cpu.pdf
# -> results/constrained_Tinference_cpu_LvsD.pdf
```

## Fit details

**Data scaling fit** (`plot_data_scaling.py`): `L(D) = L_inf + B_rep/D^{β_rep}` (L_inf free)
- Huber loss in log-space, L-BFGS-B
- 100 bootstrap resamples, 10/90 percentile band
- Horizontal shaded band shows `L_inf ± L_inf_std` per model

**Scaling law joint fit** (`plot_scaling_laws.py`): `L(N,D) = L_inf + A/N^α + B/D^β`
- `L_inf` fixed to `L_INF_FIXED` from `plot_data_scaling.py`, change in `scaling_utils.py`
- Huber loss in log-space, L-BFGS-B with multiple warm starts
- 100 bootstrap resamples, 10/90 percentile band (80% CI)
- `γ` and `a` are computed **numerically** from the compute-optimal trajectory (since `C = 3·fwd_flops(N)·D` and `fwd_flops(N)` is measured, not exactly `2N·seqlen`)


## Fit filters

**Scaling law fits** (`plot_scaling_laws.py`):
- Monotonic N-scaling: at fixed D, if increasing N gives worse loss, that point is excluded — later improvements still kept (lloca and slim can be unstable at small D with large N)

**Combined optimal training curves** (`plot_optimal_training_curves.py`):
- Only runs within 10% of the compute-optimal prediction at that run's total compute are scattered in the combined plots; per-model panels show all runs


## Scripts

| Script | Purpose |
|--------|---------|
| `scaling_utils.py` | Shared constants (`MODELS`, `MODEL_STYLE`, `L_INF_FIXED`), FLOPs interpolation, compute-optimal trajectory, measurement loader |
| `plot_style.py` | ATLAS mplhep style setup |
| `plot_scaling_laws.py` | Fit `L(N,D) = L_inf + A/N^α + B/D^β`; plot N-D surface (`scaling_law_ND_*`) and iso-FLOP curves (`scaling_law_isoflop_*`); write fits JSON + TeX |
| `plot_data_scaling.py` | Fit and plot `L(D)` with repetition; L_inf ± 1σ shaded band |
| `plot_optimal_training_curves.py` | Plot training loss vs compute / GPU-hours for optimal runs (per-model + combined) |
| `plot_compute_optimal_trajectories.py` | N-vs-D and N-vs-T_inference compute-optimal trajectories + inference-time budget-constrained loss plots |
