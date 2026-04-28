"""Plot training loss vs compute for compute-optimal runs.

Reads pre-collected data from scaling/data/optimal_runs.json.
Outputs per-model and combined PDFs for both FLOPs and GPU-hours x-axes.

Usage:
    python scaling/scripts/plot_optimal_training_curves.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scaling.scripts.scaling_utils import (
    MODELS, MODEL_STYLE, build_flops_fn, compute_opt_trajectory,
)
from scaling.scripts.plot_style import setup_style, SCATTER_KW, FILL_ALPHA, CMAP

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def smooth_ema(values, window):
    """Exponential moving average smoothing."""
    if window <= 1 or len(values) < 2:
        return values
    alpha = 2.0 / (window + 1)
    smoothed = np.zeros_like(values, dtype=float)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def main():
    parser = argparse.ArgumentParser(description="Plot compute-optimal training curves")
    parser.add_argument("--data", default="scaling/data/optimal_runs.json")
    parser.add_argument("--basics", default="scaling/data/basics.json")
    parser.add_argument("--fits", default="scaling/results/scaling_law_fits.json")
    parser.add_argument("--output-dir", default="scaling/results")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--smooth-window", type=int, default=50)
    parser.add_argument("--filter-tol", type=float, default=float("inf"),
                        help="Filter runs where |val_loss - L_opt|/L_opt > tol. Default inf = no filter.")
    parser.add_argument("--color-code", choices=["D", "N"], default="N",
                        help="Color code per-model curves by dataset size (D) or model size (N)")
    args = parser.parse_args()

    setup_style()

    with open(args.basics) as f:
        basics = json.load(f)
    with open(args.data) as f:
        opt_data = json.load(f)

    # Build FLOPs functions
    flops_fns = {}
    for model in MODELS:
        try:
            flops_fns[model] = build_flops_fn(basics, model)
        except Exception:
            pass

    # Load scaling law fits for 10% filter and overlay
    scaling_fits = {}
    if os.path.exists(args.fits):
        with open(args.fits) as f:
            scaling_fits = json.load(f)

    # Organize runs by model and apply 5% filter vs compute-optimal loss
    runs = {}  # model -> list of run dicts
    for run in opt_data["runs"]:
        model = run["model"]
        if model not in flops_fns:
            continue

        # Compare val_loss to compute-optimal loss at this run's total compute
        if model in scaling_fits and run["val_loss"] is not None and run["N"] is not None:
            fit = scaling_fits[model]
            fwd_flops = flops_fns[model](run["N"])
            C_run = 3 * fwd_flops * run["D"]
            L_opt, _ = compute_opt_trajectory(
                fit["L_inf"], fit["A"], fit["alpha"],
                fit["B"], fit["beta"], flops_fns[model], [C_run])
            L_opt_val = L_opt[0]
            rel_diff = abs(run["val_loss"] - L_opt_val) / L_opt_val
            if rel_diff > args.filter_tol:
                print(f"  SKIP {run['exp_name']}: val_loss={run['val_loss']:.4f} "
                      f"opt={L_opt_val:.4f} diff={rel_diff:.1%}")
                continue

        if model not in runs:
            runs[model] = []
        runs[model].append(run)

    if not runs:
        print("No compute-optimal runs found!")
        return

    models_with_data = [m for m in MODELS if m in runs]
    os.makedirs(args.output_dir, exist_ok=True)

    # Global colorbar range based on chosen color-code
    color_key = args.color_code  # "D" or "N"
    color_label = "Dataset size $D$" if color_key == "D" else "Model parameters $N$"
    all_vals = [r[color_key] for m in models_with_data for r in runs[m]
                if r[color_key] is not None]
    global_norm = LogNorm(vmin=min(all_vals), vmax=max(all_vals))
    cmap = plt.get_cmap(CMAP)

    # Shared y limits
    y_lo, y_hi = 0.3, 2.5

    # =====================================================================
    # Helper to plot a single model panel
    # =====================================================================
    def _plot_panel(ax, model, x_mode="flops"):
        """Plot training curves for one model. x_mode: 'flops' or 'gpuhrs'."""
        style = MODEL_STYLE[model]
        model_runs = sorted(runs[model], key=lambda r: r["D"])

        for run in model_runs:
            N = run["N"]
            if N is None or model not in flops_fns:
                continue
            fwd_flops = flops_fns[model](N)
            iters = np.array(run["train_iters"], dtype=float)
            losses = np.array(run["train_losses"], dtype=float)

            if x_mode == "flops":
                x = 3 * fwd_flops * iters * args.batch_size
            else:  # gpuhrs
                if run["elapsed_seconds"] is None or len(iters) < 2:
                    continue
                total_iters = iters[-1]
                x = run["elapsed_seconds"] * iters / total_iters / 3600

            color = cmap(global_norm(run[color_key]))
            if args.smooth_window > 0:
                losses = smooth_ema(losses, args.smooth_window)
            ax.plot(x, losses, color=color, linewidth=0.8, alpha=0.7)

            if run["val_loss"] is not None:
                ax.scatter([x[-1]], [run["val_loss"]], color=color,
                           marker='o', s=50, zorder=5, **SCATTER_KW)

        # Compute-optimal overlay
        if model in scaling_fits and model in flops_fns and x_mode == "flops":
            fit = scaling_fits[model]
            C_range = np.logspace(10, 17, 200)
            L_opt, _ = compute_opt_trajectory(
                fit["L_inf"], fit["A"], fit["alpha"],
                fit["B"], fit["beta"], flops_fns[model], C_range)
            ax.plot(C_range, L_opt, 'k--', linewidth=1.5, alpha=0.5,
                    label='Compute-optimal')

        ax.set_xscale('log')
        xlabel = 'Compute (FLOPs)' if x_mode == "flops" else 'A100 GPU-hours'
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Loss')
        ax.set_ylim(y_lo, y_hi)

    # =====================================================================
    # Per-model individual plots (FLOPs)
    # =====================================================================
    for model in models_with_data:
        fig, ax = plt.subplots(1, 1)
        _plot_panel(ax, model, x_mode="flops")
        ax.set_title(MODEL_STYLE[model]["label"])
        ax.legend(loc='upper right')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=global_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(color_label)
        fig.savefig(os.path.join(args.output_dir, f"optimal_training_curves_{model}.pdf"))
        plt.close(fig)

    # Per-model individual plots (GPU-hours)
    for model in models_with_data:
        fig, ax = plt.subplots(1, 1)
        _plot_panel(ax, model, x_mode="gpuhrs")
        ax.set_title(MODEL_STYLE[model]["label"])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=global_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(color_label)
        fig.savefig(os.path.join(args.output_dir, f"optimal_training_curves_gpuhrs_{model}.pdf"))
        plt.close(fig)

    # =====================================================================
    # Combined FLOPs plot: fit lines + final val_loss scatter per model.
    # Only scatter runs within 10% of compute-optimal prediction at that C.
    # =====================================================================
    COMBINED_TOL = 0.10

    def _close_to_optimal(model, run, tol=COMBINED_TOL):
        """True if run's val_loss is within `tol` of L_opt at the run's total C."""
        if (model not in scaling_fits or run["val_loss"] is None or run["N"] is None
                or model not in flops_fns):
            return False
        fit = scaling_fits[model]
        fwd = flops_fns[model](run["N"])
        C_run = 3 * fwd * run["D"]
        L_opt, _ = compute_opt_trajectory(
            fit["L_inf"], fit["A"], fit["alpha"],
            fit["B"], fit["beta"], flops_fns[model], [C_run])
        return abs(run["val_loss"] - L_opt[0]) / L_opt[0] <= tol

    fig, ax = plt.subplots(1, 1)
    for model in models_with_data:
        style = MODEL_STYLE[model]

        # Scatter final val_loss for each run (2% filter)
        for run in runs[model]:
            if not _close_to_optimal(model, run):
                continue
            N = run["N"]
            fwd_flops = flops_fns[model](N)
            C_final = 3 * fwd_flops * run["train_iters"][-1] * args.batch_size
            ax.scatter([C_final], [run["val_loss"]], color=style['color'],
                       marker=style['marker'], s=80, zorder=5, alpha=0.6,
                       **SCATTER_KW)

        # Compute-optimal fit line + uncertainty band
        if model in scaling_fits and model in flops_fns:
            fit = scaling_fits[model]
            C_range = np.logspace(10, 21, 300)
            L_opt, _ = compute_opt_trajectory(
                fit["L_inf"], fit["A"], fit["alpha"],
                fit["B"], fit["beta"], flops_fns[model], C_range)
            gamma = fit.get("gamma",
                fit["alpha"] * fit["beta"] / (fit["alpha"] + fit["beta"]))
            ax.plot(C_range, L_opt, color=style['color'], linestyle='--',
                    linewidth=2.5,
                    label=f"{style['label']} ($L \\propto C^{{-{gamma:.3f}}}$)")

            # Bootstrap uncertainty band
            boot = fit.get("bootstrap_samples", [])
            if len(boot) > 10:
                boot_L = []
                for bp in boot:
                    bL, _ = compute_opt_trajectory(
                        bp[0], bp[1], bp[2], bp[3], bp[4],
                        flops_fns[model], C_range)
                    boot_L.append(bL)
                boot_L = np.array(boot_L)
                ax.fill_between(C_range,
                                np.percentile(boot_L, 16, axis=0),
                                np.percentile(boot_L, 84, axis=0),
                                color=style['color'], alpha=FILL_ALPHA, zorder=1)

    ax.set_xscale('log')
    ax.set_xlabel('Compute (FLOPs)')
    ax.set_ylabel('Loss')
    ax.set_ylim(y_lo, 1.0)
    ax.legend()

    fig.savefig(os.path.join(args.output_dir, "optimal_training_curves_combined.pdf"))
    plt.close(fig)

    # =====================================================================
    # Combined GPU-hours plot: scatter final val_loss per model
    # =====================================================================
    fig, ax = plt.subplots(1, 1)
    for model in models_with_data:
        style = MODEL_STYLE[model]
        for run in runs[model]:
            if run["elapsed_seconds"] is None or len(run["train_iters"]) < 2:
                continue
            if not _close_to_optimal(model, run):
                continue
            gpu_hrs_final = run["elapsed_seconds"] / 3600
            ax.scatter([gpu_hrs_final], [run["val_loss"]], color=style['color'],
                       marker=style['marker'], s=80, zorder=5, alpha=0.6,
                       **SCATTER_KW)

        ax.scatter([], [], color=style['color'], marker=style['marker'],
                   s=80, label=style['label'], **SCATTER_KW)

    ax.set_xscale('log')
    ax.set_xlabel('A100 GPU-hours')
    ax.set_ylabel('Loss')
    ax.set_ylim(y_lo, 1.0)
    ax.legend()
    fig.savefig(os.path.join(args.output_dir, "optimal_training_curves_gpuhrs_combined.pdf"))
    plt.close(fig)

    # Summary
    print("Saved optimal training curve plots:")
    for model in models_with_data:
        n_runs = len(runs[model])
        n_val = sum(1 for r in runs[model] if r["val_loss"] is not None)
        print(f"  {MODEL_STYLE[model]['label']}: {n_runs} runs, {n_val} with val_loss")
        print(f"    -> optimal_training_curves_{model}.pdf, optimal_training_curves_gpuhrs_{model}.pdf")
    print(f"    -> optimal_training_curves_combined.pdf, optimal_training_curves_gpuhrs_combined.pdf")


if __name__ == "__main__":
    main()
