"""Compute-optimal trajectory plots + inference-time budget-constrained plots.

Reads scaling/results/scaling_law_fits.json and basics for FLOPs interpolation.
Outputs (in scaling/results/):
    compute_optimal_trajectories_NvsD.pdf              - N vs D compute-optimal trajectory
    compute_optimal_trajectories_Nvs{measurement}.pdf  - one per entry in MEASUREMENTS
    constrained_{measurement}_LvsD.pdf                 - 2x2 L vs D at 4 budgets
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scaling.scripts.scaling_utils import (
    MODELS, MODEL_STYLE, build_flops_fn, compute_opt_trajectory,
    load_measurement,
)
from scaling.scripts.plot_style import setup_style, FILL_ALPHA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def _fit_q_of_n(Ns, Qs):
    """Fit Q(N) = A + B * N^alpha. Returns (A, B, alpha). Falls back to pure
    power-law (A=0) if the 3-parameter fit fails or if there aren't enough pts."""
    def model(N, A, B, alpha):
        return A + B * N**alpha

    if len(Ns) < 3:
        slope, intercept = np.polyfit(np.log(Ns), np.log(Qs), 1)
        return 0.0, float(np.exp(intercept)), float(slope)

    # Start from pure power law
    slope, intercept = np.polyfit(np.log(Ns), np.log(Qs), 1)
    B0, a0 = float(np.exp(intercept)), float(slope)
    A0 = max(0.0, float(Qs.min()) * 0.1)
    try:
        popt, _ = curve_fit(model, Ns, Qs, p0=[A0, B0, a0],
                            bounds=([0, 0, 0.01], [np.inf, np.inf, 3.0]),
                            maxfev=5000)
        return float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        return 0.0, B0, a0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits", default="scaling/results/scaling_law_fits.json")
    parser.add_argument("--basics", default="scaling/data/basics.json")
    parser.add_argument("--output-dir", default="scaling/results")
    parser.add_argument("--C-min", type=float, default=1e10)
    parser.add_argument("--C-max", type=float, default=1e22)
    parser.add_argument("--n-bootstrap", type=int, default=50,
                        help="Number of bootstrap samples used to compute 68%% bands (0 disables)")
    args = parser.parse_args()

    setup_style()

    with open(args.fits) as f:
        fits = json.load(f)
    with open(args.basics) as f:
        basics = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    C_range = np.logspace(np.log10(args.C_min), np.log10(args.C_max), 300)

    # Precompute trajectories
    traj = {}
    boot_traj = {}
    for model in MODELS:
        if model not in fits:
            continue
        fit = fits[model]
        flops_fn = build_flops_fn(basics, model)
        L_opt, N_opt = compute_opt_trajectory(
            fit["L_inf"], fit["A"], fit["alpha"], fit["B"], fit["beta"],
            flops_fn, C_range)
        D_opt = C_range / (3 * np.array([flops_fn(n) for n in N_opt]))
        traj[model] = (L_opt, N_opt, D_opt)

        boot = fit.get("bootstrap_samples", [])
        boot_list = []
        for bp in boot[:args.n_bootstrap]:
            L_b, N_b = compute_opt_trajectory(
                bp[0], bp[1], bp[2], bp[3], bp[4], flops_fn, C_range)
            D_b = C_range / (3 * np.array([flops_fn(n) for n in N_b]))
            boot_list.append((L_b, N_b, D_b))
        boot_traj[model] = boot_list

    models = [m for m in MODELS if m in traj]

    # --- N vs D compute-optimal trajectory plot ---
    fig, ax = plt.subplots(1, 1)
    for model in models:
        style = MODEL_STYLE[model]
        L_opt, N_opt, D_opt = traj[model]
        # 68% band
        bs = boot_traj[model]
        if len(bs) > 5:
            D_boot = np.array([D_b for _, _, D_b in bs])
            D_lo = np.percentile(D_boot, 10, axis=0)
            D_hi = np.percentile(D_boot, 90, axis=0)
            ax.fill_between(N_opt, D_lo, D_hi, color=style["color"],
                            alpha=FILL_ALPHA, linewidth=0)
        ax.plot(N_opt, D_opt, color=style["color"], linewidth=3.0,
                label=style["label"])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Model parameters $N$')
    ax.set_ylabel('Training samples $D$')
    ax.legend(loc='best')
    fig.savefig(os.path.join(args.output_dir, "compute_optimal_trajectories_NvsD.pdf"))
    plt.close(fig)

    # =====================================================================
    # Measurements: load all per-N quantities used as alternative y-axes
    # =====================================================================
    MEASUREMENTS = [
        # (suffix, file, key, divisor, ylabel, log_y)
        ("Tinference_cpu", "cost_estimate/inference_cpu.json",
         "mean", 1.0, "CPU inference time per sample [ms]", True),
    ]

    measurements_loaded = {}  # suffix -> (raw dict, ylabel, log_y)
    for suffix, path, key, divisor, ylabel, log_y in MEASUREMENTS:
        if not os.path.exists(path):
            continue
        raw = load_measurement(path, key=key, divisor=divisor)
        measurements_loaded[suffix] = (raw, ylabel, log_y)

    written = ["compute_optimal_trajectories_NvsD.pdf"]

    # ---- N vs Q trajectory plots (one per measurement) ----
    for suffix, (raw, ylabel, log_y) in measurements_loaded.items():
        fig, ax = plt.subplots(1, 1)
        for model in models:
            style = MODEL_STYLE[model]

            # Measured points (N, Q)
            meas = raw.get(model, {})
            Ns_q, Qs_q = [], []
            for s, q in meas.items():
                s_key = f"{float(s)}"
                if s_key not in basics or model not in basics[s_key]:
                    continue
                N = basics[s_key][model]["params"]
                if q > 0 and N > 0:
                    Ns_q.append(N)
                    Qs_q.append(q)
            if len(Ns_q) < 2:
                continue
            Ns_q = np.array(Ns_q)
            Qs_q = np.array(Qs_q)
            order = np.argsort(Ns_q)
            Ns_q = Ns_q[order]
            Qs_q = Qs_q[order]

            ax.scatter(Ns_q, Qs_q, color=style["color"], marker=style["marker"],
                       s=80, edgecolors='black', linewidths=0.8, zorder=5)

            # Offset + power-law fit: Q(N) = A + B*N^alpha
            A, B, alpha = _fit_q_of_n(Ns_q, Qs_q)
            L_opt, N_opt, _ = traj[model]
            N_grid = np.logspace(np.log10(N_opt.min()), np.log10(N_opt.max()), 200)
            Q_fit = A + B * N_grid**alpha
            ax.plot(N_grid, Q_fit, color=style["color"], linewidth=3.0,
                    label=f"{style['label']} ($Q={A:.2g}+{B:.2g}\\,N^{{{alpha:.2f}}}$)")
        ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        ax.set_xlabel('Model parameters $N$')
        ax.set_ylabel(ylabel)
        ax.legend(loc='best', fontsize=14)
        fname = f"compute_optimal_trajectories_Nvs{suffix}.pdf"
        fig.savefig(os.path.join(args.output_dir, fname))
        plt.close(fig)
        written.append(fname)

    # =====================================================================
    # Constrained scaling: 2x2 grid of L vs (D | C) for 4 budgets
    # Two PDFs per constraint type: one with D on x-axis, one with C
    # =====================================================================
    D_grid = np.logspace(4, 12, 300)

    def _draw_one_panel(ax, model, fit, flops_fn, N_use, x_mode):
        """Draw single (loss curve + bootstrap band) for one model on one axis."""
        style = MODEL_STYLE[model]
        fwd = float(flops_fn(N_use))
        L = fit["L_inf"] + fit["A"] / N_use**fit["alpha"] + fit["B"] / D_grid**fit["beta"]
        x = D_grid if x_mode == "D" else 3 * fwd * D_grid

        boot = fit.get("bootstrap_samples", [])
        if len(boot) > 5:
            L_boot = np.array([
                bp[0] + bp[1] / N_use**bp[2] + bp[3] / D_grid**bp[4]
                for bp in boot
            ])
            ax.fill_between(x,
                            np.percentile(L_boot, 10, axis=0),
                            np.percentile(L_boot, 90, axis=0),
                            color=style["color"], alpha=FILL_ALPHA, linewidth=0)
        label = f"{style['label']} ($N$={N_use:.1e})"
        ax.plot(x, L, color=style["color"], linewidth=3.0, label=label)

    def _plot_2x2(values, x_mode, suffix, title_fmt, fname, N_for_value):
        """Make 2x2 grid: 4 sub-panels, one per Q_budget.
        N_for_value(model, budget) -> N_use (or None to skip that model).
        x_mode: 'D' or 'C'.
        """
        assert len(values) == 4
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
        axes = axes.flatten()
        xlabel = 'Training samples $D$' if x_mode == "D" else 'Training compute (FLOPs)'
        for ax, val in zip(axes, values):
            for model in models:
                N_use = N_for_value(model, val)
                if N_use is None:
                    continue
                fit = fits[model]
                flops_fn = build_flops_fn(basics, model)
                _draw_one_panel(ax, model, fit, flops_fn, N_use, x_mode)
            ax.set_xscale('log')
            ax.set_xlabel(xlabel)
            ax.set_ylim(0.3, 1.2)
            ax.set_title(title_fmt.format(val=val), fontsize=22, pad=10)
            ax.legend(loc='best', fontsize=12)
        for ax in (axes[0], axes[2]):
            ax.set_ylabel('Loss')
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, fname))
        plt.close(fig)
        return fname

    # ---- Fixed Q: 4 budgets per measurement, both x-modes ----
    def _make_inverter(raw_meas):
        """Build per-model inverter: budget Q -> N using Q(N) = A + B*N^alpha."""
        inverters = {}
        for model in models:
            meas = raw_meas.get(model, {})
            Ns_q, Qs_q = [], []
            for s, q in meas.items():
                s_key = f"{float(s)}"
                if s_key not in basics or model not in basics[s_key]:
                    continue
                N = basics[s_key][model]["params"]
                if q > 0 and N > 0:
                    Ns_q.append(N)
                    Qs_q.append(q)
            if len(Ns_q) < 2:
                continue
            Ns_q = np.array(Ns_q)
            Qs_q = np.array(Qs_q)
            A, B, alpha = _fit_q_of_n(Ns_q, Qs_q)
            if alpha == 0 or B == 0:
                continue
            inverters[model] = (A, B, alpha)
        def fn(model, budget):
            if model not in inverters:
                return None
            A, B, alpha = inverters[model]
            if budget <= A:
                return None  # below constant offset — impossible
            return float(((budget - A) / B) ** (1.0 / alpha))
        return fn

    for q_suffix, (raw, ylabel, log_y) in measurements_loaded.items():
        # Compute per-model (A, max_q) then pick budgets s.t. all models answer
        per_model_bounds = []  # (lower_bound, max_q)
        for m in models:
            Ns_q, Qs_q = [], []
            for s, q in raw.get(m, {}).items():
                s_key = f"{float(s)}"
                if s_key not in basics or m not in basics[s_key]:
                    continue
                N = basics[s_key][m]["params"]
                if q > 0 and N > 0:
                    Ns_q.append(N)
                    Qs_q.append(q)
            if len(Ns_q) < 2:
                continue
            A, B, alpha = _fit_q_of_n(np.array(Ns_q), np.array(Qs_q))
            per_model_bounds.append((A, max(Qs_q)))
        if not per_model_bounds:
            continue
        # budgets must be > max(A) so every model has a valid inverse
        q_min = max(b[0] for b in per_model_bounds) * 1.05
        q_max = max(b[1] for b in per_model_bounds)
        if q_max <= q_min:
            continue
        budgets = list(np.logspace(np.log10(q_min), np.log10(q_max), 4))
        N_for = _make_inverter(raw)
        fname = _plot_2x2(
            budgets, "D", q_suffix,
            f"{ylabel} = " + "{val:.2g}",
            f"constrained_{q_suffix}_LvsD.pdf",
            N_for,
        )
        written.append(fname)

    print("Saved:")
    for f in written:
        print(f"  {f}")


if __name__ == "__main__":
    main()
