"""Plot scaling laws for all models.

scaling_law_ND: Loss surface L(N,D) with iso-loss contours and compute-optimal trajectory
scaling_law_isoflop: Iso-FLOP curves (loss vs N at fixed compute)

Reads data from scaling/data/grid_runs.json.
Fits L(N,D) = L_inf + A/N^alpha + B/D^beta and outputs per-model + combined PDFs.

Usage:
    python scaling/scripts/plot_scaling_laws.py
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scaling.scripts.scaling_utils import (
    MODELS, MODEL_STYLE, L_INF_FIXED,
    build_flops_fn, compute_opt_trajectory,
)
from scaling.scripts.plot_style import setup_style, SCATTER_KW, FILL_ALPHA, CMAP

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Scaling law model
# ---------------------------------------------------------------------------

def scaling_law(X, L_inf, A, alpha, B, beta):
    """L(N, D) = L_inf + A/N^alpha + B/D^beta"""
    N, D = X
    return L_inf + A / N**alpha + B / D**beta


def compute_optimal_ND(C, A, alpha, B, beta, flops_fn):
    """Given total compute C, find optimal (N, D) minimizing L."""
    from scipy.optimize import minimize_scalar

    def objective(log_N):
        N = np.exp(log_N)
        fwd_flops = flops_fn(N)
        D = C / (3 * fwd_flops)
        if D <= 0:
            return 1e10
        return A / N**alpha + B / D**beta

    res = minimize_scalar(objective, bounds=(np.log(100), np.log(1e9)),
                          method='bounded')
    N_opt = np.exp(res.x)
    D_opt = C / (3 * flops_fn(N_opt))
    return N_opt, D_opt


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

N_BOOTSTRAP = 100
BAND_QUANTILE = 0.1  # 10/90 percentiles (80% CI) — matches results/evaluate.py


def _huber(x, delta=1e-3):
    """Huber loss, applied element-wise."""
    abs_x = np.abs(x)
    return np.where(abs_x <= delta, 0.5 * x**2, delta * (abs_x - 0.5 * delta))


def _do_fit(N_arr, D_arr, L_arr, L_inf=None, warm_start=None):
    """Fit L(N,D) = L_inf + A/N^alpha + B/D^beta using Huber loss in log-space.

    If L_inf is provided, it is fixed. Otherwise L_inf is a free parameter.
    If warm_start is provided (from a central fit), only that start is used (fast).
    Returns (L_inf, A, alpha, B, beta) or None.
    """
    log_L = np.log(L_arr)
    free_l_inf = (L_inf is None)

    if free_l_inf:
        def objective(params):
            l_inf, a, b, alpha, beta = params
            L_i = np.exp(l_inf)
            A, B = np.exp(a), np.exp(b)
            log_L_hat = np.log(L_i + A / N_arr**alpha + B / D_arr**beta)
            return np.sum(_huber(log_L_hat - log_L, delta=1e-3))

        if warm_start is not None:
            L_i, A, al, B, be = warm_start
            warm_starts = [[np.log(L_i), np.log(A), np.log(B), al, be]]
        else:
            warm_starts = []
            try:
                from scipy.optimize import curve_fit
                def _mse_model(X, L_i, A, alpha, B, beta):
                    N, D = X
                    return L_i + A / N**alpha + B / D**beta
                popt, _ = curve_fit(_mse_model, (N_arr, D_arr), L_arr,
                                    p0=[0.32, 10.0, 0.4, 7.0, 0.22],
                                    bounds=([0.01, 0, 0.01, 0, 0.01], [1.0, 1000, 2, 1000, 2]),
                                    maxfev=10000)
                warm_starts.append([np.log(popt[0]), np.log(popt[1]), np.log(popt[3]), popt[2], popt[4]])
            except Exception:
                pass
            for l_inf_init in [np.log(0.32), np.log(0.25), np.log(0.35)]:
                for alpha_init in [0.3, 0.5]:
                    for beta_init in [0.2, 0.4]:
                        warm_starts.append([l_inf_init, 2, 2, alpha_init, beta_init])

        best_result, best_loss = None, float('inf')
        for x0 in warm_starts:
            try:
                res = minimize(objective, x0=x0,
                               bounds=[(np.log(0.01), np.log(1.0)),
                                       (None, np.log(1000)), (None, np.log(1000)),
                                       (0.01, 2.0), (0.01, 2.0)],
                               method='L-BFGS-B', options={'maxiter': 5000})
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_result = res
            except Exception:
                pass

        if best_result is None:
            return None
        l_inf, a, b, alpha, beta = best_result.x
        return np.array([np.exp(l_inf), np.exp(a), alpha, np.exp(b), beta])

    else:
        def objective(params):
            a, b, alpha, beta = params
            A, B = np.exp(a), np.exp(b)
            log_L_hat = np.log(L_inf + A / N_arr**alpha + B / D_arr**beta)
            return np.sum(_huber(log_L_hat - log_L, delta=1e-3))

        if warm_start is not None:
            _, A, al, B, be = warm_start
            warm_starts = [[np.log(A), np.log(B), al, be]]
        else:
            warm_starts = []
            try:
                from scipy.optimize import curve_fit
                def _mse_model(X, A, alpha, B, beta):
                    N, D = X
                    return L_inf + A / N**alpha + B / D**beta
                popt, _ = curve_fit(_mse_model, (N_arr, D_arr), L_arr,
                                    p0=[10.0, 0.4, 7.0, 0.22],
                                    bounds=([0, 0.01, 0, 0.01], [1000, 2, 1000, 2]),
                                    maxfev=10000)
                warm_starts.append([np.log(popt[0]), np.log(popt[2]), popt[1], popt[3]])
            except Exception:
                pass
            for alpha_init in [0.3, 0.5]:
                for beta_init in [0.2, 0.4]:
                    warm_starts.append([2, 2, alpha_init, beta_init])

        best_result, best_loss = None, float('inf')
        for x0 in warm_starts:
            try:
                res = minimize(objective, x0=x0,
                               bounds=[(None, np.log(1000)), (None, np.log(1000)),
                                       (0.01, 2.0), (0.01, 2.0)],
                               method='L-BFGS-B', options={'maxiter': 5000})
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_result = res
            except Exception:
                pass

        if best_result is None:
            return None
        a, b, alpha, beta = best_result.x
        return np.array([L_inf, np.exp(a), alpha, np.exp(b), beta])


def fit_scaling_law(data_points, L_inf_model=None):
    """Fit L(N,D) with bootstrap. Returns (full_params, boot_samples) or (None, None)."""
    N_arr = np.array([p[0] for p in data_points], dtype=float)
    D_arr = np.array([p[1] for p in data_points], dtype=float)
    L_arr = np.array([p[2] for p in data_points], dtype=float)
    n = len(data_points)

    if n < 5:
        print(f"    Not enough points ({n}) for fit")
        return None, None

    popt = _do_fit(N_arr, D_arr, L_arr, L_inf=L_inf_model)
    if popt is None:
        print(f"    Fit failed")
        return None, None

    L_inf, A, alpha, B, beta = popt
    full_params = np.array([L_inf, A, alpha, B, beta])
    residuals = L_arr - scaling_law((N_arr, D_arr), *full_params)
    rmse = np.sqrt(np.mean(residuals**2))

    rng = np.random.default_rng(42)
    boot_samples = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n, size=n, replace=True)
        bp = _do_fit(N_arr[idx], D_arr[idx], L_arr[idx],
                     L_inf=L_inf_model, warm_start=full_params)
        if bp is not None:
            boot_samples.append(bp.tolist())
    boot_samples = np.array(boot_samples)

    lo = np.percentile(boot_samples, 100 * BAND_QUANTILE, axis=0)
    hi = np.percentile(boot_samples, 100 * (1 - BAND_QUANTILE), axis=0)
    free_str = "free" if L_inf_model is None else "fixed"
    print(f"    Fit (L_inf={L_inf:.4f} {free_str}): A={A:.2f}, alpha={alpha:.3f}, "
          f"B={B:.2f}, beta={beta:.3f}, RMSE={rmse:.4f}")
    print(f"    68% CI: L_inf=({lo[0]:.4f}, {hi[0]:.4f}), "
          f"alpha=({lo[2]:.3f}, {hi[2]:.3f}), beta=({lo[4]:.3f}, {hi[4]:.3f})")

    return full_params, boot_samples


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_scaling_law_ND_single(ax, N_arr, D_arr, L_arr, C_arr, fit_params, flops_fn,
                       style, global_L_range, global_C_range):
    """Fig ND panel: loss surface contours + iso-FLOPs + optimal trajectory."""
    L_inf, A, alpha, B, beta = fit_params
    cmap_loss = plt.get_cmap('inferno_r')
    cmap_compute = plt.get_cmap('viridis')

    # Contours
    N_grid = np.logspace(np.log10(N_arr.min()) - 0.5,
                         np.log10(N_arr.max()) + 0.5, 300)
    D_grid = np.logspace(np.log10(D_arr.min()) - 0.5,
                         np.log10(D_arr.max()) + 0.5, 300)
    NN, DD = np.meshgrid(N_grid, D_grid)
    LL = scaling_law((NN, DD), *fit_params)
    levels = np.linspace(L_inf + 0.01, min(L_arr.max() + 0.1, 2.5), 25)
    ax.contour(NN, DD, LL, levels=levels, cmap=cmap_loss, linewidths=0.8, zorder=1)

    # Iso-FLOPs
    C_min, C_max = global_C_range
    norm_C = LogNorm(vmin=C_min, vmax=C_max)
    C_values = np.logspace(np.log10(C_min), np.log10(C_max), 10)
    for C in C_values:
        N_line = np.logspace(np.log10(N_arr.min()) - 0.5,
                             np.log10(N_arr.max()) + 0.5, 200)
        D_line = C / (3 * np.array([flops_fn(n) for n in N_line]))
        mask = (D_line >= D_arr.min() * 0.2) & (D_line <= D_arr.max() * 5)
        if mask.sum() > 2:
            ax.plot(N_line[mask], D_line[mask], '--', color=cmap_compute(norm_C(C)),
                    alpha=0.6, linewidth=1.0, zorder=2)

    # Compute-optimal trajectory
    C_range = np.logspace(np.log10(C_min), np.log10(C_max), 100)
    N_opt_arr, D_opt_arr = [], []
    for C in C_range:
        N_opt, D_opt = compute_optimal_ND(C, A, alpha, B, beta, flops_fn)
        if N_opt > 0 and D_opt > 0:
            N_opt_arr.append(N_opt)
            D_opt_arr.append(D_opt)
    ax.plot(N_opt_arr, D_opt_arr, '--', color='#1f77b4', linewidth=2.5,
            label='Compute-optimal', zorder=4)

    # Scatter
    norm_L = plt.Normalize(vmin=global_L_range[0], vmax=global_L_range[1])
    sc = ax.scatter(N_arr, D_arr, c=L_arr, cmap=cmap_loss, norm=norm_L,
                    s=80, zorder=5, **SCATTER_KW)

    ax.plot([], [], '--', color=cmap_compute(0.5), linewidth=1, label='Iso-FLOPs')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Parameters $N$')
    ax.set_ylabel('Training samples $D$')
    ax.legend(loc='upper left')
    return sc


def _plot_scaling_law_isoflop_single(ax, N_arr, D_arr, L_arr, C_arr, fit_params, flops_fn,
                       style, global_C_range):
    """Fig isoflops panel: iso-FLOP curves (loss vs N)."""
    L_inf, A, alpha, B, beta = fit_params
    C_min, C_max = global_C_range
    cmap_compute = plt.get_cmap('viridis')
    norm = LogNorm(vmin=C_min, vmax=C_max)

    N_fine = np.logspace(np.log10(N_arr.min()) - 0.3,
                         np.log10(N_arr.max()) + 0.3, 200)
    C_levels = np.logspace(np.log10(C_min), np.log10(C_max), 10)
    for C in C_levels:
        D_at_C = C / (3 * np.array([flops_fn(n) for n in N_fine]))
        L_at_C = scaling_law((N_fine, D_at_C), *fit_params)
        mask = D_at_C > 0
        ax.plot(N_fine[mask], L_at_C[mask], '-', color=cmap_compute(norm(C)),
                linewidth=2.5, alpha=0.8, zorder=2)

    sc = ax.scatter(N_arr, L_arr, c=C_arr, cmap='viridis', norm=norm,
                    s=80, zorder=5, **SCATTER_KW)

    C_range = np.logspace(np.log10(C_min), np.log10(C_max), 100)
    N_opt_list, L_opt_list = [], []
    for C in C_range:
        N_opt, D_opt = compute_optimal_ND(C, A, alpha, B, beta, flops_fn)
        L_opt = scaling_law((np.array([N_opt]), np.array([D_opt])), *fit_params)[0]
        N_opt_list.append(N_opt)
        L_opt_list.append(L_opt)
    ax.plot(N_opt_list, L_opt_list, '--', color='#1f77b4', linewidth=2.5,
            label='Compute-optimal', zorder=4)

    ax.set_xscale('log')
    ax.set_ylim(0.32, 1.2)
    ax.set_xlabel('Parameters $N$')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')
    return sc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot scaling laws Fig 1a, 1b")
    parser.add_argument("--data", default="scaling/data/grid_runs.json")
    parser.add_argument("--basics", default="scaling/data/basics.json")
    parser.add_argument("--output-dir", default="scaling/results")
    parser.add_argument("--results-json", default="scaling/results/scaling_law_fits.json")
    args = parser.parse_args()

    setup_style()

    with open(args.basics) as f:
        basics = json.load(f)
    with open(args.data) as f:
        grid_data = json.load(f)

    # Build per-model FLOPs interpolators and recompute C from basics
    flops_fns = {m: build_flops_fn(basics, m) for m in MODELS
                 if any(m in basics[s] for s in basics)}

    plot_data = defaultdict(list)
    for pt in grid_data["runs"]:
        model = pt["model"]
        if model not in flops_fns:
            continue
        fwd_flops = float(flops_fns[model](pt["N"]))
        C = 3 * fwd_flops * pt["D"]
        plot_data[model].append((pt["N"], pt["D"], pt["val_loss"], C, fwd_flops))

    # --- Filtering for fit ---
    fit_data = defaultdict(list)

    for model in plot_data:
        eligible = list(plot_data[model])

        # Monotonic N-scaling: at fixed D, if increasing N gives worse loss,
        # exclude that point and all larger N at that D
        by_D = defaultdict(list)
        for p in eligible:
            by_D[p[1]].append(p)

        for D_val, pts in by_D.items():
            pts_sorted = sorted(pts, key=lambda x: x[0])
            best_loss = float("inf")
            for pt in pts_sorted:
                if pt[2] <= best_loss:
                    best_loss = pt[2]
                    fit_data[model].append(pt)

    if not fit_data:
        print("No data found!")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    fit_results = {}

    # Fit all models in parallel
    models_to_fit = {m: fit_data[m] for m in MODELS
                     if m in fit_data and len(fit_data[m]) >= 6}

    print(f"Fitting {len(models_to_fit)} models with L_inf={L_INF_FIXED}...")
    parallel_results = {}
    with ProcessPoolExecutor(max_workers=len(models_to_fit)) as executor:
        futures = {executor.submit(fit_scaling_law, pts, L_INF_FIXED): m
                   for m, pts in models_to_fit.items()}
        for future in as_completed(futures):
            m = futures[future]
            parallel_results[m] = future.result()

    for model in MODELS:
        if model not in parallel_results:
            continue
        fit_params, boot_samples = parallel_results[model]
        if fit_params is None:
            continue

        L_inf, A, alpha, B, beta = fit_params

        # Effective a, gamma from numerical compute-optimal trajectory, since
        # C = 3 * fwd_flops(N) * D with measured (non-linear-in-N) fwd_flops.
        # Analytic a = beta/(alpha+beta), gamma = alpha*beta/(alpha+beta) only
        # hold when fwd_flops ~ N (C = 6ND); we fit effective exponents from
        # the numerical trajectory.
        flops_fn = build_flops_fn(basics, model)

        def _eff_exponents(params):
            L_i, A_p, al, B_p, be = params
            C_range = np.logspace(11, 20, 80)
            L_opt, N_opt = compute_opt_trajectory(L_i, A_p, al, B_p, be,
                                                  flops_fn, C_range)
            # Fit log N_opt vs log C, and log(L_opt - L_i) vs log C
            logC = np.log(C_range)
            a_eff = float(np.polyfit(logC, np.log(N_opt), 1)[0])
            excess = L_opt - L_i
            mask = excess > 0
            if mask.sum() < 3:
                gamma_eff = float("nan")
            else:
                gamma_eff = -float(np.polyfit(logC[mask], np.log(excess[mask]), 1)[0])
            return a_eff, gamma_eff

        a, gamma = _eff_exponents([L_inf, A, alpha, B, beta])

        # Compute uncertainties from bootstrap
        boot_arr = boot_samples if boot_samples is not None and len(boot_samples) > 10 else None
        if boot_arr is not None:
            stds = np.std(boot_arr, axis=0)
            boot_a, boot_gamma = [], []
            for bp in boot_arr:
                try:
                    a_b, g_b = _eff_exponents(bp)
                    if np.isfinite(a_b) and np.isfinite(g_b):
                        boot_a.append(a_b)
                        boot_gamma.append(g_b)
                except Exception:
                    pass
            a_std = float(np.std(boot_a)) if boot_a else 0.0
            gamma_std = float(np.std(boot_gamma)) if boot_gamma else 0.0
            L_inf_std = float(stds[0])
        else:
            stds = np.zeros(5)
            a_std, gamma_std, L_inf_std = 0.0, 0.0, 0.0

        fit_results[model] = {
            "label": MODEL_STYLE[model]["label"],
            "L_inf": float(L_inf), "L_inf_std": L_inf_std,
            "A": float(A), "A_std": float(stds[1]),
            "alpha": float(alpha), "alpha_std": float(stds[2]),
            "B": float(B), "B_std": float(stds[3]),
            "beta": float(beta), "beta_std": float(stds[4]),
            "a": float(a), "a_std": a_std,
            "gamma": float(gamma), "gamma_std": gamma_std,
            "n_points": len(fit_data[model]),
            "bootstrap_samples": boot_samples.tolist() if boot_samples is not None else [],
        }

    # Global ranges for consistent color scales
    all_L, all_C = [], []
    for m in MODELS:
        for pt in plot_data.get(m, []):
            all_L.append(pt[2])
            all_C.append(pt[3])
    global_L_range = (L_INF_FIXED, max(all_L)) if all_L else (0.3, 2.0)
    global_C_range = (min(all_C), max(all_C)) if all_C else (1e10, 1e17)

    models_with_fits = [m for m in MODELS if m in fit_results]

    # ---- Per-model individual plots ----
    for model in models_with_fits:
        style = MODEL_STYLE[model]
        fr = fit_results[model]
        pts = plot_data[model]
        N_arr = np.array([p[0] for p in pts], dtype=float)
        D_arr = np.array([p[1] for p in pts], dtype=float)
        L_arr = np.array([p[2] for p in pts], dtype=float)
        C_arr = np.array([p[3] for p in pts], dtype=float)
        fp = [fr["L_inf"], fr["A"], fr["alpha"], fr["B"], fr["beta"]]
        flops_fn = build_flops_fn(basics, model)

        # ND
        fig, ax = plt.subplots(1, 1)
        sc = _plot_scaling_law_ND_single(ax, N_arr, D_arr, L_arr, C_arr, fp, flops_fn,
                                style, global_L_range, global_C_range)
        ax.set_title(style['label'])
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label('Loss')
        fig.savefig(os.path.join(args.output_dir, f"scaling_law_ND_{model}.pdf"))
        plt.close(fig)

        # isoflops
        fig, ax = plt.subplots(1, 1)
        sc = _plot_scaling_law_isoflop_single(ax, N_arr, D_arr, L_arr, C_arr, fp, flops_fn,
                                style, global_C_range)
        ax.set_title(style['label'])
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label('Compute (FLOPs)')
        fig.savefig(os.path.join(args.output_dir, f"scaling_law_isoflop_{model}.pdf"))
        plt.close(fig)

    # Save
    with open(args.results_json, "w") as f:
        json.dump(fit_results, f, indent=2)
    print(f"\nFit results saved to {args.results_json}")

    # Summary table
    print(f"\n{'='*80}")
    print("SCALING LAW FIT SUMMARY: L(N,D) = L_inf + A/N^alpha + B/D^beta")
    print(f"{'='*80}")
    print(f"  {'Model':<14} {'L_inf':>6} {'A':>7} {'alpha':>6} {'B':>7} "
          f"{'beta':>6} {'gamma':>6} {'a':>5} {'pts':>4}")
    print(f"  {'─'*14} {'─'*6} {'─'*7} {'─'*6} {'─'*7} {'─'*6} {'─'*6} {'─'*5} {'─'*4}")
    for model in models_with_fits:
        fr = fit_results[model]
        print(f"  {fr['label']:<14} {fr['L_inf']:>6.3f} {fr['A']:>7.2f} "
              f"{fr['alpha']:>6.3f} {fr['B']:>7.2f} {fr['beta']:>6.3f} "
              f"{fr['gamma']:>6.3f} {fr['a']:>5.2f} {fr['n_points']:>4}")

    # LaTeX table
    tex_path = os.path.join(args.output_dir, "scaling_law_fits.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\caption{Scaling law fit parameters: $L(N,D) = L_\\infty + A/N^\\alpha + B/D^\\beta$.}\n")
        f.write("\\label{tab:scaling_law_fits}\n")
        f.write("\\begin{tabular}{l c c c c c c c}\n\\toprule\n")
        f.write("Model & $L_\\infty$ & $A$ & $\\alpha$ & $B$ & $\\beta$ & $\\gamma$ & $a$ \\\\\n")
        f.write("\\midrule\n")
        for model in models_with_fits:
            fr = fit_results[model]
            f.write(f"{fr['label']} & {fr['L_inf']:.3f} "
                    f"& ${fr['A']:.2f} \\pm {fr['A_std']:.2f}$ "
                    f"& ${fr['alpha']:.3f} \\pm {fr['alpha_std']:.3f}$ "
                    f"& ${fr['B']:.2f} \\pm {fr['B_std']:.2f}$ "
                    f"& ${fr['beta']:.3f} \\pm {fr['beta_std']:.3f}$ "
                    f"& ${fr['gamma']:.3f} \\pm {fr['gamma_std']:.3f}$ "
                    f"& ${fr['a']:.3f} \\pm {fr['a_std']:.3f}$ \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Saved {tex_path}")

    for model in models_with_fits:
        print(f"Saved scaling_law_ND_{model}.pdf, scaling_law_isoflop_{model}.pdf")


if __name__ == "__main__":
    main()
