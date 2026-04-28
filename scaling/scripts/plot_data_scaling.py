"""Plot data-limited scaling laws for all models.

Early-stopped val loss vs dataset size D for models trained with data repetition.
Fits L(D) = L_inf + B_rep / D^beta_rep (L_inf free).

Reads pre-collected data from scaling/data/data_scaling.json.
Outputs per-model and combined PDFs.

Usage:
    python scaling/scripts/plot_data_scaling.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scaling.scripts.scaling_utils import MODELS, MODEL_STYLE
from scaling.scripts.plot_style import setup_style, SCATTER_KW, FILL_ALPHA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def _huber(x, delta=1e-3):
    abs_x = np.abs(x)
    return np.where(abs_x <= delta, 0.5 * x**2, delta * (abs_x - 0.5 * delta))


def _fit_once(D_arr, L_arr, warm_start=None):
    """Fit L(D) = L_inf + B/D^beta with Huber loss in log-space (L-BFGS-B)."""
    log_L = np.log(L_arr)

    def objective(params):
        l_inf, log_B, beta = params
        L_i = np.exp(l_inf)
        B = np.exp(log_B)
        pred = np.log(L_i + B / D_arr**beta)
        return np.sum(_huber(pred - log_L, delta=1e-3))

    if warm_start is not None:
        L_i0, B0, beta0 = warm_start
        starts = [[np.log(L_i0), np.log(B0), beta0]]
    else:
        starts = [
            [np.log(0.3), np.log(5.0), 0.22],
            [np.log(0.25), np.log(2.0), 0.3],
            [np.log(0.35), np.log(10.0), 0.15],
        ]

    best_res, best_fun = None, float("inf")
    for x0 in starts:
        try:
            res = minimize(objective, x0=x0,
                           bounds=[(np.log(0.01), np.log(1.0)),
                                   (None, np.log(1000)),
                                   (0.01, 2.0)],
                           method='L-BFGS-B', options={'maxiter': 5000})
            if res.fun < best_fun:
                best_fun = res.fun
                best_res = res
        except Exception:
            pass
    if best_res is None:
        return None
    l_inf, log_B, beta = best_res.x
    return np.array([np.exp(l_inf), np.exp(log_B), beta])


def _fit_data_scaling(D_arr, L_arr, style_label):
    """Fit L(D) = L_inf + B/D^beta with Huber (log-space) + bootstrap (100, 10/90)."""
    def scaling_free(D, L_inf, B, beta):
        return L_inf + B / D**beta

    popt = _fit_once(D_arr, L_arr)
    if popt is None:
        print(f"  {style_label}: fit failed")
        return None

    D_fit = np.logspace(np.log10(D_arr.min()) - 0.5,
                        np.log10(D_arr.max()) + 1.5, 200)
    L_fit = scaling_free(D_fit, *popt)

    # Bootstrap: 100 resamples, Huber/log-space, central fit as warm start
    rng = np.random.default_rng(42)
    n_pts = len(D_arr)
    boot_curves, boot_params = [], []
    for _ in range(100):
        idx = rng.choice(n_pts, size=n_pts, replace=True)
        bp = _fit_once(D_arr[idx], L_arr[idx], warm_start=popt)
        if bp is not None:
            boot_curves.append(scaling_free(D_fit, *bp))
            boot_params.append(bp)

    band_lo, band_hi = None, None
    if len(boot_curves) > 10:
        boot_arr = np.array(boot_curves)
        band_lo = np.percentile(boot_arr, 10, axis=0)
        band_hi = np.percentile(boot_arr, 90, axis=0)

    boot_params = np.array(boot_params) if boot_params else np.empty((0, 3))
    param_stds = np.std(boot_params, axis=0) if len(boot_params) > 1 else np.zeros(3)
    return popt, D_fit, L_fit, band_lo, band_hi, param_stds


def main():
    parser = argparse.ArgumentParser(description="Plot data scaling")
    parser.add_argument("--data", default="scaling/data/data_scaling.json")
    parser.add_argument("--basics", default="scaling/data/basics.json")
    parser.add_argument("--output-dir", default="scaling/results")
    args = parser.parse_args()

    setup_style()

    with open(args.data) as f:
        ds_data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # Group points by model
    model_data = {}  # model -> (D_arr, L_arr, d_labels)
    for pt in ds_data["points"]:
        model = pt["model"]
        if model not in model_data:
            model_data[model] = {"D": [], "L": []}
        model_data[model]["D"].append(pt["D"])
        model_data[model]["L"].append(pt["best_val_loss"])

    fit_results = {}

    # Shared axis limits
    all_D = [pt["D"] for pt in ds_data["points"]]
    all_L = [pt["best_val_loss"] for pt in ds_data["points"]]
    x_lo, x_hi = min(all_D) * 0.5, max(all_D) * 5
    y_lo, y_hi = 0.25, max(all_L) * 1.1

    models_with_data = [m for m in MODELS if m in model_data and len(model_data[m]["D"]) >= 2]

    # ---- Fit per model ----
    for model in models_with_data:
        style = MODEL_STYLE[model]
        md = model_data[model]
        order = np.argsort(md["D"])
        D_arr = np.array(md["D"], dtype=float)[order]
        L_arr = np.array(md["L"], dtype=float)[order]

        if len(D_arr) >= 4:
            result = _fit_data_scaling(D_arr, L_arr, style["label"])
            if result is not None:
                popt, D_fit, L_fit, band_lo, band_hi, param_stds = result
                L_inf_fit, B_rep, beta_rep = popt

                fit_results[model] = {
                    "label": style["label"],
                    "L_inf": float(L_inf_fit),
                    "L_inf_std": float(param_stds[0]),
                    "B_rep": float(B_rep),
                    "B_rep_std": float(param_stds[1]),
                    "beta_rep": float(beta_rep),
                    "beta_rep_std": float(param_stds[2]),
                    "n_points": len(D_arr),
                }

                print(f"  {style['label']}: L_inf={L_inf_fit:.3f}+/-{param_stds[0]:.3f}, "
                      f"B_rep={B_rep:.2f}+/-{param_stds[1]:.2f}, "
                      f"beta_rep={beta_rep:.3f}+/-{param_stds[2]:.3f} ({len(D_arr)} pts)")

    # ---- Combined plot (all models) ----
    fig, ax = plt.subplots(1, 1)

    for model in models_with_data:
        style = MODEL_STYLE[model]
        md = model_data[model]
        order = np.argsort(md["D"])
        D_arr = np.array(md["D"], dtype=float)[order]
        L_arr = np.array(md["L"], dtype=float)[order]

        ax.scatter(D_arr, L_arr, color=style["color"], marker=style["marker"],
                   s=100, zorder=5, **SCATTER_KW)

        if len(D_arr) >= 4:
            result = _fit_data_scaling(D_arr, L_arr, style["label"])
            if result is not None:
                popt, D_fit, L_fit, band_lo, band_hi, param_stds = result
                L_inf_fit, B_rep, beta_rep = popt
                L_inf_std = float(param_stds[0])

                ax.plot(D_fit, L_fit, color=style["color"], linestyle='--',
                        linewidth=2, alpha=0.7,
                        label=style['label'])

                if band_lo is not None:
                    ax.fill_between(D_fit, band_lo, band_hi,
                                    color=style["color"], alpha=FILL_ALPHA, zorder=1)

                ax.axhline(L_inf_fit, color=style["color"], linestyle=':',
                           linewidth=2.5, alpha=0.6)
                ax.axhspan(L_inf_fit - L_inf_std, L_inf_fit + L_inf_std,
                           color=style["color"], alpha=FILL_ALPHA, zorder=0)

    ax.set_xscale('log')
    ax.set_xlabel('Training samples $D$')
    ax.set_ylabel('Validation Loss')
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.legend()

    out_pdf = os.path.join(args.output_dir, "data_scaling.pdf")
    fig.savefig(out_pdf)
    plt.close(fig)

    # Mean L_inf (simple and weighted by 1/L_inf_std^2)
    if fit_results:
        l_infs = np.array([fr["L_inf"] for fr in fit_results.values()])
        l_inf_stds = np.array([fr["L_inf_std"] for fr in fit_results.values()])
        print(f"\n  Mean L_inf = {np.mean(l_infs):.4f} +/- {np.std(l_infs):.4f}")
        weights = 1.0 / l_inf_stds**2
        wmean = np.sum(weights * l_infs) / np.sum(weights)
        wstd = 1.0 / np.sqrt(np.sum(weights))
        print(f"  Weighted mean L_inf = {wmean:.4f} +/- {wstd:.4f}")

    # Save fit results JSON
    results_json = os.path.join(args.output_dir, "data_scaling_fits.json")
    with open(results_json, "w") as f:
        json.dump(fit_results, f, indent=2)

    # LaTeX table
    tex_path = os.path.join(args.output_dir, "data_scaling_fits.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\caption{Data scaling fit parameters: $L(D) = L_\\infty + B_{\\mathrm{rep}}/D^{\\beta_{\\mathrm{rep}}}$.}\n")
        f.write("\\label{tab:data_scaling_fits}\n")
        f.write("\\begin{tabular}{l c c c}\n\\toprule\n")
        f.write("Model & $L_\\infty$ & $B_{\\mathrm{rep}}$ & $\\beta_{\\mathrm{rep}}$ \\\\\n")
        f.write("\\midrule\n")
        for model in models_with_data:
            if model not in fit_results:
                continue
            fr = fit_results[model]
            f.write(f"{fr['label']} "
                    f"& ${fr['L_inf']:.3f} \\pm {fr['L_inf_std']:.3f}$ "
                    f"& ${fr['B_rep']:.2f} \\pm {fr['B_rep_std']:.2f}$ "
                    f"& ${fr['beta_rep']:.3f} \\pm {fr['beta_rep_std']:.3f}$ \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"\nSaved {out_pdf}")
    print(f"Saved {results_json}")
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
