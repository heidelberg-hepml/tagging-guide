"""Power-law scaling fits with Huber loss and bootstrap uncertainty.

Model: L = L_inf + B * x^{-beta}, with B > 0 if the metric decreases toward
the floor L_inf and B < 0 if it increases toward the ceiling L_inf. The sign
of B is auto-detected from the data; callers do not pass a direction flag.
"""

import numpy as np
from scipy.optimize import minimize


def huber_loss(residuals, delta=0.1):
    """0.5 r^2 for |r| <= delta, else delta * (|r| - 0.5 * delta)."""
    a = np.abs(residuals)
    a_clipped = np.minimum(a, delta)
    return 0.5 * a_clipped**2 + delta * np.maximum(a - delta, 0.0)


def fit_func(x, L_inf, B, beta):
    """L = L_inf + B * x^{-beta}; B > 0 for floor curves, B < 0 for ceiling curves."""
    return L_inf + B * x**-beta


def single_fit(cost, performance, delta=0.1):
    """L-BFGS-B fit with Huber loss in z-scored log-cost / performance space.

    Returns ({L_inf, B, beta}, success). Raises ValueError if cost or
    performance is constant (degenerate, no power law to fit).
    """
    cost = np.asarray(cost, dtype=float)
    performance = np.asarray(performance, dtype=float)

    log_cost = np.log(cost)
    log_cost_mean, log_cost_std = log_cost.mean(), log_cost.std()
    if log_cost_std == 0:
        raise ValueError("All cost values are identical; cannot fit a power law.")
    z = (log_cost - log_cost_mean) / log_cost_std

    perf_mean, perf_std = performance.mean(), performance.std()
    if perf_std == 0:
        raise ValueError("All performance values are identical; cannot fit a power law.")
    y = (performance - perf_mean) / perf_std
    y_range = float(y.max() - y.min())

    # Detect direction from data, then optimize a positive amplitude with the
    # sign baked in: log-amplitude is better-conditioned than signed B at the
    # optimum, where L-BFGS-B's Hessian approximation otherwise stumbles on
    # Huber's kink. Warm start: place L_inf 10% beyond the data, fit
    # log|residual| ~ log(amp) - beta * z linearly to get initial (amp, beta).
    if np.polyfit(z, y, 1)[0] > 0:
        sign = -1.0
        L_inf_z0 = float(y.max() + 0.1 * y_range)
        residual_pos = L_inf_z0 - y
    else:
        sign = 1.0
        L_inf_z0 = float(y.min() - 0.1 * y_range)
        residual_pos = y - L_inf_z0
    slope, intercept = np.polyfit(z, np.log(np.maximum(residual_pos, 1e-12)), 1)
    log_amp_z0 = float(intercept)
    log_beta_z0 = float(np.log(np.clip(-slope, 1e-8, 200.0)))

    def loss_and_grad(theta):
        log_amp_z, L_inf_z, log_beta_z = theta
        amp_z, beta_z = np.exp(log_amp_z), np.exp(log_beta_z)
        exp_term = np.exp(np.clip(-beta_z * z, -700, 700))
        residuals = (L_inf_z + sign * amp_z * exp_term) - y

        loss = float(np.sum(huber_loss(residuals, delta=delta)))
        psi = np.where(np.abs(residuals) <= delta, residuals, delta * np.sign(residuals))
        d_term = sign * amp_z * exp_term
        grad = np.array(
            [
                np.sum(psi * d_term),
                np.sum(psi),
                np.sum(psi * d_term * (-z) * beta_z),
            ]
        )
        return loss, grad

    bounds = [
        (-80, 80),
        (float(y.min() - 10 * y_range), float(y.max() + 10 * y_range)),
        (np.log(1e-8), np.log(200.0)),
    ]
    result = minimize(
        loss_and_grad,
        np.array([log_amp_z0, L_inf_z0, log_beta_z0]),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options=dict(maxiter=5000, ftol=1e-12),
    )

    log_amp_z, L_inf_z, log_beta_z = result.x
    beta = float(np.exp(log_beta_z) / log_cost_std)
    L_inf = float(perf_mean + perf_std * L_inf_z)
    B = float(perf_std * sign * np.exp(log_amp_z) * np.exp(beta * log_cost_mean))
    return {"L_inf": L_inf, "B": B, "beta": beta}, result.success


def fit_with_uncertainty(
    cost,
    performance,
    n_bootstrap=100,
    quantile=0.1,
    seed=None,
    max_attempts=None,
    delta=0.1,
):
    """Bootstrap (resample with replacement) until n_bootstrap successful fits."""
    cost = np.asarray(cost, dtype=float)
    performance = np.asarray(performance, dtype=float)
    n = len(cost)
    rng = np.random.default_rng(seed)
    if max_attempts is None:
        max_attempts = max(n_bootstrap, 1) * 3

    best_fit, success = single_fit(cost, performance, delta=delta)
    assert success

    samples = []
    for _ in range(max_attempts):
        idx = rng.integers(0, n, size=n)
        if np.unique(idx).size < 3:
            continue
        try:
            fit, success = single_fit(cost[idx], performance[idx], delta=delta)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if success and all(np.isfinite(fit[k]) for k in ("L_inf", "B", "beta")):
            samples.append(fit)
        if len(samples) == n_bootstrap:
            break
    assert samples

    def summarize(key):
        v = np.array([s[key] for s in samples], dtype=float)
        out = {p: float(np.quantile(v, p)) for p in (quantile, 0.5, 1 - quantile)}
        out["all"] = v.tolist()
        out["best"] = best_fit[key]
        return out

    return {k: summarize(k) for k in ("L_inf", "B", "beta")}


def fit_scaling_law(metric_dict, cost_dict, models, sizes, n_bootstrap=100, quantile=0.1):
    """Fit one curve per model from the nested-dict layout in evaluate.py."""
    fits = {"label_metric": metric_dict["label"], "label_cost": cost_dict["label"]}
    for model in models:
        costs, metrics = [], []
        for size in sizes:
            for m in metric_dict[model][size]:
                costs.append(cost_dict[model][size])
                metrics.append(m)
        fits[model] = fit_with_uncertainty(
            costs,
            metrics,
            n_bootstrap=n_bootstrap,
            quantile=quantile,
        )
    return fits
