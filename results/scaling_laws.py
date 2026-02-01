import numpy as np
from scipy.optimize import minimize


def huber_loss(residuals, delta=1.0):
    a = np.abs(residuals)
    return np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))

def fit_func(x, A, B, alpha):
    return B + A * x ** -alpha

def perform_fit(cost, performance, delta=1.0):
    log_cost = np.log(cost)
    log_cost_mean = log_cost.mean()
    log_cost_std = log_cost.std() or 1.0
    log_cost_z = (log_cost - log_cost_mean) / log_cost_std

    perf_mean = performance.mean()
    perf_std = performance.std() or 1.0
    perf_z = (performance - perf_mean) / perf_std
    perf_range = float(perf_z.max() - perf_z.min()) or 1.0

    Bz0 = float(perf_z.max() + 0.1 * perf_range)
    positive_part = np.maximum(Bz0 - perf_z, 1e-12)

    slope, intercept = np.polyfit(log_cost_z, np.log(positive_part), 1)
    k0 = float(np.clip(-slope, 1e-8, 200.0))
    Atilde_z0 = float(np.exp(intercept))

    theta0 = np.array([np.log(max(Atilde_z0, 1e-12)), Bz0, np.log(k0)], dtype=float)

    def objective(theta):
        log_Atilde_z, Bz, log_k = theta
        Atilde_z = np.exp(log_Atilde_z)
        k = np.exp(log_k)

        exp_term = np.exp(np.clip(-k * log_cost_z, -700, 700))
        pred_z = Bz - Atilde_z * exp_term
        residuals = pred_z - perf_z
        return float(np.sum(huber_loss(residuals, delta=delta)))

    def gradient(theta):
        log_Atilde_z, Bz, log_k = theta
        Atilde_z = np.exp(log_Atilde_z)
        k = np.exp(log_k)

        exp_term = np.exp(np.clip(-k * log_cost_z, -700, 700))
        pred_z = Bz - Atilde_z * exp_term
        residuals = pred_z - perf_z

        psi = np.where(np.abs(residuals) <= delta, residuals, delta * np.sign(residuals))

        d_pred_d_log_Atilde_z = -Atilde_z * exp_term
        d_pred_d_Bz = np.ones_like(residuals)
        d_pred_d_log_k = (-Atilde_z * exp_term) * (-log_cost_z) * k

        g0 = np.sum(psi * d_pred_d_log_Atilde_z)
        g1 = np.sum(psi * d_pred_d_Bz)
        g2 = np.sum(psi * d_pred_d_log_k)
        return np.array([g0, g1, g2], dtype=float)

    bounds = [
        (-80, 80),
        (float(perf_z.min() - 10 * perf_range), float(perf_z.max() + 10 * perf_range)),
        (np.log(1e-8), np.log(200.0)),
    ]

    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        jac=gradient,
        bounds=bounds,
        options=dict(maxiter=5000, ftol=1e-12),
    )

    log_Atilde_z, Bz_hat, log_k = result.x
    Atilde_z_hat = float(np.exp(log_Atilde_z))
    k_hat = float(np.exp(log_k))

    alpha_hat = float(k_hat / log_cost_std)
    B_hat = float(perf_mean + perf_std * Bz_hat)

    Atilde_hat = float(-perf_std * Atilde_z_hat)
    A_hat = float(Atilde_hat * np.exp(alpha_hat * log_cost_mean))

    return dict(A=A_hat, B=B_hat, alpha=alpha_hat), result.success

def bootstrap_fit(cost, performance, n_bootstrap=100, quantile=0.3, seed=None, max_attempts=None):
    cost = np.asarray(cost, dtype=float)
    performance = np.asarray(performance, dtype=float)

    n = len(cost)
    rng = np.random.default_rng(seed)

    if max_attempts is None:
        max_attempts = max(n_bootstrap, 1) * 3

    best_fit, success = perform_fit(cost, performance)
    assert success

    bootstrap_fits = []
    for attempts in range(max_attempts):
        attempts += 1
        idx = rng.integers(0, n, size=n)
        if np.unique(idx).size < 3:
            continue

        try:
            fit, success = perform_fit(cost[idx], performance[idx])
        except np.linalg.LinAlgError:
            continue

        success = success and np.isfinite(fit["A"]) and np.isfinite(fit["B"]) and np.isfinite(fit["alpha"])

        if success:
            bootstrap_fits.append(fit)
        
        if len(bootstrap_fits) == n_bootstrap:
            break
    assert len(bootstrap_fits) > 0

    def _summarize(key):
        v = np.array([f[key] for f in bootstrap_fits], dtype=float)
        summary = {p: float(np.quantile(v, p)) for p in [quantile, 0.5, 1 - quantile]}
        summary["all"] = v.tolist()
        summary["best"] = best_fit[key]
        return summary

    summary = {
        "A": _summarize("A"),
        "B": _summarize("B"),
        "alpha": _summarize("alpha"),
    }

    return summary

def scaling_law_fit(metric_dict, cost_dict, models, sizes, n_bootstrap=100, quantile=0.3):
    fits = {"label_metric": metric_dict["label"], "label_cost": cost_dict["label"]}

    for model in models:
        costs = []
        metrics = []
        for size in sizes:
            cost = cost_dict[model][size]
            metric = metric_dict[model][size]
            for m in metric:
                costs.append(cost)
                metrics.append(m)
        
        fits[model] = bootstrap_fit(costs, metrics, n_bootstrap=n_bootstrap, quantile=quantile)
    return fits
