"""Shared constants and helpers for scaling law plots."""

import json

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

MODELS = ["tr", "part", "gn3", "lloca","slim"]
MODEL_STYLE = {
    "tr":    {"color": "#E26D5C", "marker": "o", "label": "Transformer"},
    "gn3":   {"color": "#4C6E91", "marker": "^", "label": "GN3"},
    "lloca": {"color": "#8C271E", "marker": "D", "label": "LLoCa-Tr."},
    "slim":  {"color": "#419108", "marker": "X", "label": "L-GATr-slim"},
    "part":  {"color": "#E9C46A", "marker": "s", "label": "ParT"},
}

L_INF_FIXED = 0.315


def build_flops_fn(basics, model):
    """Build interpolator: N (params) -> forward FLOPs per sample, from basics.json."""
    sizes = sorted(basics.keys(), key=float)
    N_vals, F_vals = [], []
    for s in sizes:
        if model in basics[s]:
            N_vals.append(basics[s][model]["params"])
            F_vals.append(basics[s][model]["flops"])
    N_vals, F_vals = np.array(N_vals, dtype=float), np.array(F_vals, dtype=float)
    log_interp = interp1d(np.log(N_vals), np.log(F_vals),
                          kind='linear', fill_value='extrapolate')
    return lambda N: np.exp(log_interp(np.log(N)))


def load_measurement(path, key="mean", divisor=1.0):
    """Load a per-(size, model) measurement JSON.

    Returns dict {model: {size_float: value}}. Skips sub-dicts without `key`.
    Top-level keys that are not numeric (e.g. 'system_info', 'benchmarking')
    are ignored. The value is divided by `divisor` (e.g. 512 to convert
    per-batch numbers to per-sample).
    """
    with open(path) as f:
        d = json.load(f)
    out = {}
    for s_str, per_model in d.items():
        try:
            s = float(s_str)
        except ValueError:
            continue
        if not isinstance(per_model, dict):
            continue
        for m, v in per_model.items():
            if not isinstance(v, dict) or key not in v:
                continue
            out.setdefault(m, {})[s] = float(v[key]) / divisor
    return out


def compute_opt_trajectory(L_inf, A, alpha, B, beta, flops_fn, C_range):
    """Compute L(C) along the compute-optimal trajectory.

    For each compute budget C, finds N* minimizing L(N, C/(3*flops(N))).
    Returns (L_opt, N_opt) arrays.
    """
    L_opt, N_opt = [], []
    for C in C_range:
        def objective(log_N, _C=C):
            N = np.exp(log_N)
            fwd_flops = flops_fn(N)
            D = _C / (3 * fwd_flops)
            if D <= 0:
                return 1e10
            return L_inf + A / N**alpha + B / D**beta

        res = minimize_scalar(objective,
                              bounds=(np.log(100), np.log(1e9)),
                              method='bounded')
        L_opt.append(objective(res.x))
        N_opt.append(np.exp(res.x))
    return np.array(L_opt), np.array(N_opt)
