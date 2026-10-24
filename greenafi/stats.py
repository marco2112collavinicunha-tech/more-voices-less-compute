"""Pre-registered statistics (speaker = unit of analysis)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def tost_paired(diff_points: np.ndarray, margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided tests for equivalence of a paired difference.

    diff_points: per-speaker difference (system A - system B) in UAR points.
    Equivalence is declared when both one-sided tests reject at `alpha`,
    i.e. the (1 - 2 alpha) confidence interval lies inside (-margin, +margin).
    """
    d = np.asarray(diff_points, dtype=float)
    n = len(d)
    if n < 3:
        return {"n": n, "verdict": "no verdict (n < 3)"}
    mean, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n) if sd > 0 else 1e-12
    t_low = (mean + margin) / se
    t_high = (mean - margin) / se
    p_low = 1 - stats.t.cdf(t_low, n - 1)
    p_high = stats.t.cdf(t_high, n - 1)
    p = max(p_low, p_high)
    tcrit = stats.t.ppf(1 - alpha, n - 1)
    ci = (mean - tcrit * se, mean + tcrit * se)
    return {
        "n": n, "mean_diff": mean, "sd_diff": sd, "margin": margin,
        "ci_90_low": ci[0], "ci_90_high": ci[1], "p_tost": p,
        "verdict": "equivalent" if p < alpha else "equivalence not shown",
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def wilcoxon_or_none(d: np.ndarray):
    d = np.asarray(d, dtype=float)
    if len(d) < 6 or np.allclose(d, 0):
        return None
    return float(stats.wilcoxon(d).pvalue)
