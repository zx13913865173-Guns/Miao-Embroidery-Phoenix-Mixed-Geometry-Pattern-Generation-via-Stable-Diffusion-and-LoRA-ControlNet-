"""
Statistical analysis: TOST equivalence, ANOVA, Cohen's d, bootstrap CI.
"""

import numpy as np
from scipy import stats


def tost_two_sided(x, y, margin, alpha=0.05):
    """
    Two One-Sided Tests for equivalence.

    Args:
        x, y: arrays
        margin: equivalence margin (positive scalar)
        alpha: significance level

    Returns:
        p_value, equivalent (bool)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x.mean() - y.mean()
    se = np.sqrt(x.var(ddof=1)/len(x) + y.var(ddof=1)/len(y))
    t1 = (diff - (-margin)) / se
    t2 = (diff - margin) / se
    p1 = 1 - stats.norm.cdf(t1)
    p2 = stats.norm.cdf(t2)
    p = max(p1, p2)
    return float(p), bool(p < alpha)


def cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx-1)*x.var(ddof=1) + (ny-1)*y.var(ddof=1)) / (nx+ny-2))
    return float((x.mean() - y.mean()) / pooled)


def bootstrap_ci(x, y, n_boot=1000, alpha=0.05):
    rng = np.random.default_rng(42)
    d_vals = []
    for _ in range(n_boot):
        xs = rng.choice(x, size=len(x), replace=True)
        ys = rng.choice(y, size=len(y), replace=True)
        d_vals.append(cohens_d(xs, ys))
    d_vals = np.array(d_vals)
    lo = np.percentile(d_vals, 100 * alpha / 2)
    hi = np.percentile(d_vals, 100 * (1 - alpha / 2))
    return float(lo), float(hi)
