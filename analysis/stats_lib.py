"""Paired-comparison statistics used by power.py and run_stats.py.

Unit of replication = one independent world seed.  For a contrast A - B we form the
per-world paired differences d_w = W_A(w) - W_B(w) (same map, currents, tasks, tapes).

* Two-sided paired randomization test (sign-flip permutation): under H0 the sign of each
  d_w is exchangeable; p = (1 + #{|mean(s*d)| >= |mean(d)|}) / (1 + n_perm).
* Holm (1979) step-down adjustment across the pre-specified family.
* Percentile and BCa bootstrap CIs resampling WORLDS.
* Paired standardized effect size d_z = mean(d) / sd(d).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def sign_flip_test(d, n_perm=100_000, seed=0, chunk=20_000):
    d = np.asarray(d, float)
    n = len(d)
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    ge = 0
    done = 0
    while done < n_perm:
        k = min(chunk, n_perm - done)
        s = rng.integers(0, 2, size=(k, n), dtype=np.int8) * 2 - 1
        m = np.abs(s @ d) / n
        ge += int(np.sum(m >= obs - 1e-12))
        done += k
    return (1 + ge) / (1 + n_perm)


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def bootstrap_ci(d, B=20_000, seed=1, level=0.95, stat="mean"):
    d = np.asarray(d, float)
    f = {"mean": np.mean, "median": np.median,
         "dz": lambda x, axis=None: np.mean(x, axis=axis) / np.std(x, axis=axis, ddof=1)}[stat]
    res = stats.bootstrap((d,), f, n_resamples=B, confidence_level=level, method="BCa",
                          random_state=np.random.default_rng(seed), vectorized=True)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def summarize_contrast(d, n_perm=100_000, seed=0):
    d = np.asarray(d, float)
    sd = d.std(ddof=1)
    out = dict(n=len(d), mean=float(d.mean()), median=float(np.median(d)), sd=float(sd),
               dz=float(d.mean() / sd) if sd > 0 else float("nan"),
               p_perm=sign_flip_test(d, n_perm, seed),
               p_wilcoxon=float(stats.wilcoxon(d[d != 0]).pvalue) if np.any(d != 0) else 1.0,
               p_t=float(stats.ttest_1samp(d, 0).pvalue),
               frac_positive=float(np.mean(d > 0)), frac_negative=float(np.mean(d < 0)),
               frac_zero=float(np.mean(d == 0)))
    out["ci_mean"] = bootstrap_ci(d, seed=seed + 1)
    out["ci_median"] = bootstrap_ci(d, seed=seed + 2, stat="median")
    out["ci_dz"] = bootstrap_ci(d, seed=seed + 3, stat="dz")
    return out
