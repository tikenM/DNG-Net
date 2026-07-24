"""Statistical helpers for hypothesis testing and effect sizes.

Previously colocated with `metrics.metrics`. Moved here to match the layout
declared in the framework document.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats


def bootstrap_ci(x: np.ndarray, iters: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, len(x), size=len(x))
        means[i] = x[idx].mean()
    return float(x.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    stat, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(p)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt(((a.std(ddof=1) ** 2) + (b.std(ddof=1) ** 2)) / 2 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty_like(pvals, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = pvals[idx] * (m - rank)
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj
