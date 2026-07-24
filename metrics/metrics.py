"""Game-theoretic, clustering, and statistical metrics.

All metrics are named for what they measure rather than the paper section that
motivates them. The mapping between metric and hypothesis is documented in each
experiment script.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from data_pkg.synthetic import ClientData


# ---- Clustering metrics ---------------------------------------------------- #


def ari(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    return float(adjusted_rand_score(true_labels, pred_labels))


def nmi(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    return float(normalized_mutual_info_score(true_labels, pred_labels))


def cluster_purity(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    total = 0
    for k in np.unique(pred_labels):
        mask = pred_labels == k
        if mask.sum() == 0:
            continue
        (_, counts) = np.unique(true_labels[mask], return_counts=True)
        total += counts.max()
    return float(total / len(true_labels))


def admission_rate(
    cluster_assignment: np.ndarray,
    adversary_mask: np.ndarray,
    target_cluster: int,
) -> float:
    """Empirical fraction of adversarial clients assigned to the target cluster."""
    if adversary_mask.sum() == 0:
        return 0.0
    adv_assign = cluster_assignment[adversary_mask]
    return float((adv_assign == target_cluster).mean())


# ---- Game-theoretic metrics ------------------------------------------------ #


def ir_violation_rate(
    utilities: np.ndarray,
    disagreements: np.ndarray,
    allocation: np.ndarray,
    honest_mask: np.ndarray,
) -> Tuple[float, float]:
    """(fraction violated, mean shortfall over violating honest clients).

    The utility used inside the mechanism is `utilities + allocation`, matching
    the argument of the log-surplus objective. Ignoring `allocation` (as the
    previous version did) answers a different question from the one the
    optimizer is solving.
    """
    diff = (utilities[honest_mask] + allocation[honest_mask]) - disagreements[honest_mask]
    viol = diff < 0
    frac = float(viol.mean()) if viol.size else 0.0
    shortfall = float(-diff[viol].mean()) if viol.sum() > 0 else 0.0
    return frac, shortfall


def budget_balance_residual(allocation: np.ndarray) -> float:
    return float(abs(allocation.sum()))


def core_deviation_gain(
    utilities: np.ndarray,
    allocation: np.ndarray,
    coalition_value_fn: Callable[[Sequence[int]], float],
    max_size: int,
    sampled_coalitions: int = 100,
    rng: np.random.Generator | None = None,
) -> Tuple[float, np.ndarray]:
    """Return the maximum per-capita coalition deviation gain.

    Contract: `coalition_value_fn(S)` MUST return the SUM of member values that
    coalition S can achieve on its own, in the same units as `utilities`. The
    within-mechanism payoff of member i is `utilities[i] + allocation[i]`, so
    the Core condition is:

        sum_{i in S} (utilities[i] + allocation[i])  >=  v(S)   for all S.

    A violation is a coalition with strictly positive per-capita gap
    (v(S) - sum_{i in S}(u_i + x_i)) / |S|. The previous version subtracted
    a mean coalition value from a sum of utilities, which produced quantities
    whose magnitude grew with |S| in a way that made larger coalitions
    unrepresentable in the gain metric. This is fixed here.
    """
    n = len(utilities)
    if rng is None:
        rng = np.random.default_rng(0)
    payoff = utilities + allocation
    gains = []
    for size in range(1, n):
        if size <= max_size:
            iterator = combinations(range(n), size)
        else:
            def gen():
                for _ in range(sampled_coalitions):
                    yield tuple(rng.choice(n, size=size, replace=False).tolist())
            iterator = gen()
        for S in iterator:
            v = coalition_value_fn(S)      # SUM contract
            u_sum = payoff[list(S)].sum()  # SUM to match
            gains.append((v - u_sum) / size)
    arr = np.array(gains) if gains else np.array([0.0])
    return float(arr.max()), arr


# ---- Statistical utilities ------------------------------------------------- #


def bootstrap_ci(x: np.ndarray, iters: int = 2000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Return (mean, lower, upper) with a percentile bootstrap CI."""
    rng = np.random.default_rng(0)
    means = []
    for _ in range(iters):
        idx = rng.integers(0, len(x), size=len(x))
        means.append(x[idx].mean())
    m = float(x.mean())
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return m, lo, hi


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Return (statistic, p) from the paired Wilcoxon signed-rank test."""
    stat, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(p)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt(((a.std(ddof=1) ** 2) + (b.std(ddof=1) ** 2)) / 2 + 1e-12)
    return float((a.mean() - b.mean()) / pooled)


def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    """Return Holm-Bonferroni adjusted p-values."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty_like(pvals, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = pvals[idx] * (m - rank)
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj
