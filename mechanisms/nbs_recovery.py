"""Differentiable Nash Bargaining recovery layer, fixed.

Two substantive changes from the previous version:

1. `ProbabilisticLatentPartitioning` now supports a population-independent
   feature extractor. When `feature_mode='stats'` each row of W is reduced to
   nine statistical descriptors (mean, std, min, max, five quantiles) before
   the MLP. The input dimension is then constant in N. When
   `feature_mode='row'` the previous full-row behavior is preserved for
   experiments where N is fixed and small.

2. The forward path returns the raw logits alongside Z so the caller can log
   entropy and hard assignments without a second forward pass.
"""

from __future__ import annotations

from typing import List, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_ROW_STATS_DIM = 9


def _row_statistics(W: torch.Tensor) -> torch.Tensor:
    """Fixed-length descriptor per row: mean, std, min, max, quantiles."""
    mean = W.mean(dim=1, keepdim=True)
    std = W.std(dim=1, keepdim=True)
    lo = W.min(dim=1, keepdim=True).values
    hi = W.max(dim=1, keepdim=True).values
    qs = torch.quantile(
        W, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=W.device), dim=1
    ).T  # [N, 5]
    return torch.cat([mean, std, lo, hi, qs], dim=1)  # [N, 9]


class ProbabilisticLatentPartitioning(nn.Module):
    def __init__(
        self,
        num_clients: int,
        num_clusters: int,
        hidden: List[int],
        feature_mode: Literal["stats", "row"] = "stats",
    ):
        super().__init__()
        self.feature_mode = feature_mode
        in_dim = _ROW_STATS_DIM if feature_mode == "stats" else num_clients
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, num_clusters)]
        self.mlp = nn.Sequential(*layers)

    def _features(self, W: torch.Tensor) -> torch.Tensor:
        if self.feature_mode == "stats":
            return _row_statistics(W)
        mean = W.mean(dim=1, keepdim=True)
        std = W.std(dim=1, keepdim=True).clamp(min=1e-6)
        return (W - mean) / std

    def forward(self, W: torch.Tensor, tau: float, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self._features(W)
        logits = self.mlp(feats)
        Z = F.gumbel_softmax(logits, tau=tau, hard=hard)
        return Z, logits


class NashRecoveryLayer(nn.Module):
    """Mean-subtraction allocation over the peer-audit column mean.

    The learned log_scale still allows the raw column mean (in loss units) to
    be rescaled against the differentiable utility surrogate (in negative-loss
    units).
    """

    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        col_score = -W.mean(dim=0)
        s = torch.exp(self.log_scale) * col_score
        return s - s.mean()


def log_surplus_loss(
    utilities: torch.Tensor,
    disagreements: torch.Tensor,
    allocation: torch.Tensor,
    floor: float,
) -> Tuple[torch.Tensor, int]:
    surplus = utilities - disagreements + allocation
    clamped = torch.clamp(surplus, min=floor)
    return -torch.log(clamped).sum(), int((surplus < floor).sum().item())


def budget_balance_residual(allocation: torch.Tensor) -> float:
    return float(allocation.sum().abs().item())
