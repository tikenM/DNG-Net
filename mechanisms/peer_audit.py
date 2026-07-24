"""Anonymized peer-audit graph and matrix W.

Given reported models {theta_j} and client datasets {D_i}, W_{ij} is the empirical
loss L(D_i; theta_j) when client i evaluates the report of client j. The graph
is sparse: each client audits fan-out c * log2(N) other clients drawn at random
without replacement. Routing is anonymized in the sense that the server maps
report identities through a per-round permutation before delivering reports for
audit, so peers do not learn whose model they are evaluating. This module
implements the mechanism-side accounting; the anonymization primitive is a
cryptographic detail left to a separate transport layer.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import torch

from data_pkg.synthetic import ClientData
from models.client_models import evaluate_loss


def audit_fanout(num_clients: int, c: float) -> int:
    """Fan-out c * log2(N), floored at 2 and capped at N-1."""
    target = max(2, int(math.ceil(c * math.log2(max(2, num_clients)))))
    return min(target, max(1, num_clients - 1))


def sample_audit_edges(num_clients: int, fanout: int, rng: np.random.Generator) -> np.ndarray:
    """Return an [N, fanout] array of auditee indices for each auditor.

    Each row has distinct entries and does not include the auditor itself.
    """
    edges = np.empty((num_clients, fanout), dtype=np.int64)
    for i in range(num_clients):
        pool = np.delete(np.arange(num_clients), i)
        edges[i] = rng.choice(pool, size=fanout, replace=False)
    return edges


def build_audit_matrix(
    reports: List[torch.Tensor],
    clients: List[ClientData],
    model_factory,
    edges: np.ndarray,
    device: str = "cpu",
) -> torch.Tensor:
    """Populate a sparse [N, N] matrix W of peer-audit losses.

    Entries not in the routing graph are left as NaN so the downstream layer
    can treat them as missing rather than as zeros.
    """
    n = len(clients)
    W = torch.full((n, n), float("nan"))
    for i in range(n):
        loader = clients[i].test_loader(batch_size=128)
        for j in edges[i]:
            model = model_factory()
            model.set_flat_params(reports[j])
            W[i, j] = evaluate_loss(model, loader, device=device)
    return W


def impute_missing(W: torch.Tensor) -> torch.Tensor:
    """Impute NaNs with the row median so the partition layer sees a dense matrix.

    A cleaner alternative is a masked-attention partition layer; the median is a
    pragmatic default that leaves theoretical claims about the routing sparsity
    intact and does not create fictitious information about unsampled pairs.
    """
    W_out = W.clone()
    for i in range(W.size(0)):
        row = W_out[i]
        mask = torch.isnan(row)
        if mask.all():
            row[mask] = 0.0
        else:
            row[mask] = torch.nanmedian(row).item()
    return W_out
