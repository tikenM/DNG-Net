"""Synthetic federated mixture datasets.

Produces controlled Gaussian mixture data with known latent cluster labels,
sufficient for E1, E2, E3, E10, and theorem validation. Real dataset loaders
(FEMNIST, CIFAR-10 Dirichlet, Shakespeare) are in the sibling modules; they
share the ClientData interface defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ClientData:
    """Container for a single client's local dataset and latent label."""

    client_id: int
    latent_cluster: int          # ground-truth z_i in {0, ..., K-1}
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    is_adversarial: bool = False

    def train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            TensorDataset(self.train_x, self.train_y),
            batch_size=batch_size,
            shuffle=True,
        )

    def test_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            TensorDataset(self.test_x, self.test_y),
            batch_size=batch_size,
            shuffle=False,
        )


def make_synthetic_mixture(
    num_clients: int,
    num_clusters: int,
    input_dim: int,
    num_classes: int,
    samples_per_client_range: Tuple[int, int],
    intra_cluster_std: float,
    inter_cluster_shift: float,
    test_fraction: float = 0.2,
    rng: np.random.Generator | None = None,
) -> List[ClientData]:
    """Create a client population from a Gaussian mixture with K latent clusters.

    Each cluster k has a per-class mean drawn once, plus a client-level shift.
    Labels are drawn by nearest-mean rule with added Gaussian noise, so intra-
    cluster classification is nontrivial but consistent across clients in the
    same cluster.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Class means per cluster.
    cluster_class_means = rng.normal(
        loc=0.0,
        scale=inter_cluster_shift,
        size=(num_clusters, num_classes, input_dim),
    )

    clients: List[ClientData] = []
    for i in range(num_clients):
        k = int(rng.integers(0, num_clusters))
        n = int(rng.integers(samples_per_client_range[0], samples_per_client_range[1] + 1))
        y = rng.integers(0, num_classes, size=n)
        x = cluster_class_means[k, y] + rng.normal(0.0, intra_cluster_std, size=(n, input_dim))

        n_test = max(1, int(n * test_fraction))
        idx = rng.permutation(n)
        test_idx, train_idx = idx[:n_test], idx[n_test:]

        clients.append(
            ClientData(
                client_id=i,
                latent_cluster=k,
                train_x=torch.tensor(x[train_idx], dtype=torch.float32),
                train_y=torch.tensor(y[train_idx], dtype=torch.long),
                test_x=torch.tensor(x[test_idx], dtype=torch.float32),
                test_y=torch.tensor(y[test_idx], dtype=torch.long),
            )
        )
    return clients


def inject_adversaries(
    clients: List[ClientData],
    fraction: float,
    target_cluster: int,
    rng: np.random.Generator,
) -> List[ClientData]:
    """Mark a fraction of clients as adversarial and re-label their latent cluster
    to a fictitious slot (num_clusters). Their local data is left in place; the
    attack module produces the misreported parameters.
    """
    n_adv = int(len(clients) * fraction)
    idx = rng.permutation(len(clients))[:n_adv]
    for j in idx:
        clients[j].is_adversarial = True
        clients[j].latent_cluster = -1  # sentinel: "not from any true cluster"
    return clients
