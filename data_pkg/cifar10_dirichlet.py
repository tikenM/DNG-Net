"""CIFAR-10 with Dirichlet non-IID partitioning across K latent groups.

Each latent group is a Dirichlet-mixture over the ten CIFAR-10 classes with a
group-specific concentration. Clients within a group draw their class mixture
from the same distribution, so the ground-truth cluster label is the group id.
This gives a controlled non-IID benchmark with a well-defined K.

Usage:

    from data_pkg.cifar10_dirichlet import make_cifar10_dirichlet
    clients = make_cifar10_dirichlet(
        root='./data',
        num_clients=100,
        num_clusters=5,
        dirichlet_alpha=0.3,
        rng=np.random.default_rng(0),
    )

CIFAR-10 is downloaded via torchvision the first time the loader runs.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from data_pkg.synthetic import ClientData


_CIFAR10_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


def _load_cifar10(root: str):
    train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=_CIFAR10_TRANSFORM)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=_CIFAR10_TRANSFORM)
    x_train = torch.stack([x for x, _ in train])
    y_train = torch.tensor([y for _, y in train])
    x_test = torch.stack([x for x, _ in test])
    y_test = torch.tensor([y for _, y in test])
    return x_train, y_train, x_test, y_test


def make_cifar10_dirichlet(
    root: str,
    num_clients: int,
    num_clusters: int,
    dirichlet_alpha: float = 0.3,
    rng: np.random.Generator | None = None,
) -> List[ClientData]:
    if rng is None:
        rng = np.random.default_rng(0)
    x_train, y_train, x_test, y_test = _load_cifar10(root)
    num_classes = 10

    # Per-cluster class mixture.
    cluster_mix = rng.dirichlet([dirichlet_alpha] * num_classes, size=num_clusters)

    # Index each class into a pool we can draw from.
    train_by_class = [np.where(y_train.numpy() == c)[0].tolist() for c in range(num_classes)]
    test_by_class = [np.where(y_test.numpy() == c)[0].tolist() for c in range(num_classes)]
    for lst in train_by_class + test_by_class:
        rng.shuffle(lst)

    clients: List[ClientData] = []
    # Assign clients to clusters round-robin plus a random tail.
    cluster_assign = np.array([i % num_clusters for i in range(num_clients)])
    rng.shuffle(cluster_assign)

    per_client_n_train = 500
    per_client_n_test = 100

    for i in range(num_clients):
        k = int(cluster_assign[i])
        # Sample the client's class multinomial once from the cluster mixture.
        client_mix = rng.dirichlet(cluster_mix[k] * 20 + 0.1)
        counts_train = rng.multinomial(per_client_n_train, client_mix)
        counts_test = rng.multinomial(per_client_n_test, client_mix)

        tr_idx, te_idx = [], []
        for c in range(num_classes):
            if counts_train[c] > 0:
                take = min(counts_train[c], len(train_by_class[c]))
                tr_idx += train_by_class[c][:take]
                train_by_class[c] = train_by_class[c][take:]
            if counts_test[c] > 0:
                take = min(counts_test[c], len(test_by_class[c]))
                te_idx += test_by_class[c][:take]
                test_by_class[c] = test_by_class[c][take:]

        clients.append(ClientData(
            client_id=i,
            latent_cluster=k,
            train_x=x_train[tr_idx],
            train_y=y_train[tr_idx],
            test_x=x_test[te_idx],
            test_y=y_test[te_idx],
        ))
    return clients
