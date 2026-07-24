"""Baseline federated learning methods used in the SOTA comparison.

Each baseline exposes the same interface:

    fit(clients, rounds, ...) -> BaselineResult

where BaselineResult carries:
    - cluster_assignment: np.ndarray of shape [N] with integer cluster ids
    - per_client_test_acc: np.ndarray of shape [N]
    - communication_bytes_per_round: float
    - wall_time: float

Every baseline uses the same client-side optimizer and model factory as
DNG-Net so that comparisons are on the mechanism, not on backbone tricks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import torch
from sklearn.cluster import KMeans, SpectralClustering
from torch.utils.data import DataLoader

from data_pkg.synthetic import ClientData
from models.client_models import evaluate, local_train


@dataclass
class BaselineResult:
    cluster_assignment: np.ndarray
    per_client_test_acc: np.ndarray
    communication_bytes_per_round: float
    wall_time: float
    cluster_models_flat: List[torch.Tensor]


def _bytes_per_report(model_factory) -> float:
    with torch.no_grad():
        n = int(model_factory().get_flat_params().numel())
    return 4.0 * n  # float32


def _client_report(model_factory, client: ClientData, epochs: int, lr: float, init_flat) -> torch.Tensor:
    m = model_factory()
    m.set_flat_params(init_flat)
    m = local_train(m, client.train_loader(64), epochs=epochs, lr=lr)
    return m.get_flat_params()


# ---- FedAvg (single global model, no clustering) --------------------------- #


def fit_fedavg(
    clients: List[ClientData],
    model_factory: Callable,
    rounds: int,
    epochs: int,
    lr: float,
) -> BaselineResult:
    t0 = time.time()
    global_flat = model_factory().get_flat_params()
    for _ in range(rounds):
        reports = [_client_report(model_factory, c, epochs, lr, global_flat) for c in clients]
        global_flat = torch.stack(reports).mean(dim=0)
    accs = []
    for c in clients:
        m = model_factory()
        m.set_flat_params(global_flat)
        accs.append(evaluate(m, c.test_loader(128)))
    return BaselineResult(
        cluster_assignment=np.zeros(len(clients), dtype=int),
        per_client_test_acc=np.array(accs),
        communication_bytes_per_round=_bytes_per_report(model_factory) * len(clients),
        wall_time=time.time() - t0,
        cluster_models_flat=[global_flat],
    )


# ---- Local-only (disagreement point) --------------------------------------- #


def fit_local_only(
    clients: List[ClientData],
    model_factory: Callable,
    epochs: int,
    lr: float,
) -> BaselineResult:
    t0 = time.time()
    accs = []
    for c in clients:
        m = model_factory()
        m = local_train(m, c.train_loader(64), epochs=max(1, epochs), lr=lr)
        accs.append(evaluate(m, c.test_loader(128)))
    return BaselineResult(
        cluster_assignment=np.arange(len(clients)),
        per_client_test_acc=np.array(accs),
        communication_bytes_per_round=0.0,
        wall_time=time.time() - t0,
        cluster_models_flat=[],
    )


# ---- IFCA ------------------------------------------------------------------ #


def fit_ifca(
    clients: List[ClientData],
    model_factory: Callable,
    num_clusters: int,
    rounds: int,
    epochs: int,
    lr: float,
) -> BaselineResult:
    """Ghosh et al. 2020. Each round, each client picks the cluster model that
    achieves the lowest loss on its local data, then contributes to that cluster.
    """
    t0 = time.time()
    cluster_flats = [model_factory().get_flat_params() for _ in range(num_clusters)]
    for _ in range(rounds):
        # Client picks best cluster.
        assign = np.zeros(len(clients), dtype=int)
        contributions: List[List[torch.Tensor]] = [[] for _ in range(num_clusters)]
        for i, c in enumerate(clients):
            losses = []
            for k in range(num_clusters):
                m = model_factory()
                m.set_flat_params(cluster_flats[k])
                # Cheap proxy: single-batch loss.
                x, y = next(iter(c.train_loader(64)))
                with torch.no_grad():
                    loss = torch.nn.functional.cross_entropy(m(x), y).item()
                losses.append(loss)
            k_star = int(np.argmin(losses))
            assign[i] = k_star
            contributions[k_star].append(_client_report(model_factory, c, epochs, lr, cluster_flats[k_star]))
        for k in range(num_clusters):
            if contributions[k]:
                cluster_flats[k] = torch.stack(contributions[k]).mean(dim=0)

    accs = []
    for i, c in enumerate(clients):
        m = model_factory()
        m.set_flat_params(cluster_flats[assign[i]])
        accs.append(evaluate(m, c.test_loader(128)))
    return BaselineResult(
        cluster_assignment=assign,
        per_client_test_acc=np.array(accs),
        communication_bytes_per_round=_bytes_per_report(model_factory) * len(clients) * 2,
        wall_time=time.time() - t0,
        cluster_models_flat=cluster_flats,
    )


# ---- CFL ------------------------------------------------------------------- #


def fit_cfl(
    clients: List[ClientData],
    model_factory: Callable,
    num_clusters: int,
    rounds: int,
    epochs: int,
    lr: float,
) -> BaselineResult:
    """Sattler et al. 2021. Gradient cosine-similarity clustering after warm-up
    with a single global model. Uses spectral clustering on the cosine matrix.
    """
    t0 = time.time()
    warmup = fit_fedavg(clients, model_factory, rounds=rounds // 2, epochs=epochs, lr=lr)
    global_flat = warmup.cluster_models_flat[0]
    grads = []
    for c in clients:
        report = _client_report(model_factory, c, epochs, lr, global_flat)
        grads.append((report - global_flat).numpy())
    G = np.stack(grads)
    G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    S = np.abs(G @ G.T)
    assign = SpectralClustering(n_clusters=num_clusters, affinity="precomputed", random_state=0).fit_predict(S)

    # Refine per-cluster models.
    cluster_flats = []
    for k in range(num_clusters):
        members = [i for i in range(len(clients)) if assign[i] == k]
        if not members:
            cluster_flats.append(global_flat.clone())
            continue
        flat = global_flat.clone()
        for _ in range(rounds // 2):
            reports = [_client_report(model_factory, clients[i], epochs, lr, flat) for i in members]
            flat = torch.stack(reports).mean(dim=0)
        cluster_flats.append(flat)

    accs = []
    for i, c in enumerate(clients):
        m = model_factory()
        m.set_flat_params(cluster_flats[assign[i]])
        accs.append(evaluate(m, c.test_loader(128)))
    return BaselineResult(
        cluster_assignment=assign,
        per_client_test_acc=np.array(accs),
        communication_bytes_per_round=_bytes_per_report(model_factory) * len(clients),
        wall_time=time.time() - t0,
        cluster_models_flat=cluster_flats,
    )


# ---- FeSEM ----------------------------------------------------------------- #


def fit_fesem(
    clients: List[ClientData],
    model_factory: Callable,
    num_clusters: int,
    rounds: int,
    epochs: int,
    lr: float,
) -> BaselineResult:
    """Xie et al. 2021. Soft EM over cluster centroids in parameter space.
    Assignment is softmax over squared parameter distances to centroids.
    """
    t0 = time.time()
    cluster_flats = [model_factory().get_flat_params() for _ in range(num_clusters)]
    for _ in range(rounds):
        reports = [_client_report(model_factory, c, epochs, lr, cluster_flats[0]) for c in clients]
        R = torch.stack(reports)
        # E-step: soft assignments by negative squared distance.
        dists = torch.stack([((R - cf) ** 2).sum(dim=1) for cf in cluster_flats], dim=1)
        Z_soft = torch.softmax(-dists, dim=1)
        # M-step: weighted average of reports per cluster.
        for k in range(num_clusters):
            w = Z_soft[:, k:k + 1]
            cluster_flats[k] = (w * R).sum(dim=0) / (w.sum() + 1e-9)
    assign = Z_soft.argmax(dim=1).numpy()
    accs = []
    for i, c in enumerate(clients):
        m = model_factory()
        m.set_flat_params(cluster_flats[assign[i]])
        accs.append(evaluate(m, c.test_loader(128)))
    return BaselineResult(
        cluster_assignment=assign,
        per_client_test_acc=np.array(accs),
        communication_bytes_per_round=_bytes_per_report(model_factory) * len(clients),
        wall_time=time.time() - t0,
        cluster_models_flat=cluster_flats,
    )
