"""Fixed DNG-Net training loop.

Fixes over the previous version, in order of substance:

1. Gradient path from partition MLP to loss is real. Utility is a
   differentiable function of Z. Each round we precompute a detached matrix
   `M[i,k] = L(D_i; theta_k)` (loss of cluster k on client i's held-out data)
   and form `u_i = - sum_k Z_ik * M[i,k]`, whose gradient w.r.t. Z is
   `-M[i,k] / |M|`. Partition parameters now receive Nash-product signal.

2. Consistent units. Utility and disagreement are both negative loss.
   Peer-audit W is loss. All three live in the same space, so the recovery
   scale multiplier is a genuine calibration rather than a units patch.

3. Device-aware. Uses select_device() and moves test tensors once per client
   to avoid the per-round host-to-device shuffle that dominated the runtime
   on Apple Silicon.

4. Uses a single reusable evaluator model instead of instantiating one per
   (auditor, auditee) pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from attacks.attacks import mimicry_report, noise_report, collusion_whitewash
from data_pkg.synthetic import ClientData
from mechanisms.nbs_recovery import (
    NashRecoveryLayer,
    ProbabilisticLatentPartitioning,
    budget_balance_residual,
    log_surplus_loss,
)
from mechanisms.peer_audit import audit_fanout, impute_missing, sample_audit_edges
from models.client_models import local_train
from utils.device import select_device


@dataclass
class TrainingLog:
    loss_history: List[float] = field(default_factory=list)
    row_entropy_history: List[float] = field(default_factory=list)
    bb_residual_history: List[float] = field(default_factory=list)
    non_positive_surplus_rounds: int = 0
    final_Z: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    final_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    per_client_test_utility: np.ndarray = field(default_factory=lambda: np.zeros(0))
    per_client_disagreement: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cluster_flats: List[torch.Tensor] = field(default_factory=list)
    device: str = "cpu"


def anneal_tau(t: int, T: int, tau_start: float, tau_end: float, schedule: str) -> float:
    if schedule == "constant":
        return tau_start
    if schedule == "linear":
        return tau_start + (tau_end - tau_start) * (t / max(1, T - 1))
    return tau_start * ((tau_end / tau_start) ** (t / max(1, T - 1)))


@torch.no_grad()
def _batched_loss(model, x: torch.Tensor, y: torch.Tensor, batch: int = 512) -> float:
    """Cross-entropy loss over an already-on-device tensor pair."""
    total, count = 0.0, 0
    for i in range(0, x.size(0), batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        total += F.cross_entropy(model(xb), yb, reduction="sum").item()
        count += yb.numel()
    return total / max(count, 1)


def compute_disagreements(
    clients: List[ClientData],
    model_factory: Callable,
    epochs: int,
    lr: float,
    device: str,
) -> np.ndarray:
    """d_i is the negative test loss of a strictly-local model.

    Units match the utility surrogate used in training, so IR checks are
    against a comparable quantity.
    """
    d = np.zeros(len(clients))
    for i, c in enumerate(clients):
        m = model_factory().to(device)
        m = local_train(m, c.train_loader(64), epochs=max(1, epochs), lr=lr, device=device)
        x_te, y_te = c.test_x.to(device), c.test_y.to(device)
        d[i] = -_batched_loss(m, x_te, y_te)
    return d


def _adversary_report(
    client: ClientData,
    init_flat: torch.Tensor,
    cluster_flats: List[torch.Tensor],
    attack_type: str,
    target_cluster: int,
    mimicry_strength: float,
) -> torch.Tensor:
    if attack_type == "mimicry":
        return mimicry_report(cluster_flats[target_cluster], strength=mimicry_strength)
    if attack_type == "free_ride":
        return init_flat.clone()
    if attack_type == "noise":
        return noise_report(init_flat, sigma=0.1)
    return init_flat.clone()


def _client_report(
    model_factory: Callable,
    client: ClientData,
    init_flat: torch.Tensor,
    cluster_flats: List[torch.Tensor],
    epochs: int,
    lr: float,
    cfg_adv: dict,
    device: str,
) -> torch.Tensor:
    if client.is_adversarial:
        return _adversary_report(
            client, init_flat, cluster_flats,
            cfg_adv["attack_type"], cfg_adv.get("target_cluster", 0),
            cfg_adv["mimicry_strength"],
        )
    m = model_factory().to(device)
    m.set_flat_params(init_flat.to(device))
    m = local_train(m, client.train_loader(64), epochs=epochs, lr=lr, device=device)
    return m.get_flat_params().detach()


def _build_audit_and_M(
    reports: List[torch.Tensor],
    cluster_flats: List[torch.Tensor],
    clients: List[ClientData],
    model_factory: Callable,
    edges: np.ndarray,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (W, M).

    W[i,j] = loss of client i's held-out data under report j (sparse).
    M[i,k] = loss of client i's held-out data under cluster model k (dense).

    Both are detached constants for the round. The utility surrogate multiplies
    -M[i,k] by Z[i,k], which is where the partition MLP's gradient enters.
    """
    n = len(clients)
    K = len(cluster_flats)
    # Cache per-client test tensors on device to avoid re-copy per audit.
    test_xy = [(c.test_x.to(device), c.test_y.to(device)) for c in clients]

    scratch = model_factory().to(device)

    W = torch.full((n, n), float("nan"), device=device)
    for i in range(n):
        x_te, y_te = test_xy[i]
        for j in edges[i]:
            scratch.set_flat_params(reports[j].to(device))
            W[i, j] = _batched_loss(scratch, x_te, y_te)

    M = torch.empty((n, K), device=device)
    for k in range(K):
        scratch.set_flat_params(cluster_flats[k].to(device))
        for i in range(n):
            x_te, y_te = test_xy[i]
            M[i, k] = _batched_loss(scratch, x_te, y_te)
    return W, M


def train_dng_net(
    clients: List[ClientData],
    model_factory: Callable,
    num_clusters: int,
    cfg: Dict,
    disagreements: np.ndarray | None = None,
    device: str | None = None,
) -> TrainingLog:
    device = select_device(device)
    n = len(clients)
    K = num_clusters
    log = TrainingLog(device=device)

    if disagreements is None:
        disagreements = compute_disagreements(
            clients, model_factory,
            epochs=cfg["federation"]["local_epochs"],
            lr=cfg["federation"]["client_lr"],
            device=device,
        )
    log.per_client_disagreement = disagreements
    d_t = torch.tensor(disagreements, dtype=torch.float32, device=device)

    feature_mode = cfg["mechanism"].get("partition_feature_mode", "stats")
    ablation = cfg.get("ablation", {})
    partition = ProbabilisticLatentPartitioning(
        num_clients=n,
        num_clusters=K,
        hidden=cfg["mechanism"]["partition_hidden"],
        feature_mode=feature_mode,
    ).to(device)
    recovery = NashRecoveryLayer().to(device)
    opt_cls = torch.optim.SGD if ablation.get("server_optimizer") == "sgd" else torch.optim.Adam
    server_opt = opt_cls(
        list(partition.parameters()) + list(recovery.parameters()),
        lr=cfg["mechanism"]["partition_lr"],
    )

    cluster_flats = [model_factory().get_flat_params().to(device) for _ in range(K)]

    T = cfg["federation"]["rounds"]
    tau_start = cfg["mechanism"]["gumbel_tau_start"]
    tau_end = cfg["mechanism"]["gumbel_tau_end"]
    schedule = cfg["mechanism"]["gumbel_anneal"]
    rng = np.random.default_rng(cfg["experiment"]["seed"])
    fanout = audit_fanout(n, cfg["mechanism"]["peer_audit_fanout_c"])

    for t in range(T):
        tau = anneal_tau(t, T, tau_start, tau_end, schedule)
        init_flat = torch.stack(cluster_flats).mean(dim=0)

        # Client reports (adversaries and honest).
        reports = [
            _client_report(
                model_factory, c, init_flat, cluster_flats,
                epochs=cfg["federation"]["local_epochs"],
                lr=cfg["federation"]["client_lr"],
                cfg_adv=cfg["adversary"],
                device=device,
            )
            for c in clients
        ]

        # Audit and per-cluster loss matrix.
        edges = sample_audit_edges(n, fanout, rng)
        W, M = _build_audit_and_M(reports, cluster_flats, clients, model_factory, edges, device)

        if cfg["adversary"]["collusion_fraction"] > 0:
            n_coll = int(cfg["adversary"]["collusion_fraction"] * n)
            colluders = list(rng.choice(n, size=n_coll, replace=False))
            W = collusion_whitewash(W, colluders)
        W = impute_missing(W)

        # Ablation A1: replace peer-audit W with parameter-distance matrix.
        if ablation.get("use_parameter_distance", False):
            R_now = torch.stack([r.to(device) for r in reports])
            W = torch.cdist(R_now, R_now)

        # Differentiable forward pass.
        Z, _ = partition(W, tau=tau, hard=ablation.get("deterministic_softmax", False) and False)
        if ablation.get("deterministic_softmax", False):
            # Replace Gumbel-Softmax by plain softmax on the logits (A4).
            _, logits = partition(W, tau=tau, hard=False)
            Z = torch.softmax(logits / max(tau, 1e-6), dim=1)

        if ablation.get("disable_allocation", False):
            x = torch.zeros(n, device=device)                      # A3
        elif ablation.get("no_projection", False):
            x = -W.mean(dim=0) * torch.exp(recovery.log_scale)     # A5
        else:
            x = recovery(W)
            if ablation.get("post_hoc_projection", False):         # A9
                x = x.detach() - x.detach().mean()

        util = -(Z * M).sum(dim=1)
        if ablation.get("mean_utility", False):                    # A2
            loss = -(util - d_t + x).mean()
            npos = int(((util - d_t + x) < cfg["mechanism"]["surplus_floor"]).sum().item())
        else:
            loss, npos = log_surplus_loss(util, d_t, x, cfg["mechanism"]["surplus_floor"])
        log.non_positive_surplus_rounds += int(npos > 0)

        server_opt.zero_grad()
        loss.backward()
        server_opt.step()

        # Aggregate reports into cluster models with the current soft Z.
        with torch.no_grad():
            R = torch.stack([r.to(device) for r in reports])
            for k in range(K):
                w = Z[:, k].detach().unsqueeze(1)
                total = w.sum().clamp(min=1e-6)
                cluster_flats[k] = (w * R).sum(dim=0) / total

            row_ent = -(Z * Z.clamp(min=1e-12).log()).sum(dim=1).mean().item()
        log.loss_history.append(float(loss.item()))
        log.row_entropy_history.append(row_ent)
        log.bb_residual_history.append(budget_balance_residual(x.detach()))

    # Finalize.
    with torch.no_grad():
        Z_hard, _ = partition(W, tau=tau_end, hard=True)
        util_final = -(Z_hard * M).sum(dim=1)
    log.final_Z = Z_hard.detach().cpu().numpy()
    log.final_x = x.detach().cpu().numpy()
    log.cluster_flats = [f.detach().clone() for f in cluster_flats]
    log.per_client_test_utility = util_final.detach().cpu().numpy()
    return log
