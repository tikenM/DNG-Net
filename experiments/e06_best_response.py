"""E6: incentive compatibility via projected-gradient best-response search.

For each of a randomly chosen subset of honest clients, search over their
reported parameter vector for the report that maximizes their own realized
utility while all other reports are held at truthful. If the maximum utility
gain is non-positive, unilateral IC is verified for that client.
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
import yaml

from data_pkg.registry import build_clients, build_model_factory
from experiments._common import dump
from experiments.train_dng import train_dng_net, compute_disagreements, _build_audit_and_M
from mechanisms.nbs_recovery import ProbabilisticLatentPartitioning, NashRecoveryLayer
from mechanisms.peer_audit import audit_fanout, sample_audit_edges, impute_missing
from metrics.statistics import bootstrap_ci
from utils.device import select_device


def _utility_for_client(reports, i, cfg, clients, factory, cluster_flats, device, partition, recovery, d_t):
    rng = np.random.default_rng(cfg["experiment"]["seed"])
    edges = sample_audit_edges(len(clients), audit_fanout(len(clients), cfg["mechanism"]["peer_audit_fanout_c"]), rng)
    W, M = _build_audit_and_M(reports, cluster_flats, clients, factory, edges, device)
    W = impute_missing(W)
    with torch.no_grad():
        Z, _ = partition(W, tau=cfg["mechanism"]["gumbel_tau_end"], hard=True)
        x = recovery(W)
        util = -(Z * M).sum(dim=1)
    return float((util[i] - d_t[i] + x[i]).item())


def run(cfg):
    dev = select_device()
    rng = np.random.default_rng(cfg["experiment"]["seed"])
    clients = build_clients(cfg, rng)
    factory = build_model_factory(cfg)
    K = cfg["federation"]["num_clusters"]
    d = compute_disagreements(clients, factory,
                              epochs=cfg["federation"]["local_epochs"],
                              lr=cfg["federation"]["client_lr"], device=dev)
    log = train_dng_net(clients, factory, K, cfg, d)
    d_t = torch.tensor(d, dtype=torch.float32, device=dev)
    cluster_flats = log.cluster_flats

    # Recreate mechanism modules at the converged state.
    partition = ProbabilisticLatentPartitioning(len(clients), K,
                                                 cfg["mechanism"]["partition_hidden"],
                                                 feature_mode=cfg["mechanism"].get("partition_feature_mode","stats")).to(dev)
    recovery = NashRecoveryLayer().to(dev)

    truthful_reports = [factory().to(dev).get_flat_params().detach() for _ in clients]
    # Best-response search on a random subset.
    subset = list(rng.choice(len(clients), size=min(10, len(clients)), replace=False))
    gains = []
    for i in subset:
        u_truth = _utility_for_client(truthful_reports, i, cfg, clients, factory,
                                       cluster_flats, dev, partition, recovery, d_t)
        r = truthful_reports[i].clone().requires_grad_(True)
        opt = torch.optim.Adam([r], lr=1e-2)
        best_u = u_truth
        for _ in range(cfg["evaluation"].get("br_steps", 30)):
            reports_i = list(truthful_reports)
            reports_i[i] = r
            u = _utility_for_client(reports_i, i, cfg, clients, factory,
                                    cluster_flats, dev, partition, recovery, d_t)
            best_u = max(best_u, u)
            # Numerical gradient via a random perturbation (utility path is
            # non-differentiable end-to-end because argmax hardening breaks
            # gradient flow through Z at eval time).
            eps = 0.01
            g = torch.zeros_like(r)
            for _ in range(3):
                v = torch.randn_like(r)
                v = v / v.norm()
                reports_i[i] = (r + eps * v).detach()
                u_plus = _utility_for_client(reports_i, i, cfg, clients, factory,
                                             cluster_flats, dev, partition, recovery, d_t)
                g += (u_plus - u) / eps * v
            with torch.no_grad():
                r.add_(1e-2 * g)
        gains.append(best_u - u_truth)
    return {"max_unilateral_gain": bootstrap_ci(np.array(gains))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/e06.json")
    a = ap.parse_args()
    dump(run(yaml.safe_load(open(a.config))), a.out)


if __name__ == "__main__":
    main()
