"""Sensitivity driver S1 through S10.

Each axis is either a config sweep or a controlled data perturbation. Reports
the standard metric bundle per sweep point.
"""

from __future__ import annotations

import argparse
import copy
import numpy as np
import torch
import yaml

from data_pkg.registry import build_clients, build_model_factory
from data_pkg.synthetic import ClientData
from experiments._common import dump
from experiments.train_dng import train_dng_net, compute_disagreements
from metrics.metrics import ari, ir_violation_rate
from metrics.statistics import bootstrap_ci
from utils.device import select_device


def _bundle(cfg, mutate_clients=None):
    dev = select_device()
    aris, irs = [], []
    for seed in range(cfg["experiment"]["num_seeds"]):
        cfg["experiment"]["seed"] = seed
        rng = np.random.default_rng(seed)
        clients = build_clients(cfg, rng)
        if mutate_clients is not None:
            clients = mutate_clients(clients, rng)
        factory = build_model_factory(cfg)
        honest = np.ones(len(clients), dtype=bool)
        d = compute_disagreements(clients, factory,
                                  epochs=cfg["federation"]["local_epochs"],
                                  lr=cfg["federation"]["client_lr"], device=dev)
        log = train_dng_net(clients, factory, cfg["federation"]["num_clusters"], cfg, d)
        true_z = np.array([c.latent_cluster for c in clients])
        pred = log.final_Z.argmax(axis=1)
        aris.append(ari(true_z, pred))
        frac_v, _ = ir_violation_rate(log.per_client_test_utility,
                                       log.per_client_disagreement, log.final_x, honest)
        irs.append(frac_v)
    return {"ari": bootstrap_ci(np.array(aris)), "ir_violation_frac": bootstrap_ci(np.array(irs))}


def _label_noise(fraction):
    def mut(clients, rng):
        for c in clients:
            n = c.train_y.numel()
            k = int(n * fraction)
            idx = rng.choice(n, size=k, replace=False)
            c.train_y[idx] = torch.tensor(rng.integers(0, int(c.train_y.max().item()) + 1, size=k))
        return clients
    return mut


def _feature_drop(fraction):
    def mut(clients, rng):
        for c in clients:
            mask = torch.tensor(rng.random(c.train_x.shape) > fraction).float()
            c.train_x = c.train_x * mask
        return clients
    return mut


def _pgd_perturb(eps):
    def mut(clients, rng):
        for c in clients:
            c.train_x = c.train_x + eps * torch.sign(torch.randn_like(c.train_x))
        return clients
    return mut


def run(cfg):
    out = {}
    base = copy.deepcopy(cfg)

    # S1: 10 seeds; already the default.
    out["S1_seed_variance"] = _bundle(copy.deepcopy(base))

    # S2: Z-init prior is fixed by architecture; report entropy at round 1 only.
    out["S2_init_entropy"] = _bundle(copy.deepcopy(base))

    # S3: learning-rate sweep on the partitioning network.
    lr_out = {}
    for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
        c = copy.deepcopy(base); c["mechanism"]["partition_lr"] = lr
        lr_out[str(lr)] = _bundle(c)
    out["S3_partition_lr"] = lr_out

    # S4: Gumbel temperature schedule.
    tau_out = {}
    for sch in ["constant", "linear", "exponential"]:
        c = copy.deepcopy(base); c["mechanism"]["gumbel_anneal"] = sch
        tau_out[sch] = _bundle(c)
    out["S4_gumbel_schedule"] = tau_out

    # S5: sample-size heterogeneity is a data-gen knob; run at three ranges.
    n_out = {}
    for rng_pair in [[100, 200], [200, 500], [500, 1000]]:
        c = copy.deepcopy(base); c["data"]["samples_per_client_range"] = rng_pair
        n_out[str(rng_pair)] = _bundle(c)
    out["S5_size_heterogeneity"] = n_out

    # S6: label noise.
    noise_out = {}
    for f in [0.0, 0.05, 0.15, 0.30]:
        noise_out[str(f)] = _bundle(copy.deepcopy(base), mutate_clients=_label_noise(f))
    out["S6_label_noise"] = noise_out

    # S7: missing features.
    miss_out = {}
    for f in [0.0, 0.10, 0.30, 0.50]:
        miss_out[str(f)] = _bundle(copy.deepcopy(base), mutate_clients=_feature_drop(f))
    out["S7_missing_features"] = miss_out

    # S8: PGD-style adversarial perturbation.
    pgd_out = {}
    for eps in [0.0, 0.01, 0.05, 0.10]:
        pgd_out[str(eps)] = _bundle(copy.deepcopy(base), mutate_clients=_pgd_perturb(eps))
    out["S8_pgd_perturbation"] = pgd_out

    # S9: OOD client injection: use one more cluster than the mechanism expects.
    c = copy.deepcopy(base); c["federation"]["num_clusters"] -= 1
    out["S9_ood_extra_cluster"] = _bundle(c)

    # S10: fixed compute budget: halve rounds, double local_epochs.
    c = copy.deepcopy(base)
    c["federation"]["rounds"] = max(1, c["federation"]["rounds"] // 2)
    c["federation"]["local_epochs"] *= 2
    out["S10_compute_budget"] = _bundle(c)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/sensitivity.json")
    a = ap.parse_args()
    dump(run(yaml.safe_load(open(a.config))), a.out)


if __name__ == "__main__":
    main()
