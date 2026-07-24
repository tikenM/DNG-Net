"""E13: scalability with N and K, including K-misspecification.

Reports ARI, IR-violation rate, and BB residual across a grid of population
sizes and cluster counts. Also sweeps the mechanism's K against the true K to
audit robustness to misspecification.
"""

from __future__ import annotations

import argparse
import numpy as np
import yaml

from data_pkg.registry import build_clients, build_model_factory
from experiments._common import dump
from experiments.train_dng import train_dng_net, compute_disagreements
from metrics.metrics import ari, ir_violation_rate
from metrics.statistics import bootstrap_ci
from utils.device import select_device


def _one(cfg, seed, N, K_true, K_mech):
    cfg["experiment"]["seed"] = seed
    cfg["federation"]["num_clients"] = N
    cfg["federation"]["num_clusters"] = K_mech
    cfg["data"]["name"] = "synthetic_mixture"
    rng = np.random.default_rng(seed)
    # Override K in the data generator by rebuilding with K_true.
    cfg_data = dict(cfg)
    cfg_data["federation"] = dict(cfg["federation"])
    cfg_data["federation"]["num_clusters"] = K_true
    clients = build_clients(cfg_data, rng)
    honest = np.ones(N, dtype=bool)
    factory = build_model_factory(cfg)
    dev = select_device()
    d = compute_disagreements(clients, factory,
                              epochs=cfg["federation"]["local_epochs"],
                              lr=cfg["federation"]["client_lr"], device=dev)
    log = train_dng_net(clients, factory, K_mech, cfg, d)
    true_z = np.array([c.latent_cluster for c in clients])
    pred = log.final_Z.argmax(axis=1)
    frac_v, _ = ir_violation_rate(log.per_client_test_utility,
                                   log.per_client_disagreement,
                                   log.final_x, honest)
    return {
        "ari": ari(true_z, pred),
        "ir_violation_frac": frac_v,
        "bb_residual_max": max(log.bb_residual_history),
    }


def run(cfg):
    grid = cfg["experiment"].get("grid", [(50, 3, 3), (200, 5, 5), (200, 5, 3), (200, 5, 8), (1000, 10, 10)])
    out = {}
    for N, K_true, K_mech in grid:
        agg = {}
        for seed in range(cfg["experiment"]["num_seeds"]):
            r = _one(cfg, seed, N, K_true, K_mech)
            for k, v in r.items():
                agg.setdefault(k, []).append(v)
        out[f"N={N},K_true={K_true},K_mech={K_mech}"] = {
            k: bootstrap_ci(np.array(v)) for k, v in agg.items()
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/e13.json")
    a = ap.parse_args()
    dump(run(yaml.safe_load(open(a.config))), a.out)


if __name__ == "__main__":
    main()
