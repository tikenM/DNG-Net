"""E14: cross-dataset generalization.

Runs a fixed metric bundle (ARI, IR-violation, BB residual, surplus) on each
of a list of configs pointing at synthetic, cifar10, and femnist. Uses the
same mechanism configuration except for the data section.

This is a thin driver: it just iterates over configs and calls the relevant
per-dataset experiment runs internally, then collates the JSON summaries.
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


def _one_config(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    dev = select_device()
    aris, irs, bbs, surpluses = [], [], [], []
    for seed in range(cfg["experiment"]["num_seeds"]):
        cfg["experiment"]["seed"] = seed
        rng = np.random.default_rng(seed)
        clients = build_clients(cfg, rng)
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
        bbs.append(max(log.bb_residual_history))
        surpluses.append((log.per_client_test_utility - d).mean())
    return {
        "ari": bootstrap_ci(np.array(aris)),
        "ir_violation_frac": bootstrap_ci(np.array(irs)),
        "bb_residual_max": bootstrap_ci(np.array(bbs)),
        "mean_surplus": bootstrap_ci(np.array(surpluses)),
    }


def run(cfg):
    return {name: _one_config(path) for name, path in cfg["datasets"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/e14.json")
    a = ap.parse_args()
    dump(run(yaml.safe_load(open(a.config))), a.out)


if __name__ == "__main__":
    main()
