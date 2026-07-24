"""E12: positive collaboration surplus.

Compares per-client test utility inside DNG-Net against local-only training.
A positive mean improvement is the premise that a cooperative game has
non-trivial surplus to allocate.
"""

from __future__ import annotations

import argparse
import numpy as np
import yaml

from data_pkg.registry import build_clients, build_model_factory
from experiments._common import dump
from experiments.train_dng import train_dng_net, compute_disagreements
from metrics.statistics import bootstrap_ci, wilcoxon_paired
from models.baselines import fit_local_only
from utils.device import select_device


def run(cfg):
    dev = select_device()
    per_client_gain = []
    for seed in range(cfg["experiment"]["num_seeds"]):
        cfg["experiment"]["seed"] = seed
        rng = np.random.default_rng(seed)
        clients = build_clients(cfg, rng)
        factory = build_model_factory(cfg)
        d = compute_disagreements(clients, factory,
                                  epochs=cfg["federation"]["local_epochs"],
                                  lr=cfg["federation"]["client_lr"], device=dev)
        log = train_dng_net(clients, factory, cfg["federation"]["num_clusters"], cfg, d)
        # Local-only utility in matching units (already d_i).
        gains = log.per_client_test_utility - d
        per_client_gain.append(gains.mean())
    arr = np.array(per_client_gain)
    return {
        "mean_per_client_surplus": bootstrap_ci(arr),
        "positive_fraction": float((arr > 0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/e12.json")
    a = ap.parse_args()
    dump(run(yaml.safe_load(open(a.config))), a.out)


if __name__ == "__main__":
    main()
