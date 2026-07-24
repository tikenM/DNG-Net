"""E7: Gumbel-Softmax annealing study.

Compares constant, linear, and exponential temperature schedules on: final
row entropy of Z, cluster recovery ARI, and rounds-to-convergence.
"""

from __future__ import annotations

import argparse
import numpy as np
import yaml

from data_pkg.registry import build_clients, build_model_factory
from experiments._common import dump
from experiments.train_dng import train_dng_net, compute_disagreements
from metrics.metrics import ari
from metrics.statistics import bootstrap_ci
from utils.device import select_device


def run(cfg):
    schedules = ["constant", "linear", "exponential"]
    dev = select_device()
    out = {s: {"entropy_end": [], "ari": [], "rounds_to_low_entropy": []} for s in schedules}
    for schedule in schedules:
        cfg["mechanism"]["gumbel_anneal"] = schedule
        for seed in range(cfg["experiment"]["num_seeds"]):
            cfg["experiment"]["seed"] = seed
            rng = np.random.default_rng(seed)
            clients = build_clients(cfg, rng)
            factory = build_model_factory(cfg)
            true_z = np.array([c.latent_cluster for c in clients])
            d = compute_disagreements(clients, factory,
                                      epochs=cfg["federation"]["local_epochs"],
                                      lr=cfg["federation"]["client_lr"], device=dev)
            log = train_dng_net(clients, factory, cfg["federation"]["num_clusters"], cfg, d)
            pred = log.final_Z.argmax(axis=1)
            ent = np.array(log.row_entropy_history)
            r2c = int(np.argmax(ent < 0.2)) if (ent < 0.2).any() else len(ent)
            out[schedule]["entropy_end"].append(float(ent[-1]))
            out[schedule]["ari"].append(ari(true_z, pred))
            out[schedule]["rounds_to_low_entropy"].append(r2c)
    return {s: {k: bootstrap_ci(np.array(v)) for k, v in kv.items()} for s, kv in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/e07.json")
    a = ap.parse_args()
    dump(run(yaml.safe_load(open(a.config))), a.out)


if __name__ == "__main__":
    main()
