"""E10: colluding-clique threshold.

Sweeps the colluding fraction of the target cluster and reports admission
rate and cluster purity. Expected phase transition near 0.5 per
Assumption A4.
"""

from __future__ import annotations

import argparse
import numpy as np
import yaml

from data_pkg.registry import build_clients, build_model_factory
from data_pkg.synthetic import inject_adversaries
from experiments._common import dump
from experiments.train_dng import train_dng_net, compute_disagreements
from metrics.metrics import admission_rate, cluster_purity
from metrics.statistics import bootstrap_ci
from utils.device import select_device


def run(cfg):
    dev = select_device()
    rhos = cfg["experiment"].get("collusion_fractions", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    out = {}
    for rho in rhos:
        adm, pur = [], []
        for seed in range(cfg["experiment"]["num_seeds"]):
            cfg["experiment"]["seed"] = seed
            cfg["adversary"]["collusion_fraction"] = rho
            cfg["adversary"]["fraction_adversarial"] = rho
            rng = np.random.default_rng(seed)
            clients = build_clients(cfg, rng)
            clients = inject_adversaries(clients, rho, target_cluster=0, rng=rng)
            adv = np.array([c.is_adversarial for c in clients])
            true_z = np.array([c.latent_cluster for c in clients])
            factory = build_model_factory(cfg)
            d = compute_disagreements(clients, factory,
                                      epochs=cfg["federation"]["local_epochs"],
                                      lr=cfg["federation"]["client_lr"], device=dev)
            log = train_dng_net(clients, factory, cfg["federation"]["num_clusters"], cfg, d)
            pred = log.final_Z.argmax(axis=1)
            adm.append(admission_rate(pred, adv, target_cluster=0))
            pur.append(cluster_purity(true_z, pred))
        out[str(rho)] = {
            "admission_rate": bootstrap_ci(np.array(adm)),
            "cluster_purity": bootstrap_ci(np.array(pur)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/e10.json")
    a = ap.parse_args()
    dump(run(yaml.safe_load(open(a.config))), a.out)


if __name__ == "__main__":
    main()
