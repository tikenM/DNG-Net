"""E2: adversarial admission rate under mimicry.

Sweeps adversary fraction and reports the rate at which mimicked clients end
up in the target cluster. Compares DNG-Net to IFCA, FeSEM, CFL, and
FedAvg+kmeans.
"""

from __future__ import annotations

import argparse
import numpy as np
import yaml
from sklearn.cluster import KMeans

from data_pkg.registry import build_clients, build_model_factory
from data_pkg.synthetic import inject_adversaries
from experiments._common import dump
from experiments.train_dng import train_dng_net, compute_disagreements
from metrics.metrics import admission_rate
from metrics.statistics import bootstrap_ci
from models.baselines import fit_ifca, fit_fesem, fit_cfl, fit_fedavg
from utils.device import select_device


def _run_one(cfg, seed, adv_frac):
    cfg["experiment"]["seed"] = seed
    cfg["adversary"]["fraction_adversarial"] = adv_frac
    rng = np.random.default_rng(seed)
    clients = build_clients(cfg, rng)
    clients = inject_adversaries(clients, adv_frac, target_cluster=0, rng=rng)
    adv = np.array([c.is_adversarial for c in clients])
    factory = build_model_factory(cfg)
    dev = select_device()
    K = cfg["federation"]["num_clusters"]
    R = cfg["federation"]["rounds"]
    E = cfg["federation"]["local_epochs"]
    lr = cfg["federation"]["client_lr"]

    d = compute_disagreements(clients, factory, epochs=E, lr=lr, device=dev)
    dng = train_dng_net(clients, factory, K, cfg, d)
    dng_assign = dng.final_Z.argmax(axis=1)

    ifca = fit_ifca(clients, factory, K, R, E, lr)
    fesem = fit_fesem(clients, factory, K, R, E, lr)
    cfl = fit_cfl(clients, factory, K, R, E, lr)

    return {
        "dng": admission_rate(dng_assign, adv, target_cluster=0),
        "ifca": admission_rate(ifca.cluster_assignment, adv, target_cluster=0),
        "fesem": admission_rate(fesem.cluster_assignment, adv, target_cluster=0),
        "cfl": admission_rate(cfl.cluster_assignment, adv, target_cluster=0),
    }


def run(cfg):
    fractions = cfg["experiment"].get("adversary_fractions", [0.05, 0.10, 0.20, 0.30])
    result = {}
    for f in fractions:
        per = {}
        for seed in range(cfg["experiment"]["num_seeds"]):
            r = _run_one(cfg, seed, f)
            for k, v in r.items():
                per.setdefault(k, []).append(v)
        result[str(f)] = {k: bootstrap_ci(np.array(v)) for k, v in per.items()}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/e02.json")
    a = ap.parse_args()
    dump(run(yaml.safe_load(open(a.config))), a.out)


if __name__ == "__main__":
    main()
