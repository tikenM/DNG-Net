"""E11: latent cluster recovery under honest reports.

Compares ARI, NMI, and cluster purity of DNG-Net against IFCA, FeSEM, and CFL.
Ground truth is the client's latent cluster label.
"""

from __future__ import annotations

import argparse
import numpy as np
import yaml

from data_pkg.registry import build_clients, build_model_factory
from experiments._common import dump
from experiments.train_dng import train_dng_net, compute_disagreements
from metrics.metrics import ari, nmi, cluster_purity
from metrics.statistics import bootstrap_ci, wilcoxon_paired
from models.baselines import fit_ifca, fit_fesem, fit_cfl
from utils.device import select_device


def run(cfg):
    dev = select_device()
    methods = ["dng", "ifca", "fesem", "cfl"]
    scores = {m: {"ari": [], "nmi": [], "purity": []} for m in methods}
    K = cfg["federation"]["num_clusters"]
    R = cfg["federation"]["rounds"]
    E = cfg["federation"]["local_epochs"]
    lr = cfg["federation"]["client_lr"]
    for seed in range(cfg["experiment"]["num_seeds"]):
        cfg["experiment"]["seed"] = seed
        rng = np.random.default_rng(seed)
        clients = build_clients(cfg, rng)
        factory = build_model_factory(cfg)
        true_z = np.array([c.latent_cluster for c in clients])

        d = compute_disagreements(clients, factory, epochs=E, lr=lr, device=dev)
        dng = train_dng_net(clients, factory, K, cfg, d)
        preds = {
            "dng": dng.final_Z.argmax(axis=1),
            "ifca": fit_ifca(clients, factory, K, R, E, lr).cluster_assignment,
            "fesem": fit_fesem(clients, factory, K, R, E, lr).cluster_assignment,
            "cfl": fit_cfl(clients, factory, K, R, E, lr).cluster_assignment,
        }
        for m, p in preds.items():
            scores[m]["ari"].append(ari(true_z, p))
            scores[m]["nmi"].append(nmi(true_z, p))
            scores[m]["purity"].append(cluster_purity(true_z, p))

    out = {m: {k: bootstrap_ci(np.array(v)) for k, v in kv.items()} for m, kv in scores.items()}
    # Paired Wilcoxon: DNG vs each baseline on ARI.
    tests = {}
    if len(scores["dng"]["ari"]) >= 2:
        dng_ari = np.array(scores["dng"]["ari"])
        for b in ["ifca", "fesem", "cfl"]:
            try:
                _, p = wilcoxon_paired(dng_ari, np.array(scores[b]["ari"]))
            except ValueError:
                p = float("nan")
            tests[f"dng_vs_{b}_ari_wilcoxon_p"] = p
    out["paired_tests"] = tests
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/e11.json")
    a = ap.parse_args()
    dump(run(yaml.safe_load(open(a.config))), a.out)


if __name__ == "__main__":
    main()
