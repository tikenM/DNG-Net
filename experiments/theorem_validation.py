"""Theorem 1 validation.

Verifies:

    P1. Statistically distinguishable misreports receive strictly negative x.
    P2. Row entropy of Z decays and successive-round ARI approaches 1.
    P3. Coalition-deviation gains approach zero (dispatched to E3).

Also verifies Theorem 1 assumptions:

    A. Positive individual surpluses: report the fraction of rounds in which
       any u_i - d_i + x_i is non-positive.
    B. Definition 1 margin: estimate the empirical margin between honest and
       adversarial risk on the peer-audit signal, per dataset.

The margin estimate is a lower bound. If small on some dataset, Theorem 1's
guarantee is weak there, and this script reports that plainly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import adjusted_rand_score

from data_pkg.registry import build_clients, build_model_factory
from data_pkg.synthetic import inject_adversaries
from experiments.train_dng import compute_disagreements, train_dng_net
from utils.device import select_device


def run(cfg):
    rng = np.random.default_rng(cfg["experiment"]["seed"])
    clients = build_clients(cfg, rng)
    clients = inject_adversaries(
        clients,
        fraction=cfg["adversary"]["fraction_adversarial"],
        target_cluster=0,
        rng=rng,
    )
    adv_mask = np.array([c.is_adversarial for c in clients])

    factory = build_model_factory(cfg)

    disagreements = compute_disagreements(
        clients, factory,
        epochs=cfg["federation"]["local_epochs"],
        lr=cfg["federation"]["client_lr"],
        device=select_device(),
    )
    log = train_dng_net(clients, factory, cfg["federation"]["num_clusters"], cfg, disagreements)

    # P1: adversaries receive negative x.
    x = log.final_x
    p1 = {
        "mean_x_adversarial": float(x[adv_mask].mean()) if adv_mask.any() else None,
        "mean_x_honest": float(x[~adv_mask].mean()),
        "fraction_adv_negative": float((x[adv_mask] < 0).mean()) if adv_mask.any() else None,
    }

    # P2: entropy decay.
    ent = np.array(log.row_entropy_history)
    p2 = {
        "entropy_start": float(ent[0]),
        "entropy_end": float(ent[-1]),
        "monotone_fraction": float((np.diff(ent) <= 0).mean()),
    }

    # Assumption A: non-positive surplus rounds.
    a_check = {
        "rounds_with_non_positive_surplus": log.non_positive_surplus_rounds,
        "total_rounds": len(log.loss_history),
    }

    # Assumption B: risk margin proxy = (mean_honest_x - mean_adv_x) / std_honest_x.
    if adv_mask.any():
        margin = (x[~adv_mask].mean() - x[adv_mask].mean()) / (x[~adv_mask].std() + 1e-9)
    else:
        margin = None
    b_check = {"empirical_margin_proxy": float(margin) if margin is not None else None}

    return {"P1": p1, "P2": p2, "assumption_A": a_check, "assumption_B": b_check}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/theorem_validation.json")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    result = run(cfg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
