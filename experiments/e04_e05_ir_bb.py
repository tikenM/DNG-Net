"""E4 (IR check) and E5 (budget-balance residual).

Uses the fixed `metrics.ir_violation_rate` that includes the transfer in the
utility. Also reports the maximum absolute round-by-round budget-balance
residual, on log scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from data_pkg.registry import build_clients, build_model_factory
from data_pkg.synthetic import inject_adversaries
from experiments.train_dng import compute_disagreements, train_dng_net
from metrics.metrics import ir_violation_rate
from metrics.statistics import bootstrap_ci
from utils.device import select_device


def run(cfg):
    ir_fracs, ir_shortfalls, bb_residuals = [], [], []
    for seed in range(cfg["experiment"]["num_seeds"]):
        cfg["experiment"]["seed"] = seed
        rng = np.random.default_rng(seed)
        clients = build_clients(cfg, rng)
        if cfg["adversary"]["fraction_adversarial"] > 0:
            clients = inject_adversaries(
                clients, fraction=cfg["adversary"]["fraction_adversarial"],
                target_cluster=0, rng=rng,
            )
        honest_mask = np.array([not c.is_adversarial for c in clients])
        factory = build_model_factory(cfg)
        device = select_device()
        d = compute_disagreements(
            clients, factory,
            epochs=cfg["federation"]["local_epochs"],
            lr=cfg["federation"]["client_lr"],
            device=device,
        )
        log = train_dng_net(clients, factory, cfg["federation"]["num_clusters"], cfg, d)

        frac, short = ir_violation_rate(
            utilities=log.per_client_test_utility,
            disagreements=log.per_client_disagreement,
            allocation=log.final_x,
            honest_mask=honest_mask,
        )
        ir_fracs.append(frac)
        ir_shortfalls.append(short)
        bb_residuals.append(max(log.bb_residual_history))

    return {
        "ir_violation_fraction": bootstrap_ci(np.array(ir_fracs)),
        "ir_mean_shortfall": bootstrap_ci(np.array(ir_shortfalls)),
        "bb_residual_max_over_rounds": bootstrap_ci(np.array(bb_residuals)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/e04_e05.json")
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
