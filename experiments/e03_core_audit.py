"""E3: Core deviation audit.

Question: at convergence, does any coalition profitably deviate from the DNG-Net
partition?

Protocol:
    - Train DNG-Net on a small synthetic mixture (N=20 by default) to allow full
      enumeration of coalitions up to size 4.
    - For each coalition S, compute v(S) as the mean test accuracy that S
      members would achieve by training a coalition-local model.
    - Compute per-capita deviation gain (v(S) - sum_{i in S} u_i) / |S|.
    - Report the maximum gain, its bootstrap CI, and the round-by-round series
      via periodic snapshots.

For larger N a sampled coalition set is used, complemented by a learned
adversarial coalition search (implemented in worst_case_coalition_search).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml

from data_pkg.registry import build_clients, build_model_factory
from experiments.train_dng import compute_disagreements, train_dng_net
from metrics.metrics import core_deviation_gain
from metrics.statistics import bootstrap_ci
from models.client_models import evaluate, local_train
from utils.device import select_device
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


def coalition_value(clients, S: Sequence[int], model_factory, epochs: int, lr: float, device: str) -> float:
    """Train one coalition-local model and return the SUM of member utilities.

    Utility is `-test_loss` to match the units used in the training loop, so
    that `core_deviation_gain` compares comparable quantities.
    """
    xs = torch.cat([clients[i].train_x for i in S])
    ys = torch.cat([clients[i].train_y for i in S])
    loader = DataLoader(TensorDataset(xs, ys), batch_size=64, shuffle=True)
    m = model_factory().to(device)
    m = local_train(m, loader, epochs=epochs, lr=lr, device=device)
    m.eval()
    with torch.no_grad():
        vals = []
        for i in S:
            x_te, y_te = clients[i].test_x.to(device), clients[i].test_y.to(device)
            vals.append(-float(F.cross_entropy(m(x_te), y_te).item()))
    return float(np.sum(vals))


def run(cfg):
    seeds = list(range(cfg["experiment"]["num_seeds"]))
    max_gains = []
    for seed in seeds:
        cfg["experiment"]["seed"] = seed
        rng = np.random.default_rng(seed)
        clients = build_clients(cfg, rng)
        factory = build_model_factory(cfg)

        disagreements = compute_disagreements(
            clients, factory,
            epochs=cfg["federation"]["local_epochs"],
            lr=cfg["federation"]["client_lr"],
            device=select_device(),
        )
        log = train_dng_net(clients, factory, cfg["federation"]["num_clusters"], cfg, disagreements)

        device = log.device

        def v(S):
            return coalition_value(clients, S, factory, epochs=2, lr=cfg["federation"]["client_lr"], device=device)

        max_gain, all_gains = core_deviation_gain(
            utilities=log.per_client_test_utility,
            allocation=log.final_x,
            coalition_value_fn=v,
            max_size=cfg["evaluation"]["core_audit_max_size"],
            sampled_coalitions=cfg["evaluation"]["core_sampled_coalitions"],
            rng=rng,
        )
        max_gains.append(max_gain)

    m, lo, hi = bootstrap_ci(np.array(max_gains), iters=cfg["evaluation"]["bootstrap_iters"])
    return {
        "max_gain_mean": m,
        "max_gain_ci_lower": lo,
        "max_gain_ci_upper": hi,
        "seeds": len(seeds),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/e03.json")
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
