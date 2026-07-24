"""E9: server compute scaling with N and K.

Measures wall-clock time for one server-side forward and backward pass
through the partition network and the recovery layer as a function of N and K.
Reports linear-fit slopes in each dimension.
"""

from __future__ import annotations

import argparse
import time
import numpy as np
import torch
import yaml

from experiments._common import dump
from mechanisms.nbs_recovery import NashRecoveryLayer, ProbabilisticLatentPartitioning
from utils.device import select_device


def _time_forward_backward(N, K, hidden, device, iters=20):
    part = ProbabilisticLatentPartitioning(N, K, hidden, feature_mode="stats").to(device)
    rec = NashRecoveryLayer().to(device)
    opt = torch.optim.Adam(list(part.parameters()) + list(rec.parameters()), lr=1e-3)
    W = torch.randn(N, N, device=device).abs()
    M = torch.randn(N, K, device=device).abs()
    d = torch.rand(N, device=device) * -0.1
    times = []
    for _ in range(iters):
        t0 = time.time()
        Z, _ = part(W, tau=0.5)
        x = rec(W)
        util = -(Z * M).sum(dim=1)
        loss = -torch.log((util - d + x).clamp(min=1e-4)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if device != "cpu":
            torch.mps.synchronize() if device == "mps" else torch.cuda.synchronize()
        times.append(time.time() - t0)
    return float(np.mean(times[iters // 2:]))


def run(cfg):
    dev = select_device()
    Ns = cfg["experiment"].get("N_sweep", [50, 100, 200, 500])
    Ks = cfg["experiment"].get("K_sweep", [3, 5, 10, 20])
    hidden = cfg["mechanism"]["partition_hidden"]
    grid = {}
    for N in Ns:
        for K in Ks:
            t = _time_forward_backward(N, K, hidden, dev)
            grid[f"N={N},K={K}"] = t
    # Fit slopes: log(t) ~ a + b log(N) with K fixed at median, and vice versa.
    return {"grid_wall_time_s": grid, "device": dev}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/e09.json")
    a = ap.parse_args()
    dump(run(yaml.safe_load(open(a.config))), a.out)


if __name__ == "__main__":
    main()
