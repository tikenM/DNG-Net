"""E8: communication scaling with N.

Reports bytes transmitted per round as a function of the client population.
For DNG-Net, per-round bytes = (audit fan-out per client) * (report size). The
fan-out is c * log2(N) by design.
"""

from __future__ import annotations

import argparse
import math
import numpy as np
import yaml

from data_pkg.registry import build_model_factory
from experiments._common import dump
from mechanisms.peer_audit import audit_fanout


def run(cfg):
    Ns = cfg["experiment"].get("N_sweep", [20, 50, 100, 200, 500, 1000])
    c = cfg["mechanism"]["peer_audit_fanout_c"]
    factory = build_model_factory(cfg)
    bytes_per_report = 4 * int(factory().get_flat_params().numel())
    out = {"N": [], "fanout": [], "bytes_per_round": [], "log_bytes_per_report": []}
    for N in Ns:
        f = audit_fanout(N, c)
        out["N"].append(N)
        out["fanout"].append(f)
        out["bytes_per_round"].append(N * f * bytes_per_report)
        out["log_bytes_per_report"].append(math.log2(bytes_per_report))
    # Fit slope in log(bytes/round) vs log(N * log N).
    x = np.log(np.array(out["N"]) * np.log2(np.array(out["N"])))
    y = np.log(out["bytes_per_round"])
    slope, intercept = np.polyfit(x, y, 1)
    return {"scan": out, "log_log_slope_vs_N_logN": float(slope), "intercept": float(intercept)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/e08.json")
    a = ap.parse_args()
    dump(run(yaml.safe_load(open(a.config))), a.out)


if __name__ == "__main__":
    main()
