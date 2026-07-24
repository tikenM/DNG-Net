"""Orchestration entry point.

Runs the headline experiments in order and writes their JSON summaries under
`runs/`. Individual experiments can also be launched directly, e.g.:

    python -m experiments.e01_signal_separation --config configs/base.yaml

For full reproducibility, seeds and configs live under configs/. Ablations
A1-A10 and sensitivity sweeps S1-S10 have their own configs derived from
`configs/base.yaml` by shallow override; see `configs/README.md`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENTS = [
    "experiments.e01_signal_separation",
    "experiments.e02_admission_rate",
    "experiments.e03_core_audit",
    "experiments.e04_e05_ir_bb",
    "experiments.e06_best_response",
    "experiments.e07_gumbel_annealing",
    "experiments.e08_communication_scaling",
    "experiments.e09_server_compute_scaling",
    "experiments.e10_collusion_sweep",
    "experiments.e11_latent_recovery",
    "experiments.e12_surplus_quantification",
    "experiments.e13_scalability",
    "experiments.theorem_validation",
    "experiments.ablations",
    "experiments.sensitivity",
    # E14 is a driver over multiple dataset configs; invoke it separately with
    # a config that lists them, e.g. configs/e14_cross_dataset.yaml.
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default="runs")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for mod in EXPERIMENTS:
        out = Path(args.out_dir) / f"{mod.split('.')[-1]}.json"
        print(f"[run_all] {mod} -> {out}", flush=True)
        subprocess.run(
            [sys.executable, "-m", mod, "--config", args.config, "--out", str(out)],
            check=True,
        )


if __name__ == "__main__":
    main()
