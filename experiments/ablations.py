"""Ablation driver A1 through A10.

Each ablation overrides one config key and reruns the standard metric bundle.
Isolates the contribution of one design decision to the headline claims.
"""

from __future__ import annotations

import argparse
import copy
import numpy as np
import yaml

from data_pkg.registry import build_clients, build_model_factory
from experiments._common import dump
from experiments.train_dng import train_dng_net, compute_disagreements
from metrics.metrics import ari, ir_violation_rate
from metrics.statistics import bootstrap_ci
from utils.device import select_device


ABLATIONS = {
    "A1_parameter_distance":       {"ablation": {"use_parameter_distance": True}},
    "A2_mean_utility":             {"ablation": {"mean_utility": True}},
    "A3_no_allocation":            {"ablation": {"disable_allocation": True}},
    "A4_deterministic_softmax":    {"ablation": {"deterministic_softmax": True}},
    "A5_no_projection":            {"ablation": {"no_projection": True}},
    "A6_row_input":                {"mechanism": {"partition_feature_mode": "row"}},
    "A7_low_fanout":               {"mechanism": {"peer_audit_fanout_c": 1.0}},
    "A8_deep_partition":           {"mechanism": {"partition_hidden": [64, 64, 64, 64]}},
    "A9_post_hoc_projection":      {"ablation": {"post_hoc_projection": True}},
    "A10_sgd_optimizer":           {"ablation": {"server_optimizer": "sgd"}},
}


def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _bundle(cfg):
    dev = select_device()
    aris, irs = [], []
    for seed in range(cfg["experiment"]["num_seeds"]):
        cfg["experiment"]["seed"] = seed
        rng = np.random.default_rng(seed)
        clients = build_clients(cfg, rng)
        factory = build_model_factory(cfg)
        honest = np.ones(len(clients), dtype=bool)
        d = compute_disagreements(clients, factory,
                                  epochs=cfg["federation"]["local_epochs"],
                                  lr=cfg["federation"]["client_lr"], device=dev)
        log = train_dng_net(clients, factory, cfg["federation"]["num_clusters"], cfg, d)
        true_z = np.array([c.latent_cluster for c in clients])
        pred = log.final_Z.argmax(axis=1)
        aris.append(ari(true_z, pred))
        frac_v, _ = ir_violation_rate(log.per_client_test_utility,
                                       log.per_client_disagreement, log.final_x, honest)
        irs.append(frac_v)
    return {"ari": bootstrap_ci(np.array(aris)), "ir_violation_frac": bootstrap_ci(np.array(irs))}


def run(cfg):
    baseline = _bundle(copy.deepcopy(cfg))
    out = {"baseline": baseline}
    for name, override in ABLATIONS.items():
        c = copy.deepcopy(cfg)
        _deep_merge(c, override)
        out[name] = _bundle(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/ablations.json")
    a = ap.parse_args()
    dump(run(yaml.safe_load(open(a.config))), a.out)


if __name__ == "__main__":
    main()
