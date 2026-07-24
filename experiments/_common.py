"""Shared harness for every experiment script.

`run_with_seeds(cfg, per_seed_fn)` calls per_seed_fn once per seed, collects the
returned scalar or dict of scalars, and returns bootstrap CIs. Every script
below relies on this so the reporting style is uniform.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from metrics.statistics import bootstrap_ci


def run_with_seeds(cfg, per_seed_fn: Callable) -> dict:
    out: dict = {}
    for seed in range(cfg["experiment"]["num_seeds"]):
        cfg["experiment"]["seed"] = seed
        res = per_seed_fn(cfg, seed)
        if isinstance(res, (int, float)):
            out.setdefault("value", []).append(float(res))
        else:
            for k, v in res.items():
                out.setdefault(k, []).append(float(v))
    return {k: bootstrap_ci(np.array(v), iters=cfg["evaluation"].get("bootstrap_iters", 2000))
            for k, v in out.items()}


def dump(result: dict, path: str) -> None:
    import json, pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
