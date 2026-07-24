"""FEMNIST loader for the LEAF preprocessed format.

LEAF (https://leaf.cmu.edu) ships FEMNIST as a directory of JSON files. Each
JSON contains a `users` list and a `user_data` dict mapping each user id to
`{'x': [[float]*784], 'y': [int]}`. The preprocessing script produces two
directories, `train/` and `test/`, that share the same set of user ids.

This loader reads those directories and returns one ClientData per user id. It
does not download LEAF for you; the user must run the LEAF preprocessing
script once and point this loader at the resulting paths. See
https://github.com/TalwalkarLab/leaf for instructions.

Two options for ground-truth clustering:
    - `cluster_by='writer'`: every writer is their own cluster. This gives
      high K (about 3500) and is unsuitable for K=5-10 recovery experiments.
    - `cluster_by='digit_split'`: writers are grouped by which class subset
      appears most in their data (0-4 vs 5-9, or a finer partition). Suitable
      for E11-style recovery when a small K is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal

import numpy as np
import torch

from data_pkg.synthetic import ClientData


def _load_users(dir_path: Path) -> dict:
    users_data = {}
    for jf in sorted(dir_path.glob("*.json")):
        with open(jf) as f:
            blob = json.load(f)
        for u in blob["users"]:
            ud = blob["user_data"][u]
            x = np.asarray(ud["x"], dtype=np.float32).reshape(-1, 1, 28, 28)
            y = np.asarray(ud["y"], dtype=np.int64)
            users_data[u] = (x, y)
    return users_data


def _assign_digit_split(y_train: np.ndarray, num_clusters: int) -> int:
    """Group writers by which digit range dominates their samples."""
    counts = np.bincount(y_train, minlength=10)
    dominant_class = int(np.argmax(counts))
    return dominant_class * num_clusters // 10


def make_femnist(
    train_dir: str,
    test_dir: str,
    max_users: int | None = None,
    cluster_by: Literal["writer", "digit_split"] = "digit_split",
    num_clusters: int = 5,
) -> List[ClientData]:
    train_map = _load_users(Path(train_dir))
    test_map = _load_users(Path(test_dir))
    user_ids = sorted(set(train_map.keys()) & set(test_map.keys()))
    if max_users is not None:
        user_ids = user_ids[:max_users]

    clients: List[ClientData] = []
    for i, u in enumerate(user_ids):
        x_tr, y_tr = train_map[u]
        x_te, y_te = test_map[u]
        if cluster_by == "writer":
            k = i
        else:
            k = _assign_digit_split(y_tr, num_clusters)
        clients.append(ClientData(
            client_id=i,
            latent_cluster=k,
            train_x=torch.tensor(x_tr),
            train_y=torch.tensor(y_tr),
            test_x=torch.tensor(x_te),
            test_y=torch.tensor(y_te),
        ))
    return clients
