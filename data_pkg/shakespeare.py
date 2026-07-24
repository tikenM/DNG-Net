"""LEAF Shakespeare loader for next-character prediction.

LEAF preprocesses Shakespeare into per-role JSON dumps of 80-character input
sequences and their next-character targets. The vocabulary is the printable
ASCII subset used by LEAF (80 characters).

This loader returns one ClientData per Shakespearean role, ready to feed a
character-level LSTM. It does not download or preprocess LEAF for you; run
LEAF's preprocess.sh once and point `train_dir` and `test_dir` at the results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch

from data_pkg.synthetic import ClientData


LEAF_ALL_LETTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
)
VOCAB_SIZE = len(LEAF_ALL_LETTERS)
_CHAR2IDX = {ch: i for i, ch in enumerate(LEAF_ALL_LETTERS)}


def _encode(seq: str) -> np.ndarray:
    return np.array([_CHAR2IDX.get(c, 0) for c in seq], dtype=np.int64)


def _load(dir_path: Path) -> dict:
    data = {}
    for jf in sorted(dir_path.glob("*.json")):
        with open(jf) as f:
            blob = json.load(f)
        for u in blob["users"]:
            ud = blob["user_data"][u]
            xs = np.stack([_encode(s) for s in ud["x"]])
            ys = np.array([_CHAR2IDX.get(c, 0) for c in ud["y"]], dtype=np.int64)
            data[u] = (xs, ys)
    return data


def make_shakespeare(
    train_dir: str,
    test_dir: str,
    max_users: int | None = None,
    num_clusters: int = 5,
) -> List[ClientData]:
    """Return one ClientData per role. Latent cluster is a length-modulo hash of
    the role name, which gives a stable but arbitrary K-way partition suitable
    for recovery experiments. Replace with an act- or genre-based partition if
    a semantically meaningful ground truth is required.
    """
    train_map = _load(Path(train_dir))
    test_map = _load(Path(test_dir))
    ids = sorted(set(train_map) & set(test_map))
    if max_users is not None:
        ids = ids[:max_users]
    clients: List[ClientData] = []
    for i, u in enumerate(ids):
        x_tr, y_tr = train_map[u]
        x_te, y_te = test_map[u]
        k = hash(u) % num_clusters
        clients.append(ClientData(
            client_id=i,
            latent_cluster=k,
            train_x=torch.tensor(x_tr),
            train_y=torch.tensor(y_tr),
            test_x=torch.tensor(x_te),
            test_y=torch.tensor(y_te),
        ))
    return clients
