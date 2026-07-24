"""Dataset and model registry.

Central dispatch by config name. Adding a new dataset means writing a loader
and a model factory and registering them here. Every experiment script should
route through `build_clients` and `build_model_factory` rather than importing
loaders directly so ablations can swap datasets without touching experiment
code.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np

from data_pkg.synthetic import ClientData, make_synthetic_mixture
from models.client_models import CNNClassifier, MLPClassifier


def build_clients(cfg: dict, rng: np.random.Generator) -> List[ClientData]:
    name = cfg["data"]["name"]
    if name == "synthetic_mixture":
        return make_synthetic_mixture(
            num_clients=cfg["federation"]["num_clients"],
            num_clusters=cfg["federation"]["num_clusters"],
            input_dim=cfg["data"]["input_dim"],
            num_classes=cfg["data"]["num_classes"],
            samples_per_client_range=tuple(cfg["data"]["samples_per_client_range"]),
            intra_cluster_std=cfg["data"]["intra_cluster_std"],
            inter_cluster_shift=cfg["data"]["inter_cluster_shift"],
            rng=rng,
        )
    if name == "cifar10_dirichlet":
        from data_pkg.cifar10_dirichlet import make_cifar10_dirichlet
        return make_cifar10_dirichlet(
            root=cfg["data"].get("root", "./data"),
            num_clients=cfg["federation"]["num_clients"],
            num_clusters=cfg["federation"]["num_clusters"],
            dirichlet_alpha=cfg["data"].get("dirichlet_alpha", 0.3),
            rng=rng,
        )
    if name == "femnist":
        from data_pkg.femnist import make_femnist
        return make_femnist(
            train_dir=cfg["data"]["train_dir"],
            test_dir=cfg["data"]["test_dir"],
            max_users=cfg["federation"]["num_clients"],
            cluster_by=cfg["data"].get("cluster_by", "digit_split"),
            num_clusters=cfg["federation"]["num_clusters"],
        )
    if name == "shakespeare":
        from data_pkg.shakespeare import make_shakespeare, VOCAB_SIZE
        return make_shakespeare(
            train_dir=cfg["data"]["train_dir"],
            test_dir=cfg["data"]["test_dir"],
            max_users=cfg["federation"]["num_clients"],
            num_clusters=cfg["federation"]["num_clusters"],
        )
    raise ValueError(f"unknown dataset: {name}")


def build_model_factory(cfg: dict) -> Callable:
    name = cfg["data"]["name"]
    if name == "synthetic_mixture":
        input_dim = cfg["data"]["input_dim"]
        num_classes = cfg["data"]["num_classes"]
        return lambda: MLPClassifier(input_dim, [64, 64], num_classes)
    if name == "cifar10_dirichlet":
        return lambda: CNNClassifier(in_channels=3, num_classes=10, input_hw=32)
    if name == "femnist":
        num_classes = cfg["data"].get("num_classes", 62)
        return lambda: CNNClassifier(in_channels=1, num_classes=num_classes, input_hw=28)
    if name == "shakespeare":
        from data_pkg.shakespeare import VOCAB_SIZE
        from models.client_models import CharLSTMClassifier
        return lambda: CharLSTMClassifier(vocab_size=VOCAB_SIZE)
    raise ValueError(f"unknown dataset: {name}")
