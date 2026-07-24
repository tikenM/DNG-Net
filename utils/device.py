"""Device selection with Apple Silicon (MPS) support.

Selection order:
    1. Explicit user override.
    2. CUDA if available.
    3. Apple MPS if available (M-series chips).
    4. CPU.

MPS falls back to CPU for a small set of ops. This is silent in recent PyTorch
and does not require additional handling. Two flags help when a fallback is
observed:

    export PYTORCH_ENABLE_MPS_FALLBACK=1

is recommended for training on Apple Silicon.
"""

from __future__ import annotations

import os

import torch


def select_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    env = os.environ.get("DNG_DEVICE")
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_device(x, device: str):
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(y, device) for y in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x
