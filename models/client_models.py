"""Client-side model architectures.

The paper does not tie itself to a particular backbone. For the synthetic and
tabular experiments a small MLP suffices. For FEMNIST and CIFAR-10 a
convolutional backbone is provided. All models expose:

    forward(x) -> logits
    get_flat_params() -> 1-D tensor
    set_flat_params(vec) -> None

so that the mechanism layer can treat a client's report as a vector without
knowing its shape.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlatParamMixin:
    """Provides get_flat_params / set_flat_params for any nn.Module."""

    def get_flat_params(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.parameters()])

    def set_flat_params(self, vec: torch.Tensor) -> None:
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(vec[offset : offset + n].view_as(p))
            offset += n


class MLPClassifier(nn.Module, FlatParamMixin):
    def __init__(self, input_dim: int, hidden: List[int], num_classes: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNClassifier(nn.Module, FlatParamMixin):
    """Small CNN for FEMNIST / CIFAR-10 experiments.

    Input shape is inferred from a dummy forward pass at construction time so
    the same class serves 28x28 grayscale (FEMNIST) and 32x32 RGB (CIFAR-10).
    """

    def __init__(self, in_channels: int, num_classes: int, input_hw: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_hw, input_hw)
            h = self.pool(F.relu(self.conv1(dummy)))
            h = self.pool(F.relu(self.conv2(h)))
            flat_dim = h.reshape(1, -1).size(1)
        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def local_train(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    device: str = "cpu",
) -> nn.Module:
    """Standard local SGD training used by both DNG-Net and every baseline."""
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: str = "cpu") -> float:
    """Return top-1 accuracy on the given loader."""
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: torch.utils.data.DataLoader, device: str = "cpu") -> float:
    """Return mean cross-entropy loss on the given loader.

    This is the peer-audit signal used to populate W_ij when client i evaluates
    the model reported by client j.
    """
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x), y, reduction="sum")
        total_loss += loss.item()
        total += y.numel()
    return total_loss / max(total, 1)


class CharLSTMClassifier(nn.Module, FlatParamMixin):
    """Small character-level LSTM for the Shakespeare next-character task.

    Input shape: [B, T] of long indices. Output: [B, vocab_size] logits for the
    next character. Suitable for LEAF Shakespeare with vocab_size = 80 and
    T = 80. Small enough to run on CPU and MPS at modest client counts.
    """

    def __init__(self, vocab_size: int = 80, embed: int = 32, hidden: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embed(x)
        h, _ = self.lstm(e)
        return self.out(h[:, -1])
