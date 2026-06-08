"""Incremento 1 — Baseline: CNN convencional sobre LFCC (TC1 §4.7, §5.3).

Serve de referência para medir o ganho dos incrementos de fusão e atenção.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import CNNEncoder


class BaselineCNN(nn.Module):
    def __init__(self, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.encoder = CNNEncoder(in_ch=1, channels=(16, 32, 64))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.out_channels, n_classes),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.encoder(features["lfcc"])
        h = self.pool(h)
        return self.classifier(h)
