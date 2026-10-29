"""Incremento 1: LFCC + encoder convolucional (TC1 §4.7, §5.3).

`encoder="cnn"` é a CNN de referência do TC1; `encoder="lcnn"` usa a Light CNN.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import build_encoder
from .pooling import build_pooling


class BaselineCNN(nn.Module):
    """Classificador de um ramo só (LFCC). Modos de `pooling` em `build_pooling`."""

    def __init__(self, n_classes: int = 2, dropout: float = 0.3, pooling: str = "avg",
                 channels: tuple[int, ...] = (16, 32, 64), encoder: str = "cnn",
                 freq_bins: int = 4):
        super().__init__()
        self.encoder = build_encoder(encoder, in_ch=1, channels=channels)
        self.pool, feat_dim = build_pooling(
            pooling, self.encoder.out_channels, freq_bins=freq_bins)

        # Flatten fica em primeiro (no-op com pooling 2D) para manter as chaves
        # do state_dict dos checkpoints antigos.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_classes),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.pool(self.encoder(features["lfcc"]))
        return self.classifier(h)
