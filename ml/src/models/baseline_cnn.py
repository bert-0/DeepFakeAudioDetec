"""Incremento 1 — Baseline: CNN convencional sobre LFCC (TC1 §4.7, §5.3).

Serve de referência para medir o ganho dos incrementos de fusão e atenção.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import CNNEncoder, StatsPool


class BaselineCNN(nn.Module):
    """CNN sobre LFCC.

    `pooling`:
      - "avg"   (padrão): média global — comportamento original.
      - "stats": média + desvio-padrão temporais, preservando a variação do sinal.
    """

    def __init__(self, n_classes: int = 2, dropout: float = 0.3, pooling: str = "avg",
                 channels: tuple[int, ...] = (16, 32, 64)):
        super().__init__()
        self.encoder = CNNEncoder(in_ch=1, channels=tuple(channels))
        self.pooling = pooling

        if pooling == "stats":
            self.pool = StatsPool(self.encoder.out_channels)
            feat_dim = self.pool.out_dim
        elif pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            feat_dim = self.encoder.out_channels
        else:
            raise ValueError(f"pooling desconhecido: {pooling!r} (use 'avg' ou 'stats')")

        # O nn.Flatten() é mantido como primeira camada do Sequential (mesmo com
        # pooling "stats", onde ele é um no-op sobre um tensor já 2D) para que os
        # índices das camadas — e portanto as chaves do state_dict — continuem
        # compatíveis com checkpoints treinados antes do pooling configurável.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_classes),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.encoder(features["lfcc"])
        h = self.pool(h)
        return self.classifier(h)
