"""Incremento 1 — LFCC + encoder convolucional (TC1 §4.7, §5.3).

Com `encoder="cnn"` é a CNN convencional exigida pelo TC1 como referência.
Com `encoder="lcnn"` usa a Light CNN com Max-Feature-Map.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import build_encoder
from .pooling import build_pooling


class BaselineCNN(nn.Module):
    """Classificador de um único ramo (LFCC).

    `pooling`:
      - "avg"        (padrão): média global — comportamento original.
      - "stats":     média + desvio temporais (descarta o eixo de frequência).
      - "freq_stats": preserva a estrutura em frequência antes das estatísticas.
    """

    def __init__(self, n_classes: int = 2, dropout: float = 0.3, pooling: str = "avg",
                 channels: tuple[int, ...] = (16, 32, 64), encoder: str = "cnn",
                 freq_bins: int = 4):
        super().__init__()
        self.encoder = build_encoder(encoder, in_ch=1, channels=channels)
        self.pool, feat_dim = build_pooling(
            pooling, self.encoder.out_channels, freq_bins=freq_bins)

        # O nn.Flatten() é mantido como primeira camada do Sequential (no-op
        # quando o pooling já devolve um tensor 2D) para que os índices das
        # camadas — e portanto as chaves do state_dict — continuem compatíveis
        # com checkpoints treinados antes do pooling configurável.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_classes),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.pool(self.encoder(features["lfcc"]))
        return self.classifier(h)
