"""Incremento 2 — Fusão de características: LFCC + espectrograma (TC1 §4.7).

Dois ramos CNN independentes (um por representação) cujas saídas, após pooling
global, são concatenadas e classificadas. A ideia é explorar informações
complementares das duas representações (fusão tardia / late fusion).

NOTA (julho/2026): este é o ponto de partida funcional do Incremento 2.
Refinamentos previstos: ajustar a profundidade/canais de cada ramo, testar
fusão intermediária (concatenar mapas antes do pooling) e regularização.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import CNNEncoder, StatsPool


class FeatureFusionNet(nn.Module):
    """Fusão tardia de dois ramos CNN. Ver `pooling` em BaselineCNN."""

    def __init__(self, n_classes: int = 2, dropout: float = 0.3, pooling: str = "avg"):
        super().__init__()
        self.lfcc_branch = CNNEncoder(in_ch=1, channels=(16, 32, 64))
        self.spec_branch = CNNEncoder(in_ch=1, channels=(16, 32, 64))
        self.pooling = pooling

        if pooling == "stats":
            self.lfcc_pool = StatsPool(self.lfcc_branch.out_channels)
            self.spec_pool = StatsPool(self.spec_branch.out_channels)
            fused_dim = self.lfcc_pool.out_dim + self.spec_pool.out_dim
        elif pooling == "avg":
            self.lfcc_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.spec_pool = nn.AdaptiveAvgPool2d((1, 1))
            fused_dim = self.lfcc_branch.out_channels + self.spec_branch.out_channels
        else:
            raise ValueError(f"pooling desconhecido: {pooling!r} (use 'avg' ou 'stats')")

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def _encode(self, branch: nn.Module, pool: nn.Module, x: torch.Tensor) -> torch.Tensor:
        h = pool(branch(x))
        return torch.flatten(h, 1) if self.pooling == "avg" else h

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        a = self._encode(self.lfcc_branch, self.lfcc_pool, features["lfcc"])
        b = self._encode(self.spec_branch, self.spec_pool, features["spectrogram"])
        fused = torch.cat([a, b], dim=1)
        return self.classifier(fused)
