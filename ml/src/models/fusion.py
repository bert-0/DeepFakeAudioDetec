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

from .blocks import CNNEncoder


class FeatureFusionNet(nn.Module):
    def __init__(self, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lfcc_branch = CNNEncoder(in_ch=1, channels=(16, 32, 64))
        self.spec_branch = CNNEncoder(in_ch=1, channels=(16, 32, 64))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        fused_dim = self.lfcc_branch.out_channels + self.spec_branch.out_channels
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def _encode(self, branch: nn.Module, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(branch(x))           # (B, C, 1, 1)
        return torch.flatten(h, 1)         # (B, C)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        a = self._encode(self.lfcc_branch, features["lfcc"])
        b = self._encode(self.spec_branch, features["spectrogram"])
        fused = torch.cat([a, b], dim=1)
        return self.classifier(fused)
