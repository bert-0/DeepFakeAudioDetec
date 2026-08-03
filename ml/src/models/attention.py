"""Incremento 3 — Fusão + mecanismo de atenção (TC1 §4.7, §3.4).

Estende o Incremento 2 substituindo o pooling global por um pooling com atenção
temporal em cada ramo, permitindo que o modelo priorize os frames mais
relevantes do sinal antes da fusão (inspirado em Vaswani et al., 2017).

NOTA (julho/2026): ponto de partida funcional do Incremento 3. Refinamentos
previstos: atenção multi-cabeça, atenção cruzada entre os ramos (cross-attention)
e atenção também no eixo de frequência.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import AttentiveStatsPool, CNNEncoder, TemporalAttentionPool


class AttentionFusionNet(nn.Module):
    """Fusão com pooling por atenção.

    `pooling`:
      - "avg"   (padrão): atenção sobre a média — comportamento original.
      - "stats": Attentive Statistics Pooling (média + desvio ponderados).
    """

    def __init__(self, n_classes: int = 2, dropout: float = 0.3, pooling: str = "avg"):
        super().__init__()
        self.lfcc_branch = CNNEncoder(in_ch=1, channels=(16, 32, 64))
        self.spec_branch = CNNEncoder(in_ch=1, channels=(16, 32, 64))

        if pooling == "stats":
            pool_cls = AttentiveStatsPool
        elif pooling == "avg":
            pool_cls = TemporalAttentionPool
        else:
            raise ValueError(f"pooling desconhecido: {pooling!r} (use 'avg' ou 'stats')")

        self.lfcc_attn = pool_cls(self.lfcc_branch.out_channels)
        self.spec_attn = pool_cls(self.spec_branch.out_channels)
        fused_dim = self.lfcc_attn.out_dim + self.spec_attn.out_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        a = self.lfcc_attn(self.lfcc_branch(features["lfcc"]))
        b = self.spec_attn(self.spec_branch(features["spectrogram"]))
        fused = torch.cat([a, b], dim=1)
        return self.classifier(fused)
