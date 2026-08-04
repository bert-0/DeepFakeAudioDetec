"""Incremento 3 — Fusão + mecanismo de atenção (TC1 §4.7, §3.4).

Estende o Incremento 2 substituindo o pooling por um pooling com atenção
temporal em cada ramo, permitindo que o modelo priorize os frames mais
relevantes do sinal antes da fusão (inspirado em Vaswani et al., 2017).

Com `pooling` estatístico a atenção vira Attentive Statistics Pooling.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import build_encoder
from .pooling import build_attention_pooling


class AttentionFusionNet(nn.Module):
    """Fusão com pooling por atenção. Ver `encoder`/`pooling` em BaselineCNN."""

    def __init__(self, n_classes: int = 2, dropout: float = 0.3, pooling: str = "avg",
                 channels: tuple[int, ...] = (16, 32, 64), encoder: str = "cnn",
                 freq_bins: int = 4):
        super().__init__()
        self.lfcc_branch = build_encoder(encoder, in_ch=1, channels=channels)
        self.spec_branch = build_encoder(encoder, in_ch=1, channels=channels)
        self.lfcc_attn, dim_a = build_attention_pooling(
            pooling, self.lfcc_branch.out_channels, freq_bins=freq_bins)
        self.spec_attn, dim_b = build_attention_pooling(
            pooling, self.spec_branch.out_channels, freq_bins=freq_bins)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_a + dim_b, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        a = self.lfcc_attn(self.lfcc_branch(features["lfcc"]))
        b = self.spec_attn(self.spec_branch(features["spectrogram"]))
        return self.classifier(torch.cat([a, b], dim=1))
