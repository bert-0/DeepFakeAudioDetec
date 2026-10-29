"""Incremento 2: fusão tardia de LFCC + espectrograma (TC1 §4.7).

Dois ramos convolucionais independentes, concatenados após o pooling.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import build_encoder
from .pooling import build_pooling


class FeatureFusionNet(nn.Module):
    """Fusão tardia de dois ramos.

    `branches` escolhe a feature de cada ramo: qualquer par de `features.types`,
    inclusive dois LFCC de resoluções diferentes.
    """

    def __init__(self, n_classes: int = 2, dropout: float = 0.3, pooling: str = "avg",
                 channels: tuple[int, ...] = (16, 32, 64), encoder: str = "cnn",
                 freq_bins: int = 4,
                 branches: tuple[str, str] = ("lfcc", "spectrogram")):
        super().__init__()
        self.branches = tuple(branches)
        # Nomes lfcc_branch/spec_branch mantidos pelas chaves do state_dict.
        self.lfcc_branch = build_encoder(encoder, in_ch=1, channels=channels)
        self.spec_branch = build_encoder(encoder, in_ch=1, channels=channels)
        self.lfcc_pool, dim_a = build_pooling(
            pooling, self.lfcc_branch.out_channels, freq_bins=freq_bins)
        self.spec_pool, dim_b = build_pooling(
            pooling, self.spec_branch.out_channels, freq_bins=freq_bins)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_a + dim_b, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def _encode(self, branch: nn.Module, pool: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(pool(branch(x)), 1)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        first, second = self.branches
        a = self._encode(self.lfcc_branch, self.lfcc_pool, features[first])
        b = self._encode(self.spec_branch, self.spec_pool, features[second])
        return self.classifier(torch.cat([a, b], dim=1))
