"""LCNN: Light CNN com Max-Feature-Map (Lavrentyeva et al., ASVspoof 2019).

Usa MFM no lugar da ReLU e alterna blocos 1x1 e 3x3. Tem a mesma interface do
`CNNEncoder`, então serve aos três incrementos.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import MFM


def _conv_mfm(in_ch: int, out_ch: int, kernel: int, padding: int = 0) -> nn.Sequential:
    """Conv -> MFM. `out_ch` conta os canais após a MFM (a conv gera o dobro)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * 2, kernel_size=kernel, padding=padding),
        MFM(),
    )


class LCNNEncoder(nn.Module):
    """Pilha convolucional da LCNN, com 4 max-pools de 2x em cada eixo.

    (B, in_ch, freq, frames) -> (B, out_channels, freq', frames')
    """

    def __init__(self, in_ch: int = 1, width: int = 1.0):
        super().__init__()
        # Larguras da LCNN-9 clássica (canais já após a MFM).
        c1, c2, c3, c4 = (int(w * width) for w in (32, 48, 64, 32))

        self.net = nn.Sequential(
            _conv_mfm(in_ch, c1, 5, padding=2),
            nn.MaxPool2d(2),

            _conv_mfm(c1, c1, 1),
            _conv_mfm(c1, c2, 3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(c2),

            _conv_mfm(c2, c2, 1),
            nn.BatchNorm2d(c2),
            _conv_mfm(c2, c3, 3, padding=1),
            nn.MaxPool2d(2),

            _conv_mfm(c3, c3, 1),
            nn.BatchNorm2d(c3),
            _conv_mfm(c3, c4, 3, padding=1),
            nn.BatchNorm2d(c4),

            _conv_mfm(c4, c4, 1),
            nn.BatchNorm2d(c4),
            _conv_mfm(c4, c4, 3, padding=1),
            nn.MaxPool2d(2),
        )
        self.out_channels = c4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
