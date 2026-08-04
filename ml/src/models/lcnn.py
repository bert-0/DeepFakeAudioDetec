"""LCNN — Light CNN com Max-Feature-Map, aplicada a LFCC.

Arquitetura amplamente usada em detecção de spoofing de voz (Lavrentyeva et al.,
sistema do STC para o ASVspoof 2019). Duas características a distinguem da CNN
convencional do Incremento 1:

1. **Max-Feature-Map (MFM)** no lugar da ReLU: em vez de zerar ativações
   negativas, ela põe pares de mapas para competir e mantém o mais forte,
   funcionando como seleção de características.
2. **Blocos 1x1 + 3x3 alternados**, que refinam os canais antes de cada
   convolução espacial, dando profundidade com poucos parâmetros.

O encoder é um substituto direto do `CNNEncoder`: mesma interface de entrada
(B, 1, freq, frames) e saída (B, C, freq', frames'), então serve aos três
incrementos (baseline, fusão e atenção).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import MFM


def _conv_mfm(in_ch: int, out_ch: int, kernel: int, padding: int = 0) -> nn.Sequential:
    """Conv -> MFM. `out_ch` é o número de canais APÓS a MFM.

    A convolução produz 2*out_ch canais porque a MFM os consome aos pares.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * 2, kernel_size=kernel, padding=padding),
        MFM(),
    )


class LCNNEncoder(nn.Module):
    """Pilha convolucional da LCNN.

    Entrada: (B, in_ch, freq, frames)
    Saída:   (B, out_channels, freq', frames')  —  4 reduções de 2x em cada eixo.
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
