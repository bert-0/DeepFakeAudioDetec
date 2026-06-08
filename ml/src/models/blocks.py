"""Blocos de rede compartilhados entre os modelos dos três incrementos."""

from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CNNEncoder(nn.Module):
    """Pilha de blocos convolucionais que extrai mapas de características 2D.

    Entrada:  (B, in_ch, freq, frames)
    Saída:    (B, channels[-1], freq', frames')
    """

    def __init__(self, in_ch: int = 1, channels: tuple[int, ...] = (16, 32, 64)):
        super().__init__()
        blocks = []
        prev = in_ch
        for ch in channels:
            blocks.append(conv_block(prev, ch))
            prev = ch
        self.net = nn.Sequential(*blocks)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalAttentionPool(nn.Module):
    """Pooling com atenção sobre o eixo temporal (mecanismo de atenção).

    Reduz o mapa (B, C, freq, frames) a um vetor (B, C), ponderando os frames
    pela sua relevância em vez de fazer uma média simples. É o componente que
    diferencia o Incremento 3.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=2)                       # média na frequência -> (B, C, T)
        weights = torch.softmax(self.score(x), dim=-1)  # (B, 1, T)
        return (x * weights).sum(dim=-1)        # soma ponderada -> (B, C)
