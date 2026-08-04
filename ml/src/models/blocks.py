"""Blocos de rede compartilhados entre os modelos dos três incrementos."""

from __future__ import annotations

import torch
import torch.nn as nn

# Piso da variância antes do sqrt. Precisa ser representável em float16 caso o
# tensor chegue em meia precisão (o menor normal do fp16 é ~6e-5), por isso
# 1e-4 em vez de valores como 1e-8, que virariam zero.
EPS_VAR = 1e-4


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


class StatsPool(nn.Module):
    """Pooling estatístico: concatena média e desvio-padrão no eixo temporal.

    O pooling por média global (`AdaptiveAvgPool2d((1,1))`) descarta a *variação*
    do sinal ao longo do tempo — justamente onde ficam os artefatos transientes
    da síntese de voz. Guardar também o desvio-padrão preserva essa informação
    e dobra a dimensão do vetor de saída (2*C em vez de C).

    Entrada: (B, C, freq, frames)  ->  Saída: (B, 2*C)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.out_dim = 2 * channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Estatísticas sempre em float32: sob AMP (fp16) a variância é uma soma
        # de quadrados que estoura facilmente o alcance do tipo (máx. ~65504),
        # virando inf -> NaN. O NaN então contamina as estatísticas do BatchNorm
        # e o modelo passa a produzir NaN para sempre em modo eval.
        with torch.amp.autocast(x.device.type, enabled=False):
            x = x.float().mean(dim=2)           # média na frequência -> (B, C, T)
            mean = x.mean(dim=-1)
            # clamp evita gradiente infinito do sqrt quando a variância é ~0.
            std = x.var(dim=-1, unbiased=False).clamp(min=EPS_VAR).sqrt()
            return torch.cat([mean, std], dim=1)  # (B, 2*C)


class TemporalAttentionPool(nn.Module):
    """Pooling com atenção sobre o eixo temporal (mecanismo de atenção).

    Reduz o mapa (B, C, freq, frames) a um vetor (B, C), ponderando os frames
    pela sua relevância em vez de fazer uma média simples. É o componente que
    diferencia o Incremento 3.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)
        self.out_dim = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=2)                       # média na frequência -> (B, C, T)
        weights = torch.softmax(self.score(x), dim=-1)  # (B, 1, T)
        return (x * weights).sum(dim=-1)        # soma ponderada -> (B, C)


class AttentiveStatsPool(nn.Module):
    """Attentive Statistics Pooling: média E desvio-padrão ponderados por atenção.

    Une as duas ideias anteriores — a atenção escolhe quais frames importam, e a
    estatística preserva tanto o nível médio quanto a variação temporal desses
    frames. Técnica consagrada em tarefas de voz (Okabe et al., 2018).

    Entrada: (B, C, freq, frames)  ->  Saída: (B, 2*C)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)
        self.out_dim = 2 * channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x.mean(dim=2)), dim=-1)  # (B, 1, T)
        # Mesma proteção do StatsPool: a soma de quadrados sai do alcance do
        # fp16 sob AMP, então as estatísticas são calculadas em float32.
        with torch.amp.autocast(x.device.type, enabled=False):
            x = x.float().mean(dim=2)                   # (B, C, T)
            w = weights.float()
            mean = (x * w).sum(dim=-1)                  # (B, C)
            var = (x.pow(2) * w).sum(dim=-1) - mean.pow(2)
            std = var.clamp(min=EPS_VAR).sqrt()
            return torch.cat([mean, std], dim=1)        # (B, 2*C)
