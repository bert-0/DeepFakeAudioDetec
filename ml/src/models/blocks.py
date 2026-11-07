"""Blocos de rede compartilhados entre os modelos dos três incrementos."""

from __future__ import annotations

import torch
import torch.nn as nn

# Piso da variância antes do sqrt; 1e-8 viraria zero em fp16 (menor normal ~6e-5).
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
    """Pilha de blocos conv 3x3 + BN + ReLU + MaxPool.

    (B, in_ch, freq, frames) -> (B, channels[-1], freq', frames')
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
    """Média e desvio-padrão no tempo: (B, C, freq, frames) -> (B, 2*C).

    O desvio guarda a variação temporal (artefatos transientes) que a média perde.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.out_dim = 2 * channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Em float32: sob AMP a soma de quadrados estoura o fp16 (máx. ~65504) e
        # o NaN contamina o BatchNorm de vez.
        with torch.amp.autocast(x.device.type, enabled=False):
            x = x.float().mean(dim=2)           # média na frequência -> (B, C, T)
            mean = x.mean(dim=-1)
            # clamp evita gradiente infinito do sqrt quando a variância é ~0.
            std = x.var(dim=-1, unbiased=False).clamp(min=EPS_VAR).sqrt()
            return torch.cat([mean, std], dim=1)  # (B, 2*C)


class FreqStatsPool(nn.Module):
    """Como o StatsPool, mas mantém `freq_bins` faixas de frequência em vez de 1.

    Os artefatos de síntese dependem da faixa, e a média na frequência os apaga.
    (B, C, freq, frames) -> (B, 2 * C * freq_bins)
    """

    def __init__(self, channels: int, freq_bins: int = 4):
        super().__init__()
        # None: mantém o eixo temporal.
        self.freq = nn.AdaptiveAvgPool2d((freq_bins, None))
        self.out_dim = 2 * channels * freq_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.freq(x)                                # (B, C, freq_bins, T)
        with torch.amp.autocast(x.device.type, enabled=False):
            x = x.float()
            b, c, f, t = x.shape
            x = x.reshape(b, c * f, t)                  # frequência vira canal
            mean = x.mean(dim=-1)
            std = x.var(dim=-1, unbiased=False).clamp(min=EPS_VAR).sqrt()
            return torch.cat([mean, std], dim=1)        # (B, 2*C*freq_bins)


class MFM(nn.Module):
    """Max-Feature-Map (Wu et al.): máximo entre as duas metades dos canais.

    Ativação da LCNN; (B, 2n, ...) -> (B, n, ...).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.chunk(x, 2, dim=1)
        return torch.max(a, b)


class TemporalAttentionPool(nn.Module):
    """Média dos frames ponderada por atenção: (B, C, freq, frames) -> (B, C)."""

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)
        self.out_dim = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=2)                       # média na frequência -> (B, C, T)
        weights = torch.softmax(self.score(x), dim=-1)  # (B, 1, T)
        return (x * weights).sum(dim=-1)        # soma ponderada -> (B, C)


class AttentiveFreqStatsPool(nn.Module):
    """Attentive Statistics Pooling mantendo `freq_bins` faixas de frequência.

    É o `freq_stats` do Incremento 3, para não perder o eixo espectral que a
    fusão mantém. (B, C, freq, frames) -> (B, 2 * C * freq_bins)
    """

    def __init__(self, channels: int, freq_bins: int = 4):
        super().__init__()
        self.freq = nn.AdaptiveAvgPool2d((freq_bins, None))
        self.score = nn.Conv1d(channels * freq_bins, 1, kernel_size=1)
        self.out_dim = 2 * channels * freq_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.freq(x)                                 # (B, C, freq_bins, T)
        b, c, f, t = x.shape
        x = x.reshape(b, c * f, t)                       # frequência vira canal
        weights = torch.softmax(self.score(x), dim=-1)   # (B, 1, T)
        with torch.amp.autocast(x.device.type, enabled=False):
            x32, w = x.float(), weights.float()
            mean = (x32 * w).sum(dim=-1)
            var = (x32.pow(2) * w).sum(dim=-1) - mean.pow(2)
            std = var.clamp(min=EPS_VAR).sqrt()
            return torch.cat([mean, std], dim=1)         # (B, 2*C*freq_bins)


class AttentiveStatsPool(nn.Module):
    """Attentive Statistics Pooling (Okabe et al., 2018): média e desvio ponderados.

    (B, C, freq, frames) -> (B, 2*C)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Conv1d(channels, 1, kernel_size=1)
        self.out_dim = 2 * channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x.mean(dim=2)), dim=-1)  # (B, 1, T)
        # Em float32, como no StatsPool.
        with torch.amp.autocast(x.device.type, enabled=False):
            x = x.float().mean(dim=2)                   # (B, C, T)
            w = weights.float()
            mean = (x * w).sum(dim=-1)                  # (B, C)
            var = (x.pow(2) * w).sum(dim=-1) - mean.pow(2)
            std = var.clamp(min=EPS_VAR).sqrt()
            return torch.cat([mean, std], dim=1)        # (B, 2*C)
