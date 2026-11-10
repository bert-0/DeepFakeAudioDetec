"""Fábrica dos modos de pooling, compartilhada pelos três incrementos."""

from __future__ import annotations

import torch.nn as nn

from .blocks import (
    AttentiveFreqStatsPool,
    AttentiveStatsPool,
    FreqStatsPool,
    StatsPool,
    TemporalAttentionPool,
)


def build_pooling(
    name: str, channels: int, freq_bins: int = 4
) -> tuple[nn.Module, int]:
    """Devolve (módulo de pooling, dimensão de saída).

    `avg` é a média global; `stats` faz média e desvio no tempo; `freq_stats`
    faz o mesmo mantendo `freq_bins` faixas de frequência.
    """
    if name == "avg":
        return nn.AdaptiveAvgPool2d((1, 1)), channels
    if name == "stats":
        pool = StatsPool(channels)
        return pool, pool.out_dim
    if name == "freq_stats":
        pool = FreqStatsPool(channels, freq_bins=freq_bins)
        return pool, pool.out_dim
    raise ValueError(
        f"pooling desconhecido: {name!r} (use 'avg', 'stats' ou 'freq_stats')")


def build_attention_pooling(
    name: str, channels: int, freq_bins: int = 4
) -> tuple[nn.Module, int]:
    """Versões com atenção de `build_pooling` (Incremento 3).

    Cada modo espelha o sem atenção, para a comparação isolar o efeito da atenção.
    """
    if name == "avg":
        pool = TemporalAttentionPool(channels)
    elif name == "stats":
        pool = AttentiveStatsPool(channels)
    elif name == "freq_stats":
        pool = AttentiveFreqStatsPool(channels, freq_bins=freq_bins)
    else:
        raise ValueError(
            f"pooling desconhecido: {name!r} (use 'avg', 'stats' ou 'freq_stats')")
    return pool, pool.out_dim
