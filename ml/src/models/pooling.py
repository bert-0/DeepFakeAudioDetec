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

    - "avg"        média global (freq e tempo) — comportamento original
    - "stats"      média + desvio no tempo (freq é mediada antes)
    - "freq_stats" preserva a frequência: fixa `freq_bins` faixas e as achata
                   nos canais antes das estatísticas temporais
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
    """Versões com atenção, usadas pelo Incremento 3.

    Cada modo espelha o equivalente sem atenção em `build_pooling`, para que a
    comparação entre os incrementos isole de fato o efeito da atenção:

    - "avg"        atenção sobre a média (frequência mediada)
    - "stats"      Attentive Statistics Pooling (frequência mediada)
    - "freq_stats" idem, mas **preservando** as faixas de frequência
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
