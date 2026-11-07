"""Fábrica de encoders: `cnn` (baseline do TC1 §4.7) ou `lcnn` (Light CNN com MFM)."""

from __future__ import annotations

import torch.nn as nn

from .blocks import CNNEncoder
from .lcnn import LCNNEncoder


def build_encoder(
    name: str = "cnn",
    in_ch: int = 1,
    channels: tuple[int, ...] = (16, 32, 64),
) -> nn.Module:
    """Instancia o encoder pedido. `channels` só se aplica ao encoder `cnn`."""
    if name == "cnn":
        return CNNEncoder(in_ch=in_ch, channels=tuple(channels))
    if name == "lcnn":
        return LCNNEncoder(in_ch=in_ch)
    raise ValueError(f"encoder desconhecido: {name!r} (use 'cnn' ou 'lcnn')")
