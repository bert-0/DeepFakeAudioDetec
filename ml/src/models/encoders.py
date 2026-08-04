"""Fábrica de encoders convolucionais compartilhada pelos três incrementos.

- `cnn`  — CNN convencional; é o baseline exigido pelo TC1 §4.7 e a referência
           dos resultados já medidos. Mantido inalterado.
- `lcnn` — Light CNN com Max-Feature-Map, arquitetura de referência da
           literatura para detecção de spoofing com features LFCC.
"""

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
