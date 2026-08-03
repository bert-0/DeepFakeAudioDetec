"""Fábrica de modelos — mapeia o nome no config para a classe correspondente."""

from __future__ import annotations

import torch.nn as nn

from .attention import AttentionFusionNet
from .baseline_cnn import BaselineCNN
from .fusion import FeatureFusionNet

_MODELS = {
    "baseline_cnn": BaselineCNN,
    "fusion": FeatureFusionNet,
    "attention": AttentionFusionNet,
}


def build_model(model_cfg: dict) -> nn.Module:
    """Instancia o modelo definido em `model_cfg['name']`."""
    name = model_cfg["name"]
    if name not in _MODELS:
        raise ValueError(
            f"Modelo desconhecido: {name!r}. Opções: {sorted(_MODELS)}"
        )
    return _MODELS[name](
        n_classes=model_cfg.get("n_classes", 2),
        dropout=model_cfg.get("dropout", 0.3),
        pooling=model_cfg.get("pooling", "avg"),
    )
