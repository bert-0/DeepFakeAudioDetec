"""Carga de configuração (YAML) e utilidades de reprodutibilidade/dispositivo."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Lê um arquivo YAML de configuração e devolve um dicionário."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int) -> None:
    """Fixa as sementes para tornar os experimentos reprodutíveis."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "cuda") -> torch.device:
    """Devolve o dispositivo pedido, caindo para CPU se não houver GPU."""
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def make_generator(seed: int) -> torch.Generator:
    """Gerador semeado para o embaralhamento reprodutível do DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - assinatura exigida pelo DataLoader
    """Semeia cada worker do DataLoader (reprodutibilidade com num_workers > 0)."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
