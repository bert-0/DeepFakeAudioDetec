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


def output_name(config: dict, smoke: bool = False) -> str:
    """Nome-base dos artefatos de um experimento (checkpoints, JSONs, gráficos).

    Execuções `--smoke` recebem o sufixo `_smoke`. Sem isso, um teste rápido de
    30 segundos com áudio sintético sobrescreve o checkpoint e os resultados de
    um treino real de horas — e como `checkpoints/` e `outputs/` estão no
    `.gitignore`, a perda é irrecuperável.
    """
    nome = config["experiment"]["name"]
    return f"{nome}_smoke" if smoke else nome


def make_generator(seed: int) -> torch.Generator:
    """Gerador semeado para o embaralhamento reprodutível do DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - assinatura exigida pelo DataLoader
    """Prepara cada worker do DataLoader: semente + limite de threads.

    **Semente** — reprodutibilidade com `num_workers > 0`.

    **Threads** — cada worker é um processo separado, e o OpenBLAS/OpenMP abre
    por padrão uma thread por núcleo *em cada um deles*. Com 4 workers numa
    máquina de 4 núcleos são 16 threads disputando 4 núcleos: o tempo se perde
    em espera ativa, não em cálculo. As matrizes aqui (banco de filtros × STFT)
    são pequenas demais para compensar a paralelização interna.

    Medido neste projeto: **3,3× mais rápido** no estágio de dados
    (14,1 s → 4,3 s para 512 áudios; 36 → 120 amostras/s), sem qualquer
    alteração numérica.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    _limit_worker_threads()


def _limit_worker_threads() -> None:
    """Restringe as bibliotecas numéricas a uma thread dentro do worker."""
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:
        # Sem threadpoolctl, as variáveis de ambiente só valem se definidas
        # antes do import do numpy — então aqui resta limitar o próprio torch.
        pass
    torch.set_num_threads(1)
