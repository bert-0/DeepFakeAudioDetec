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


def memoria_total_gb() -> float | None:
    """RAM física da máquina, em GB. `None` se não der para descobrir.

    Sem `psutil` (não é dependência do projeto) e sem depender do SO: no Windows
    a informação vem da API Win32, no Linux/macOS de `sysconf`. Serve para o
    treino avisar *antes* de começar quando a configuração não cabe na máquina —
    no Windows, estourar a RAM não dá `MemoryError`, dá paginação, e a máquina
    trava a ponto de exigir desligamento no botão.
    """
    import ctypes
    import os

    try:
        if os.name == "nt":
            class _Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return status.ullTotalPhys / 1e9
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (AttributeError, ValueError, OSError):
        return None


# Custo medido de um processo worker no Windows: 512 MB. O `spawn` do Windows
# recria o processo do zero, reimportando torch e librosa em cada um — no Linux,
# com `fork`, essas páginas seriam compartilhadas com o pai e o custo real seria
# uma fração disto.
RAM_POR_WORKER_GB = 0.5
RAM_PROCESSO_PRINCIPAL_GB = 1.5   # inclui o contexto CUDA
RAM_SISTEMA_GB = 3.0              # Windows + serviços, sem navegador


def estimativa_ram_gb(n_workers_treino: int, n_workers_dev: int) -> dict[str, float]:
    """Quanto o treino deve ocupar, por parcela. Ver `RAM_POR_WORKER_GB`."""
    treino = n_workers_treino * RAM_POR_WORKER_GB
    dev = n_workers_dev * RAM_POR_WORKER_GB
    return {"workers_treino": treino, "workers_dev": dev,
            "principal": RAM_PROCESSO_PRINCIPAL_GB,
            "sistema": RAM_SISTEMA_GB,
            "pico": treino + dev + RAM_PROCESSO_PRINCIPAL_GB + RAM_SISTEMA_GB}
