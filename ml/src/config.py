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
    """Nome-base dos artefatos do experimento (checkpoints, JSONs, gráficos).

    O sufixo `_smoke` impede que um teste rápido sobrescreva um treino real.
    """
    nome = config["experiment"]["name"]
    return f"{nome}_smoke" if smoke else nome


def config_derivado(base: dict, sufixo: str, protocolo_eval: str | Path,
                    audio_eval: str | Path, ext: str = ".flac") -> dict[str, Any]:
    """Config para avaliar um modelo já treinado em outro conjunto de áudio.

    O nome ganha um sufixo e o cache é desligado, para não sobrescrever os
    resultados nem o cache de features do `eval` original.
    """
    import copy

    cfg = copy.deepcopy(base)
    sufixo = "".join(c if c.isalnum() or c in "-_" else "_" for c in sufixo)
    cfg["experiment"]["name"] = f"{base['experiment']['name']}__{sufixo}"
    cfg.setdefault("data", {}).setdefault("protocols", {})["eval"] = str(protocolo_eval)
    cfg["data"].setdefault("audio_dir", {})["eval"] = str(audio_eval)
    if ext != ".flac":
        cfg["data"]["file_ext"] = ext
    cfg.setdefault("train", {})["cache_features"] = False
    return cfg


def salvar_config(cfg: dict, destino: str | Path) -> Path:
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                       encoding="utf-8")
    return destino


def make_generator(seed: int) -> torch.Generator:
    """Gerador semeado para o embaralhamento reprodutível do DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - assinatura exigida pelo DataLoader
    """Semeia o worker do DataLoader e limita suas threads a uma.

    Sem o limite, cada worker abre uma thread BLAS por núcleo e eles disputam
    a CPU; com ele o estágio de dados ficou ~3,3x mais rápido.
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
        # Aqui já é tarde para variáveis de ambiente; resta limitar o torch.
        pass
    torch.set_num_threads(1)


def memoria_total_gb() -> float | None:
    """RAM física da máquina em GB, ou `None` se não der para descobrir.

    Sem `psutil`: usa a API Win32 no Windows e `sysconf` no Linux/macOS.
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


# Medido no Windows (spawn): ~512 MB por worker. Com fork, no Linux, é bem menos.
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
