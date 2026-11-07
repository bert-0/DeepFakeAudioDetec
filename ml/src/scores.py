"""Scores de uma partição salvos em `.npz`, para não repetir a inferência.

O arquivo guarda ids, partição e um hash dos pesos; se algo não bater, o
reuso é recusado. Guarda também os log-odds, usados no EER (a probabilidade
satura em 1,0).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from src.metrics import logodds_de_probabilidade

VERSION = 1


def checkpoint_fingerprint(state_dict: dict) -> str:
    """Hash dos pesos do modelo (não do mtime, que muda ao copiar o arquivo)."""
    h = hashlib.md5()
    for chave in sorted(state_dict):
        tensor = state_dict[chave]
        h.update(chave.encode("utf-8"))
        if torch.is_tensor(tensor):
            h.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:16]


def scores_path(output_dir: str | Path, name: str, partition: str) -> Path:
    return Path(output_dir) / f"{name}_{partition}_scores.npz"


def save_scores(path: str | Path, *, ids, labels, scores, systems,
                fingerprint: str, partition: str, logodds=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = {} if logodds is None else {"logodds": np.asarray(logodds, dtype=np.float64)}
    np.savez_compressed(
        path,
        **extras,
        version=VERSION,
        ids=np.asarray([str(i) for i in ids]),
        labels=np.asarray(labels),
        scores=np.asarray(scores, dtype=np.float64),
        systems=np.asarray([str(s) for s in systems]),
        fingerprint=fingerprint,
        partition=partition,
    )


def load_scores(path: str | Path, *, ids, fingerprint: str, partition: str):
    """Relê os scores se ainda valem para este modelo e partição.

    Devolve `((labels, scores, systems, logodds), motivo)` ou `(None, motivo)`.
    Arquivo antigo sem log-odds só é aceito se nenhum bonafide saturou.
    """
    path = Path(path)
    if not path.exists():
        return None, "nenhum arquivo de scores salvo"
    try:
        dados = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as erro:
        return None, f"arquivo ilegível ({erro})"

    faltando = {"version", "partition", "fingerprint", "ids", "labels",
                "scores", "systems"} - set(dados.files)
    if faltando or int(dados["version"]) != VERSION:
        return None, "formato antigo"
    if str(dados["partition"]) != partition:
        return None, f"foi salvo para a partição '{dados['partition']}'"
    if str(dados["fingerprint"]) != fingerprint:
        return None, "o checkpoint mudou desde que foram salvos"
    guardados = dados["ids"].tolist()
    if guardados != [str(i) for i in ids]:
        return None, "a lista de áudios do protocolo mudou"
    labels, scores = dados["labels"], dados["scores"]
    if "logodds" in dados.files:
        logodds = dados["logodds"]
    elif (scores[labels == 0] >= 1.0).any():
        return None, ("formato antigo com bonafide saturados em 1.0 — o EER "
                      "precisa dos log-odds")
    else:
        logodds = logodds_de_probabilidade(scores)
    return (labels, scores, dados["systems"], logodds), "reaproveitados"
