"""Persistência dos scores de uma partição, para não refazer a mesma inferência.

`run_pipeline.py` roda, em sequência, `evaluate.py` e depois `per_attack_eval.py`
sobre o **mesmo modelo** e a **mesma partição**. Os dois faziam a passada inteira
de inferência: no `eval` do ASVspoof LA são 71.237 áudios percorridos duas vezes
por experimento, para chegar exatamente aos mesmos scores. `score_fusion.py`
repetia a passada de novo, uma por modelo combinado.

Aqui os scores são gravados uma vez e relidos pelas análises seguintes. As
métricas não mudam: são os mesmos números, calculados uma vez só.

**Reusar score errado seria pior que recalcular.** Por isso o arquivo guarda,
além dos scores, a identidade do que os produziu — a lista de ids na ordem, a
partição e uma impressão digital dos *pesos* do checkpoint. Se qualquer um dos
três não bater (o modelo foi retreinado, o protocolo mudou), `load_scores`
devolve `None` e quem chamou recalcula.

O arquivo `.txt` no estilo ASVspoof continua sendo gerado pelo `evaluate.py`:
ele é o artefato legível/auditável (e a entrada do script oficial de t-DCF).
Este `.npz` é o formato interno, em precisão total — o `.txt` arredonda para
seis casas, o que criaria empates artificiais no cálculo do EER.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

VERSION = 1


def checkpoint_fingerprint(state_dict: dict) -> str:
    """Hash dos pesos do modelo — muda sempre que o modelo é retreinado.

    Usa os pesos, e não a data do arquivo: copiar ou restaurar um checkpoint não
    deve invalidar scores que continuam corretos, e um retreino que por acaso
    preserve o mtime não pode passar batido.
    """
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
                fingerprint: str, partition: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=VERSION,
        ids=np.asarray([str(i) for i in ids]),
        labels=np.asarray(labels),
        scores=np.asarray(scores, dtype=np.float64),
        systems=np.asarray([str(s) for s in systems]),
        fingerprint=fingerprint,
        partition=partition,
    )


def load_scores(path: str | Path, *, ids, fingerprint: str, partition: str):
    """Relê os scores se ainda descreverem este modelo nesta partição.

    Devolve `(labels, scores, systems)` ou `None`, com o motivo da recusa. O
    motivo é devolvido em vez de impresso porque quem chama sabe se aquilo é uma
    informação útil ou ruído.
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
    return (dados["labels"], dados["scores"], dados["systems"]), "reaproveitados"
