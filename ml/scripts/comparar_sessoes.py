"""Põe lado a lado as sessões de um teste ao vivo (limpo, controle, chamada).

Lê os scores do `evaluate.py` de cada sessão e mostra o EER e, por áudio, como
o score mudou. Com playlists pequenas o EER é grosseiro; a tabela é o principal.

Uso:
    python scripts/comparar_sessoes.py \\
        outputs/canal_real/limpo/config_canal_real.yaml \\
        outputs/canal_real/controle/config_canal_real.yaml \\
        outputs/canal_real/chamada/config_canal_real.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, output_name  # noqa: E402
from src.metrics import compute_eer  # noqa: E402
from src.scores import scores_path  # noqa: E402

OUTPUT_DIR = Path("outputs")


def carregar_sessao(config_path: str | Path, output_dir: Path = OUTPUT_DIR) -> dict:
    """Scores de uma sessão, pelo nome do experimento do config derivado."""
    config_path = Path(config_path)
    nome = output_name(load_config(config_path))
    npz = scores_path(output_dir, nome, "eval")
    if not npz.is_file():
        raise FileNotFoundError(
            f"{npz} não existe — rode antes:\n  python evaluate.py --config "
            f"{config_path.as_posix()} --checkpoint <checkpoint> --partition eval")
    dados = np.load(npz, allow_pickle=False)
    limiar = None
    metricas = output_dir / f"{nome}_eval_metrics.json"
    if metricas.is_file():
        limiar = json.loads(metricas.read_text(encoding="utf-8")).get("threshold")
    logodds = dados["logodds"] if "logodds" in dados.files else dados["scores"]
    return {
        "sessao": config_path.parent.name,
        "ids": [str(i) for i in dados["ids"]],
        "labels": dados["labels"],
        "scores": dados["scores"],
        "logodds": logodds,
        "systems": [str(s) for s in dados["systems"]],
        "limiar": limiar,
    }


def resumir(sessao: dict, ids: list[str]) -> dict:
    """EER e acertos no limiar, restritos aos áudios comuns a todas as sessões."""
    pos = {i: k for k, i in enumerate(sessao["ids"])}
    idx = [pos[i] for i in ids]
    labels = sessao["labels"][idx]
    scores = sessao["scores"][idx]
    out = {"n": len(idx), "eer": compute_eer(labels, sessao["logodds"][idx])}
    if sessao["limiar"] is not None:
        preds = (scores >= sessao["limiar"]).astype(int)
        out["acertos"] = int((preds == labels).sum())
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compara sessões do teste ao vivo")
    p.add_argument("configs", nargs="+", help="config_canal_real.yaml de cada sessão")
    p.add_argument("--csv", default=None, help="grava a tabela por arquivo neste CSV")
    args = p.parse_args(argv)

    try:
        sessoes = [carregar_sessao(c, OUTPUT_DIR) for c in args.configs]
    except FileNotFoundError as erro:
        print(f"[ERRO] {erro}")
        return 1

    # Só áudios presentes em todas as sessões, para isolar o efeito do canal.
    comuns = set(sessoes[0]["ids"])
    for s in sessoes[1:]:
        comuns &= set(s["ids"])
    ids = [i for i in sessoes[0]["ids"] if i in comuns]
    if not ids:
        print("[ERRO] nenhum áudio em comum entre as sessões.")
        return 1
    ref = sessoes[0]
    pos_ref = {i: k for k, i in enumerate(ref["ids"])}

    print(f"Áudios em comum: {len(ids)}\n")
    print(f"{'sessão':12s} {'EER':>8s} {'acertos no limiar':>20s}")
    for s in sessoes:
        r = resumir(s, ids)
        acertos = (f"{r['acertos']}/{r['n']}" if "acertos" in r else "—")
        print(f"{s['sessao']:12s} {r['eer'] * 100:7.2f}% {acertos:>20s}")
    n_bona = int(sum(ref["labels"][pos_ref[i]] == 0 for i in ids))
    print(f"\nCom {n_bona} bonafide, cada bonafide mal ordenado move o EER em "
          f"~{50 / max(n_bona, 1):.1f} pp. Leia a tabela abaixo, não só o EER.\n")

    cab = ["id", "rotulo", "ataque"] + [s["sessao"] for s in sessoes]
    print(f"{'id':14s} {'rótulo':9s} {'ataque':6s} "
          + " ".join(f"{s['sessao'][:9]:>9s}" for s in sessoes))
    linhas = []
    for i in sorted(ids, key=lambda x: (ref["labels"][pos_ref[x]], x)):
        k = pos_ref[i]
        rotulo = "spoof" if ref["labels"][k] else "bonafide"
        valores = []
        for s in sessoes:
            j = s["ids"].index(i)
            valores.append(float(s["scores"][j]))
        linhas.append([i, rotulo, ref["systems"][k]] + [f"{v:.4f}" for v in valores])
        print(f"{i:14s} {rotulo:9s} {ref['systems'][k]:6s} "
              + " ".join(f"{v:9.3f}" for v in valores))

    if args.csv:
        destino = Path(args.csv)
        destino.parent.mkdir(parents=True, exist_ok=True)
        with open(destino, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cab)
            w.writerows(linhas)
        print(f"\nTabela por arquivo: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
