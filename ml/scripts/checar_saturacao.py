"""Conta quantos scores saturaram em exatamente 0 ou 1 nos arquivos de scores.

O score salvo é o softmax em float32. Quando a diferença entre as duas saídas da
rede passa de ~17, ele arredonda para EXATAMENTE 1.0 — e a ordem entre esses
áudios se perde. O `score_fusion.py` já registrava ~60 mil das 71 mil amostras
do eval de 2019 nessa situação.

Isso importa para o EER, que depende só da ordem dos scores: se muitos bonafide
e muitos spoof empatam em 1.0, o limiar não consegue separá-los, e o EER mede o
arredondamento em vez do modelo. O sintoma foi visto no ASVspoof 2021 LA com
Opus: quatro ataques com EER idêntico (48,65%), com números de amostras
diferentes — o EER passou a depender só dos bonafide.

Uso:
    python scripts/checar_saturacao.py                 # todos em outputs/
    python scripts/checar_saturacao.py outputs/x_eval_scores.npz ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def contar(scores: np.ndarray, labels: np.ndarray) -> dict[str, tuple[int, int]]:
    """Por classe: (quantos saturaram em 1.0, total). Também em 0.0."""
    saida = {}
    for classe, nome in ((0, "bonafide"), (1, "spoof")):
        s = scores[labels == classe]
        saida[nome] = (int((s >= 1.0).sum()), int((s <= 0.0).sum()), len(s))
    return saida


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    arquivos = [Path(a) for a in argv] or sorted(Path("outputs").glob("*_scores.npz"))
    if not arquivos:
        print("Nenhum arquivo *_scores.npz encontrado em outputs/.")
        return 1

    print(f"{'arquivo':58s} {'bonafide em 1.0':>18s} {'spoof em 1.0':>16s} "
          f"{'valores distintos':>18s}")
    print("-" * 114)
    for arquivo in arquivos:
        dados = np.load(arquivo)
        scores, labels = dados["scores"], dados["labels"]
        c = contar(scores, labels)
        b1, _, bn = c["bonafide"]
        s1, _, sn = c["spoof"]
        nome = arquivo.name.replace("_eval_scores.npz", "")
        print(f"{nome[:58]:58s} {b1:>7,}/{bn:<7,} {100*b1/max(bn,1):3.0f}% "
              f"{s1:>6,}/{sn:<6,} {100*s1/max(sn,1):3.0f}% {len(np.unique(scores)):>12,}")

    print("\nComo ler: se uma fração grande dos BONAFIDE e dos SPOOF estão em 1.0, os")
    print("dois grupos empatam no topo e o EER daquele arquivo mede o arredondamento,")
    print("não o modelo. Spoof em 1.0 com bonafide abaixo não é problema: é acerto.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
