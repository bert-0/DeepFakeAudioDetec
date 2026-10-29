"""Amostra do In-the-Wild (Müller et al., Interspeech 2022): falsos e reais tirados da internet.

É a régua das vozes de IA que circulam na internet (RESUMO_TCC §10.2.47 e §10.2.48):
58 figuras públicas, em inglês, com falsos de ferramentas que nenhuma outra régua tem.
O `meta.csv` da base traz file, speaker e label (bona-fide ou spoof).

Seleção: em cada pessoa, até `--por-classe` reais e o mesmo número de falsos, com
semente fixa. A lista sai em `data/recortes/inthewild_<n>_s<semente>.tsv` (arquivo,
classe, rótulo, locutor), no formato que o `estudo_ssl.py extrair --lista` lê. Os
áudios não são copiados.

Uso (de dentro de ml/):
    python scripts/montar_amostra_inthewild.py --pasta data/InTheWild --por-classe 50
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

COLUNAS = ("arquivo", "classe", "rotulo", "locutor")


def ler_meta(pasta: Path) -> list[dict]:
    with open(pasta / "meta.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def selecionar(linhas: list[dict], por_classe: int, semente: int = 42) -> list[dict]:
    rng = random.Random(semente)
    grupos: dict[tuple[str, str], list[dict]] = {}
    for r in linhas:
        grupos.setdefault((r["speaker"], r["label"]), []).append(r)
    saida = []
    for chave in sorted(grupos):
        lista = sorted(grupos[chave], key=lambda r: r["file"])
        saida.extend(rng.sample(lista, min(por_classe, len(lista))))
    return saida


def linha(pasta: Path, r: dict) -> list[str]:
    real = r["label"] == "bona-fide"
    return [str(pasta / r["file"]).replace("\\", "/"), "bonafide" if real else "spoof",
            "0" if real else "1", r["speaker"]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Amostra do In-the-Wild por pessoa")
    p.add_argument("--pasta", required=True, help="a pasta com o meta.csv e os .wav")
    p.add_argument("--por-classe", type=int, default=50)
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--lista", default=None)
    args = p.parse_args(argv)

    pasta = Path(args.pasta)
    linhas = ler_meta(pasta)
    rotulos = {r["label"] for r in linhas}
    if not rotulos <= {"bona-fide", "spoof"}:
        p.error(f"rótulos inesperados no meta.csv: {sorted(rotulos)}")
    sel = selecionar(linhas, args.por_classe, args.semente)
    lista = Path(args.lista or f"data/recortes/inthewild_{args.por_classe}_s{args.semente}.tsv")
    lista.parent.mkdir(parents=True, exist_ok=True)
    with open(lista, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(COLUNAS) + "\n")
        f.writelines("\t".join(linha(pasta, r)) + "\n" for r in sel)
    n_real = sum(r["label"] == "bona-fide" for r in sel)
    print(f"{len(sel)} áudios ({n_real} reais, {len(sel) - n_real} falsos) de "
          f"{len({r['speaker'] for r in sel})} pessoas -> {lista}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
