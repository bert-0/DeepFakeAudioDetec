"""Amostra do In-the-Wild (Müller et al., Interspeech 2022): falsos e reais tirados da internet.

É a régua das vozes de IA que circulam na internet (RESUMO_TCC §10.2.47 e §10.2.48):
58 figuras públicas, em inglês, com falsos de ferramentas que nenhuma outra régua tem.
O `meta.csv` da base traz file, speaker e label (bona-fide ou spoof).

Seleção: em cada pessoa, até `--por-classe` reais e o mesmo número de falsos, com
semente fixa. A lista sai em `data/recortes/inthewild_<n>_s<semente>.tsv` (arquivo,
classe, rótulo, locutor), no formato que o `estudo_ssl.py extrair --lista` lê. Os
áudios não são copiados.

Divisão por pessoa (RESUMO_TCC §10.2.49): `--dividir TREINO DEV` sorteia as pessoas
da amostra já medida em treino, dev e régua (o resto), grava a lista em
`data/recortes/inthewild_pessoas_s<semente>.tsv` e, com `--npz`, separa as
características já extraídas em `<npz>_treino`, `_dev` e `_regua`, sem extrair de novo.
Uma pessoa fica inteira numa parte só: o treino não vê a voz de quem está na régua.

Uso (de dentro de ml/):
    python scripts/montar_amostra_inthewild.py --pasta data/InTheWild --por-classe 50
    python scripts/montar_amostra_inthewild.py --dividir 30 8 --npz outputs/ssl/inthewild.npz
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


def dividir_pessoas(pessoas, n_treino: int, n_dev: int, semente: int = 42) -> dict[str, list[str]]:
    u = sorted(set(pessoas))
    if n_treino + n_dev >= len(u):
        raise ValueError(f"{len(u)} pessoas não dão {n_treino} no treino, {n_dev} no dev e uma régua")
    random.Random(semente).shuffle(u)
    return {"treino": sorted(u[:n_treino]), "dev": sorted(u[n_treino:n_treino + n_dev]),
            "regua": sorted(u[n_treino + n_dev:])}


def _nome(caminho: str) -> str:
    return caminho.replace("\\", "/").rsplit("/", 1)[-1]


def separar_npz(npz: Path, pessoa_do_arquivo: dict[str, str], partes: dict[str, list[str]]) -> dict[str, Path]:
    """Um .npz por parte, com as mesmas chaves; arquivo sem pessoa na lista é erro."""
    import numpy as np

    with np.load(npz, allow_pickle=False) as d:
        dados = {k: d[k] for k in d.files}
    n = len(dados["arquivo"])
    pessoa = np.array([pessoa_do_arquivo[_nome(str(a))] for a in dados["arquivo"]])
    saida = {}
    for parte, pessoas in partes.items():
        m = np.isin(pessoa, pessoas)
        destino = npz.with_name(f"{npz.stem}_{parte}.npz")
        np.savez_compressed(destino, **{k: (v[m] if v.ndim and len(v) == n else v) for k, v in dados.items()})
        saida[parte] = destino
    return saida


def dividir(args) -> int:
    with open(args.amostra, encoding="utf-8", newline="") as f:
        amostra = list(csv.DictReader(f, delimiter="\t"))
    partes = dividir_pessoas([r["locutor"] for r in amostra], *args.dividir, semente=args.semente)
    lista = Path(args.lista_pessoas or f"data/recortes/inthewild_pessoas_s{args.semente}.tsv")
    lista.parent.mkdir(parents=True, exist_ok=True)
    with open(lista, "w", encoding="utf-8", newline="") as f:
        f.write("pessoa\tparte\n")
        f.writelines(f"{u}\t{k}\n" for k, us in partes.items() for u in us)
    contagem = {k: sum(r["locutor"] in set(v) for r in amostra) for k, v in partes.items()}
    print("Pessoas: " + ", ".join(f"{k} {len(v)} ({contagem[k]} áudios)" for k, v in partes.items())
          + f" -> {lista}")
    if args.npz:
        pessoa_do_arquivo = {_nome(r["arquivo"]): r["locutor"] for r in amostra}
        for parte, destino in separar_npz(Path(args.npz), pessoa_do_arquivo, partes).items():
            print(f"  {parte}: {destino}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Amostra do In-the-Wild por pessoa")
    p.add_argument("--pasta", help="a pasta com o meta.csv e os .wav")
    p.add_argument("--por-classe", type=int, default=50)
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--lista", default=None)
    p.add_argument("--dividir", nargs=2, type=int, metavar=("TREINO", "DEV"), default=None)
    p.add_argument("--amostra", default="data/recortes/inthewild_50_s42.tsv",
                   help="com --dividir: a lista da amostra já medida (§10.2.48)")
    p.add_argument("--npz", help="com --dividir: as características dessa amostra")
    p.add_argument("--lista-pessoas", default=None, help="com --dividir: onde gravar a divisão")
    args = p.parse_args(argv)
    if args.dividir:
        return dividir(args)
    if not args.pasta:
        p.error("diga a pasta da base (--pasta) ou use --dividir")

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
