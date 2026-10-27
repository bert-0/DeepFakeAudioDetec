"""Amostra do ReplayDF (Interspeech 2025): real e falso tocados por alto-falante e regravados.

É o caminho "pelo ar" da demonstração, medido por outros em 110 montagens de
alto-falante e microfone, com as duas classes pelo mesmo canal. Cada montagem
tem um `meta.csv` (separado por `|`): original_file, recorded_file, label
(bona-fide ou spoof), architecture (bark, vits, xtts_v1.1, xtts_v2.0), language,
mic, speaker e uid.

Seleção: em cada montagem, até `--por-montagem` reais e o mesmo número de falsos,
os falsos repartidos entre as ferramentas; semente fixa. A lista sai em
`data/recortes/replaydf_<n>_s<semente>.tsv` (arquivo, classe, rótulo, montagem,
microfone, alto-falante e língua), no formato que o `estudo_ssl.py extrair --lista`
lê. Os áudios não são copiados.

Uso (de dentro de ml/):
    python scripts/montar_amostra_replaydf.py --pasta data/ReplayDF --por-montagem 10
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

COLUNAS = ("arquivo", "classe", "rotulo", "montagem", "microfone", "alto_falante", "lingua")


def ler_meta(pasta: Path) -> list[dict]:
    linhas = []
    for meta in sorted(pasta.glob("wav/*/meta.csv")):
        with open(meta, encoding="utf-8", newline="") as f:
            linhas.extend(csv.DictReader(f, delimiter="|"))
    return linhas


def selecionar(linhas: list[dict], por_montagem: int, semente: int = 42) -> list[dict]:
    rng = random.Random(semente)
    por_uid: dict[str, list[dict]] = {}
    for r in linhas:
        por_uid.setdefault(r["uid"], []).append(r)
    saida = []
    for uid in sorted(por_uid):
        reais = sorted((r for r in por_uid[uid] if r["label"] == "bona-fide"), key=lambda r: r["recorded_file"])
        falsos: dict[str, list[dict]] = {}
        for r in por_uid[uid]:
            if r["label"] == "spoof":
                falsos.setdefault(r["architecture"], []).append(r)
        saida.extend(rng.sample(reais, min(por_montagem, len(reais))))
        # Falsos repartidos entre as ferramentas da montagem, em rodízio.
        filas = {a: rng.sample(sorted(v, key=lambda r: r["recorded_file"]), len(v)) for a, v in sorted(falsos.items())}
        escolhidos = []
        while len(escolhidos) < por_montagem and any(filas.values()):
            for a in sorted(filas):
                if filas[a] and len(escolhidos) < por_montagem:
                    escolhidos.append(filas[a].pop())
        saida.extend(escolhidos)
    return saida


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Amostra do ReplayDF por montagem")
    p.add_argument("--pasta", required=True)
    p.add_argument("--por-montagem", type=int, default=10)
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--lista", default=None)
    args = p.parse_args(argv)

    pasta = Path(args.pasta)
    sel = selecionar(ler_meta(pasta), args.por_montagem, args.semente)
    if not sel:
        p.error(f"nenhum meta.csv em {pasta}/wav/*/")
    lista = Path(args.lista or f"data/recortes/replaydf_{args.por_montagem}_s{args.semente}.tsv")
    lista.parent.mkdir(parents=True, exist_ok=True)
    with open(lista, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(COLUNAS) + "\n")
        for r in sel:
            real = r["label"] == "bona-fide"
            f.write("\t".join([str(pasta / r["recorded_file"]).replace("\\", "/"),
                               "bonafide" if real else r["architecture"], "0" if real else "1", r["uid"],
                               r["mic"], r["speaker"], r["language"]]) + "\n")
    n_real = sum(r["label"] == "bona-fide" for r in sel)
    print(f"{len(sel)} áudios ({n_real} reais, {len(sel) - n_real} falsos) de "
          f"{len({r['uid'] for r in sel})} montagens -> {lista}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
