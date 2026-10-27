"""Amostra do Common Voice em português para o piloto 3 de destilação.

O aluno imita mal a camada 7 do XLS-R no áudio que passou por microfone e sala
(RESUMO_TCC §10.2.37). A destilação não usa rótulos: qualquer fala real serve,
e o Common Voice traz milhares de pessoas gravando com o próprio microfone
(notebook, celular, fone), em casa. Os clipes são lidos direto do `.tar.gz`, sem
extrair a base inteira.

Seleção: só clipes validados, com pelo menos `--min-s` segundos, sem a variante
de Portugal, no máximo `--por-locutor` por pessoa (muitas vozes e microfones),
com semente fixa. Os locutores do `dev` são outros que os do `treino`: o dev
acompanha a perda de destilação sem rótulo. Saídas:

- `<saida>/<treino|dev>/<clipe>.mp3`: só o nome do arquivo, nunca o caminho do tar;
- `data/recortes/commonvoice_pt_<treino>_<dev>_s<semente>.tsv`: a seleção, para
  versionar (nome do clipe, parte e duração; sem o identificador da pessoa).

Termos da base: não redistribuir os áudios e não tentar identificar quem fala.

Uso (de dentro de ml/):
    python scripts/montar_amostra_commonvoice.py --tar "D:/CommonVoice_pt/<arquivo>.tar.gz" \\
        --tabelas D:/CommonVoice_pt/tabelas --saida D:/CommonVoice_pt/amostra
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

PORTUGAL = "Portuguese (Portugal)"


def ler_tsv(caminho: Path) -> list[dict]:
    csv.field_size_limit(10 ** 7)
    with open(caminho, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def selecionar(validados: list[dict], duracoes_ms: dict[str, int], n_treino: int, n_dev: int,
               min_s: float = 5.0, por_locutor: int = 3, semente: int = 42) -> dict[str, list[tuple[str, float]]]:
    """{"treino": [(clipe, segundos)], "dev": [...]}, com locutores diferentes nas duas partes."""
    por_pessoa: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in validados:
        s = duracoes_ms.get(r["path"], 0) / 1000
        if s >= min_s and r.get("variant", "") != PORTUGAL:
            por_pessoa[r["client_id"]].append((r["path"], s))
    rng = random.Random(semente)
    pessoas = sorted(por_pessoa)
    rng.shuffle(pessoas)
    for p in pessoas:
        por_pessoa[p].sort()
        rng.shuffle(por_pessoa[p])
        por_pessoa[p] = por_pessoa[p][:por_locutor]
    saida: dict[str, list[tuple[str, float]]] = {"dev": [], "treino": []}
    for p in pessoas:
        parte = "dev" if len(saida["dev"]) < n_dev else "treino"
        if parte == "treino" and len(saida["treino"]) >= n_treino:
            break
        falta = (n_dev - len(saida["dev"])) if parte == "dev" else (n_treino - len(saida["treino"]))
        saida[parte].extend(por_pessoa[p][:falta])
    return saida


def extrair(tar_caminho: Path, escolhidos: dict[str, str], saida: Path) -> int:
    """Uma passada pelo tar; grava só os clipes escolhidos ({clipe: parte}), pelo nome."""
    gravados = 0
    with tarfile.open(tar_caminho, "r|gz") as tar:
        for m in tar:
            nome = Path(m.name).name
            if not m.isfile() or nome not in escolhidos:
                continue
            destino = saida / escolhidos[nome] / nome
            if not destino.exists():
                destino.parent.mkdir(parents=True, exist_ok=True)
                destino.write_bytes(tar.extractfile(m).read())
            gravados += 1
            if gravados == len(escolhidos):
                break
    return gravados


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Amostra do Common Voice em português (piloto 3)")
    p.add_argument("--tar", required=True)
    p.add_argument("--tabelas", required=True, help="pasta com validated.tsv e clip_durations.tsv")
    p.add_argument("--saida", required=True)
    p.add_argument("--treino", type=int, default=4000)
    p.add_argument("--dev", type=int, default=400)
    p.add_argument("--min-s", type=float, default=5.0)
    p.add_argument("--por-locutor", type=int, default=3)
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--lista", default=None, help="padrão: data/recortes/commonvoice_pt_<treino>_<dev>_s<semente>.tsv")
    args = p.parse_args(argv)

    tabelas = Path(args.tabelas)
    duracoes = {r["clip"]: int(r["duration[ms]"]) for r in ler_tsv(tabelas / "clip_durations.tsv")}
    sel = selecionar(ler_tsv(tabelas / "validated.tsv"), duracoes, args.treino, args.dev,
                     args.min_s, args.por_locutor, args.semente)
    lista = Path(args.lista or f"data/recortes/commonvoice_pt_{args.treino}_{args.dev}_s{args.semente}.tsv")
    lista.parent.mkdir(parents=True, exist_ok=True)
    with open(lista, "w", encoding="utf-8", newline="") as f:
        f.write("clipe\tparte\tsegundos\n")
        for parte in ("treino", "dev"):
            for clipe, s in sel[parte]:
                f.write(f"{clipe}\t{parte}\t{s:.2f}\n")
    print(f"Seleção: {len(sel['treino'])} de treino, {len(sel['dev'])} de dev -> {lista}", flush=True)
    escolhidos = {c: parte for parte in ("treino", "dev") for c, _ in sel[parte]}
    n = extrair(Path(args.tar), escolhidos, Path(args.saida))
    print(f"Clipes em {args.saida}: {n} de {len(escolhidos)}")
    return 0 if n == len(escolhidos) else 1


if __name__ == "__main__":
    sys.exit(main())
