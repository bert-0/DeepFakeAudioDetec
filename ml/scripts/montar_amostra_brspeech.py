"""Amostra do BRSpeech-DF (EMNLP 2025): deepfakes em português, para régua e treino.

A base vem em Parquet (áudio FLAC dentro da coluna `audio`). Aqui uma parte dela
vira arquivos, como as outras bases: todas as vozes reais da parte pedida e até
`--por-modelo` clones de cada ferramenta (XTTS, F5-TTS, Fish-Speech, YourTTS e
ToucanTTS), com semente fixa. As partes da base têm leitores diferentes (10 no
teste, 10 na validação), e os clones de cada parte imitam os leitores dela.

Saídas:
- `<saida>/<parte>/bonafide/<arquivo>.flac` e `<saida>/<parte>/<modelo>/<n>.flac`
  (os clones não têm nome na base; ganham o número da linha);
- `data/recortes/brspeech_df_<parte>_<por-modelo>_s<semente>.tsv`: a seleção.

Uso (de dentro de ml/):
    python scripts/montar_amostra_brspeech.py --pasta D:/BRSpeech-DF --parte test \\
        --saida D:/BRSpeech-DF/amostra
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


def _linhas(arquivos: list[Path], colunas: list[str]):
    """(arquivo, índice global, linha) de cada linha, em ordem fixa."""
    import pyarrow.parquet as pq

    k = 0
    for a in arquivos:
        f = pq.ParquetFile(a)
        for g in range(f.num_row_groups):
            for linha in f.read_row_group(g, columns=colunas).to_pylist():
                yield a, k, linha
                k += 1


def _nome_seguro(nome: str) -> str:
    return "".join(c for c in Path(nome).name if c.isalnum() or c in "._-")


def selecionar(pasta: Path, parte: str, por_modelo: int, semente: int = 42,
               max_reais: int | None = None) -> dict:
    """{"bonafide": [índices], modelo: [índices]} das linhas dos Parquet da parte."""
    spoof = sorted((pasta / "spoof").glob(f"{parte}-*.parquet"))
    por: dict[str, list[int]] = {}
    for _, k, linha in _linhas(spoof, ["model"]):
        por.setdefault(linha["model"], []).append(k)
    rng = random.Random(semente)
    escolha = {m: sorted(rng.sample(idx, min(por_modelo, len(idx)))) for m, idx in sorted(por.items())}
    reais = sorted((pasta / "bonafide").glob(f"{parte}-*.parquet"))
    todas = [k for _, k, _ in _linhas(reais, ["label"])]
    # No treino baixado, um leitor tem 6.665 das 8.020 reais: o teto evita que ele domine.
    escolha["bonafide"] = todas if max_reais is None else sorted(rng.sample(todas, min(max_reais, len(todas))))
    return escolha


def exportar(pasta: Path, parte: str, escolha: dict, saida: Path) -> list[tuple[str, str]]:
    """Grava os FLAC escolhidos; devolve (caminho relativo, classe)."""
    gravados = []
    for classe_pasta, colunas in (("bonafide", ["audio"]), ("spoof", ["audio", "model"])):
        arquivos = sorted((pasta / classe_pasta).glob(f"{parte}-*.parquet"))
        alvos = ({k: "bonafide" for k in escolha["bonafide"]} if classe_pasta == "bonafide" else
                 {k: m for m, idx in escolha.items() if m != "bonafide" for k in idx})
        for _, k, linha in _linhas(arquivos, colunas):
            classe = alvos.get(k)
            if classe is None:
                continue
            nome = _nome_seguro(linha["audio"].get("path") or "") if classe == "bonafide" else f"{k:06d}.flac"
            destino = saida / parte / classe / (nome or f"{k:06d}.flac")
            if not destino.exists():
                destino.parent.mkdir(parents=True, exist_ok=True)
                destino.write_bytes(linha["audio"]["bytes"])
            gravados.append((f"{parte}/{classe}/{destino.name}", classe))
    return gravados


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Amostra do BRSpeech-DF em arquivos")
    p.add_argument("--pasta", required=True, help="a pasta baixada, com bonafide/ e spoof/")
    p.add_argument("--parte", default="test", choices=("train", "validation", "test"))
    p.add_argument("--por-modelo", type=int, default=300)
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--max-reais", type=int, default=None, help="padrão: todas as reais da parte")
    p.add_argument("--saida", required=True)
    p.add_argument("--lista", default=None)
    args = p.parse_args(argv)

    pasta = Path(args.pasta)
    escolha = selecionar(pasta, args.parte, args.por_modelo, args.semente, args.max_reais)
    gravados = exportar(pasta, args.parte, escolha, Path(args.saida))
    lista = Path(args.lista or f"data/recortes/brspeech_df_{args.parte}_{args.por_modelo}_s{args.semente}.tsv")
    lista.parent.mkdir(parents=True, exist_ok=True)
    with open(lista, "w", encoding="utf-8", newline="") as f:
        f.write("arquivo\tclasse\n")
        f.writelines(f"{a}\t{c}\n" for a, c in sorted(gravados))
    contagem = {c: sum(1 for _, x in gravados if x == c) for c in sorted({c for _, c in gravados})}
    print(f"{len(gravados)} arquivos em {args.saida}: {contagem}; lista em {lista}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
