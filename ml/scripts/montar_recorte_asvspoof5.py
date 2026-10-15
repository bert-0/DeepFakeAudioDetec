"""Monta o recorte da ASVspoof 5 lendo os FLAC direto dos `.tar`, sem extrair.

Seleção (`src.data.asvspoof5.recorte`): todos os bonafide e até N spoof por
ataque, com semente fixa. Cada áudio vira um FLAC de 16 bits com no máximo 6 s,
começando 0,5 s antes da primeira fala. Saídas:

- `<saida>/flac/<id>.flac` e `<saida>/protocolo.txt`, só com os áudios que já
  estão na pasta, no formato de 2019: entram no config do treino como
  `protocols.<partição>` e `audio_dir.<partição>`;
- `data/recortes/asvspoof5_<partição>_<N>_s<semente>.txt`: a seleção inteira, no
  mesmo formato, para versionar.

Pode ser interrompido e retomado: o que já foi escrito é pulado. Com só parte dos
`.tar` baixados, rode com eles e depois de novo com os outros.

Uso (de dentro de ml/):
    python scripts/montar_recorte_asvspoof5.py \\
        --protocolo data/ASVspoof5/ASVspoof5.train.tsv \\
        --tars "data/ASVspoof5/flac_T_a*.tar" \\
        --saida <pasta no NVMe>/ASVspoof5_recorte/train --md5
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.asvspoof5 import (  # noqa: E402
    ProtocoloInvalido,
    conferir_arquivo,
    conferir_tamanho,
    cortar_trecho,
    escrever_protocolo,
    ler_protocolo,
    ler_tar,
    recorte,
    resumo,
)

TAXA = 16000
#: Do plano: 19 mil humanos e cerca de 5 mil falsos por ataque no treino.
FALSOS_POR_ATAQUE = 5000


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monta o recorte da ASVspoof 5 a partir dos .tar")
    p.add_argument("--protocolo", required=True, help="ASVspoof5.train.tsv ou .dev.track_1.tsv")
    p.add_argument("--tars", nargs="+", required=True,
                   help="os .tar da partição; aceita padrões como \"flac_T_a*.tar\"")
    p.add_argument("--saida", required=True, help="pasta do recorte (de preferência no NVMe)")
    p.add_argument("--falsos-por-ataque", type=int, default=FALSOS_POR_ATAQUE)
    p.add_argument("--segundos", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--md5", action="store_true",
                   help="confere o MD5 de cada .tar antes de ler (lê o arquivo inteiro)")
    p.add_argument("--lista", default=None,
                   help="onde gravar a seleção versionada (padrão: data/recortes/...)")
    return p.parse_args(argv)


def _expandir(padroes: list[str]) -> list[Path]:
    # O PowerShell não expande curingas para programas externos.
    caminhos = []
    for padrao in padroes:
        achados = sorted(glob.glob(padrao)) or [padrao]
        caminhos.extend(Path(a) for a in achados)
    return caminhos


def _gravar(destino: Path, trecho) -> None:
    import soundfile as sf

    temporario = destino.with_name(destino.name + ".parcial")
    sf.write(temporario, trecho, TAXA, subtype="PCM_16", format="FLAC")
    os.replace(temporario, destino)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        enunciados = ler_protocolo(args.protocolo)
    except ProtocoloInvalido as erro:
        print(f"[ERRO] {erro}")
        return 1
    aviso = conferir_tamanho(args.protocolo, enunciados)
    if aviso:
        print(f"[ERRO] {aviso}")
        return 1

    selecao = recorte(enunciados, args.falsos_por_ataque, args.seed)
    particao = selecao[0].particao
    contagem = resumo(selecao)
    print(f"Seleção ({particao}): {len(selecao)} áudios — {contagem['chave']}")
    print("  por ataque: " + ", ".join(f"{a} {n}" for a, n in contagem["ataque"].items()))

    lista = Path(args.lista or f"data/recortes/asvspoof5_{particao}_"
                               f"{args.falsos_por_ataque}_s{args.seed}.txt")
    escrever_protocolo(selecao, lista)

    pasta = Path(args.saida) / "flac"
    pasta.mkdir(parents=True, exist_ok=True)
    faltam = {e.arquivo for e in selecao} - {p.stem for p in pasta.glob("*.flac")}
    ja_havia = len(selecao) - len(faltam)
    escritos, inicio = 0, time.time()
    for tar in _expandir(args.tars):
        if not faltam:
            break
        if not tar.is_file():
            print(f"[AVISO] {tar} não existe; pulando")
            continue
        problema = conferir_arquivo(tar, md5=args.md5)
        if problema:
            print(f"[AVISO] {problema} Pulando {tar.name}.")
            continue
        print(f"Lendo {tar.name}...")
        for nome, wav, taxa in ler_tar(tar, faltam):
            if taxa != TAXA:
                from src.preprocess.reamostragem import para_taxa

                wav = para_taxa(wav, taxa, TAXA)
            _gravar(pasta / f"{nome}.flac", cortar_trecho(wav, TAXA, args.segundos))
            faltam.discard(nome)
            escritos += 1
            if escritos % 2000 == 0:
                print(f"  {escritos} escritos ({escritos / (time.time() - inicio):.0f}/s), "
                      f"faltam {len(faltam)}")

    presentes = [e for e in selecao if e.arquivo not in faltam]
    escrever_protocolo(presentes, Path(args.saida) / "protocolo.txt")
    print(f"\nEscritos agora: {escritos}; já existiam: {ja_havia}; faltam: {len(faltam)}")
    if faltam:
        print("  Os que faltam estão em .tar que não foram lidos (ou não baixados): rode de "
              "novo com eles; o que já está na pasta é pulado.")
    print(f"Seleção versionada: {lista}")
    print(f"No config: protocols.{particao} = {Path(args.saida) / 'protocolo.txt'}")
    print(f"           audio_dir.{particao} = {pasta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
