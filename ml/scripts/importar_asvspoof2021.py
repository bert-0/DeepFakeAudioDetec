"""Converte o metadado do ASVspoof 2021 no protocolo que este projeto lê.

O ASVspoof 2021 LA transmite os **mesmos ataques A07–A19** do eval de 2019 por
redes reais (VoIP e PSTN) com codecs reais, e traz uma condição de referência
**sem codec e sem transmissão**. Isso dá o experimento pareado que a simulação
da Seção 5 só consegue aproximar: mesmo ataque, mesmo áudio, canal real.

Uso:
    # 1. ver quais condições existem no metadado
    python scripts/importar_asvspoof2021.py --metadata keys/LA/CM/trial_metadata.txt --listar

    # 2. gerar um protocolo por condição
    python scripts/importar_asvspoof2021.py --metadata keys/LA/CM/trial_metadata.txt \\
        --condicao nocodec/nocodec --saida outputs/2021/protocolo_limpo.txt
    python scripts/importar_asvspoof2021.py --metadata keys/LA/CM/trial_metadata.txt \\
        --codec opus --saida outputs/2021/protocolo_opus.txt

    # 3. avaliar cada um com o config apontado para o áudio do 2021
    python evaluate.py --config configs/asvspoof2021.yaml --checkpoint ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.asvspoof2021 import (  # noqa: E402
    MetadadoInvalido,
    condicoes,
    escrever_protocolo,
    filtrar,
    ler_metadata,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Importa o ASVspoof 2021")
    p.add_argument("--metadata", required=True,
                   help="trial_metadata.txt do eval-package (LA/CM/)")
    p.add_argument("--listar", action="store_true",
                   help="só lista as condições encontradas e sai")
    p.add_argument("--condicao", default=None, help="condição exata `codec/canal`")
    p.add_argument("--codec", default=None, help="filtra só pelo codec")
    p.add_argument("--saida", default=None, help="protocolo de saída")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        trials = ler_metadata(args.metadata)
    except (OSError, MetadadoInvalido) as erro:
        print(f"[ERRO] {erro}")
        return 1

    ataques = sorted({t.ataque for t in trials if t.ataque != "-"})
    print(f"{len(trials):,} trials | {len(ataques)} ataques: {', '.join(ataques)}")

    if args.listar or not args.saida:
        print(f"\n{'condição (codec/canal)':32s} {'bonafide':>10s} {'spoof':>10s} "
              f"{'total':>10s}")
        print("-" * 66)
        for nome, bona, spoof in condicoes(trials):
            print(f"{nome:32s} {bona:10,} {spoof:10,} {bona + spoof:10,}")
        print("\nA condição de referência (sem codec e sem transmissão) é a que "
              "reproduz\no cenário do eval de 2019 — use-a como controle pareado.")
        if not args.saida:
            print("\nInforme --saida para gerar um protocolo.")
        return 0

    selecao = filtrar(trials, condicao=args.condicao, codec=args.codec)
    if not selecao:
        print("[ERRO] nenhum trial casa com o filtro pedido. Use --listar.")
        return 1

    bona = sum(1 for t in selecao if t.chave == "bonafide")
    if bona == 0 or bona == len(selecao):
        print("[ERRO] a seleção tem uma classe só; não dá para calcular EER.")
        return 1

    n = escrever_protocolo(selecao, args.saida)
    print(f"\n{n:,} linhas em {args.saida} "
          f"({bona:,} bonafide, {n - bona:,} spoof)")
    print("Aponte `data.protocols.eval` e `data.audio_dir.eval` do config para "
          "este arquivo\ne para a pasta de .flac do 2021, e rode o evaluate.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
