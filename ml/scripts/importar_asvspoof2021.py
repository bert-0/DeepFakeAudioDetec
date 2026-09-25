"""Converte o metadado do ASVspoof 2021 no protocolo — e no config — deste projeto.

O ASVspoof 2021 LA transmite os **mesmos ataques A07–A19** do eval de 2019 por
redes reais (VoIP e PSTN) com codecs reais, e traz uma condição de referência
sem codec e sem transmissão. Isso dá o experimento pareado que a simulação da
Seção 5 só consegue aproximar: mesmo ataque, canal real.

Uso:
    # 1. ver quais condições existem no metadado
    python scripts/importar_asvspoof2021.py --metadata <keys>/LA/CM/trial_metadata.txt --listar

    # 2. gerar protocolo + config de uma condição, subamostrada
    python scripts/importar_asvspoof2021.py --metadata <...>/trial_metadata.txt \\
        --codec opus --amostra 10000 \\
        --config-base configs/fusion_v4.yaml \\
        --audio-dir data/ASVspoof2021_LA_eval/flac

    # 3. o passo 2 imprime o comando do evaluate.py, já com o config gerado

**Por que o config é gerado, e não copiado à mão.** Os artefatos do `evaluate.py`
levam o nome `experiment.name` + partição. Copiar `fusion_v4.yaml` e trocar só
os caminhos do `eval` faria a avaliação do 2021 gravar **por cima** dos
resultados do eval de 2019 — métricas, scores reaproveitados e o cache de
features. O config gerado muda o nome e desliga o cache (ver
`src/config.config_derivado`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config_derivado, load_config, salvar_config  # noqa: E402
from src.data.asvspoof2021 import (  # noqa: E402
    MetadadoInvalido,
    condicoes,
    escrever_protocolo,
    filtrar,
    ler_metadata,
    subamostrar,
)

SAIDA = Path("outputs/asvspoof2021")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Importa o ASVspoof 2021")
    p.add_argument("--metadata", required=True,
                   help="trial_metadata.txt do eval-package (LA/CM/)")
    p.add_argument("--listar", action="store_true",
                   help="só lista as condições encontradas e sai")
    p.add_argument("--condicao", default=None, help="condição exata `codec/canal`")
    p.add_argument("--codec", default=None, help="filtra só pelo codec")
    p.add_argument("--amostra", type=int, default=0,
                   help="subamostra estratificada de N trials (0 = todos)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config-base", default=None,
                   help="config do modelo treinado; gera um config derivado seguro")
    p.add_argument("--audio-dir", default=None,
                   help="pasta de .flac do eval do 2021 (exigida com --config-base)")
    p.add_argument("--saida", default=None,
                   help="protocolo de saída (padrão: outputs/asvspoof2021/<rótulo>.txt)")
    return p.parse_args(argv)


def rotulo(args) -> str:
    base = args.condicao or args.codec or "todas"
    rot = base.replace("/", "-")
    return f"{rot}_n{args.amostra}" if args.amostra else rot


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        trials = ler_metadata(args.metadata)
    except (OSError, MetadadoInvalido) as erro:
        print(f"[ERRO] {erro}")
        return 1

    ataques = sorted({t.ataque for t in trials if t.ataque != "-"})
    print(f"{len(trials):,} trials | {len(ataques)} ataques: {', '.join(ataques)}")

    if args.listar:
        print(f"\n{'condição (codec/canal)':32s} {'bonafide':>10s} {'spoof':>10s} "
              f"{'total':>10s}")
        print("-" * 66)
        for nome, bona, spoof in condicoes(trials):
            print(f"{nome:32s} {bona:10,} {spoof:10,} {bona + spoof:10,}")
        print("\nA condição sem codec e sem transmissão reproduz o cenário do eval"
              "\nde 2019 — é o controle pareado. Gere ela e a do Opus com a mesma"
              "\n--seed e o mesmo --amostra.")
        return 0

    if args.config_base and not args.audio_dir:
        print("[ERRO] --config-base exige --audio-dir (a pasta de .flac do 2021).")
        return 1

    selecao = filtrar(trials, condicao=args.condicao, codec=args.codec)
    if not selecao:
        print("[ERRO] nenhum trial casa com o filtro pedido. Use --listar.")
        return 1
    selecao = subamostrar(selecao, args.amostra, seed=args.seed)

    bona = sum(1 for t in selecao if t.chave == "bonafide")
    if bona == 0 or bona == len(selecao):
        print("[ERRO] a seleção tem uma classe só; não dá para calcular EER.")
        return 1

    tag = rotulo(args)
    protocolo = Path(args.saida) if args.saida else SAIDA / f"{tag}.txt"
    n = escrever_protocolo(selecao, protocolo)
    print(f"\n{n:,} trials em {protocolo} ({bona:,} bonafide, {n - bona:,} spoof, "
          f"{len({t.ataque for t in selecao if t.ataque != '-'})} ataques)")

    if not args.config_base:
        print("\nSem --config-base: nenhum config gerado. NÃO copie o config do"
              "\nmodelo trocando só os caminhos — isso sobrescreve os resultados de"
              "\n2019. Rode de novo com --config-base e --audio-dir.")
        return 0

    base = load_config(args.config_base)
    cfg = config_derivado(base, f"2021_{tag}", protocolo, args.audio_dir)
    destino = salvar_config(cfg, protocolo.with_suffix(".yaml"))
    print(f"Config derivado: {destino}")
    print(f"   experimento: {cfg['experiment']['name']}  (os de 2019 ficam intactos)")
    print("\nPróximo passo:")
    print(f"   python evaluate.py --config {destino.as_posix()} "
          "--checkpoint checkpoints/<modelo>.pt")
    print(f"   python scripts/per_attack_eval.py --config {destino.as_posix()} "
          "--checkpoint checkpoints/<modelo>.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
