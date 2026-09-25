"""Camada 2: mede o sistema através de uma chamada REAL (Teams, Meet, Zoom).

A avaliação de robustez do projeto (`robustness_eval.py`) simula o canal em
software: codec Opus e limitação de banda. Isso é um **limite inferior** da
degradação, porque não inclui o que só existe num cliente de conferência de
verdade — supressão de ruído, cancelamento de eco e ganho automático. Nenhum
dos três é simulável de forma honesta.

Este script fecha essa lacuna em três passos:

1. `preparar`  — monta uma playlist de áudios ROTULADOS do eval, com silêncio
                 entre eles, e grava o mapa de posições.
2. (manual)    — você toca `referencia.wav` dentro de uma chamada e grava o que
                 chega do outro lado (veja o roteiro impresso no passo 1).
3. `alinhar`   — localiza cada áudio dentro da gravação por correlação de
                 envelope, recorta e emite um protocolo no formato ASVspoof.

O que sai do passo 3 entra direto no `evaluate.py`. O EER resultante é o número
da camada 2: o desempenho no canal real, não no simulado.

Uso:
    python scripts/canal_real.py preparar --config configs/fusion_v4.yaml \\
        --n-por-classe 40 --saida outputs/canal_real
    python scripts/canal_real.py alinhar --pasta outputs/canal_real \\
        --gravacao chamada_teams.wav
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture.alinhamento import (  # noqa: E402
    Trecho,
    alinhar,
    carregar_mapa,
    linha_de_protocolo,
    montar_referencia,
    recortar,
    salvar_mapa,
)
from src.config import config_derivado, load_config, salvar_config  # noqa: E402
from src.data.dataset import parse_protocol_with_systems  # noqa: E402
from src.preprocess import load_audio  # noqa: E402

#: Silêncio entre os áudios. Um segundo dá folga ao recorte e evita que a
#: supressão de ruído trate a emenda como um fluxo contínuo de fala.
GAP_S = 1.0

#: Uma chamada longa demais acumula deriva e cansa quem está segurando o
#: procedimento. 80 áudios de ~4 s dão ~7 min, que é operável de uma sentada.
AVISO_MINUTOS = 12.0


def sortear(registros, n_por_classe: int, seed: int):
    """Amostra equilibrada: metade bonafide, metade spoof, espalhada por ataque.

    Sortear ao acaso puxaria os ataques na proporção do eval (4.914 de cada um
    contra 7.355 bonafide no total), e uma playlist pequena ficaria sem A10 ou
    sem A12 — justamente os dois que dominam o erro neste projeto.
    """
    rng = random.Random(seed)
    bonafide = [r for r in registros if r[1] == 0]
    por_ataque: dict[str, list] = defaultdict(list)
    for r in registros:
        if r[1] == 1:
            por_ataque[r[2]].append(r)

    escolhidos = rng.sample(bonafide, min(n_por_classe, len(bonafide)))
    ataques = sorted(por_ataque)
    for i in range(n_por_classe):                     # rodízio entre os ataques
        pool = por_ataque[ataques[i % len(ataques)]]
        if pool:
            escolhidos.append(pool.pop(rng.randrange(len(pool))))
    rng.shuffle(escolhidos)
    return escolhidos


def cmd_preparar(args) -> int:
    config = load_config(args.config)
    sr = config["audio"]["sample_rate"]
    registros = parse_protocol_with_systems(config["data"]["protocols"]["eval"])
    if not registros:
        print("[ERRO] protocolo de eval vazio ou inexistente.")
        return 1

    escolhidos = sortear(registros, args.n_por_classe, args.seed)
    audio_dir = Path(config["data"]["audio_dir"]["eval"])

    audios = []
    for nome, rotulo, sistema in escolhidos:
        caminho = audio_dir / f"{nome}.flac"
        if not caminho.is_file():
            print(f"[AVISO] falta {caminho}, pulando")
            continue
        wav = load_audio(caminho, sr)
        audios.append((Trecho(nome, "spoof" if rotulo else "bonafide",
                              sistema, 0, 0), wav))

    if not audios:
        print("[ERRO] nenhum áudio lido — confira data.audio_dir no config.")
        return 1

    referencia, mapa = montar_referencia(audios, sr, GAP_S)
    saida = Path(args.saida); saida.mkdir(parents=True, exist_ok=True)
    sf.write(saida / "referencia.wav", referencia, sr)
    salvar_mapa(mapa, saida / "mapa.json", sr)

    minutos = len(referencia) / sr / 60
    n_spoof = sum(1 for t in mapa if t.rotulo == "spoof")
    print(f"Playlist: {len(mapa)} áudios ({len(mapa)-n_spoof} bonafide, "
          f"{n_spoof} spoof) — {minutos:.1f} min")
    ataques = sorted({t.sistema for t in mapa if t.sistema != "-"})
    print(f"Ataques cobertos: {', '.join(ataques)}")
    print(f"Gravado em {saida}/referencia.wav e {saida}/mapa.json")
    if minutos > AVISO_MINUTOS:
        print(f"[AVISO] {minutos:.0f} min é uma chamada longa; a deriva de "
              f"relógio cresce e a chance de queda também. Considere "
              f"--n-por-classe menor e repetir o procedimento.")
    print(_roteiro(saida))
    return 0


def _roteiro(saida: Path) -> str:
    return f"""
──────────────────────────── ROTEIRO DA CHAMADA ────────────────────────────
Precisa de DUAS pontas. Podem ser dois notebooks, ou um notebook e um celular.

PONTA A (quem toca)
  1. Entre na chamada (Teams/Meet/Zoom).
  2. Selecione o dispositivo de saída "CABLE Input" (VB-Cable) como MICROFONE
     da chamada, e toque {saida}/referencia.wav para esse dispositivo.
     Sem o VB-Cable dá para usar o alto-falante e o microfone no ar, mas aí a
     acústica da sala entra na medida e o resultado deixa de ser só do canal.
  3. Deixe o volume estável. NÃO mexa no ganho no meio da reprodução.

PONTA B (quem grava)
  4. ANTES de a ponta A começar, inicie a gravação:
       python monitor.py --config configs/fusion_v4.yaml \\
           --checkpoint checkpoints/fusion_lcnn_v4.pt \\
           --gravar chamada.wav
     Comece a gravar ANTES e pare DEPOIS — a folga é o que o alinhamento usa.

DEPOIS
  5. python scripts/canal_real.py alinhar --pasta {saida} \\
         --gravacao chamada.wav
  6. O passo 5 imprime o comando do evaluate.py para obter o EER da camada 2.

CONTROLE (importante)
  Repita tudo com as duas pontas na MESMA máquina, sem chamada nenhuma
  (referencia.wav -> VB-Cable -> gravação). Esse é o controle: se o EER do
  controle já divergir do eval limpo, a diferença é do procedimento, não do
  Teams.
────────────────────────────────────────────────────────────────────────────"""


def cmd_alinhar(args) -> int:
    pasta = Path(args.pasta)
    mapa, sr = carregar_mapa(pasta / "mapa.json")
    referencia, _ = sf.read(pasta / "referencia.wav", dtype="float32")
    captura = load_audio(args.gravacao, sr)

    print(f"Referência: {len(referencia)/sr/60:.1f} min | "
          f"gravação: {len(captura)/sr/60:.1f} min | {len(mapa)} trechos")
    if len(captura) < len(referencia):
        print(f"[AVISO] a gravação é mais curta que a playlist "
              f"({(len(referencia)-len(captura))/sr:.0f}s a menos). "
              f"Os trechos do fim vão faltar.")

    encaixes = alinhar(mapa, referencia, captura, sr)
    pedacos = recortar(captura, encaixes)

    saida = pasta / "capturado"; saida.mkdir(exist_ok=True)
    for arquivo in saida.glob("*.flac"):
        arquivo.unlink()
    for trecho, wav in pedacos:
        sf.write(saida / f"{trecho.id}.flac", wav, sr)
    proto = pasta / "protocolo_canal_real.txt"
    proto.write_text("\n".join(linha_de_protocolo(t) for t, _ in pedacos) + "\n",
                     encoding="utf-8")

    corrs = [e.correlacao for e in encaixes]
    perdidos = [e for e in encaixes if not e.confiavel]
    n_spoof = sum(1 for t, _ in pedacos if t.rotulo == "spoof")
    print(f"\nCorrelação: mediana {np.median(corrs):.3f} | "
          f"pior {min(corrs):.3f} | limiar {0.5:.2f}")
    print(f"Recuperados: {len(pedacos)}/{len(mapa)} "
          f"({len(pedacos)-n_spoof} bonafide, {n_spoof} spoof)")
    if perdidos:
        print(f"Descartados por alinhamento fraco: {len(perdidos)} "
              f"({', '.join(e.trecho.id for e in perdidos[:5])}"
              f"{'…' if len(perdidos) > 5 else ''})")
    if len(pedacos) < len(mapa) * 0.8:
        print("[AVISO] menos de 80% recuperado. Confira se a gravação começou "
              "antes da reprodução e se as duas pontas usaram a mesma playlist.")
    if n_spoof == 0 or n_spoof == len(pedacos):
        print("[ERRO] só sobrou uma classe; não dá para calcular EER.")
        return 1

    print(f"\nÁudio recortado em {saida}/  ({len(pedacos)} arquivos)")
    print(f"Protocolo em {proto}")

    # Config gerado, não copiado à mão: com o `experiment.name` original, a
    # avaliação gravaria por cima dos resultados do eval de 2019.
    base_path = getattr(args, "config_base", None) or "configs/fusion_v4.yaml"
    cfg = config_derivado(load_config(base_path), f"canal_real_{pasta.name}",
                          proto, saida)
    destino = salvar_config(cfg, pasta / "config_canal_real.yaml")
    print(f"Config derivado: {destino}  (experimento {cfg['experiment']['name']})")
    print(_como_avaliar(destino))
    return 0


def _como_avaliar(config: Path) -> str:
    return f"""
─────────────────────────── COMO OBTER O EER ───────────────────────────
  python evaluate.py --config {config.as_posix()} \\
      --checkpoint checkpoints/fusion_lcnn_v4.pt --partition eval

O config acima foi GERADO com outro nome de experimento e sem cache. Não copie
o config do modelo trocando só os caminhos: os resultados sairiam com o mesmo
nome dos do eval de 2019 e gravariam por cima deles.

COMPARE com três números que você já tem:
  eval limpo (mesmo subconjunto)   ..... rode com o config original
  canal simulado (opus + 8 kHz)    ..... scripts/robustness_eval.py
  canal real (este)                ..... o comando acima

A distância entre o simulado e o real é exatamente o que a camada 1 não
consegue medir: supressão de ruído, cancelamento de eco e AGC.
────────────────────────────────────────────────────────────────────────"""


def main() -> int:
    p = argparse.ArgumentParser(description="Avaliação através de chamada real")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("preparar", help="monta a playlist rotulada")
    a.add_argument("--config", required=True)
    a.add_argument("--n-por-classe", type=int, default=40)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--saida", default="outputs/canal_real")
    a.set_defaults(func=cmd_preparar)

    b = sub.add_parser("alinhar", help="recorta a gravação e emite o protocolo")
    b.add_argument("--pasta", default="outputs/canal_real")
    b.add_argument("--gravacao", required=True)
    b.add_argument("--config-base", default="configs/fusion_v4.yaml",
                   help="config do modelo; o de avaliação é derivado dele")
    b.set_defaults(func=cmd_alinhar)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
