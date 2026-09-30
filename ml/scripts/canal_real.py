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
    python scripts/canal_real.py preparar --config configs/baseline_v2.yaml \\
        --n-por-classe 20 --saida outputs/canal_real
    python scripts/canal_real.py alinhar --pasta outputs/canal_real \\
        --gravacao chamada.wav --sessao chamada

Sem VB-Cable e sem VLC:
    # sessão controle: toca a playlist no alto-falante, pelo próprio Python
    python scripts/canal_real.py tocar --pasta outputs/canal_real
    # sessão chamada: abre um Chrome separado cujo MICROFONE é a playlist
    python scripts/canal_real.py chrome --pasta outputs/canal_real

`--sessao` separa as gravações da mesma playlist (limpo, controle, chamada):
cada uma ganha a própria pasta e o próprio nome de experimento, e nenhuma
sobrescreve a outra. `scripts/comparar_sessoes.py` põe as sessões lado a lado.
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

#: Modelo recomendado para chamada (canal real do ASVspoof 2021 LA, Seção 5.3
#: do resumo). O config de avaliação é derivado dele, e o checkpoint precisa
#: ser o do mesmo modelo — senão o evaluate falha ao carregar os pesos.
CONFIG_PADRAO = "configs/baseline_v2.yaml"
CHECKPOINT_PADRAO = "checkpoints/baseline_lfcc_cnn_v2.pt"

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

PONTA A (quem toca) — sem instalar nada
  1. python scripts/canal_real.py chrome --pasta {saida}
     Abre um Chrome separado cujo MICROFONE é a playlist (modo de teste do
     WebRTC). Entre na reunião por ele e desligue a câmera.
     Alternativa: VB-Cable como microfone da chamada e
       python scripts/canal_real.py tocar --pasta {saida} --dispositivo "CABLE Input"
  2. Não mexa no volume nem nas opções de áudio no meio da reprodução.
  3. Microfone da PONTA B desligado (evita eco).

PONTA B (quem grava)
  4. ANTES de a ponta A começar, inicie a gravação:
       python monitor.py --config configs/baseline_v2.yaml \\
           --checkpoint checkpoints/baseline_lfcc_cnn_v2.pt \\
           --gravar {saida}/chamada.wav --json {saida}/chamada.json
     Comece a gravar ANTES e pare DEPOIS (Ctrl+C) — a folga é o que o
     alinhamento usa.

DEPOIS
  5. python scripts/canal_real.py alinhar --pasta {saida} \\
         --gravacao {saida}/chamada.wav --sessao chamada
  6. O passo 5 imprime o comando do evaluate.py para obter o EER da sessão.

CONTROLE (importante)
  Grave com o monitor enquanto a playlist toca no alto-falante, sem chamada:
    python scripts/canal_real.py tocar --pasta {saida}
  Se o controle já divergir do limpo, a diferença é do procedimento, não do
  Meet/Teams. Passo a passo completo: docs/ROTEIRO_TESTE_AO_VIVO.md
────────────────────────────────────────────────────────────────────────────"""


#: Silêncio antes da playlist no arquivo que o Chrome usa como microfone. O
#: Chrome começa a "tocar" o arquivo quando a página abre o microfone — na tela
#: de prévia do Meet, antes de entrar na reunião. Sem essa espera, o começo da
#: playlist passaria enquanto ninguém ainda está ouvindo.
ESPERA_CHROME_S = 60.0

#: Silêncio depois da playlist. Medido no Chromium: com `%noloop`, quando o
#: arquivo acaba o microfone falso continua emitindo o ÚLTIMO bloco, repetido.
#: Se o arquivo terminasse em fala, a chamada ouviria um zumbido até alguém
#: fechar a janela; terminando em silêncio, o que se repete é silêncio.
SILENCIO_FINAL_S = 2.0

#: Taxa do arquivo para o Chrome. 48 kHz é a taxa nativa do WebRTC; entregar
#: nela evita depender da reamostragem interna do dispositivo falso.
TAXA_CHROME = 48000


def tocar_audio(wav: np.ndarray, sample_rate: int, dispositivo: str | None = None) -> None:
    """Toca um áudio num dispositivo de saída (o padrão, se nenhum for dado).

    Substitui o VLC: o `soundcard`, que o monitor já usa para capturar, também
    reproduz, e deixa escolher o dispositivo pelo nome.
    """
    import soundcard
    import soxr

    alto_falante = (soundcard.get_speaker(dispositivo) if dispositivo
                    else soundcard.default_speaker())
    print(f"Tocando em: {alto_falante.name}  ({len(wav) / sample_rate / 60:.1f} min)")
    # Toca a 48 kHz, a mesma taxa em que a captura grava: 16 kHz obrigaria o
    # Windows a converter no meio, e uma conversão errada muda a duração e o
    # tom da gravação inteira (ver scripts/diagnosticar_captura.py).
    wav48 = soxr.resample(np.asarray(wav, dtype=np.float32), sample_rate, TAXA_CHROME)
    # Buffer de 1 s também na reprodução, pelo mesmo motivo da captura: com o
    # padrão de ~10 ms, qualquer pausa do processo vira engasgo no que sai.
    alto_falante.play(wav48, samplerate=TAXA_CHROME, blocksize=TAXA_CHROME)


def cmd_tocar(args) -> int:
    arquivo = Path(args.arquivo) if getattr(args, "arquivo", None) else \
        Path(args.pasta) / "referencia.wav"
    referencia, sr = sf.read(arquivo, dtype="float32")
    print("Confira que o monitor JÁ está gravando no outro terminal: ele leva alguns")
    print("segundos carregando o modelo. Espere aparecer o cabeçalho da tabela")
    print("('t  score  média ...') antes de dar Enter.")
    if not args.sem_pausa:
        input("Enter para começar... ")
    try:
        tocar_audio(referencia, sr, args.dispositivo)
    except ImportError:
        print("[ERRO] instale o pacote de áudio: pip install soundcard")
        return 1
    print("Fim da playlist. Espere ~5 s e pare o monitor com Ctrl+C.")
    return 0


def arquivo_para_chrome(referencia: np.ndarray, sr: int, destino: Path,
                        espera_s: float = ESPERA_CHROME_S) -> Path:
    """Grava a playlist no formato do microfone falso do Chrome.

    WAV PCM de 16 bits a 48 kHz, com `espera_s` de silêncio no começo e
    `SILENCIO_FINAL_S` no fim. O alinhamento não precisa saber da espera: ele
    acha o atraso global sozinho.

    Conferido no Chromium: o arquivo começa a tocar quando a página abre o
    microfone, respeita `%noloop`, e sem ele se repete indefinidamente.
    """
    from scipy.signal import resample_poly

    from math import gcd

    g = gcd(TAXA_CHROME, sr)
    wav = resample_poly(np.asarray(referencia, dtype=np.float64), TAXA_CHROME // g, sr // g)
    silencio = np.zeros(int(round(espera_s * TAXA_CHROME)))
    final = np.zeros(int(round(SILENCIO_FINAL_S * TAXA_CHROME)))
    wav = np.clip(np.concatenate([silencio, wav, final]), -1.0, 1.0)
    destino.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destino, wav.astype(np.float32), TAXA_CHROME, subtype="PCM_16")
    return destino


def achar_navegador(explicito: str | None = None) -> str | None:
    """Chrome ou Edge (os dois são Chromium e aceitam as mesmas opções)."""
    import os
    import shutil

    if explicito:
        return explicito if Path(explicito).is_file() else None
    candidatos = []
    for var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        base = os.environ.get(var)
        if base:
            candidatos.append(Path(base) / "Google/Chrome/Application/chrome.exe")
    for var in ("PROGRAMFILES(X86)", "PROGRAMFILES"):
        base = os.environ.get(var)
        if base:
            candidatos.append(Path(base) / "Microsoft/Edge/Application/msedge.exe")
    for c in candidatos:
        if c.is_file():
            return str(c)
    for nome in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge"):
        achado = shutil.which(nome)
        if achado:
            return achado
    return None


def comando_chrome(navegador: str, arquivo_wav: Path, perfil: Path,
                   url: str) -> list[str]:
    """Linha de comando do Chrome com o arquivo no lugar do microfone.

    - `--use-fake-device-for-media-stream` troca câmera e microfone por
      dispositivos falsos; `--use-file-for-fake-audio-capture` faz o microfone
      falso tocar o WAV. São opções de teste do próprio WebRTC.
    - `%noloop` impede o arquivo de recomeçar quando termina.
    - `--user-data-dir` força uma instância NOVA: com o Chrome já aberto, sem
      isso, a janela nasceria dentro do processo existente e as opções seriam
      ignoradas em silêncio.
    """
    return [
        navegador,
        f"--user-data-dir={perfil.resolve()}",
        "--use-fake-device-for-media-stream",
        f"--use-file-for-fake-audio-capture={arquivo_wav.resolve()}%noloop",
        "--no-first-run",
        "--no-default-browser-check",
        url,
    ]


def cmd_chrome(args) -> int:
    import subprocess

    pasta = Path(args.pasta)
    referencia, sr = sf.read(pasta / "referencia.wav", dtype="float32")
    wav_chrome = arquivo_para_chrome(referencia, sr, pasta / "referencia_chrome.wav",
                                     args.espera)
    print(f"Microfone do Chrome: {wav_chrome}  ({args.espera:.0f} s de espera + playlist)")

    navegador = achar_navegador(args.navegador)
    if navegador is None:
        print("[ERRO] Chrome/Edge não encontrado. Informe o caminho com "
              "--navegador \"C:\\...\\chrome.exe\"")
        return 1
    cmd = comando_chrome(navegador, wav_chrome, pasta / "perfil_navegador", args.url)
    if args.so_mostrar:
        print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
        return 0
    subprocess.Popen(cmd)
    print(f"""
Aberto: {Path(navegador).name}, numa instância separada (perfil próprio).

1. No terminal do monitor, a gravação da ponta B já deve estar rodando.
2. Nesta janela (ponta A): entre na reunião e DESLIGUE a câmera. O microfone
   é o arquivo — não precisa escolher nada. A playlist começa sozinha depois de
   ~{args.espera:.0f} s de silêncio contados de quando o Meet abriu o microfone.
3. Na ponta B (seu navegador normal): microfone DESLIGADO, alto-falante padrão.
4. Quando a playlist acabar (~{len(referencia) / sr / 60:.0f} min depois da espera),
   espere ~5 s e pare o monitor com Ctrl+C. Pode fechar esta janela.
""")
    return 0


def cmd_alinhar(args) -> int:
    pasta = Path(args.pasta)
    # Sem sessão, o layout antigo (tudo direto na pasta). Com sessão, cada
    # gravação da mesma playlist vive em pasta/<sessao>/ e não apaga as outras.
    sessao = getattr(args, "sessao", None)
    destino_sessao = pasta / sessao if sessao else pasta
    mapa, sr = carregar_mapa(pasta / "mapa.json")
    referencia, _ = sf.read(pasta / "referencia.wav", dtype="float32")
    captura = load_audio(args.gravacao, sr)

    print(f"Referência: {len(referencia)/sr/60:.1f} min | "
          f"gravação: {len(captura)/sr/60:.1f} min | {len(mapa)} trechos")
    if len(captura) < len(referencia):
        print(f"[AVISO] a gravação é mais curta que a playlist "
              f"({(len(referencia)-len(captura))/sr:.0f}s a menos). "
              f"Os trechos do fim vão faltar.")

    from src.capture.alinhamento import TAXA_ENVELOPE_HZ, atraso_global, envelope

    atraso_s = atraso_global(envelope(referencia, sr), envelope(captura, sr)) / TAXA_ENVELOPE_HZ
    print(f"Atraso da gravação: {atraso_s:+.2f} s")
    if atraso_s < 0:
        perdidos_ini = sum(1 for t in mapa if t.inicio / sr < -atraso_s)
        print(f"[AVISO] a gravação começou {-atraso_s:.1f} s DEPOIS da reprodução: "
              f"~{perdidos_ini} trecho(s) do início ficaram de fora. Da próxima vez, "
              "só dê Enter no `tocar` quando o monitor já mostrar a tabela de scores.")

    encaixes = alinhar(mapa, referencia, captura, sr)
    pedacos = recortar(captura, encaixes)

    saida = destino_sessao / "capturado"; saida.mkdir(parents=True, exist_ok=True)
    for arquivo in saida.glob("*.flac"):
        arquivo.unlink()
    for trecho, wav in pedacos:
        sf.write(saida / f"{trecho.id}.flac", wav, sr)
    proto = destino_sessao / "protocolo_canal_real.txt"
    proto.write_text("\n".join(linha_de_protocolo(t) for t, _ in pedacos) + "\n",
                     encoding="utf-8")

    corrs = [e.correlacao for e in encaixes]
    perdidos = [e for e in encaixes if not e.confiavel]
    n_spoof = sum(1 for t, _ in pedacos if t.rotulo == "spoof")
    print(f"\nCorrelação: mediana {np.median(corrs):.3f} | "
          f"pior {min(corrs):.3f} | limiar {0.5:.2f}")
    print(f"Recuperados: {len(pedacos)}/{len(mapa)} "
          f"({len(pedacos)-n_spoof} bonafide, {n_spoof} spoof)")
    n_fino = sum(1 for e in encaixes if e.confiavel and e.refinado)
    print(f"Ajuste fino à amostra: {n_fino}/{sum(1 for e in encaixes if e.confiavel)} "
          "(os demais ficam na posição do envelope, com erro de até 5 ms)")
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
    base_path = getattr(args, "config_base", None) or CONFIG_PADRAO
    checkpoint = getattr(args, "checkpoint", None) or CHECKPOINT_PADRAO
    sufixo = f"canal_real_{pasta.name}" + (f"_{sessao}" if sessao else "")
    cfg = config_derivado(load_config(base_path), sufixo, proto, saida)
    destino = salvar_config(cfg, destino_sessao / "config_canal_real.yaml")
    print(f"Config derivado: {destino}  (experimento {cfg['experiment']['name']})")
    print(_como_avaliar(destino, checkpoint))
    return 0


def _como_avaliar(config: Path, checkpoint: str) -> str:
    return f"""
─────────────────────────── COMO OBTER O EER ───────────────────────────
  python evaluate.py --config {config.as_posix()} \\
      --checkpoint {checkpoint} --partition eval

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
    a.add_argument("--n-por-classe", type=int, default=20)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--saida", default="outputs/canal_real")
    a.set_defaults(func=cmd_preparar)

    t = sub.add_parser("tocar", help="toca a playlist num alto-falante (sem VLC)")
    t.add_argument("--pasta", default="outputs/canal_real")
    t.add_argument("--dispositivo", default=None,
                   help="nome do dispositivo de saída (padrão: o do sistema)")
    t.add_argument("--sem-pausa", action="store_true",
                   help="começa a tocar sem esperar Enter")
    t.add_argument("--arquivo", default=None,
                   help="toca este .wav em vez de <pasta>/referencia.wav "
                        "(ex.: as playlists de scripts/montar_demo.py)")
    t.set_defaults(func=cmd_tocar)

    c = sub.add_parser("chrome", help="abre um Chrome cujo microfone é a playlist "
                                      "(sem VB-Cable)")
    c.add_argument("--pasta", default="outputs/canal_real")
    c.add_argument("--url", default="https://meet.google.com")
    c.add_argument("--espera", type=float, default=ESPERA_CHROME_S,
                   help="segundos de silêncio antes da playlist, para dar tempo "
                        "de entrar na reunião")
    c.add_argument("--navegador", default=None, help="caminho do chrome.exe/msedge.exe")
    c.add_argument("--so-mostrar", action="store_true",
                   help="só imprime o comando, sem abrir o navegador")
    c.set_defaults(func=cmd_chrome)

    b = sub.add_parser("alinhar", help="recorta a gravação e emite o protocolo")
    b.add_argument("--pasta", default="outputs/canal_real")
    b.add_argument("--gravacao", required=True)
    b.add_argument("--sessao", default=None,
                   help="nome desta gravação (ex.: limpo, controle, chamada); "
                        "cada sessão fica em <pasta>/<sessao>/")
    b.add_argument("--config-base", default=CONFIG_PADRAO,
                   help="config do modelo; o de avaliação é derivado dele")
    b.add_argument("--checkpoint", default=CHECKPOINT_PADRAO,
                   help="checkpoint do MESMO modelo do --config-base")
    b.set_defaults(func=cmd_alinhar)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
