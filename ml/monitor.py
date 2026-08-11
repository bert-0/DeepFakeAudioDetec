"""Monitor de chamada ao vivo (RF04–RF07 da APS).

Escuta a **saída de áudio do sistema** e classifica o que passa, janela a
janela. Como pega o que sai da caixa de som, funciona com Microsoft Teams,
Meet, Zoom ou qualquer outro, sem publicar aplicativo em tenant nenhum.

    # ao vivo, durante uma chamada
    python monitor.py --config configs/baseline_v2.yaml \\
        --checkpoint checkpoints/baseline_lfcc_cnn_v2.pt

    # ao vivo, gravando o que ouviu (é assim que se mede o canal real)
    python monitor.py --config ... --checkpoint ... --gravar chamada.wav

    # reprocessar uma gravação, de forma reprodutível
    python monitor.py --config ... --checkpoint ... --arquivo chamada.wav

    # ver os dispositivos disponíveis
    python monitor.py --listar-dispositivos

**Sobre o número que ele mostra.** O modelo foi treinado no ASVspoof: áudio
limpo, 16 kHz, sem codec. O áudio de uma chamada passou por microfone, sala,
supressão de ruído, ganho automático e o codec Opus. Metade do banco de filtros
do LFCC olha acima de 4 kHz, que é justamente o que um canal estreito não
transmite. Enquanto essa degradação não for medida, o monitor mostra **score**,
não veredito — e é para isso que serve o `--gravar`: toque áudios de rótulo
conhecido numa chamada real, capture, e avalie o resultado.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.capture import CaptureError, FileSource, WasapiLoopbackSource
from src.capture.analyzer import AnalisadorContinuo, Agregador
from src.config import load_config, resolve_device

OUTPUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor de chamada ao vivo")
    p.add_argument("--config", help="YAML de configuração")
    p.add_argument("--checkpoint", help="checkpoint treinado")
    p.add_argument("--arquivo", help="analisa um .wav em vez do áudio ao vivo")
    p.add_argument("--gravar", help="salva o áudio capturado neste .wav")
    p.add_argument("--dispositivo-audio", default=None,
                   help="nome do dispositivo de saída a escutar (padrão: o do sistema)")
    p.add_argument("--hop", type=float, default=None,
                   help="passo entre janelas em segundos (padrão: metade da janela)")
    p.add_argument("--device", default=None, help="cuda | cpu")
    p.add_argument("--segundos", type=float, default=None,
                   help="encerra depois de N segundos (padrão: até Ctrl+C)")
    p.add_argument("--listar-dispositivos", action="store_true",
                   help="mostra os dispositivos de saída e sai")
    p.add_argument("--json", help="grava o histórico de scores neste arquivo")
    return p.parse_args()


def listar_dispositivos() -> int:
    try:
        import soundcard
    except ImportError:
        print("[ERRO] instale o pacote de captura: pip install soundcard")
        return 1
    print("Dispositivos de saída (use o nome com --dispositivo-audio):\n")
    padrao = soundcard.default_speaker().name
    for alto_falante in soundcard.all_speakers():
        marca = " <- padrão" if alto_falante.name == padrao else ""
        print(f"  {alto_falante.name}{marca}")
    return 0


def barra(score: float, largura: int = 28) -> str:
    cheio = int(round(score * largura))
    return "#" * cheio + "." * (largura - cheio)


def main() -> int:
    args = parse_args()
    if args.listar_dispositivos:
        return listar_dispositivos()
    if not args.config or not args.checkpoint:
        print("[ERRO] --config e --checkpoint são obrigatórios "
              "(ou use --listar-dispositivos)")
        return 1

    config = load_config(args.config)
    device = resolve_device(args.device or config["train"]["device"])
    analisador = AnalisadorContinuo(config, args.checkpoint, device, hop_s=args.hop)
    agregador = Agregador()

    origem = "arquivo" if args.arquivo else "saída do sistema (loopback)"
    print(f"Modelo: {analisador.config['model']['name']} | dispositivo: {device}")
    print(f"Fonte:  {origem}")
    print(f"Janela: {analisador.janela.tamanho / analisador.sample_rate:.1f}s | "
          f"passo: {analisador.janela.passo / analisador.sample_rate:.1f}s")
    if analisador.threshold is not None:
        print(f"Threshold do checkpoint: {analisador.threshold:.4f} "
              "(calibrado no dev, em áudio LIMPO — ver aviso abaixo)")
    print("\n[AVISO] o modelo foi treinado em áudio de laboratório, sem codec. "
          "Numa chamada\n        real o score ainda não tem degradação medida: "
          "trate como indício, não\n        como veredito. Use --gravar para "
          "medir o canal.\n")

    try:
        fonte = (FileSource(args.arquivo, analisador.sample_rate) if args.arquivo
                 else WasapiLoopbackSource(analisador.sample_rate,
                                           nome_dispositivo=args.dispositivo_audio))
    except CaptureError as erro:
        print(f"[ERRO] {erro}")
        return 1

    gravado: list[np.ndarray] = []
    limite = args.segundos
    print(f"{'t':>8s}  {'score':>6s}  {'média':>6s}  sinal")
    print("-" * 60)
    try:
        with fonte:
            for bloco in fonte.blocos():
                if args.gravar:
                    gravado.append(bloco)
                for leitura in analisador.processar(bloco):
                    agregador.adicionar(leitura)
                    if leitura.silencio:
                        print(f"{leitura.instante:7.1f}s  {'—':>6s}  {'—':>6s}  "
                              "(silêncio)")
                        continue
                    media = agregador.media_movel()
                    print(f"{leitura.instante:7.1f}s  {leitura.score:6.3f}  "
                          f"{media:6.3f}  {barra(leitura.score)}")
                    if limite is not None and leitura.instante >= limite:
                        raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("\nEncerrado.")
    finally:
        if args.gravar and gravado:
            salvar(np.concatenate(gravado), analisador.sample_rate, args.gravar)
        relatar(agregador, args.json)
    return 0


def salvar(wav: np.ndarray, sample_rate: int, destino: str) -> None:
    import soundfile as sf

    caminho = Path(destino)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    sf.write(caminho, wav, sample_rate)
    print(f"Áudio capturado: {caminho}  ({len(wav) / sample_rate:.1f}s)")
    print("  Reprocesse com --arquivo para obter exatamente o mesmo resultado.")


def relatar(agregador: Agregador, destino: str | None) -> None:
    resumo = agregador.resumo()
    print("\n" + "=" * 60)
    print(f"Janelas analisadas: {resumo['janelas_total']} "
          f"({resumo['janelas_uteis']} com áudio, "
          f"{resumo['janelas_silencio']} em silêncio)")
    if resumo["score_medio"] is None:
        print("Nenhuma janela com áudio — nada a resumir.")
        return
    print(f"Score  médio {resumo['score_medio']:.3f} | "
          f"mediano {resumo['score_mediano']:.3f} | "
          f"máximo {resumo['score_maximo']:.3f}")
    print("Lembrete: score alto indica *indício* de síntese. A taxa de erro "
          "deste modelo\nem áudio de chamada ainda não foi medida.")
    if destino:
        caminho = Path(destino)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        historico = [{"indice": x.indice, "instante": x.instante,
                      "score": x.score, "rms": x.rms, "silencio": x.silencio}
                     for x in agregador.leituras]
        caminho.write_text(json.dumps({"resumo": resumo, "leituras": historico},
                                      indent=2), encoding="utf-8")
        print(f"Histórico: {caminho}")


if __name__ == "__main__":
    raise SystemExit(main())
