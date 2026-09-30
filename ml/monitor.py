"""Monitor de chamada ao vivo (RF04–RF07 da APS).

Escuta a saída de áudio do sistema (loopback) e mostra um score por janela;
funciona com Teams, Meet, Zoom ou qualquer outro. O limiar é escolhido pelo
tipo de áudio (src/limiares.py); o score é indício, não veredito.

    # ao vivo (recomendado: o baseline_v2 sozinho; a fusão não ganhou no canal real)
    python monitor.py --config configs/baseline_v2.yaml \\
        --checkpoint checkpoints/baseline_lfcc_cnn_v2.pt

    # ao vivo, com fusão pela média de dois modelos
    python monitor.py --config configs/fusion_v4.yaml \\
        --checkpoint checkpoints/fusion_lcnn_v4.pt \\
        --checkpoint checkpoints/baseline_lfcc_cnn_v2.pt

    # ao vivo, gravando o que ouviu (é assim que se mede o canal real)
    python monitor.py --config ... --checkpoint ... --gravar chamada.wav

    # reprocessar uma gravação, de forma reprodutível
    python monitor.py --config ... --checkpoint ... --arquivo chamada.wav

    # ver os dispositivos disponíveis
    python monitor.py --listar-dispositivos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.capture import CaptureError, FileSource, WasapiLoopbackSource
from src.capture.analyzer import AnalisadorContinuo, Agregador
from src.config import load_config, resolve_device
from src.data.dataset import buscar_rotulo
from src.limiares import escolher_limiar, limiares_do_checkpoint, taxa_nativa

OUTPUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor de chamada ao vivo")
    p.add_argument("--config", help="YAML de configuração")
    p.add_argument("--checkpoint", action="append",
                   help="checkpoint treinado; repita a opção para fundir modelos")
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
    p.add_argument("--parar-com", default=None, metavar="ARQUIVO",
                   help="encerra como um Ctrl+C quando este arquivo aparecer (usado "
                        "pela interface web, que não consegue mandar Ctrl+C no Windows)")
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


def barra(score: float, limiar: float | None = None, largura: int = 28) -> str:
    """Barra do score, com o limiar marcado por `|`."""
    cheio = min(largura, int(round(score * largura)))
    celulas = ["#"] * cheio + ["."] * (largura - cheio)
    if limiar is not None:
        i = min(largura - 1, max(0, int(round(limiar * largura))))
        celulas[i] = "|"
    return "".join(celulas)


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
    # Ao vivo e arquivos acima de 16 kHz usam o limiar recalibrado para a
    # conversão de taxa; nativos de 16 kHz, o original (src/limiares.py).
    import torch

    primeiro = args.checkpoint[0]
    lims = limiares_do_checkpoint(
        primeiro, torch.load(primeiro, map_location="cpu", weights_only=False))
    escolha = escolher_limiar(lims, analisador.sample_rate,
                              taxa_nativa(args.arquivo) if args.arquivo else None,
                              ao_vivo=not args.arquivo)
    if analisador.threshold is not None:
        analisador.threshold = escolha.limiar
    agregador = Agregador(
        janela_s=analisador.janela.tamanho / analisador.sample_rate,
        passo_s=analisador.janela.passo / analisador.sample_rate)

    # Arquivo do dataset: mostra o rótulo verdadeiro ao lado do score.
    verdade = None
    if args.arquivo:
        achado = buscar_rotulo(config, Path(str(args.arquivo).replace("\\", "/")).stem)
        if achado:
            particao, label, sistema = achado
            verdade = "spoof" if label else "bonafide"

    origem = "arquivo" if args.arquivo else "saída do sistema (loopback)"
    if analisador.n_modelos > 1:
        print(f"Modelos: {analisador.n_modelos} em fusão (média) | dispositivo: {device}")
    else:
        print(f"Modelo: {analisador.config['model']['name']} | dispositivo: {device}")
    print(f"Fonte:  {origem}")
    if verdade:
        extra = f" (ataque {sistema})" if sistema != "-" else ""
        print(f"Rótulo verdadeiro: {verdade.upper()}{extra}  "
              f"— do protocolo de '{particao}'")
    elif args.arquivo:
        print("Rótulo verdadeiro: desconhecido (o id não está em nenhum "
              "protocolo)")
    print(f"Janela: {analisador.janela.tamanho / analisador.sample_rate:.1f}s | "
          f"passo: {analisador.janela.passo / analisador.sample_rate:.1f}s")
    if analisador.threshold is not None:
        print(f"Limiar: {analisador.threshold:.4f} ({escolha.origem})")
    if escolha.aviso:
        print(f"[AVISO] {escolha.aviso}")
    print("\n[AVISO] codec, supressão de ruído e AGC da plataforma de chamada ainda "
          "deslocam o score;\n        desligue os aprimoramentos de áudio do "
          "Windows. Trate o número como indício.\n")

    try:
        fonte = (FileSource(args.arquivo, analisador.sample_rate) if args.arquivo
                 else WasapiLoopbackSource(analisador.sample_rate,
                                           nome_dispositivo=args.dispositivo_audio))
    except CaptureError as erro:
        print(f"[ERRO] {erro}")
        return 1

    gravado: list[np.ndarray] = []
    limite = args.segundos
    print(f"{'t':>8s}  {'score':>6s}  {'média':>6s}  {'canal':>6s}  {'peso':>5s}  sinal")
    print("          score = P(síntese): 0,00 = voz humana | 1,00 = sintético"
          + (f"   ('|' marca o limiar {analisador.threshold:.3f})"
             if analisador.threshold is not None else ""))
    print("-" * 78)
    def _mostrar(leitura) -> None:
        """Imprime uma leitura (do fluxo ou da janela final)."""
        agregador.adicionar(leitura)
        if args.json:
            gravar_json(agregador, args.json, analisador.threshold, escolha.origem, ativo=True)
        if leitura.silencio:
            print(f"{leitura.instante:7.1f}s  {'—':>6s}  {'—':>6s}  "
                  f"{'—':>6s}  {'—':>5s}  (silêncio)")
            return
        banda = (f"{100 * leitura.qualidade.fracao_alta:5.1f}%"
                 if leitura.qualidade else "    —")
        if not leitura.confiavel:
            print(f"{leitura.instante:7.1f}s  {'—':>6s}  {'—':>6s}  "
                  f"{banda:>6s}  {'—':>5s}  {analisador.canal.descricao()}")
            return
        media = agregador.media_movel()
        marca = "" if leitura.peso >= 0.95 else "  (janela parcial)"
        print(f"{leitura.instante:7.1f}s  {leitura.score:6.3f}  "
              f"{media:6.3f}  {banda:>6s}  {leitura.peso:5.2f}  "
              f"{barra(leitura.score, analisador.threshold)}{marca}")

    try:
        with fonte:
            for bloco in fonte.blocos():
                # Parada pedida pela interface web; vai pelo Ctrl+C para salvar WAV e JSON.
                if args.parar_com and Path(args.parar_com).exists():
                    raise KeyboardInterrupt
                if args.gravar:
                    gravado.append(bloco)
                for leitura in analisador.processar(bloco):
                    _mostrar(leitura)
                    if limite is not None and leitura.instante >= limite:
                        raise KeyboardInterrupt
            # Trecho final que não completou uma janela (comum no ASVspoof: janela de
            # 4 s, enunciados mais curtos).
            for leitura in analisador.finalizar():
                _mostrar(leitura)
    except KeyboardInterrupt:
        print("\nEncerrado.")
    finally:
        if args.gravar and gravado:
            salvar(np.concatenate(gravado), analisador.sample_rate, args.gravar)
        perdas = getattr(fonte, "descontinuidades", 0)
        if perdas:
            print(f"\n[AVISO] a captura perdeu amostras {perdas} vez(es) (o WASAPI "
                  "avisou de descontinuidade). Cada perda é um salto na forma de "
                  "onda; se forem muitas, feche outros programas de áudio e grave "
                  "de novo.")
        relatar(agregador, args.json, analisador.canal,
                ao_vivo=not args.arquivo, limiar=analisador.threshold,
                verdade=verdade, origem_limiar=escolha.origem)
    return 0


def salvar(wav: np.ndarray, sample_rate: int, destino: str) -> None:
    import soundfile as sf

    caminho = Path(destino)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    sf.write(caminho, wav, sample_rate)
    print(f"Áudio capturado: {caminho}  ({len(wav) / sample_rate:.1f}s)")
    print("  Reprocesse com --arquivo para obter exatamente o mesmo resultado.")


def gravar_json(agregador: Agregador, destino: str | Path, limiar: float | None = None,
                origem_limiar: str = "", ativo: bool = False) -> Path:
    """Grava o histórico de scores. Chamado a cada janela durante a captura
    (`ativo=True`); escrita atômica para a aba Ao vivo não ler arquivo pela metade."""
    import time

    caminho = Path(destino)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    uteis = {id(x) for x in agregador.uteis}
    dados = {
        "ativo": ativo,
        "atualizado_em": time.time(),
        "limiar": limiar,
        "origem_limiar": origem_limiar,
        "resumo": agregador.resumo(),
        "leituras": [{"indice": x.indice, "instante": x.instante, "score": x.score,
                      "rms": x.rms, "silencio": x.silencio, "util": id(x) in uteis,
                      "peso": round(x.peso, 3)}
                     for x in agregador.leituras],
    }
    tmp = caminho.with_suffix(caminho.suffix + ".tmp")
    tmp.write_text(json.dumps(dados, indent=2), encoding="utf-8")
    tmp.replace(caminho)
    return caminho


def relatar(agregador: Agregador, destino: str | None, canal=None,
            ao_vivo: bool = False, limiar: float | None = None,
            verdade: str | None = None, origem_limiar: str = "") -> None:
    resumo = agregador.resumo()
    print("\n" + "=" * 60)
    print(f"Janelas analisadas: {resumo['janelas_total']} "
          f"({resumo['janelas_uteis']} úteis, "
          f"{resumo['janelas_silencio']} em silêncio, "
          f"{resumo['janelas_canal_ruim']} com canal ruim)")
    # Com passo de meia janela, 5 janelas equivalem a 3 observações independentes.
    print(f"Janelas independentes (descontando a sobreposição): "
          f"{resumo['janelas_independentes']}")
    if canal is not None:
        print(f"Canal: {canal.descricao()}")
    if destino and resumo["score_medio"] is None:
        # Grava mesmo sem janelas, para a aba Ao vivo saber que a sessão acabou.
        gravar_json(agregador, destino, limiar, origem_limiar, ativo=False)
    if resumo["score_medio"] is None:
        print("Nenhuma janela com áudio — nada a resumir.")
        if ao_vivo and resumo["janelas_silencio"] == resumo["janelas_total"]:
            # No Windows o loopback devolve silêncio sem erro se nada toca ou se o
            # dispositivo aberto não é o que o sistema usa.
            print("\nTodas as janelas vieram em silêncio. As duas causas comuns:")
            print("  1. Não havia áudio tocando. O loopback captura a SAÍDA do")
            print("     sistema — se nada toca, não há o que capturar.")
            print("  2. O dispositivo aberto não é o que o Windows está usando")
            print("     (ex.: som indo para o fone e a captura no alto-falante).")
            print("     Liste com --listar-dispositivos e escolha com")
            print("     --dispositivo-audio \"<nome exato>\".")
        return
    print(f"Score  médio {resumo['score_medio']:.3f} (ponderado) | "
          f"{resumo['score_medio_simples']:.3f} (simples) | "
          f"mediano {resumo['score_mediano']:.3f} | "
          f"máximo {resumo['score_maximo']:.3f}")
    print(f"Peso médio das janelas: {resumo['peso_medio']:.2f} "
          "(1,00 = janela cheia de fala)")
    if limiar is not None:
        lado = ("ABAIXO do limiar (indício de voz humana)"
                if resumo["score_medio"] < limiar
                else "ACIMA do limiar (indício de síntese)")
        print(f"Média ponderada {resumo['score_medio']:.3f} {lado} "
              f"— limiar {limiar:.3f}")
        if verdade:
            decidiu = "spoof" if resumo["score_medio"] >= limiar else "bonafide"
            veredito = "COERENTE" if decidiu == verdade else "DIVERGENTE"
            print(f"Contra o rótulo verdadeiro ({verdade}): {veredito}. "
                  "Um caso não mede taxa de erro — para isso, evaluate.py.")
    print("Lembrete: score alto indica *indício* de síntese. A taxa de erro "
          "deste modelo\nem áudio de chamada ainda não foi medida.")
    if destino:
        caminho = gravar_json(agregador, destino, limiar, origem_limiar, ativo=False)
        print(f"Histórico: {caminho}")


if __name__ == "__main__":
    raise SystemExit(main())
