"""Monitor de chamada ao vivo (RF04–RF07 da APS).

Escuta a **saída de áudio do sistema** e classifica o que passa, janela a
janela. Como pega o que sai da caixa de som, funciona com Microsoft Teams,
Meet, Zoom ou qualquer outro, sem publicar aplicativo em tenant nenhum.

    # ao vivo, com FUSÃO dos dois modelos (recomendado: 14,03% contra 20,18%)
    python monitor.py --config configs/fusion_v4.yaml \\
        --checkpoint checkpoints/fusion_lcnn_v4.pt \\
        --checkpoint checkpoints/baseline_lfcc_cnn_v2.pt

    # ao vivo, um modelo só
    python monitor.py --config configs/fusion_v4.yaml \\
        --checkpoint checkpoints/fusion_lcnn_v4.pt

    # ao vivo, gravando o que ouviu (é assim que se mede o canal real)
    python monitor.py --config ... --checkpoint ... --gravar chamada.wav

    # reprocessar uma gravação, de forma reprodutível
    python monitor.py --config ... --checkpoint ... --arquivo chamada.wav

    # ver os dispositivos disponíveis
    python monitor.py --listar-dispositivos

**Qual modelo usar.** O `fusion_lcnn_v4`, e não o melhor modelo do benchmark.
Em áudio limpo o `baseline_lfcc_cnn_v2` ganha por 1,19 pp, mas sob as condições
de uma chamada a ordem **se inverte** (medido com `robustness_eval.py` no eval
completo, 71.237 áudios):

    condição              v2      fusion_v4
    limpo              18,99%        20,18%
    opus 25 kbps       20,04%        22,08%
    banda estreita     35,95%        25,53%   <- v4 ganha por 10,4 pp
    ruído 5 dB SNR     42,41%        27,42%   <- v4 ganha por 15,0 pp
    degradação máx.   +23,42 pp     +7,24 pp

**O que degrada, e quanto.** O codec Opus custa pouco: +1,2 a +2,8 pp mesmo a
15 kbps, abaixo do que o Teams usa. Quem derruba é perder a **banda alta**
(+5,4 pp) e o **ruído acústico** do interlocutor (+7,2 pp a 5 dB de SNR). Uma
chamada em banda larga e ambiente silencioso é terreno viável; uma que caiu para
banda estreita, não.

**Sobre o número que ele mostra.** O limiar gravado no checkpoint foi calibrado
em áudio limpo, e fora do domínio ele se comporta de forma imprevisível: sob o
mesmo Opus a 15 kbps o recall do v2 sobe (0,72 -> 0,86) e o do v4 cai
(0,55 -> 0,39). Por isso o monitor mostra **score**, não veredito. Para um ponto
de operação confiável, recalibre no canal de destino — é para isso que serve o
`--gravar`: toque áudios de rótulo conhecido numa chamada real, capture, e
avalie o resultado.
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
    """Barra do score, com o limiar marcado por `|`.

    Sem a marca, um score de 0,048 e um de 0,71 parecem só "duas barras" — e
    nada na tela diz de que lado fica a decisão. A marca torna a comparação
    visível sem transformar o score em veredito.
    """
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
    agregador = Agregador(
        janela_s=analisador.janela.tamanho / analisador.sample_rate,
        passo_s=analisador.janela.passo / analisador.sample_rate)

    # Com um arquivo do dataset dá para mostrar o rótulo verdadeiro ao lado do
    # score. É o que transforma a execução em verificação: sem rótulo, o score
    # só mostra que o sistema opera.
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
        print(f"Threshold do checkpoint: {analisador.threshold:.4f} "
              "(calibrado no dev, em áudio LIMPO — ver aviso abaixo)")
    print("\n[AVISO] o limiar acima foi calibrado em áudio LIMPO. Medido no "
          "eval completo, o canal\n        de uma chamada custa +1 a +3 pp de "
          "EER pelo codec, +5 pp por banda\n        estreita e +7 pp por ruído — e o recall no limiar herdado varia de\n        forma imprevisível. Trate o número como indício, não como veredito.\n")

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
        """Imprime uma leitura — usada tanto no fluxo quanto na janela final."""
        agregador.adicionar(leitura)
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
                if args.gravar:
                    gravado.append(bloco)
                for leitura in analisador.processar(bloco):
                    _mostrar(leitura)
                    if limite is not None and leitura.instante >= limite:
                        raise KeyboardInterrupt
            # O fim da fonte pode deixar um trecho que não completou uma janela.
            # Num arquivo do ASVspoof isso é a regra, não a exceção: a janela
            # tem 4 s e o enunciado típico é mais curto.
            for leitura in analisador.finalizar():
                _mostrar(leitura)
    except KeyboardInterrupt:
        print("\nEncerrado.")
    finally:
        if args.gravar and gravado:
            salvar(np.concatenate(gravado), analisador.sample_rate, args.gravar)
        relatar(agregador, args.json, analisador.canal,
                ao_vivo=not args.arquivo, limiar=analisador.threshold,
                verdade=verdade)
    return 0


def salvar(wav: np.ndarray, sample_rate: int, destino: str) -> None:
    import soundfile as sf

    caminho = Path(destino)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    sf.write(caminho, wav, sample_rate)
    print(f"Áudio capturado: {caminho}  ({len(wav) / sample_rate:.1f}s)")
    print("  Reprocesse com --arquivo para obter exatamente o mesmo resultado.")


def relatar(agregador: Agregador, destino: str | None, canal=None,
            ao_vivo: bool = False, limiar: float | None = None,
            verdade: str | None = None) -> None:
    resumo = agregador.resumo()
    print("\n" + "=" * 60)
    print(f"Janelas analisadas: {resumo['janelas_total']} "
          f"({resumo['janelas_uteis']} úteis, "
          f"{resumo['janelas_silencio']} em silêncio, "
          f"{resumo['janelas_canal_ruim']} com canal ruim)")
    # As janelas se sobrepõem, então a contagem infla a confiança: com passo de
    # metade da janela, 5 janelas equivalem a 3 observações independentes.
    print(f"Janelas independentes (descontando a sobreposição): "
          f"{resumo['janelas_independentes']}")
    if canal is not None:
        print(f"Canal: {canal.descricao()}")
    if resumo["score_medio"] is None:
        print("Nenhuma janela com áudio — nada a resumir.")
        if ao_vivo and resumo["janelas_silencio"] == resumo["janelas_total"]:
            # No Windows o loopback devolve silêncio sem erro nenhum quando não
            # há nada tocando ou quando o dispositivo aberto não é o que o
            # sistema está usando. Sem esta dica o sintoma é indistinguível de
            # uma falha do modelo.
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
