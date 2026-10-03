"""Teste do microfone da coleta: o caminho processa o áudio?

Grava a mesma cena duas vezes, em WASAPI exclusivo e compartilhado, com um ruído
constante ao fundo (ventilador, ou ruído branco tocando num celular parado):
silêncio, uma frase, silêncio. Depois compara:

- o ruído constante cai durante a gravação: supressão de ruído no caminho. No
  exclusivo ela é do driver ou do hardware, e o Windows não a desliga;
- o compartilhado tem menos ruído, outro nível de fala ou perde 7,6–8 kHz: os
  aprimoramentos ou a conversão de taxa do Windows estão no caminho.

Uso:
    python scripts/testar_microfone.py [--dispositivo "<nome>"] [--segundos 12]
    python scripts/testar_microfone.py --analisar exclusivo.wav compartilhado.wav

As gravações vão para outputs/teste_microfone/, fora do Git. É teste de
equipamento: nada disso entra no conjunto de dados.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture.checagens import checar, energia_por_faixa  # noqa: E402
from src.capture.sources import CaptureError, MicrofoneWasapiSource  # noqa: E402

TAXA = 48000
#: Abaixo disso é a variação natural entre duas leituras da mesma frase.
LIMIAR_DB = 6.0
#: Ruído de fundo mais baixo que isso nos dois modos: ou não havia ruído
#: constante, ou o hardware o remove; o teste não separa os dois casos.
RUIDO_MINIMO_DBFS = -65.0
SILENCIO_S = 3.0
FRASE = ("O detector compara a voz gravada com a voz sintética, frase por frase, "
         "em janelas de quatro segundos.")


def _db(x) -> np.ndarray:
    return 20 * np.log10(np.maximum(x, 1e-12))


def medir(wav: np.ndarray, sr: int) -> dict:
    """Níveis da gravação em dBFS e a energia por faixa."""
    import librosa

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    hop = 512
    rms = _db(librosa.feature.rms(y=wav, frame_length=2048, hop_length=hop)[0])
    t = np.arange(len(rms)) * hop / sr
    dur = len(wav) / sr

    def mediana(de, ate):
        sel = rms[(t >= de) & (t < ate)]
        return float(np.median(sel)) if sel.size else float("nan")

    return {
        "pico": float(_db(np.abs(wav).max())) if wav.size else float("-inf"),
        "fala": float(np.percentile(rms, 95)),
        # logo no começo, antes de uma supressão de ruído convergir
        "ruido_inicio": mediana(0.25, 0.75),
        "ruido_fim": mediana(dur - 2.0, dur - 0.5),
        "faixas": energia_por_faixa(wav, sr),
        "checagens": checar(wav, sr),
    }


def comparar(exclusivo: dict | None, compartilhado: dict | None) -> list[str]:
    """Indícios de processamento no caminho, um por linha."""
    saida = []
    for modo, m in (("exclusivo", exclusivo), ("compartilhado", compartilhado)):
        if m is None:
            continue
        queda = m["ruido_fim"] - m["ruido_inicio"]
        if queda <= -LIMIAR_DB:
            onde = "driver ou hardware" if modo == "exclusivo" else "Windows, driver ou hardware"
            saida.append(f"{modo}: o ruído constante caiu {-queda:.0f} dB durante a gravação; "
                         f"supressão de ruído no caminho ({onde})")
    if exclusivo is None or compartilhado is None:
        return saida

    if max(exclusivo["ruido_fim"], compartilhado["ruido_fim"]) < RUIDO_MINIMO_DBFS:
        saida.append(f"ruído de fundo abaixo de {RUIDO_MINIMO_DBFS:.0f} dBFS nos dois modos: "
                     "confira se o ruído constante estava tocando; se estava, o hardware o remove")
    dif_ruido = compartilhado["ruido_fim"] - exclusivo["ruido_fim"]
    if dif_ruido <= -LIMIAR_DB:
        saida.append(f"o compartilhado tem {-dif_ruido:.0f} dB menos ruído: aprimoramentos do "
                     "Windows ligados (supressão de ruído)")
    dif_fala = compartilhado["fala"] - exclusivo["fala"]
    if abs(dif_fala) >= LIMIAR_DB:
        saida.append(f"o nível da fala muda {dif_fala:+.0f} dB do exclusivo para o compartilhado: "
                     "ganho ou AGC do Windows")
    alta_e = exclusivo["faixas"].get("7,6–8 kHz", 0.0)
    alta_c = compartilhado["faixas"].get("7,6–8 kHz", 0.0)
    if alta_e > 1e-5 and alta_c < alta_e / 4:
        saida.append("o compartilhado perde a faixa de 7,6–8 kHz: o Windows converte a taxa "
                     "(formato do dispositivo diferente de 48 kHz)")
    if not saida:
        saida.append(f"nenhuma diferença de {LIMIAR_DB:.0f} dB ou mais: os dois modos entregam o "
                     "mesmo áudio e o ruído constante não foi suprimido")
    return saida


def relatar(medidas: dict[str, dict], indicios: list[str]) -> None:
    modos = list(medidas)
    print("\n" + f"{'':22s}" + "".join(f"{m:>16s}" for m in modos))
    for chave, nome in (("pico", "pico"), ("fala", "fala (p95)"),
                        ("ruido_inicio", "ruído no começo"), ("ruido_fim", "ruído no fim")):
        print(f"{nome:22s}" + "".join(f"{medidas[m][chave]:11.1f} dBFS" for m in modos))
    for faixa in medidas[modos[0]]["faixas"]:
        print(f"{'energia ' + faixa:22s}"
              + "".join(f"{100 * medidas[m]['faixas'].get(faixa, 0.0):15.2f}%" for m in modos))
    for i, c in enumerate(medidas[modos[0]]["checagens"]):
        estados = ["ok" if medidas[m]["checagens"][i].ok else "regravar" for m in modos]
        print(f"{'checagem ' + c.nome:22s}" + "".join(f"{e:>16s}" for e in estados))
    print("\nIndícios:")
    for linha in indicios:
        print(f"  - {linha}")


def gravar(segundos: float, exclusivo: bool, dispositivo: str | None) -> tuple[np.ndarray, str, str]:
    """Grava `segundos` a 48 kHz, sem conversão, e devolve (áudio, modo, nome)."""
    fonte = MicrofoneWasapiSource(TAXA, dispositivo, exclusivo=exclusivo)
    total, blocos, avisos = int(segundos * TAXA), [], [SILENCIO_S, segundos - SILENCIO_S]
    for bloco in fonte.blocos():
        blocos.append(bloco)
        decorrido = sum(len(b) for b in blocos) / TAXA
        if avisos and decorrido >= avisos[0]:
            print("    >> leia a frase" if len(avisos) == 2 else "    >> silêncio até o fim")
            avisos.pop(0)
        if decorrido * TAXA >= total:
            break
    fonte.fechar()
    return np.concatenate(blocos)[:total], fonte.modo, fonte.nome


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Teste do microfone: exclusivo x compartilhado")
    p.add_argument("--dispositivo", default=None, help="parte do nome do microfone (padrão do Windows)")
    p.add_argument("--segundos", type=float, default=12.0)
    p.add_argument("--pasta", default="outputs/teste_microfone")
    p.add_argument("--analisar", nargs=2, metavar=("EXCLUSIVO", "COMPARTILHADO"), default=None,
                   help="só analisa duas gravações já feitas")
    args = p.parse_args(argv)

    import soundfile as sf

    if args.analisar:
        medidas = {}
        for modo, caminho in zip(("exclusivo", "compartilhado"), args.analisar):
            wav, sr = sf.read(caminho, dtype="float32", always_2d=True)
            medidas[modo] = medir(wav.mean(axis=1), sr)
        relatar(medidas, comparar(medidas["exclusivo"], medidas["compartilhado"]))
        return 0

    pasta = Path(args.pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    print("Deixe um ruído constante tocando ao fundo (ventilador ou ruído branco no "
          "celular), no mesmo lugar e volume nas duas gravações.")
    print(f"Frase: \"{FRASE}\"")
    medidas, carimbo = {}, time.strftime("%Y%m%d_%H%M%S")
    for pedido in ("exclusivo", "compartilhado"):
        input(f"\n[{pedido}] Enter para gravar {args.segundos:.0f} s "
              f"({SILENCIO_S:.0f} s de silêncio, a frase, silêncio)...")
        try:
            wav, modo, nome = gravar(args.segundos, pedido == "exclusivo", args.dispositivo)
        except CaptureError as erro:
            print(f"[ERRO] {erro}")
            return 1
        destino = pasta / f"{carimbo}_{pedido}.wav"
        sf.write(destino, wav, TAXA)
        print(f"    {nome} (WASAPI {modo}) -> {destino}")
        if modo != pedido:
            print(f"[AVISO] pedido {pedido}, o driver abriu {modo}: a comparação entre os "
                  "modos não vale")
        medidas[pedido] = medir(wav, TAXA)
    relatar(medidas, comparar(medidas["exclusivo"], medidas["compartilhado"]))
    print("\nOuça também os dois arquivos: se o ruído some ou \"respira\" junto com a fala, "
          "há supressão de ruído.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
