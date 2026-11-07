"""Prepara a gravação de referência de uma voz para a clonagem com o XTTS.

O mesmo preparo da referência dos ensaios 1 e 2 (RESUMO_TCC §10.2.12), que foi
feito à mão: só o trecho com fala, com 0,5 s de folga em cada ponta; pico levado
a −3 dBFS (a gravação no microfone embutido chega baixa, perto de −26 dBFS); 16
bits. O ruído sobe junto com a voz: a relação entre os dois não muda.

Com várias gravações (as frases de uma pessoa que não gravou o texto de
referência), junta a fala de cada uma, medida pelo detector de frases do modelo,
com a mesma folga, e leva o conjunto ao mesmo pico (06/11/2026). As frases de 12 s
têm uns 5 s de ruído da sala que o corte pelo nível não tira (RESUMO_TCC
§10.2.44).

Uso (de dentro de ml/, no ambiente do projeto):
    python scripts/preparar_referencia.py data/vozes/L02/referencia/f01.wav
    -> data/vozes/L02/referencia/referencia_envio.wav
    python scripts/preparar_referencia.py data/vozes/calibracao/L03/*.wav \\
        --saida data/vozes/referencias/L03.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture.checagens import trechos_de_fala  # noqa: E402
from src.capture.frases import medir_frases  # noqa: E402

FOLGA_S = 0.5
PICO_DB = -3.0
#: A referência dos ensaios tinha ~70 s de fala; o XTTS aceita bem menos, mas a
#: voz clonada fica menos parecida.
FALA_MINIMA_S = 20.0


def preparar(wav: np.ndarray, taxa: int, folga_s: float = FOLGA_S,
             pico_db: float = PICO_DB) -> tuple[np.ndarray, float]:
    """Devolve (trecho preparado, ganho em dB)."""
    trechos = trechos_de_fala(wav, taxa)
    if not trechos:
        raise ValueError("nenhuma fala encontrada na gravação")
    folga = int(folga_s * taxa)
    ini, fim = max(0, trechos[0][0] - folga), min(len(wav), trechos[-1][1] + folga)
    trecho = np.asarray(wav[ini:fim], dtype=np.float64)
    ganho_db = pico_db - 20 * np.log10(np.abs(trecho).max())
    return (trecho * 10 ** (ganho_db / 20)).astype(np.float32), float(ganho_db)


def juntar_frases(wavs: list[np.ndarray], taxa: int, folga_s: float = FOLGA_S,
                  pico_db: float = PICO_DB) -> tuple[np.ndarray, float]:
    """Devolve (fala das gravações em sequência, ganho em dB): um ganho só, porque as
    frases vêm da mesma sessão e a diferença de volume entre elas é da voz."""
    folga = int(folga_s * taxa)
    partes = []
    for i, wav in enumerate(wavs, start=1):
        frases = medir_frases(wav, taxa)
        if not frases:
            raise ValueError(f"nenhuma fala encontrada na gravação {i}")
        ini = max(0, int(frases[0][0] * taxa) - folga)
        fim = min(len(wav), int(frases[-1][1] * taxa) + folga)
        partes.append(np.asarray(wav[ini:fim], dtype=np.float64))
    trecho = np.concatenate(partes)
    ganho_db = pico_db - 20 * np.log10(np.abs(trecho).max())
    return (trecho * 10 ** (ganho_db / 20)).astype(np.float32), float(ganho_db)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Prepara a referência de uma voz para a clonagem")
    p.add_argument("entrada", nargs="+",
                   help="a gravação do texto de referência (gravar_frases.py), ou várias frases")
    p.add_argument("--saida", default=None,
                   help="padrão, com uma gravação: referencia_envio.wav na mesma pasta")
    args = p.parse_args(argv)

    import soundfile as sf

    entradas = [Path(e) for e in args.entrada]
    if len(entradas) > 1 and not args.saida:
        # Ao lado das frases, a referência viraria mais uma frase da pasta.
        print("[ERRO] com várias gravações, diga onde salvar (--saida)")
        return 1
    wavs, taxas = [], set()
    for entrada in entradas:
        if not entrada.is_file():
            print(f"[ERRO] gravação não encontrada: {entrada}")
            return 1
        wav, taxa = sf.read(entrada, dtype="float32", always_2d=False)
        wavs.append(wav.mean(axis=1) if wav.ndim > 1 else wav)
        taxas.add(taxa)
    if len(taxas) > 1:
        print(f"[ERRO] gravações com taxas diferentes: {sorted(taxas)}")
        return 1
    try:
        pronto, ganho = preparar(wavs[0], taxa) if len(wavs) == 1 else juntar_frases(wavs, taxa)
    except ValueError as erro:
        print(f"[ERRO] {erro}")
        return 1
    saida = Path(args.saida) if args.saida else entradas[0].with_name("referencia_envio.wav")
    saida.parent.mkdir(parents=True, exist_ok=True)
    sf.write(saida, pronto, taxa, subtype="PCM_16")
    duracao = len(pronto) / taxa
    total = sum(len(w) for w in wavs) / taxa
    print(f"{saida}: {duracao:.1f} s de {total:.1f} s, ganho de {ganho:+.1f} dB, "
          f"{saida.stat().st_size / 1e6:.1f} MB")
    if duracao < FALA_MINIMA_S:
        print(f"[AVISO] menos de {FALA_MINIMA_S:.0f} s de fala: grave o texto de referência "
              "inteiro, o clone fica mais parecido")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
