"""Prepara a gravação de referência de uma voz para a clonagem com o XTTS.

O mesmo preparo da referência dos ensaios 1 e 2 (RESUMO_TCC §10.2.12), que foi
feito à mão: só o trecho com fala, com 0,5 s de folga em cada ponta; pico levado
a −3 dBFS (a gravação no microfone embutido chega baixa, perto de −26 dBFS); 16
bits. O ruído sobe junto com a voz: a relação entre os dois não muda.

Uso (de dentro de ml/, no ambiente do projeto):
    python scripts/preparar_referencia.py data/vozes/L02/referencia/f01.wav
    -> data/vozes/L02/referencia/referencia_envio.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture.checagens import trechos_de_fala  # noqa: E402

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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Prepara a referência de uma voz para a clonagem")
    p.add_argument("entrada", help="a gravação do texto de referência (gravar_frases.py)")
    p.add_argument("--saida", default=None,
                   help="padrão: referencia_envio.wav na mesma pasta da entrada")
    args = p.parse_args(argv)

    import soundfile as sf

    entrada = Path(args.entrada)
    if not entrada.is_file():
        print(f"[ERRO] gravação não encontrada: {entrada}")
        return 1
    wav, taxa = sf.read(entrada, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    try:
        pronto, ganho = preparar(wav, taxa)
    except ValueError as erro:
        print(f"[ERRO] {erro}")
        return 1
    saida = Path(args.saida) if args.saida else entrada.with_name("referencia_envio.wav")
    sf.write(saida, pronto, taxa, subtype="PCM_16")
    duracao = len(pronto) / taxa
    print(f"{saida}: {duracao:.1f} s de {len(wav) / taxa:.1f} s, ganho de {ganho:+.1f} dB, "
          f"{saida.stat().st_size / 1e6:.1f} MB")
    if duracao < FALA_MINIMA_S:
        print(f"[AVISO] menos de {FALA_MINIMA_S:.0f} s de fala: grave o texto de referência "
              "inteiro, o clone fica mais parecido")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
