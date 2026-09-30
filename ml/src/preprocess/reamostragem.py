"""Conversão 16 <-> 48 kHz com FIR longo, que preserva a banda até ~7,9 kHz.

O `soxr` corta a partir de ~7,5 kHz (-8 dB em 7,6-7,8 kHz, -33 dB acima de
7,8), e o baseline_v2 depende dessa faixa. Um FIR de 2047 coeficientes com
corte em 7,9 kHz deixa 7,6-7,8 kHz intacto e 7,8-7,95 kHz a -2 dB.
"""

from __future__ import annotations

from functools import lru_cache
from math import gcd

import numpy as np

TAXA_DISPOSITIVO = 48000
CORTE_HZ = 7900.0
COEFICIENTES = 2047


@lru_cache(maxsize=4)
def filtro(taxa_alta: int = TAXA_DISPOSITIVO, corte: float = CORTE_HZ,
           coeficientes: int = COEFICIENTES) -> np.ndarray:
    from scipy.signal import firwin

    h = firwin(coeficientes, corte, fs=taxa_alta, window=("kaiser", 10.0))
    h.setflags(write=False)
    return h


def _razao(de: int, para: int) -> tuple[int, int]:
    g = gcd(int(de), int(para))
    return int(para) // g, int(de) // g


def _polifasica(x: np.ndarray, up: int, down: int, h: np.ndarray) -> np.ndarray:
    """O mesmo que `resample_poly(x, up, down, window=h)`, por FFT e em fases.

    Só razões 1:n ou n:1. Cada fase convolui com 1/n do filtro, na taxa baixa:
    ~15x mais rápido que o resample_poly com 2047 coeficientes.
    """
    from scipy.signal import oaconvolve

    x = np.asarray(x, np.float64)
    atraso = (len(h) - 1) // 2
    if down == 1:                       # subir: saída intercalada das fases
        L = up
        fases = [oaconvolve(x, h[r::L]) for r in range(L)]
        cheio = np.zeros(max(len(f) for f in fases) * L)
        for r, f in enumerate(fases):
            cheio[r::L][: len(f)] = f
        return (L * cheio)[atraso: atraso + L * len(x)]
    M = down                            # descer: soma das fases
    w = np.concatenate([np.zeros(M - 1), x])
    total = None
    for r in range(M):
        wr = w[M - 1 - r::M]
        c = oaconvolve(wr, h[r::M])
        total = c if total is None else total[: len(c)] + c[: len(total)]
    inicio = atraso // M
    return total[inicio: inicio + -(-len(x) // M)]


def _reamostrar(x: np.ndarray, de: int, para: int) -> np.ndarray:
    up, down = _razao(de, para)
    h = filtro(max(de, para))
    if up == 1 or down == 1:
        return _polifasica(x, up, down, h).astype(np.float32)
    from scipy.signal import resample_poly

    return resample_poly(np.asarray(x, np.float64), up, down, window=h).astype(np.float32)


def subir(x: np.ndarray, de: int, para: int = TAXA_DISPOSITIVO) -> np.ndarray:
    """Sobe a taxa (ex.: 16 -> 48 kHz) sem apagar o topo da banda original."""
    return _reamostrar(x, de, para)


def descer(x: np.ndarray, de: int = TAXA_DISPOSITIVO, para: int = 16000) -> np.ndarray:
    """Desce a taxa (ex.: 48 -> 16 kHz) de uma vez, para uso fora do tempo real."""
    return _reamostrar(x, de, para)


def ida_e_volta(x: np.ndarray, sr: int, taxa: int = TAXA_DISPOSITIVO) -> np.ndarray:
    """O caminho tocar -> dispositivo -> captura, com o FIR nas duas pontas."""
    return descer(subir(x, sr, taxa), taxa, sr)[: len(x)]


class DecimadorFIR:
    """Descida contínua bloco a bloco (captura ao vivo), para razão inteira.

    Guarda o final do bloco anterior, então não há borda entre blocos. Atraso
    fixo de (coeficientes-1)/2 amostras na taxa alta (~21 ms), irrelevante ao vivo.
    """

    def __init__(self, de: int = TAXA_DISPOSITIVO, para: int = 16000):
        if de % para:
            raise ValueError(f"razão não inteira: {de} -> {para}")
        self.m = de // para
        self.h = filtro(de)[::-1].copy()
        self._resto = np.zeros(len(self.h) - 1)
        self._fase = 0

    def __call__(self, bloco: np.ndarray) -> np.ndarray:
        from numpy.lib.stride_tricks import sliding_window_view

        z = np.concatenate([self._resto, np.asarray(bloco, np.float64)])
        janelas = sliding_window_view(z, len(self.h))[self._fase::self.m]
        saida = janelas @ self.h
        self._fase = (self._fase - len(bloco)) % self.m
        self._resto = z[-(len(self.h) - 1):]
        return saida.astype(np.float32)
