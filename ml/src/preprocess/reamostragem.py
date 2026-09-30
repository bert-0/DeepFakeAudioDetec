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


def subir(x: np.ndarray, de: int, para: int = TAXA_DISPOSITIVO) -> np.ndarray:
    """Sobe a taxa (ex.: 16 -> 48 kHz) sem apagar o topo da banda original."""
    from scipy.signal import resample_poly

    up, down = _razao(de, para)
    return resample_poly(np.asarray(x, np.float64), up, down,
                         window=filtro(max(de, para))).astype(np.float32)


def descer(x: np.ndarray, de: int = TAXA_DISPOSITIVO, para: int = 16000) -> np.ndarray:
    """Desce a taxa (ex.: 48 -> 16 kHz) de uma vez, para uso fora do tempo real."""
    from scipy.signal import resample_poly

    up, down = _razao(de, para)
    return resample_poly(np.asarray(x, np.float64), up, down,
                         window=filtro(max(de, para))).astype(np.float32)


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
