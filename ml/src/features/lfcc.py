"""LFCC — Linear Frequency Cepstral Coefficients (TC1 §3.3).

Filtros lineares em frequência (não mel), preservando a alta frequência.
Pipeline: STFT -> potência -> banco de filtros linear -> log -> DCT.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.fftpack import dct

from .deltas import add_deltas
from .stft import power_spectrum


@lru_cache(maxsize=8)
def linear_filterbank(n_filter: int, n_fft: int, sample_rate: int) -> np.ndarray:
    """Banco de `n_filter` filtros triangulares igualmente espaçados em Hz.

    Devolve (n_filter, n_fft // 2 + 1), memoizado e somente-leitura.
    """
    f_max = sample_rate / 2.0
    bin_freqs = np.linspace(0.0, f_max, n_fft // 2 + 1)
    points = np.linspace(0.0, f_max, n_filter + 2)  # bordas dos triângulos

    # Linha m: triângulo (points[m-1], points[m], points[m+1]).
    esq, centro, dir_ = points[:-2, None], points[1:-1, None], points[2:, None]
    subida = (bin_freqs[None, :] - esq) / (centro - esq)
    descida = (dir_ - bin_freqs[None, :]) / (dir_ - centro)
    fb = np.clip(np.minimum(subida, descida), 0.0, None).astype(np.float32)

    fb.flags.writeable = False   # o mesmo objeto vai para todos os chamadores
    return fb


def compute_lfcc(wav: np.ndarray, sample_rate: int, cfg: dict,
                 spec: np.ndarray | None = None) -> np.ndarray:
    """Extrai LFCC (deltas opcionais): shape (n_lfcc ou 3*n_lfcc, n_frames).

    `spec`, se passado, é o espectro de potência já calculado.
    """
    n_fft = cfg["n_fft"]
    if spec is None:
        spec = power_spectrum(wav, n_fft, cfg["win_length"], cfg["hop_length"])

    fb = linear_filterbank(cfg["n_filter"], n_fft, sample_rate)
    filtered = fb @ spec                       # (n_filter, n_frames)
    log_energy = np.log(filtered + 1e-10)
    cepstra = dct(log_energy, type=2, axis=0, norm="ortho")[: cfg["n_lfcc"]]

    if cfg.get("deltas", False):
        cepstra = add_deltas(cepstra)
    return cepstra.astype(np.float32)
