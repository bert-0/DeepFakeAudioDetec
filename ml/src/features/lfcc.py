"""LFCC — Linear Frequency Cepstral Coefficients (TC1 §3.3).

Ao contrário do MFCC (escala Mel, perceptual), o LFCC usa um banco de filtros
distribuídos *linearmente* ao longo da frequência, preservando melhor as
características de alta frequência associadas a artefatos de síntese de voz.

Pipeline: STFT -> espectro de potência -> banco de filtros linear -> log -> DCT.
"""

from __future__ import annotations

import librosa
import numpy as np
from scipy.fftpack import dct

from .deltas import add_deltas


def linear_filterbank(n_filter: int, n_fft: int, sample_rate: int) -> np.ndarray:
    """Banco de `n_filter` filtros triangulares igualmente espaçados em Hz.

    Devolve uma matriz (n_filter, n_fft // 2 + 1).
    """
    f_max = sample_rate / 2.0
    bin_freqs = np.linspace(0.0, f_max, n_fft // 2 + 1)
    points = np.linspace(0.0, f_max, n_filter + 2)  # bordas dos triângulos

    fb = np.zeros((n_filter, bin_freqs.size), dtype=np.float32)
    for m in range(1, n_filter + 1):
        f_left, f_center, f_right = points[m - 1], points[m], points[m + 1]
        left = (bin_freqs - f_left) / (f_center - f_left)
        right = (f_right - bin_freqs) / (f_right - f_center)
        fb[m - 1] = np.clip(np.minimum(left, right), 0.0, None)
    return fb


def compute_lfcc(wav: np.ndarray, sample_rate: int, cfg: dict) -> np.ndarray:
    """Extrai LFCC (com delta/delta-delta opcionais) de um waveform.

    Saída: shape (n_coef, n_frames), onde n_coef = n_lfcc ou 3*n_lfcc com deltas.
    """
    n_fft = cfg["n_fft"]
    spec = np.abs(
        librosa.stft(
            wav,
            n_fft=n_fft,
            win_length=cfg["win_length"],
            hop_length=cfg["hop_length"],
        )
    ) ** 2  # espectro de potência (n_fft//2+1, n_frames)

    fb = linear_filterbank(cfg["n_filter"], n_fft, sample_rate)
    filtered = fb @ spec                       # (n_filter, n_frames)
    log_energy = np.log(filtered + 1e-10)
    cepstra = dct(log_energy, type=2, axis=0, norm="ortho")[: cfg["n_lfcc"]]

    if cfg.get("deltas", False):
        cepstra = add_deltas(cepstra)
    return cepstra.astype(np.float32)
