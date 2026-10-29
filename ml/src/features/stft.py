"""Espectro de potência (|STFT|²), a etapa comum ao LFCC e ao log-mel.

Ramos com a mesma janela compartilham o cálculo (29% da extração em fusion_v4).
"""

from __future__ import annotations

import librosa
import numpy as np


def stft_params(cfg: dict) -> tuple[int, int, int]:
    """Tripla (n_fft, win_length, hop_length) que identifica o STFT de um ramo."""
    return (int(cfg["n_fft"]), int(cfg["win_length"]), int(cfg["hop_length"]))


def power_spectrum(wav: np.ndarray, n_fft: int, win_length: int,
                   hop_length: int) -> np.ndarray:
    """|STFT|², shape (n_fft // 2 + 1, n_frames)."""
    return np.abs(
        librosa.stft(wav, n_fft=n_fft, win_length=win_length,
                     hop_length=hop_length)
    ) ** 2
