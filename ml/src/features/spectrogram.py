"""Espectrograma log-mel (TC1 §3.2).

Representa a distribuição de energia do sinal ao longo do tempo e da frequência.
Usado como segundo ramo nos incrementos de fusão e atenção, e também para a
visualização espectral exibida ao operador (RF10 da APS).
"""

from __future__ import annotations

import librosa
import numpy as np


def compute_log_mel(wav: np.ndarray, sample_rate: int, cfg: dict) -> np.ndarray:
    """Extrai o espectrograma log-mel de um waveform.

    Saída: shape (n_mels, n_frames), em decibéis.
    """
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sample_rate,
        n_fft=cfg["n_fft"],
        win_length=cfg["win_length"],
        hop_length=cfg["hop_length"],
        n_mels=cfg["n_mels"],
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)
