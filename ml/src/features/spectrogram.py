"""Espectrograma log-mel (TC1 §3.2).

Segundo ramo dos modelos de fusão/atenção e visualização ao operador (RF10).
"""

from __future__ import annotations

from functools import lru_cache

import librosa
import numpy as np

from .stft import power_spectrum


@lru_cache(maxsize=8)
def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Banco mel memoizado (o cache do librosa é no-op sem `LIBROSA_CACHE_DIR`)."""
    fb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
    fb.flags.writeable = False   # o mesmo objeto vai para todos os chamadores
    return fb


def compute_log_mel(wav: np.ndarray, sample_rate: int, cfg: dict,
                    spec: np.ndarray | None = None) -> np.ndarray:
    """Extrai o espectrograma log-mel em dB: shape (n_mels, n_frames).

    `spec`, se passado, é o espectro de potência já calculado.
    """
    if spec is None:
        spec = power_spectrum(wav, cfg["n_fft"], cfg["win_length"],
                              cfg["hop_length"])
    fb = mel_filterbank(sample_rate, cfg["n_fft"], cfg["n_mels"])
    mel = fb @ spec
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)
