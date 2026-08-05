"""Espectrograma log-mel (TC1 §3.2).

Representa a distribuição de energia do sinal ao longo do tempo e da frequência.
Usado como segundo ramo nos incrementos de fusão e atenção, e também para a
visualização espectral exibida ao operador (RF10 da APS).
"""

from __future__ import annotations

from functools import lru_cache

import librosa
import numpy as np


@lru_cache(maxsize=8)
def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Banco mel memoizado.

    `librosa.feature.melspectrogram` reconstrói o banco a cada chamada: o
    decorador `@cache` do librosa é um `Memory(location=None)`, ou seja, um
    no-op enquanto `LIBROSA_CACHE_DIR` não estiver definido. Custava ~0,85 ms
    por áudio.
    """
    fb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
    fb.flags.writeable = False   # o mesmo objeto vai para todos os chamadores
    return fb


def compute_log_mel(wav: np.ndarray, sample_rate: int, cfg: dict) -> np.ndarray:
    """Extrai o espectrograma log-mel de um waveform.

    Saída: shape (n_mels, n_frames), em decibéis.
    """
    spec = np.abs(
        librosa.stft(y=wav, n_fft=cfg["n_fft"], win_length=cfg["win_length"],
                     hop_length=cfg["hop_length"])
    ) ** 2
    fb = mel_filterbank(sample_rate, cfg["n_fft"], cfg["n_mels"])
    mel = fb @ spec
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)
