"""Espectro de potência — a etapa comum a todas as features deste projeto.

Tanto o LFCC quanto o espectrograma log-mel partem do mesmo |STFT|². Quando um
config de fusão usa os dois ramos com os mesmos parâmetros de janela — o caso de
`fusion_v4.yaml`, com `n_fft: 512`, `win_length: 400`, `hop_length: 160` nos
dois — o cálculo era feito duas vezes com resultado idêntico.

Medido: 2,09 ms dos 7,27 ms de extração por áudio, ou 29%. Como as configs v3/v4
ligam `random_crop` e `augment`, a partição de treino não tem cache (features
estocásticas não podem ser congeladas em disco), então esse desperdício era pago
nas 50 épocas, não uma vez só.
"""

from __future__ import annotations

import librosa
import numpy as np


def stft_params(cfg: dict) -> tuple[int, int, int]:
    """Chave de identidade do STFT de um bloco de config de feature.

    Dois ramos com a mesma tripla produzem espectros idênticos e podem
    compartilhar o cálculo.
    """
    return (int(cfg["n_fft"]), int(cfg["win_length"]), int(cfg["hop_length"]))


def power_spectrum(wav: np.ndarray, n_fft: int, win_length: int,
                   hop_length: int) -> np.ndarray:
    """|STFT|², shape (n_fft // 2 + 1, n_frames)."""
    return np.abs(
        librosa.stft(wav, n_fft=n_fft, win_length=win_length,
                     hop_length=hop_length)
    ) ** 2
