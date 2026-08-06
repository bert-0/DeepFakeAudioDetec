"""LFCC — Linear Frequency Cepstral Coefficients (TC1 §3.3).

Ao contrário do MFCC (escala Mel, perceptual), o LFCC usa um banco de filtros
distribuídos *linearmente* ao longo da frequência, preservando melhor as
características de alta frequência associadas a artefatos de síntese de voz.

Pipeline: STFT -> espectro de potência -> banco de filtros linear -> log -> DCT.
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

    Devolve uma matriz (n_filter, n_fft // 2 + 1).

    O resultado depende apenas dos três argumentos, mas a função era chamada uma
    vez por áudio — ~0,85 ms desperdiçados em cada amostra, o mesmo custo de
    *aplicar* o banco. Com `lru_cache` ele é construído uma vez por processo.

    A matriz devolvida é marcada como somente-leitura: como todos os chamadores
    recebem **o mesmo objeto**, uma escrita acidental contaminaria as chamadas
    seguintes.
    """
    f_max = sample_rate / 2.0
    bin_freqs = np.linspace(0.0, f_max, n_fft // 2 + 1)
    points = np.linspace(0.0, f_max, n_filter + 2)  # bordas dos triângulos

    # Vetorizado (sem laço em Python): cada linha m usa o triângulo
    # (points[m-1], points[m], points[m+1]).
    esq, centro, dir_ = points[:-2, None], points[1:-1, None], points[2:, None]
    subida = (bin_freqs[None, :] - esq) / (centro - esq)
    descida = (dir_ - bin_freqs[None, :]) / (dir_ - centro)
    fb = np.clip(np.minimum(subida, descida), 0.0, None).astype(np.float32)

    fb.flags.writeable = False
    return fb


def compute_lfcc(wav: np.ndarray, sample_rate: int, cfg: dict,
                 spec: np.ndarray | None = None) -> np.ndarray:
    """Extrai LFCC (com delta/delta-delta opcionais) de um waveform.

    Saída: shape (n_coef, n_frames), onde n_coef = n_lfcc ou 3*n_lfcc com deltas.

    `spec` é o espectro de potência já calculado. Configs de fusão costumam ter
    dois ramos com o mesmo `n_fft`/`win_length`/`hop_length` (é o caso de
    `fusion_v4.yaml`, com LFCC e espectrograma), e aí o STFT seria idêntico nos
    dois — 2,09 ms dos 7,27 ms de extração, medidos. O `FeatureExtractor`
    calcula uma vez e passa aqui. Sem ele, o cálculo é feito normalmente.
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
