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
def linear_filterbank(n_filter: int, n_fft: int, sample_rate: int,
                      f_min: float = 0.0, f_max: float | None = None) -> np.ndarray:
    """Banco de `n_filter` filtros triangulares igualmente espaçados em Hz,
    entre `f_min` e `f_max` (padrão: a banda inteira).

    Fora da faixa os filtros valem zero: o LFCC não vê o que fica acima de
    `f_max` (o Incremento 4 testa 0 a 4 kHz contra a dependência de 7,6–8 kHz,
    RESUMO_TCC §10.2.11). Devolve (n_filter, n_fft // 2 + 1), memoizado e
    somente-leitura.
    """
    nyquist = sample_rate / 2.0
    f_max = nyquist if f_max is None else float(f_max)
    if not 0.0 <= f_min < f_max <= nyquist:
        raise ValueError(f"LFCC: precisa 0 <= f_min < f_max <= {nyquist:g} Hz, "
                         f"não f_min={f_min:g}, f_max={f_max:g}")
    bin_freqs = np.linspace(0.0, nyquist, n_fft // 2 + 1)
    points = np.linspace(f_min, f_max, n_filter + 2)  # bordas dos triângulos

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

    fb = linear_filterbank(cfg["n_filter"], n_fft, sample_rate,
                           float(cfg.get("f_min", 0.0)), cfg.get("f_max"))
    filtered = fb @ spec                       # (n_filter, n_frames)
    log_energy = np.log(filtered + 1e-10)
    cepstra = dct(log_energy, type=2, axis=0, norm="ortho")[: cfg["n_lfcc"]]

    if cfg.get("deltas", False):
        cepstra = add_deltas(cepstra)
    return cepstra.astype(np.float32)
