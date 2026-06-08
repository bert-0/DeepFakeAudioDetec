"""Configuração compartilhada dos testes (pytest)."""

import os
import sys

import numpy as np
import pytest

# Permite `import src...` ao rodar pytest a partir de ml/.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def audio_cfg():
    """Config de áudio curta (1 s) para testes rápidos."""
    return {
        "sample_rate": 16000,
        "duration": 1.0,
        "trim_silence": False,
        "top_db": 30,
        "peak_normalize": True,
    }


@pytest.fixture
def feat_cfg():
    return {
        "types": ["lfcc", "spectrogram"],
        "lfcc": {"n_lfcc": 20, "n_fft": 512, "win_length": 400,
                 "hop_length": 160, "n_filter": 20, "deltas": True},
        "spectrogram": {"n_fft": 512, "win_length": 400, "hop_length": 160, "n_mels": 40},
    }


@pytest.fixture
def sine_wave():
    """1 s de um tom de 180 Hz a 16 kHz."""
    sr = 16000
    t = np.arange(sr) / sr
    return (0.5 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
