"""Testes do pré-processamento de áudio."""

import numpy as np

from src.preprocess.audio import fix_length, peak_normalize, preprocess_waveform


def test_fix_length_pads_short_signal():
    wav = np.ones(100, dtype=np.float32)
    out = fix_length(wav, 1000)
    assert out.shape[0] == 1000


def test_fix_length_crops_long_signal():
    wav = np.ones(2000, dtype=np.float32)
    out = fix_length(wav, 1000)
    assert out.shape[0] == 1000


def test_peak_normalize_peak_is_one():
    wav = np.array([0.0, 0.5, -0.25], dtype=np.float32)
    out = peak_normalize(wav)
    assert np.isclose(np.max(np.abs(out)), 1.0)


def test_preprocess_waveform_fixed_length(audio_cfg, sine_wave):
    out = preprocess_waveform(sine_wave, audio_cfg)
    expected = int(audio_cfg["sample_rate"] * audio_cfg["duration"])
    assert out.shape[0] == expected
    assert out.dtype == np.float32
