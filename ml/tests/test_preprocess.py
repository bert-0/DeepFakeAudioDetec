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


def test_fix_length_without_rng_is_deterministic_start():
    wav = np.arange(2000, dtype=np.float32)
    out = fix_length(wav, 1000)
    assert np.array_equal(out, wav[:1000])  # sempre o início


def test_random_crop_changes_position_and_keeps_length():
    wav = np.arange(2000, dtype=np.float32)
    crops = {fix_length(wav, 1000, rng=np.random.default_rng(s))[0] for s in range(20)}
    assert len(crops) > 1                    # posições diferentes entre execuções
    for s in range(5):
        assert fix_length(wav, 1000, rng=np.random.default_rng(s)).shape[0] == 1000


def test_random_crop_is_reproducible_with_same_seed():
    wav = np.arange(2000, dtype=np.float32)
    a = fix_length(wav, 1000, rng=np.random.default_rng(7))
    b = fix_length(wav, 1000, rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_random_crop_stays_in_bounds():
    wav = np.arange(1500, dtype=np.float32)
    for s in range(50):
        out = fix_length(wav, 1000, rng=np.random.default_rng(s))
        assert out.shape[0] == 1000
        assert out[0] <= 500                 # start máximo = 1500-1000


def test_random_crop_ignored_when_signal_is_short():
    """Sinal menor que o alvo: faz padding por repetição, sem usar o rng."""
    wav = np.ones(100, dtype=np.float32)
    out = fix_length(wav, 1000, rng=np.random.default_rng(0))
    assert out.shape[0] == 1000


def test_preprocess_waveform_fixed_length(audio_cfg, sine_wave):
    out = preprocess_waveform(sine_wave, audio_cfg)
    expected = int(audio_cfg["sample_rate"] * audio_cfg["duration"])
    assert out.shape[0] == expected
    assert out.dtype == np.float32
