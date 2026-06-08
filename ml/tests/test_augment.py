"""Testes da aumentação/perturbação de áudio."""

import numpy as np

from src.preprocess.augment import Augmenter, add_noise, apply_gain, make_perturbation, time_shift


def test_add_noise_changes_signal_keeps_shape(sine_wave):
    rng = np.random.default_rng(0)
    noisy = add_noise(sine_wave, snr_db=10, rng=rng)
    assert noisy.shape == sine_wave.shape
    assert not np.allclose(noisy, sine_wave)


def test_add_noise_snr_is_approximately_correct(sine_wave):
    rng = np.random.default_rng(0)
    snr_db = 10
    noisy = add_noise(sine_wave, snr_db=snr_db, rng=rng)
    noise = noisy - sine_wave
    measured = 10 * np.log10(np.mean(sine_wave ** 2) / np.mean(noise ** 2))
    assert abs(measured - snr_db) < 1.5  # tolerância estatística


def test_apply_gain_plus6db_roughly_doubles(sine_wave):
    out = apply_gain(sine_wave, 6.0)
    ratio = np.max(np.abs(out)) / np.max(np.abs(sine_wave))
    assert 1.9 < ratio < 2.1  # 10^(6/20) ~= 1.995


def test_time_shift_preserves_length(sine_wave):
    out = time_shift(sine_wave, 100)
    assert out.shape == sine_wave.shape


def test_augmenter_disabled_is_noop(sine_wave):
    aug = Augmenter({"enabled": False}, seed=0)
    assert np.array_equal(aug(sine_wave), sine_wave)


def test_augmenter_enabled_keeps_shape(sine_wave):
    cfg = {"enabled": True,
           "noise": {"prob": 1.0, "snr_db": [10, 10]},
           "gain": {"prob": 1.0, "gain_db": [3, 3]},
           "shift": {"prob": 1.0, "max_fraction": 0.1}}
    aug = Augmenter(cfg, seed=0)
    out = aug(sine_wave.copy())
    assert out.shape == sine_wave.shape


def test_make_perturbation_clean_is_identity(sine_wave):
    pert = make_perturbation("clean", None)
    assert np.array_equal(pert(sine_wave), sine_wave)
