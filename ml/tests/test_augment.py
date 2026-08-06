"""Testes da aumentação/perturbação de áudio."""

import numpy as np
import torch
import pytest

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


@pytest.mark.parametrize("kind,level", [("clean", None), ("noise", 10),
                                        ("gain", 3), ("shift", 100)])
def test_perturbations_accept_rng_kwarg(kind, level, sine_wave):
    """Mesma assinatura do Augmenter, para o dataset usar os dois igualmente."""
    pert = make_perturbation(kind, level)
    out = pert(sine_wave.copy(), rng=np.random.default_rng(0))
    assert out.shape == sine_wave.shape


# --------------------------------------------------------------------------- #
# Regressão: o Augmenter não pode guardar estado aleatório próprio.
#
# Com DataLoader(num_workers>0) cada worker recebe uma CÓPIA do dataset. Se o
# rng fosse atributo do objeto, todos os workers gerariam a mesma sequência —
# e ela se repetiria a cada época, quando os workers são recriados.
# --------------------------------------------------------------------------- #
AUG_CFG = {"enabled": True,
           "noise": {"prob": 1.0, "snr_db": [10, 30]},
           "gain": {"prob": 1.0, "gain_db": [-6, 6]},
           "shift": {"prob": 1.0, "max_fraction": 0.1}}


def test_augmenter_has_no_internal_rng_state():
    aug = Augmenter(AUG_CFG, seed=42)
    assert not any(isinstance(v, np.random.Generator) for v in vars(aug).values()), \
        "o Augmenter voltou a guardar um rng — quebra com num_workers>0"


def test_copies_of_augmenter_differ_when_given_different_rngs(sine_wave):
    """Simula dois workers processando índices diferentes."""
    import copy

    base = Augmenter(AUG_CFG, seed=42)
    w0, w1 = copy.deepcopy(base), copy.deepcopy(base)
    a = w0(sine_wave.copy(), rng=np.random.default_rng((42, 1, 0)))   # idx 0
    b = w1(sine_wave.copy(), rng=np.random.default_rng((42, 1, 1)))   # idx 1
    assert not np.array_equal(a, b)


def test_same_sample_differs_between_epochs(sine_wave):
    aug = Augmenter(AUG_CFG, seed=42)
    ep1 = aug(sine_wave.copy(), rng=np.random.default_rng((42, 1, 7)))
    ep2 = aug(sine_wave.copy(), rng=np.random.default_rng((42, 2, 7)))
    assert not np.array_equal(ep1, ep2)


def test_same_sample_same_epoch_is_reproducible(sine_wave):
    aug = Augmenter(AUG_CFG, seed=42)
    a = aug(sine_wave.copy(), rng=np.random.default_rng((42, 3, 5)))
    b = aug(sine_wave.copy(), rng=np.random.default_rng((42, 3, 5)))
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Perturbações precisam atravessar os workers do DataLoader.
#
# Enquanto eram `lambda`, o pickle não as serializava e a avaliação de robustez
# ficava presa a num_workers=0 — um processo só para 71.237 áudios por condição.
# --------------------------------------------------------------------------- #
def test_perturbation_is_picklable():
    import pickle

    for kind, level in [("clean", None), ("noise", 10), ("gain", -6), ("shift", 100)]:
        clone = pickle.loads(pickle.dumps(make_perturbation(kind, level, seed=3)))
        assert clone.kind == kind


def test_perturbation_rejects_unknown_kind():
    with pytest.raises(ValueError, match="desconhecida"):
        make_perturbation("reverb", 1)


def test_noise_is_reproducible_per_sample(sine_wave):
    """O ruído tem de vir do rng da amostra, não de um gerador compartilhado.

    Com um gerador compartilhado, o ruído de cada áudio dependeria da ORDEM em
    que ele fosse processado — e essa ordem muda com o nº de workers, o que faria
    o resultado da robustez variar conforme a máquina.
    """
    pert = make_perturbation("noise", 10, seed=7)
    a = pert(sine_wave, rng=np.random.default_rng((0, 0, 42)))
    b = pert(sine_wave, rng=np.random.default_rng((0, 0, 42)))
    np.testing.assert_array_equal(a, b)

    outro = pert(sine_wave, rng=np.random.default_rng((0, 0, 43)))
    assert not np.array_equal(a, outro), "amostras diferentes receberam o mesmo ruído"


def test_robustness_is_independent_of_worker_count(audio_cfg, feat_cfg):
    """Mesmos scores com 0 e com 2 workers — o ponto da mudança acima."""
    from torch.utils.data import DataLoader

    from src.data import SmokeDataset
    from src.features import FeatureExtractor

    def lfccs(num_workers):
        ds = SmokeDataset(8, audio_cfg, FeatureExtractor(audio_cfg, feat_cfg), seed=1,
                          augmenter=make_perturbation("noise", 10, seed=7))
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=num_workers)
        return torch.cat([lote["lfcc"] for lote, _ in loader])

    assert torch.equal(lfccs(0), lfccs(2))


def test_deterministic_perturbations_ignore_the_sample_rng(sine_wave):
    """Ganho e deslocamento têm nível fixo: o rng não pode alterá-los."""
    for kind, level in [("gain", -6), ("shift", 100), ("clean", None)]:
        pert = make_perturbation(kind, level)
        a = pert(sine_wave, rng=np.random.default_rng(1))
        b = pert(sine_wave, rng=np.random.default_rng(999))
        np.testing.assert_array_equal(a, b)
