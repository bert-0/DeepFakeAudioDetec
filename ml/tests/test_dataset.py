"""Testes de parsing de protocolo e dos datasets."""

import pytest

from src.data.dataset import SmokeDataset, build_dataset, parse_protocol
from src.features import FeatureExtractor


def test_parse_protocol_with_systems(tmp_path):
    proto = tmp_path / "proto.txt"
    proto.write_text(
        "LA_0001 utt_0 - - bonafide\n"
        "LA_0001 utt_1 - A07 spoof\n"
        "LA_0002 utt_2 - A17 spoof\n"
    )
    from src.data.dataset import parse_protocol_with_systems

    items = parse_protocol_with_systems(proto)
    assert items == [("utt_0", 0, "-"), ("utt_1", 1, "A07"), ("utt_2", 1, "A17")]


def test_smoke_dataset_exposes_system_ids(audio_cfg, feat_cfg):
    extractor = FeatureExtractor(audio_cfg, feat_cfg)
    ds = SmokeDataset(6, audio_cfg, extractor, seed=1)
    assert len(ds.system_ids) == 6
    # bonafide sempre "-"; spoof sempre um Axx
    for label, system in zip(ds.labels, ds.system_ids):
        assert (system == "-") == (label == 0)


def test_parse_protocol(tmp_path):
    proto = tmp_path / "proto.txt"
    proto.write_text(
        "LA_0001 utt_0 - - bonafide\n"
        "LA_0001 utt_1 - A07 spoof\n"
        "linha_invalida\n"  # deve ser ignorada
    )
    items = parse_protocol(proto)
    assert items == [("utt_0", 0), ("utt_1", 1)]


def test_smoke_dataset_balanced(audio_cfg, feat_cfg):
    extractor = FeatureExtractor(audio_cfg, feat_cfg)
    ds = SmokeDataset(10, audio_cfg, extractor, seed=1)
    assert len(ds) == 10
    assert ds.labels.count(0) == ds.labels.count(1) == 5
    features, label = ds[0]
    assert set(features) == {"lfcc", "spectrogram"}
    assert label in (0, 1)
    assert len(ds.ids) == 10


def test_build_dataset_smoke_returns_smoke(audio_cfg, feat_cfg):
    config = {"audio": audio_cfg, "features": feat_cfg,
              "smoke": {"n_train": 8, "n_dev": 4, "n_eval": 4}}
    extractor = FeatureExtractor(audio_cfg, feat_cfg)
    ds = build_dataset(config, "train", extractor, smoke=True)
    assert isinstance(ds, SmokeDataset)
    assert len(ds) == 8


def test_cache_fingerprint_changes_with_feature_params(audio_cfg, feat_cfg):
    """Mudar n_filter PRECISA invalidar o cache, senão reusa features antigas."""
    from src.data.dataset import _config_fingerprint

    a = FeatureExtractor(audio_cfg, feat_cfg)
    other = {**feat_cfg, "lfcc": {**feat_cfg["lfcc"], "n_filter": 70}}
    b = FeatureExtractor(audio_cfg, other)
    assert _config_fingerprint(audio_cfg, a) != _config_fingerprint(audio_cfg, b)


def test_cache_fingerprint_stable_for_same_config(audio_cfg, feat_cfg):
    from src.data.dataset import _config_fingerprint

    a = FeatureExtractor(audio_cfg, feat_cfg)
    b = FeatureExtractor(audio_cfg, {**feat_cfg})
    assert _config_fingerprint(audio_cfg, a) == _config_fingerprint(audio_cfg, b)


def test_cache_fingerprint_ignores_augment_and_random_crop(audio_cfg, feat_cfg):
    """Configs iguais salvo aumentação/recorte devem COMPARTILHAR o cache.

    Esses parâmetros não podem alterar o conteúdo cacheado: quando ativos, o
    cache é desligado para aquele dataset, e dev/eval nunca os recebem. Incluí-los
    no fingerprint duplicaria pastas idênticas (~11 GB cada, no ASVspoof LA).
    """
    from src.data.dataset import _config_fingerprint

    sem = {**audio_cfg, "random_crop": False, "augment": {"enabled": False}}
    com = {**audio_cfg, "random_crop": True,
           "augment": {"enabled": True, "noise": {"prob": 0.5, "snr_db": [10, 30]}}}
    ex = FeatureExtractor(audio_cfg, feat_cfg)
    assert _config_fingerprint(sem, ex) == _config_fingerprint(com, ex)


def test_cache_fingerprint_still_reacts_to_real_audio_params(audio_cfg, feat_cfg):
    """A exclusão acima não pode ter afrouxado o resto do bloco `audio`."""
    from src.data.dataset import _config_fingerprint

    ex = FeatureExtractor(audio_cfg, feat_cfg)
    base = _config_fingerprint(audio_cfg, ex)
    for chave, valor in [("sample_rate", 8000), ("duration", 9.0),
                         ("trim_silence", not audio_cfg["trim_silence"]),
                         ("top_db", 99), ("peak_normalize", False)]:
        assert _config_fingerprint({**audio_cfg, chave: valor}, ex) != base, chave


def test_cache_fingerprint_ignores_unused_feature(audio_cfg, feat_cfg):
    """Alterar o espectrograma não deve invalidar o cache de um modelo só-LFCC."""
    from src.data.dataset import _config_fingerprint

    lfcc_only = {**feat_cfg, "types": ["lfcc"]}
    a = FeatureExtractor(audio_cfg, lfcc_only)
    changed = {**lfcc_only, "spectrogram": {**feat_cfg["spectrogram"], "n_mels": 123}}
    b = FeatureExtractor(audio_cfg, changed)
    assert _config_fingerprint(audio_cfg, a) == _config_fingerprint(audio_cfg, b)


def test_augmenter_applied_in_smoke(audio_cfg, feat_cfg):
    """Com augmenter, o dataset ainda deve devolver features no formato certo."""
    from src.preprocess.augment import make_perturbation

    extractor = FeatureExtractor(audio_cfg, feat_cfg)
    pert = make_perturbation("noise", 10, seed=0)
    ds = SmokeDataset(4, audio_cfg, extractor, seed=1, augmenter=pert)
    features, _ = ds[0]
    assert features["lfcc"].ndim == 3


# --------------------------------------------------------------------------- #
# Regressão: set_epoch precisa alcançar workers PERSISTENTES.
#
# Com persistent_workers=True os workers recebem uma cópia do dataset e nunca
# mais a atualizam. Se a época fosse um int comum, o recorte aleatório e a
# aumentação ficariam congelados na época em que o worker nasceu — repetindo o
# mesmo trecho de áudio em todas as épocas, sem nenhum aviso.
# --------------------------------------------------------------------------- #
def test_epoch_counter_lives_in_shared_memory(audio_cfg, feat_cfg):
    import torch

    ds = SmokeDataset(4, audio_cfg, FeatureExtractor(audio_cfg, feat_cfg), seed=1)
    assert isinstance(ds._epoch, torch.Tensor), "época voltou a ser um int comum"
    assert ds._epoch.is_shared(), "o tensor da época não está em memória compartilhada"


@pytest.mark.parametrize("persistent", [False, True])
def test_set_epoch_reaches_workers(audio_cfg, feat_cfg, persistent):
    from torch.utils.data import DataLoader

    from src.preprocess.augment import Augmenter

    aug = {"enabled": True, "noise": {"prob": 1.0, "snr_db": [10, 30]}}
    ds = SmokeDataset(8, audio_cfg, FeatureExtractor(audio_cfg, feat_cfg), seed=1,
                      augmenter=Augmenter(aug, seed=42), random_crop=True)
    loader = DataLoader(ds, batch_size=4, num_workers=2,
                        persistent_workers=persistent)
    assinaturas = []
    for epoca in (1, 2, 3):
        ds.set_epoch(epoca)
        assinaturas.append(float(next(iter(loader))[0]["lfcc"].sum()))
    assert len(set(assinaturas)) > 1, "aumentação congelada: set_epoch não chegou aos workers"


def test_same_epoch_is_reproducible_across_workers(audio_cfg, feat_cfg):
    from torch.utils.data import DataLoader

    ds = SmokeDataset(8, audio_cfg, FeatureExtractor(audio_cfg, feat_cfg),
                      seed=1, random_crop=True)
    loader = DataLoader(ds, batch_size=4, num_workers=2, persistent_workers=True)
    ds.set_epoch(7); a = float(next(iter(loader))[0]["lfcc"].sum())
    ds.set_epoch(9); next(iter(loader))
    ds.set_epoch(7); b = float(next(iter(loader))[0]["lfcc"].sum())
    assert abs(a - b) < 1e-9
