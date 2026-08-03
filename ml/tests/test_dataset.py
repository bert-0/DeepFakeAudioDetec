"""Testes de parsing de protocolo e dos datasets."""

from src.data.dataset import SmokeDataset, build_dataset, parse_protocol
from src.features import FeatureExtractor


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
