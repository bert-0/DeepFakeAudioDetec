"""Testes da extração de características (formatos esperados)."""

import torch

from src.features import FeatureExtractor
from src.features.lfcc import compute_lfcc
from src.features.spectrogram import compute_log_mel


def test_lfcc_with_deltas_has_triple_rows(sine_wave, feat_cfg):
    feat = compute_lfcc(sine_wave, 16000, feat_cfg["lfcc"])
    # 20 coeficientes * 3 (estático + delta + delta-delta)
    assert feat.shape[0] == 60
    assert feat.shape[1] > 0


def test_lfcc_without_deltas(sine_wave, feat_cfg):
    cfg = {**feat_cfg["lfcc"], "deltas": False}
    feat = compute_lfcc(sine_wave, 16000, cfg)
    assert feat.shape[0] == 20


def test_spectrogram_rows_equal_n_mels(sine_wave, feat_cfg):
    feat = compute_log_mel(sine_wave, 16000, feat_cfg["spectrogram"])
    assert feat.shape[0] == feat_cfg["spectrogram"]["n_mels"]


def test_extractor_returns_dict_with_channel_dim(sine_wave, audio_cfg, feat_cfg):
    extractor = FeatureExtractor(audio_cfg, feat_cfg)
    out = extractor(sine_wave)
    assert set(out) == {"lfcc", "spectrogram"}
    for tensor in out.values():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 3  # (canal, freq, frames)
        assert tensor.shape[0] == 1
