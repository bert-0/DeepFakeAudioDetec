"""Testes dos modelos dos três incrementos (formato da saída)."""

import pytest
import torch

from src.models import build_model


def _features(batch=2, freq_lfcc=60, freq_spec=40, frames=64):
    return {
        "lfcc": torch.randn(batch, 1, freq_lfcc, frames),
        "spectrogram": torch.randn(batch, 1, freq_spec, frames),
    }


@pytest.mark.parametrize("name", ["baseline_cnn", "fusion", "attention"])
def test_model_forward_output_shape(name):
    model = build_model({"name": name, "n_classes": 2, "dropout": 0.3})
    model.eval()
    with torch.no_grad():
        logits = model(_features())
    assert logits.shape == (2, 2)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model({"name": "inexistente"})
