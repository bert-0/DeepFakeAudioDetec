"""Testes das métricas, com foco no EER."""

import math

import numpy as np

from src.metrics import compute_eer, compute_metrics, save_score_file


def test_eer_perfectly_separable():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])  # spoof tem score alto
    assert compute_eer(labels, scores) == 0.0


def test_eer_fully_inverted_is_high():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.2, 0.1])  # invertido
    assert compute_eer(labels, scores) > 0.9


def test_eer_single_class_is_nan():
    labels = np.array([1, 1, 1])
    scores = np.array([0.2, 0.5, 0.9])
    assert math.isnan(compute_eer(labels, scores))


def test_compute_metrics_keys_and_range():
    labels = np.array([0, 1, 0, 1])
    preds = np.array([0, 1, 1, 1])
    scores = np.array([0.2, 0.9, 0.6, 0.8])
    m = compute_metrics(labels, preds, scores)
    assert set(m) == {"accuracy", "precision", "recall", "f1", "eer"}
    for key in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= m[key] <= 1.0


def test_save_score_file(tmp_path):
    out = tmp_path / "scores.txt"
    save_score_file(["a", "b"], [0, 1], [0.1, 0.9], out)
    lines = out.read_text().strip().splitlines()
    assert lines[0].split() == ["a", "bonafide", "0.100000"]
    assert lines[1].split() == ["b", "spoof", "0.900000"]
