"""Testes das métricas, com foco no EER."""

import math

import numpy as np

from src.metrics import (
    compute_eer,
    compute_eer_with_threshold,
    compute_metrics,
    save_score_file,
)


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


def test_eer_with_threshold_returns_usable_cut():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    eer, thr = compute_eer_with_threshold(labels, scores)
    assert eer == 0.0
    # O corte deve separar as duas classes.
    assert (scores >= thr).astype(int).tolist() == labels.tolist()


def test_eer_with_threshold_single_class_is_nan():
    eer, thr = compute_eer_with_threshold(np.array([1, 1]), np.array([0.3, 0.7]))
    assert math.isnan(eer)
    assert thr == 0.5


def test_compute_metrics_with_threshold_overrides_preds():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.30, 0.35, 0.60, 0.70])
    bad_preds = np.array([0, 0, 0, 0])  # o que o argmax@0.5 daria
    m = compute_metrics(labels, bad_preds, scores, threshold=0.5)
    assert m["recall"] == 1.0          # o threshold recalcula, ignorando bad_preds
    assert m["threshold"] == 0.5


def test_compute_metrics_without_threshold_uses_preds():
    labels = np.array([0, 1])
    preds = np.array([0, 0])
    scores = np.array([0.1, 0.9])
    m = compute_metrics(labels, preds, scores)
    assert m["recall"] == 0.0
    assert "threshold" not in m


def test_save_score_file(tmp_path):
    out = tmp_path / "scores.txt"
    save_score_file(["a", "b"], [0, 1], [0.1, 0.9], out)
    lines = out.read_text().strip().splitlines()
    assert lines[0].split() == ["a", "bonafide", "0.100000"]
    assert lines[1].split() == ["b", "spoof", "0.900000"]
