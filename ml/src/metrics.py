"""Métricas de avaliação (TC1 §4.11): accuracy, precision, recall, F1, EER e
matriz de confusão. A classe positiva é `spoof` (rótulo 1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Equal Error Rate: ponto em que falso positivo (FPR) e falso negativo
    (FNR) se igualam. `scores` = probabilidade da classe spoof. Menor é melhor.
    """
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def compute_metrics(
    labels: np.ndarray, preds: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    """Calcula todas as métricas a partir dos rótulos, predições e scores."""
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "f1": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "eer": compute_eer(labels, scores),
    }


def plot_confusion_matrix(
    labels: np.ndarray, preds: np.ndarray, out_path: str | Path
) -> None:
    """Salva a matriz de confusão como imagem PNG."""
    import matplotlib

    matplotlib.use("Agg")  # backend sem display (servidores/headless)
    import matplotlib.pyplot as plt

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    class_names = ["bonafide", "spoof"]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=class_names)
    ax.set_yticks([0, 1], labels=class_names)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def format_metrics(metrics: dict[str, float]) -> str:
    """Formata as métricas em uma linha legível."""
    return (
        f"accuracy={metrics['accuracy']:.4f}  "
        f"precision={metrics['precision']:.4f}  "
        f"recall={metrics['recall']:.4f}  "
        f"f1={metrics['f1']:.4f}  "
        f"EER={metrics['eer'] * 100:.2f}%"
    )
