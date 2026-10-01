"""Métricas de avaliação (TC1 §4.11): accuracy, precision, recall, F1, EER e
matriz de confusão. Classe positiva: spoof (rótulo 1)."""

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
    """Equal Error Rate (FPR = FNR). Scores maiores indicam spoof."""
    eer, _ = compute_eer_with_threshold(labels, scores)
    return eer


def compute_eer_with_threshold(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    """Devolve (EER, threshold) no ponto de erro igual."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if np.unique(labels).size < 2:
        # EER indefinido sem as duas classes.
        return float("nan"), 0.5
    if not np.isfinite(scores).all():
        # Modelo divergiu; NaN em vez de exceção do sklearn.
        return float("nan"), 0.5
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    # roc_curve pode devolver +inf no primeiro ponto.
    if not np.isfinite(threshold):
        threshold = 1.0
    return eer, threshold


def probabilidade_e_logodds(logits) -> tuple[np.ndarray, np.ndarray]:
    """Das saídas (N, 2) da rede: P(spoof) e o log-odds logit[1] - logit[0].

    O EER usa o log-odds porque o softmax em float32 satura em 1,0 a partir de
    uma margem de ~17 e cria empates.
    """
    import torch

    logits = logits.detach().float()
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    logodds = (logits[:, 1] - logits[:, 0]).cpu().numpy().astype(np.float64)
    return probs, logodds


def logodds_de_probabilidade(scores: np.ndarray) -> np.ndarray:
    """Log-odds a partir de probabilidades salvas em arquivos antigos.

    Não desfaz os empates em 1,0 que a saturação já criou.
    """
    p = np.asarray(scores, dtype=np.float64)
    eps = float(np.finfo(np.float32).eps) / 2
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
    eer_scores: np.ndarray | None = None,
) -> dict[str, float]:
    """Calcula as métricas a partir dos rótulos, predições e scores.

    Com `threshold`, as predições viram `scores >= threshold` (ignora `preds`).
    `eer_scores` (tipicamente os log-odds) substitui `scores` só no EER.
    """
    if threshold is not None:
        preds = (np.asarray(scores) >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "f1": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "eer": compute_eer(labels, scores if eer_scores is None else eer_scores),
    }
    if threshold is not None:
        metrics["threshold"] = float(threshold)
    return metrics


def plot_confusion_matrix(
    labels: np.ndarray, preds: np.ndarray, out_path: str | Path
) -> None:
    """Salva a matriz de confusão como imagem PNG."""
    import matplotlib

    matplotlib.use("Agg")  # sem display
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


def save_score_file(ids, labels, scores, out_path: str | Path) -> None:
    """Salva scores por utterance no estilo ASVspoof: `utt_id key P(spoof)`.

    Serve de entrada para o script oficial de t-DCF.
    """
    inv = {0: "bonafide", 1: "spoof"}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for uid, lab, score in zip(ids, labels, scores):
            fh.write(f"{uid} {inv.get(int(lab), '-')} {float(score):.6f}\n")


def plot_history(history: list[dict], out_path: str | Path) -> None:
    """Salva as curvas de treino (loss) e validação (EER e F1) por época."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, [h["train_loss"] for h in history], marker="o", color="tab:red")
    ax1.set_title("Loss de treino")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [h["dev_eer"] * 100 for h in history], marker="o", label="EER (%)")
    ax2.plot(epochs, [h["dev_f1"] * 100 for h in history], marker="s", label="F1 (%)")
    ax2.set_title("Validação")
    ax2.set_xlabel("Época")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
