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
    eer, _ = compute_eer_with_threshold(labels, scores)
    return eer


def compute_eer_with_threshold(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    """Devolve (EER, threshold) no ponto de erro igual.

    O threshold é o ponto de corte que equilibra falsos positivos e falsos
    negativos. Calibrá-lo no conjunto de validação evita o corte fixo de 0,5,
    que é fortemente enviesado quando as classes são desbalanceadas ou quando
    a perda usa pesos de classe.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if np.unique(labels).size < 2:
        # Sem ambas as classes (bonafide e spoof) o EER é indefinido.
        return float("nan"), 0.5
    if not np.isfinite(scores).all():
        # Modelo divergiu (NaN/inf). Devolve NaN em vez de estourar dentro do
        # sklearn, para que quem chamou possa tratar o caso com uma mensagem útil.
        return float("nan"), 0.5
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    # roc_curve pode devolver +inf no primeiro ponto; nesse caso não há corte útil.
    if not np.isfinite(threshold):
        threshold = 1.0
    return eer, threshold


def probabilidade_e_logodds(logits) -> tuple[np.ndarray, np.ndarray]:
    """Das saídas da rede (N, 2): P(spoof) e o log-odds logit[1] - logit[0].

    Com duas classes, P(spoof) = sigmoide(log-odds): os dois ordenam os áudios
    igual — até o float32 arredondar. A partir de uma margem de ~17 o softmax
    vira EXATAMENTE 1,0 e áudios diferentes empatam; o log-odds não satura.
    Como o EER depende só da ordem, ele é calculado sobre o log-odds. A
    probabilidade continua sendo o score exibido e a unidade do threshold.
    """
    import torch

    logits = logits.detach().float()
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    logodds = (logits[:, 1] - logits[:, 0]).cpu().numpy().astype(np.float64)
    return probs, logodds


def logodds_de_probabilidade(scores: np.ndarray) -> np.ndarray:
    """Log-odds a partir de probabilidades já salvas (arquivos antigos).

    Preserva a ordem das probabilidades — inclusive os empates em 1,0, que já
    estão perdidos. Só serve para pôr arquivos antigos na mesma escala dos
    novos; não recupera a informação que a saturação apagou.
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
    """Calcula todas as métricas a partir dos rótulos, predições e scores.

    Se `threshold` for informado, as predições são recalculadas como
    `scores >= threshold` (ignorando `preds`), permitindo reportar as métricas
    em um ponto de corte calibrado em vez do 0,5 implícito do argmax.

    `eer_scores`, se informado, é usado só no EER — tipicamente os log-odds
    (ver `probabilidade_e_logodds`). O threshold continua em probabilidade,
    que é a unidade guardada no checkpoint e usada pelo monitor.
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


def save_score_file(ids, labels, scores, out_path: str | Path) -> None:
    """Salva um arquivo de scores por utterance no estilo ASVspoof.

    Cada linha: `utt_id  key  score`, onde `key` é bonafide/spoof e `score` é a
    probabilidade de spoof. Útil para auditoria e para ferramentas externas
    (ex.: cálculo do t-DCF com o script oficial da ASVspoof).
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
