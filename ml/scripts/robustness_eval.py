"""Avaliação de robustez (TC1 §5.5).

Mede a degradação do modelo quando o áudio de avaliação é perturbado com ruído
de fundo e variações de ganho — verificando se o sistema mantém desempenho em
condições acústicas adversas (RNF de robustez).

Uso:
    python scripts/robustness_eval.py --config configs/baseline.yaml \
        --checkpoint checkpoints/baseline_lfcc_cnn.pt
    python scripts/robustness_eval.py --config configs/attention.yaml \
        --checkpoint checkpoints/attention_fusion.pt --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_device, set_seed  # noqa: E402
from src.data import build_dataset  # noqa: E402
from src.features import FeatureExtractor  # noqa: E402
from src.metrics import compute_metrics, format_metrics  # noqa: E402
from src.models import build_model  # noqa: E402
from src.preprocess.augment import make_perturbation  # noqa: E402

OUTPUT_DIR = Path("outputs")

# (rótulo, tipo de perturbação, nível). Nível: SNR(dB) p/ ruído, dB p/ ganho.
CONDITIONS = [
    ("clean", None, None),
    ("noise_20dB", "noise", 20),
    ("noise_10dB", "noise", 10),
    ("noise_5dB", "noise", 5),
    ("gain_-6dB", "gain", -6),
    ("gain_+6dB", "gain", 6),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Avaliação de robustez sob perturbações")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--partition", default="eval", choices=["train", "dev", "eval"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    labels, preds, scores = [], [], []
    for features, y in loader:
        features = {k: v.to(device) for k, v in features.items()}
        logits = model(features)
        probs = torch.softmax(logits, dim=1)[:, 1]
        scores.append(probs.cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(y.numpy())
    return (np.concatenate(labels), np.concatenate(preds), np.concatenate(scores))


def plot_robustness(results: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = list(results.keys())
    eers = [results[c]["eer"] * 100 for c in conds]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(conds, eers, color="tab:blue")
    ax.set_ylabel("EER (%)")
    ax.set_title("Robustez: EER por condição")
    ax.tick_params(axis="x", rotation=30)
    for bar, eer in zip(bars, eers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{eer:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = config["experiment"]["seed"]
    set_seed(seed)
    device = resolve_device(args.device or config["train"]["device"])

    extractor = FeatureExtractor(config["audio"], config["features"])
    model = build_model(config["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    batch_size = config["smoke"]["batch_size"] if args.smoke else config["train"]["batch_size"]
    print(f"Robustez | modelo: {config['model']['name']} | partição: {args.partition} | "
          f"dispositivo: {device}\n")

    results: dict[str, dict] = {}
    for label, kind, level in CONDITIONS:
        perturbation = None if kind is None else make_perturbation(kind, level, seed=seed)
        ds = build_dataset(config, args.partition, extractor, args.smoke, augmenter=perturbation)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        labels, preds, scores = run_inference(model, loader, device)
        metrics = compute_metrics(labels, preds, scores)
        results[label] = metrics
        print(f"{label:12s} -> {format_metrics(metrics)}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    name = config["experiment"]["name"]
    json_path = OUTPUT_DIR / f"{name}_robustness.json"
    plot_path = OUTPUT_DIR / f"{name}_robustness.png"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    plot_robustness(results, plot_path)
    print(f"\nResultados (JSON): {json_path}")
    print(f"Gráfico:           {plot_path}")


if __name__ == "__main__":
    main()
