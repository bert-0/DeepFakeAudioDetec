"""Avaliação de um modelo treinado: métricas + matriz de confusão.

Exemplos:
    python evaluate.py --config configs/baseline.yaml --smoke \
                       --checkpoint checkpoints/baseline_lfcc_cnn.pt
    python evaluate.py --config configs/baseline.yaml --partition eval \
                       --checkpoint checkpoints/baseline_lfcc_cnn.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config, resolve_device, set_seed
from src.data import build_dataset
from src.features import FeatureExtractor
from src.metrics import compute_metrics, format_metrics, plot_confusion_matrix, save_score_file
from src.models import build_model

OUTPUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Avaliação do detector de deepfakes")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--partition", default="eval", choices=["train", "dev", "eval"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--score-file", default=None,
                   help="se definido, salva um arquivo de scores por utterance (estilo ASVspoof)")
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])
    device = resolve_device(args.device or config["train"]["device"])

    extractor = FeatureExtractor(config["audio"], config["features"])
    ds = build_dataset(config, args.partition, extractor, args.smoke)
    batch_size = config["smoke"]["batch_size"] if args.smoke else config["train"]["batch_size"]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0 if args.smoke else config["train"]["num_workers"])

    model = build_model(config["model"]).to(device)
    # weights_only=False: o checkpoint é nosso e inclui o dicionário de config.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"Modelo: {config['model']['name']} | partição: {args.partition} | "
          f"dispositivo: {device}")

    labels, preds, scores = run_inference(model, loader, device)
    metrics = compute_metrics(labels, preds, scores)
    print(f"\nResultados ({args.partition}): {format_metrics(metrics)}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    name = config["experiment"]["name"]
    cm_path = OUTPUT_DIR / f"{name}_{args.partition}_confusion.png"
    metrics_path = OUTPUT_DIR / f"{name}_{args.partition}_metrics.json"
    plot_confusion_matrix(labels, preds, cm_path)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Matriz de confusão: {cm_path}")
    print(f"Métricas (JSON):    {metrics_path}")

    if args.score_file:
        save_score_file(ds.ids, labels, scores, args.score_file)
        print(f"Arquivo de scores:  {args.score_file}")


if __name__ == "__main__":
    main()
