"""Treino de um incremento do detector de deepfakes.

Exemplos:
    python train.py --config configs/baseline.yaml --smoke
    python train.py --config configs/fusion.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config, resolve_device, set_seed
from src.data import build_dataset
from src.features import FeatureExtractor
from src.metrics import compute_metrics, format_metrics
from src.models import build_model

CHECKPOINT_DIR = Path("checkpoints")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treino do detector de deepfakes em áudio")
    p.add_argument("--config", required=True, help="caminho do YAML de configuração")
    p.add_argument("--smoke", action="store_true", help="usa dados sintéticos (sem dataset)")
    p.add_argument("--epochs", type=int, default=None, help="sobrescreve o nº de épocas")
    p.add_argument("--device", default=None, help="cuda | cpu (padrão: do config)")
    return p.parse_args()


@torch.no_grad()
def evaluate_loader(model, loader, device) -> dict[str, float]:
    """Roda o modelo em um DataLoader e devolve as métricas."""
    model.eval()
    all_labels, all_preds, all_scores = [], [], []
    for features, labels in loader:
        features = {k: v.to(device) for k, v in features.items()}
        logits = model(features)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(spoof)
        all_scores.append(probs.cpu().numpy())
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    return compute_metrics(
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_scores),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    train_cfg = config["train"]
    if args.smoke:
        train_cfg = {**train_cfg, **{k: config["smoke"][k] for k in ("epochs", "batch_size")}}
    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]
    device = resolve_device(args.device or train_cfg["device"])
    print(f"Dispositivo: {device} | épocas: {epochs} | smoke: {args.smoke}")

    extractor = FeatureExtractor(config["audio"], config["features"])
    train_ds = build_dataset(config, "train", extractor, args.smoke)
    dev_ds = build_dataset(config, "dev", extractor, args.smoke)

    num_workers = 0 if args.smoke else train_cfg["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                              shuffle=True, num_workers=num_workers)
    dev_loader = DataLoader(dev_ds, batch_size=train_cfg["batch_size"],
                            shuffle=False, num_workers=num_workers)

    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"],
                                 weight_decay=train_cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{config['experiment']['name']}.pt"
    best_eer = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for features, labels in tqdm(train_loader, desc=f"Época {epoch}/{epochs}", leave=False):
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_ds)
        dev_metrics = evaluate_loader(model, dev_loader, device)
        print(f"Época {epoch:3d} | loss={train_loss:.4f} | dev: {format_metrics(dev_metrics)}")

        # Salva o melhor modelo pelo EER de validação (menor é melhor).
        if dev_metrics["eer"] <= best_eer:
            best_eer = dev_metrics["eer"]
            torch.save({"model_state": model.state_dict(), "config": config}, ckpt_path)

    print(f"\nMelhor EER de validação: {best_eer * 100:.2f}%")
    print(f"Checkpoint salvo em: {ckpt_path}")


if __name__ == "__main__":
    main()
