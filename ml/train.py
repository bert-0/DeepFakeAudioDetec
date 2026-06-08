"""Treino de um incremento do detector de deepfakes.

Recursos: pesos de classe (desbalanceamento), scheduler de LR, early stopping,
mixed precision (AMP) em GPU, aumentação opcional no treino, e registro do
histórico (JSON) + curvas (PNG) para o relatório.

Exemplos:
    python train.py --config configs/baseline.yaml --smoke
    python train.py --config configs/fusion.yaml
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config, make_generator, resolve_device, seed_worker, set_seed
from src.data import build_dataset
from src.features import FeatureExtractor
from src.metrics import compute_metrics, format_metrics, plot_history
from src.models import build_model
from src.preprocess.augment import Augmenter

CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treino do detector de deepfakes em áudio")
    p.add_argument("--config", required=True, help="caminho do YAML de configuração")
    p.add_argument("--smoke", action="store_true", help="usa dados sintéticos (sem dataset)")
    p.add_argument("--epochs", type=int, default=None, help="sobrescreve o nº de épocas")
    p.add_argument("--device", default=None, help="cuda | cpu (padrão: do config)")
    return p.parse_args()


def class_weights_from(labels, n_classes: int, device) -> torch.Tensor:
    """Pesos inversamente proporcionais à frequência de cada classe.

    Compensa o desbalanceamento da base (ASVspoof LA tem muito mais spoof do que
    bonafide), evitando que o modelo aprenda a prever sempre a classe majoritária.
    """
    counts = Counter(int(label) for label in labels)
    total = len(labels)
    weights = [total / (n_classes * max(counts.get(c, 0), 1)) for c in range(n_classes)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


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
    seed = config["experiment"]["seed"]
    set_seed(seed)

    train_cfg = config["train"]
    if args.smoke:
        train_cfg = {**train_cfg, **{k: config["smoke"][k] for k in ("epochs", "batch_size")}}
    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]
    device = resolve_device(args.device or train_cfg["device"])
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    print(f"Dispositivo: {device} | épocas: {epochs} | smoke: {args.smoke} | AMP: {use_amp}")

    # ----- dados (aumentação só no treino) -----
    extractor = FeatureExtractor(config["audio"], config["features"])
    aug_cfg = config["audio"].get("augment", {})
    augmenter = Augmenter(aug_cfg, seed=seed) if aug_cfg.get("enabled", False) else None
    if augmenter is not None:
        print("Aumentação de treino: ATIVADA")
    train_ds = build_dataset(config, "train", extractor, args.smoke, augmenter=augmenter)
    dev_ds = build_dataset(config, "dev", extractor, args.smoke)

    num_workers = 0 if args.smoke else train_cfg["num_workers"]
    generator = make_generator(seed)
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
                              num_workers=num_workers, generator=generator,
                              worker_init_fn=seed_worker)
    dev_loader = DataLoader(dev_ds, batch_size=train_cfg["batch_size"], shuffle=False,
                            num_workers=num_workers)

    # ----- modelo, perda, otimizador -----
    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"],
                                 weight_decay=train_cfg["weight_decay"])

    weight = None
    if train_cfg.get("class_weights", "none") == "auto":
        weight = class_weights_from(train_ds.labels, config["model"].get("n_classes", 2), device)
        print(f"Pesos de classe (auto): bonafide={weight[0]:.3f}  spoof={weight[1]:.3f}")
    criterion = nn.CrossEntropyLoss(weight=weight)

    scheduler = None
    sch_cfg = train_cfg.get("scheduler", {})
    if sch_cfg.get("enabled", False):
        scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                      factor=sch_cfg.get("factor", 0.5),
                                      patience=sch_cfg.get("patience", 3))

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    early_patience = int(train_cfg.get("early_stopping_patience", 0))

    # ----- loop de treino -----
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    name = config["experiment"]["name"]
    best_ckpt = CHECKPOINT_DIR / f"{name}.pt"
    last_ckpt = CHECKPOINT_DIR / f"{name}_last.pt"
    best_eer = float("inf")
    epochs_no_improve = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for features, labels in tqdm(train_loader, desc=f"Época {epoch}/{epochs}", leave=False):
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(features), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_ds)
        dev_metrics = evaluate_loader(model, dev_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Época {epoch:3d} | lr={lr:.2e} | loss={train_loss:.4f} | "
              f"dev: {format_metrics(dev_metrics)}")
        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss,
                        **{f"dev_{k}": v for k, v in dev_metrics.items()}})

        if scheduler is not None:
            scheduler.step(dev_metrics["eer"])
        torch.save({"model_state": model.state_dict(), "config": config}, last_ckpt)

        # Melhor modelo pelo EER de validação (NaN é tratado como "pior").
        current_eer = dev_metrics["eer"]
        if not np.isnan(current_eer) and current_eer <= best_eer:
            best_eer = current_eer
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "config": config}, best_ckpt)
        else:
            epochs_no_improve += 1
            if early_patience and epochs_no_improve >= early_patience:
                print(f"Early stopping: sem melhora no EER por {early_patience} épocas.")
                break

    # ----- histórico + curvas -----
    OUTPUT_DIR.mkdir(exist_ok=True)
    hist_path = OUTPUT_DIR / f"{name}_history.json"
    curves_path = OUTPUT_DIR / f"{name}_curves.png"
    with open(hist_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    plot_history(history, curves_path)

    print(f"\nMelhor EER de validação: {best_eer * 100:.2f}%")
    print(f"Melhor checkpoint: {best_ckpt}")
    print(f"Histórico:         {hist_path}")
    print(f"Curvas:            {curves_path}")


if __name__ == "__main__":
    main()
