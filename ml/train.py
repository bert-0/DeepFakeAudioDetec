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

from src.config import (
    load_config,
    make_generator,
    output_name,
    resolve_device,
    seed_worker,
    set_seed,
)
from src.data import build_dataset
from src.features import FeatureExtractor
from src.metrics import (
    compute_eer_with_threshold,
    compute_metrics,
    format_metrics,
    plot_history,
)
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


def class_weights_from(labels, n_classes: int, device, mode: str = "auto") -> torch.Tensor:
    """Pesos de classe para compensar o desbalanceamento da base.

    O ASVspoof LA tem ~9 spoof para cada bonafide; sem ponderação o modelo tende
    a prever sempre a classe majoritária.

    - "auto": peso inversamente proporcional à frequência (compensação total).
      No LA isso gera ~8,8x mais peso no bonafide, o que desloca fortemente o
      ponto de decisão e aumenta os falsos positivos.
    - "sqrt": raiz quadrada da razão (~3x no LA) — compensação mais suave, que
      mantém o benefício sem desestabilizar tanto o threshold.
    """
    counts = Counter(int(label) for label in labels)
    total = len(labels)
    weights = [total / (n_classes * max(counts.get(c, 0), 1)) for c in range(n_classes)]
    if mode == "sqrt":
        weights = [float(np.sqrt(w)) for w in weights]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def archive_previous_checkpoints(*paths: Path) -> None:
    """Renomeia checkpoints de execuções anteriores em vez de sobrescrevê-los.

    Na primeira época de um treino novo o melhor EER ainda é infinito, então
    qualquer resultado é considerado "melhor" e o arquivo antigo seria perdido —
    mesmo que a execução anterior tivesse chegado a um modelo muito superior.
    Por isso o arquivo existente vira `<nome>_prev.pt` antes do treino começar.
    """
    for path in paths:
        if not path.exists():
            continue
        backup = path.with_name(f"{path.stem}_prev{path.suffix}")
        backup.unlink(missing_ok=True)
        path.rename(backup)
        print(f"Checkpoint anterior preservado em: {backup}")


def save_history(history: list[dict], name: str) -> None:
    """Grava o histórico de forma atômica (escreve num temporário e renomeia).

    Gravar direto no destino deixaria um JSON truncado se o processo morresse
    no meio da escrita — e é justamente numa queda que o histórico importa.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    destino = OUTPUT_DIR / f"{name}_history.json"
    temporario = destino.with_suffix(".json.tmp")
    with open(temporario, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    temporario.replace(destino)


@torch.no_grad()
def evaluate_loader(model, loader, device) -> tuple[dict[str, float], float]:
    """Roda o modelo em um DataLoader.

    Devolve (métricas, threshold_do_EER). As métricas são calculadas no ponto de
    corte calibrado quando `calibrate` está ativo — ver `main`.
    """
    model.eval()
    all_labels, all_preds, all_scores = [], [], []
    for features, labels in loader:
        features = {k: v.to(device) for k, v in features.items()}
        logits = model(features)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(spoof)
        all_scores.append(probs.cpu().numpy())
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    labels_arr = np.concatenate(all_labels)
    preds_arr = np.concatenate(all_preds)
    scores_arr = np.concatenate(all_scores)
    _, threshold = compute_eer_with_threshold(labels_arr, scores_arr)
    return (labels_arr, preds_arr, scores_arr), threshold


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
    random_crop = bool(config["audio"].get("random_crop", False))
    if random_crop:
        print("Recorte aleatório no treino: ATIVADO")
    calibrate = bool(train_cfg.get("calibrate_threshold", False))
    train_ds = build_dataset(config, "train", extractor, args.smoke,
                             augmenter=augmenter, random_crop=random_crop)
    dev_ds = build_dataset(config, "dev", extractor, args.smoke)

    num_workers = 0 if args.smoke else train_cfg["num_workers"]
    generator = make_generator(seed)
    # persistent_workers evita recriar os processos a cada época. No Windows,
    # que usa `spawn`, cada criação custa a reimportação de torch+librosa
    # (~2,4 s medidos): com 4 workers e 50 épocas seriam ~8 minutos só de
    # inicialização. Só é seguro porque `set_epoch` grava num tensor em memória
    # compartilhada — com um int comum, os workers persistentes ficariam presos
    # à época em que nasceram e repetiriam o mesmo recorte/aumentação sempre.
    extras = {"persistent_workers": True} if num_workers > 0 else {}
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
                              num_workers=num_workers, generator=generator,
                              worker_init_fn=seed_worker, **extras)
    # dev também recebe worker_init_fn: sem ele os workers da validação abrem
    # uma thread BLAS por núcleo cada um e disputam CPU entre si.
    dev_loader = DataLoader(dev_ds, batch_size=train_cfg["batch_size"], shuffle=False,
                            num_workers=num_workers, worker_init_fn=seed_worker, **extras)

    # ----- modelo, perda, otimizador -----
    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"],
                                 weight_decay=train_cfg["weight_decay"])

    weight = None
    cw_mode = train_cfg.get("class_weights", "none")
    if cw_mode in ("auto", "sqrt"):
        weight = class_weights_from(train_ds.labels, config["model"].get("n_classes", 2),
                                    device, mode=cw_mode)
        print(f"Pesos de classe ({cw_mode}): bonafide={weight[0]:.3f}  spoof={weight[1]:.3f}")
    criterion = nn.CrossEntropyLoss(weight=weight)

    scheduler = None
    sch_cfg = train_cfg.get("scheduler", {})
    if sch_cfg.get("enabled", False):
        scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                      factor=sch_cfg.get("factor", 0.5),
                                      patience=sch_cfg.get("patience", 3))

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    early_patience = int(train_cfg.get("early_stopping_patience", 0))
    grad_clip = float(train_cfg.get("grad_clip", 0) or 0)
    if grad_clip:
        print(f"Clipping de gradiente: norma máxima {grad_clip}")

    # ----- loop de treino -----
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    name = output_name(config, args.smoke)
    best_ckpt = CHECKPOINT_DIR / f"{name}.pt"
    last_ckpt = CHECKPOINT_DIR / f"{name}_last.pt"
    archive_previous_checkpoints(best_ckpt, last_ckpt)
    best_eer = float("inf")
    epochs_no_improve = 0
    history: list[dict] = []

    best_threshold = 0.5
    for epoch in range(1, epochs + 1):
        # Varia a semente do recorte aleatório a cada época (no-op sem random_crop).
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        seen = 0
        skipped = 0
        for features, labels in tqdm(train_loader, desc=f"Época {epoch}/{epochs}", leave=False):
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(features), labels)

            # Um batch com loss inf/NaN propagaria o estrago para os pesos e para
            # as estatísticas do BatchNorm; descartar é mais seguro que treinar
            # com ele. Se acontecer sempre, o aviso no fim da época denuncia.
            if not torch.isfinite(loss):
                skipped += 1
                continue

            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)  # desfaz a escala antes de medir a norma
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        if skipped:
            print(f"  [aviso] {skipped} batch(es) descartados por loss inf/NaN nesta época.")
        if seen == 0:
            # Nenhum batch sobreviveu: `running_loss / max(seen, 1)` reportaria
            # 0.0000 — uma loss "perfeita" — e um checkpoint com pesos não
            # treinados poderia ser salvo como o melhor.
            print(f"\n[ERRO] Época {epoch}: todos os {skipped} batches foram descartados "
                  "por loss inf/NaN. O treino não avançou.")
            print("       Sugestões: desligar AMP (train.amp: false), reduzir o "
                  "learning rate ou ativar train.grad_clip.")
            break
        train_loss = running_loss / seen
        (dev_labels, dev_preds, dev_scores), dev_threshold = evaluate_loader(
            model, dev_loader, device)

        # Se o modelo passou a emitir NaN, treinar mais não recupera (pesos e/ou
        # estatísticas do BatchNorm já estão contaminados). Encerra de forma
        # limpa preservando o melhor checkpoint, em vez de estourar exceção.
        if not np.isfinite(dev_scores).all():
            print(f"\n[ERRO] A época {epoch} produziu scores NaN/inf no dev — "
                  "o modelo divergiu numericamente.")
            print("       O melhor checkpoint anterior foi preservado.")
            print("       Sugestões: desligar AMP (train.amp: false), reduzir o "
                  "learning rate ou ativar train.grad_clip.")
            break
        # Com calibração, as métricas usam o corte do EER (medido no dev) em vez
        # do 0,5 implícito do argmax — que oscila muito com pesos de classe.
        dev_metrics = compute_metrics(dev_labels, dev_preds, dev_scores,
                                      threshold=dev_threshold if calibrate else None)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Época {epoch:3d} | lr={lr:.2e} | loss={train_loss:.4f} | "
              f"dev: {format_metrics(dev_metrics)}")
        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss,
                        **{f"dev_{k}": v for k, v in dev_metrics.items()}})
        # Gravado a cada época: se o processo cair na época 34 de 50, as curvas
        # do relatório sobrevivem. Não há retomada de treino no checkpoint.
        save_history(history, name)

        if scheduler is not None:
            scheduler.step(dev_metrics["eer"])
        torch.save({"model_state": model.state_dict(), "config": config,
                    "threshold": dev_threshold}, last_ckpt)

        # Melhor modelo pelo EER de validação (NaN é tratado como "pior").
        # Comparação estrita: com `<=`, um platô perfeito zeraria o contador de
        # paciência a cada época e o early stopping nunca dispararia.
        current_eer = dev_metrics["eer"]
        if not np.isnan(current_eer) and current_eer < best_eer:
            best_eer = current_eer
            best_threshold = dev_threshold
            epochs_no_improve = 0
            # O threshold calibrado viaja junto com os pesos: assim evaluate.py e
            # infer.py usam o mesmo ponto de corte escolhido no dev.
            torch.save({"model_state": model.state_dict(), "config": config,
                        "threshold": dev_threshold}, best_ckpt)
        else:
            epochs_no_improve += 1
            if early_patience and epochs_no_improve >= early_patience:
                print(f"Early stopping: sem melhora no EER por {early_patience} épocas.")
                break

    # ----- curvas (o histórico já foi gravado a cada época) -----
    curves_path = OUTPUT_DIR / f"{name}_curves.png"
    if history:
        plot_history(history, curves_path)

    print(f"\nMelhor EER de validação: {best_eer * 100:.2f}%")
    if calibrate:
        print(f"Threshold calibrado no dev: {best_threshold:.4f} (salvo no checkpoint)")
    print(f"Melhor checkpoint: {best_ckpt}")
    print(f"Histórico:         {OUTPUT_DIR / f'{name}_history.json'}")
    print(f"Curvas:            {curves_path}")


if __name__ == "__main__":
    main()
