"""Treino do detector de deepfakes.

Pesos de classe, scheduler de LR, early stopping, AMP em GPU e aumentação
opcional; grava histórico (JSON) e curvas (PNG).

Exemplos:
    python train.py --config configs/baseline.yaml --smoke
    python train.py --config configs/fusion.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    estimativa_ram_gb,
    load_config,
    make_generator,
    memoria_total_gb,
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
    probabilidade_e_logodds,
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
    """Pesos de classe contra o desbalanceamento (~9 spoof por bonafide no LA).

    "auto" usa o inverso da frequência (~8,8x no bonafide); "sqrt" usa a raiz
    (~3x), que desloca menos o threshold.
    """
    counts = Counter(int(label) for label in labels)
    total = len(labels)
    weights = [total / (n_classes * max(counts.get(c, 0), 1)) for c in range(n_classes)]
    if mode == "sqrt":
        weights = [float(np.sqrt(w)) for w in weights]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def archive_previous_checkpoints(*paths: Path) -> None:
    """Renomeia checkpoints anteriores para `<nome>_prev.pt`.

    Na 1ª época o melhor EER é infinito e o arquivo antigo seria sobrescrito.
    """
    for path in paths:
        if not path.exists():
            continue
        backup = path.with_name(f"{path.stem}_prev{path.suffix}")
        backup.unlink(missing_ok=True)
        path.rename(backup)
        print(f"Checkpoint anterior preservado em: {backup}")


class EpochTimer:
    """Separa o tempo da época em espera por dados e cálculo na GPU.

    O tempo de cálculo só é confiável porque `loss.item()` sincroniza com a GPU.
    """

    def __init__(self):
        self.dados = 0.0
        self.calculo = 0.0
        self.dev = 0.0
        self._t0 = time.perf_counter()

    def batches(self, loader):
        """Itera o loader medindo quanto se esperou por cada lote."""
        inicio = time.perf_counter()
        for lote in loader:
            self.dados += time.perf_counter() - inicio
            yield lote
            inicio = time.perf_counter()

    @contextmanager
    def medindo(self, campo: str):
        inicio = time.perf_counter()
        yield
        setattr(self, campo, getattr(self, campo) + time.perf_counter() - inicio)

    @property
    def total(self) -> float:
        return time.perf_counter() - self._t0

    def resumo(self) -> str:
        return (f"tempo {self.total:6.1f}s  (dados {self.dados:5.1f}s | "
                f"GPU {self.calculo:5.1f}s | dev {self.dev:5.1f}s)")

    def as_dict(self) -> dict[str, float]:
        return {"t_epoch": self.total, "t_data": self.dados,
                "t_compute": self.calculo, "t_dev": self.dev}


def format_duration(segundos: float) -> str:
    h, resto = divmod(int(segundos), 3600)
    m, s = divmod(resto, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def print_time_report(history: list[dict], num_workers: int) -> None:
    """Mostra onde o tempo do treino foi gasto e o que otimizar.

    A 1ª época fica fora da média (enche o cache e roda o benchmark do cuDNN).
    """
    total = sum(h.get("t_epoch", 0.0) for h in history)
    print(f"\nTempo de treino: {format_duration(total)} em {len(history)} época(s)")
    regime = history[1:]
    if not regime:
        return

    dados = sum(h.get("t_data", 0.0) for h in regime) / len(regime)
    calculo = sum(h.get("t_compute", 0.0) for h in regime) / len(regime)
    dev = sum(h.get("t_dev", 0.0) for h in regime) / len(regime)
    epoca = sum(h.get("t_epoch", 0.0) for h in regime) / len(regime)
    if epoca <= 0:
        return

    print(f"Por época (média das {len(regime)} últimas): {epoca:.0f}s = "
          f"dados {dados:.0f}s + GPU {calculo:.0f}s + dev {dev:.0f}s")

    fracao = dados / epoca
    if fracao > 0.35:
        print(f"Gargalo: espera por dados ({fracao * 100:.0f}% da época) — a GPU "
              "fica ociosa esperando o carregamento.")
        print(f"  - suba train.num_workers (está em {num_workers}); um bom ponto "
              "de partida é o nº de núcleos físicos da CPU")
        print("  - ative train.cache_features se estiver desligado")
        print("  - se `audio.augment` e `audio.random_crop` estão ligados, o cache "
              "do treino é desativado de propósito e as features são recalculadas "
              "toda época: esse é o preço da aumentação")
    else:
        print(f"Gargalo: cálculo na GPU ({calculo / epoca * 100:.0f}% da época) — "
              "o carregamento já acompanha o treino.")
        print("  - mexer em num_workers ou no cache não vai ajudar")
        print("  - o que reduz tempo aqui é modelo menor, batch maior ou "
              "menos épocas (train.early_stopping_patience já corta o excesso)")


# Fora de um terminal (ex.: run_pipeline.py lendo por pipe) cada refresh da
# barra vira uma linha no log; por isso o intervalo longo.
_INTERVALO_BARRA = 0.1 if sys.stderr.isatty() else 30.0


def aviso_de_memoria(n_workers_treino: int, n_workers_dev: int) -> bool:
    """Estima a RAM do treino e avisa se não couber. Devolve True se avisou.

    No Windows estourar a RAM não gera `MemoryError`: a máquina pagina e trava.
    """
    est = estimativa_ram_gb(n_workers_treino, n_workers_dev)
    total = memoria_total_gb()
    print(f"Workers: {n_workers_treino} treino + {n_workers_dev} dev  |  "
          f"RAM estimada no pico: ~{est['pico']:.1f} GB "
          f"({est['workers_treino']:.1f} treino + {est['workers_dev']:.1f} dev + "
          f"{est['principal']:.1f} principal + {est['sistema']:.1f} sistema)")
    if total is None:
        return False
    # 2 GB de folga para navegador, editor etc.
    if est["pico"] <= total - 2.0:
        return False
    print(f"[AVISO] a máquina tem {total:.1f} GB. Com ~{est['pico']:.1f} GB de "
          f"pico a folga é pequena e o sistema pode paginar até travar.\n"
          f"        Reduza `train.num_workers` e/ou `train.dev_num_workers` "
          f"(0 desliga os processos do dev),\n"
          f"        ou use `train.pin_memory: false` para liberar memória "
          f"não-paginável.")
    return True


def save_checkpoint(payload: dict, destino: Path) -> None:
    """Grava o checkpoint de forma atômica (temporário + rename).

    Evita um `.pt` truncado se a máquina cair no meio da escrita.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(destino.suffix + ".tmp")
    torch.save(payload, temporario)
    temporario.replace(destino)


def save_history(history: list[dict], name: str) -> None:
    """Grava o histórico de forma atômica (temporário + rename)."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    destino = OUTPUT_DIR / f"{name}_history.json"
    temporario = destino.with_suffix(".json.tmp")
    with open(temporario, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    temporario.replace(destino)


@torch.no_grad()
def evaluate_loader(model, loader, device) -> tuple[dict[str, float], float]:
    """Roda o modelo no loader.

    Devolve ((labels, preds, P(spoof), log-odds), threshold do EER).
    """
    model.eval()
    all_labels, all_preds, all_scores, all_logodds = [], [], [], []
    for features, labels in loader:
        features = {k: v.to(device) for k, v in features.items()}
        logits = model(features)
        probs, logodds = probabilidade_e_logodds(logits)
        all_scores.append(probs)
        all_logodds.append(logodds)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    labels_arr = np.concatenate(all_labels)
    preds_arr = np.concatenate(all_preds)
    scores_arr = np.concatenate(all_scores)
    # Threshold em probabilidade (unidade do checkpoint); o EER sai dos log-odds.
    _, threshold = compute_eer_with_threshold(labels_arr, scores_arr)
    return (labels_arr, preds_arr, scores_arr, np.concatenate(all_logodds)), threshold


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

    # Shape fixo (audio.duration), então o benchmark do cuDNN compensa. Pode
    # mudar os últimos dígitos entre máquinas; `cudnn_benchmark: false` desliga.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", True))
        if not torch.backends.cudnn.benchmark:
            print("cuDNN benchmark: DESLIGADO (cudnn_benchmark: false)")

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
    # O dev lê só do cache (exceto na 1ª época), então basta menos workers;
    # `dev_num_workers: 0` elimina os processos.
    dev_workers = int(train_cfg.get("dev_num_workers", min(2, num_workers)))
    dev_workers = 0 if args.smoke else max(0, min(dev_workers, num_workers))

    # Memória fixada acelera a cópia para a GPU, mas o SO não a libera sob
    # pressão; use `pin_memory: false` se faltar RAM.
    pin = bool(train_cfg.get("pin_memory", True)) and device.type == "cuda"
    aviso_de_memoria(num_workers, dev_workers)

    # Persistentes para não recriar os processos a cada época (~2,4 s cada no
    # Windows). Só funciona porque `set_epoch` usa memória compartilhada.
    extras = {"persistent_workers": True} if num_workers > 0 else {}
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
                              num_workers=num_workers, generator=generator,
                              worker_init_fn=seed_worker, pin_memory=pin, **extras)
    # Dev sem persistent_workers: não ocupa RAM durante o treino.
    dev_loader = DataLoader(dev_ds, batch_size=train_cfg["batch_size"], shuffle=False,
                            num_workers=dev_workers, worker_init_fn=seed_worker,
                            pin_memory=pin)

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
        # Novo recorte aleatório a cada época.
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        seen = 0
        skipped = 0
        cronometro = EpochTimer()
        barra = tqdm(train_loader, desc=f"Época {epoch}/{epochs}", leave=False,
                     mininterval=_INTERVALO_BARRA)
        for features, labels in cronometro.batches(barra):
            with cronometro.medindo("calculo"):
                # `pin` vem do config: redefini-lo aqui anularia `pin_memory: false`.
                features = {k: v.to(device, non_blocking=pin) for k, v in features.items()}
                labels = labels.to(device, non_blocking=pin)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    loss = criterion(model(features), labels)

                # Loss inf/NaN estragaria pesos e BatchNorm; descarta o batch.
                if not torch.isfinite(loss):
                    skipped += 1
                    continue

                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)  # a norma é medida sem a escala do AMP
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * labels.size(0)
                seen += labels.size(0)

        if skipped:
            print(f"  [aviso] {skipped} batch(es) descartados por loss inf/NaN nesta época.")
        if seen == 0:
            # Sem nenhum batch válido a loss sairia 0 e pesos não treinados
            # poderiam virar o melhor checkpoint.
            print(f"\n[ERRO] Época {epoch}: todos os {skipped} batches foram descartados "
                  "por loss inf/NaN. O treino não avançou.")
            print("       Sugestões: desligar AMP (train.amp: false), reduzir o "
                  "learning rate ou ativar train.grad_clip.")
            break
        train_loss = running_loss / seen
        with cronometro.medindo("dev"):
            (dev_labels, dev_preds, dev_scores, dev_logodds), dev_threshold = evaluate_loader(
                model, dev_loader, device)

        # Modelo divergiu: treinar mais não recupera, então encerra.
        if not np.isfinite(dev_scores).all():
            print(f"\n[ERRO] A época {epoch} produziu scores NaN/inf no dev — "
                  "o modelo divergiu numericamente.")
            print("       O melhor checkpoint anterior foi preservado.")
            print("       Sugestões: desligar AMP (train.amp: false), reduzir o "
                  "learning rate ou ativar train.grad_clip.")
            break
        # Com calibração, usa o corte do EER no lugar do 0,5 do argmax.
        dev_metrics = compute_metrics(dev_labels, dev_preds, dev_scores,
                                      threshold=dev_threshold if calibrate else None,
                                      eer_scores=dev_logodds)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Época {epoch:3d} | lr={lr:.2e} | loss={train_loss:.4f} | "
              f"dev: {format_metrics(dev_metrics)}")
        print(f"           {cronometro.resumo()}")
        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss,
                        **{f"dev_{k}": v for k, v in dev_metrics.items()},
                        **cronometro.as_dict()})
        # A cada época, para as curvas sobreviverem a uma queda.
        save_history(history, name)

        if scheduler is not None:
            scheduler.step(dev_metrics["eer"])
        save_checkpoint({"model_state": model.state_dict(), "config": config,
                         "threshold": dev_threshold}, last_ckpt)

        # `<` estrito: com `<=` um platô zeraria a paciência e o early stopping
        # nunca dispararia.
        current_eer = dev_metrics["eer"]
        if not np.isnan(current_eer) and current_eer < best_eer:
            best_eer = current_eer
            best_threshold = dev_threshold
            epochs_no_improve = 0
            # O threshold vai junto para evaluate.py e infer.py.
            save_checkpoint({"model_state": model.state_dict(), "config": config,
                             "threshold": dev_threshold}, best_ckpt)
        else:
            epochs_no_improve += 1
            if early_patience and epochs_no_improve >= early_patience:
                print(f"Early stopping: sem melhora no EER por {early_patience} épocas.")
                break

    # ----- curvas -----
    curves_path = OUTPUT_DIR / f"{name}_curves.png"
    if history:
        plot_history(history, curves_path)

    print_time_report(history, num_workers)

    print(f"\nMelhor EER de validação: {best_eer * 100:.2f}%")
    if calibrate:
        print(f"Threshold calibrado no dev: {best_threshold:.4f} (salvo no checkpoint)")
    print(f"Melhor checkpoint: {best_ckpt}")
    print(f"Histórico:         {OUTPUT_DIR / f'{name}_history.json'}")
    print(f"Curvas:            {curves_path}")


if __name__ == "__main__":
    main()
