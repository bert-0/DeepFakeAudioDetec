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


class EpochTimer:
    """Separa o tempo da época em *espera por dados* e *cálculo na GPU*.

    É a medida que decide qual otimização vale a pena: se a maior parte do tempo
    é espera por dados, o gargalo é CPU/disco (mais `num_workers`, cache de
    features); se é cálculo, o gargalo é a GPU e mexer no carregamento não muda
    nada. Sem separar os dois, a intuição erra com frequência.

    A cronometragem do cálculo é honesta porque `loss.item()` sincroniza com a
    GPU a cada iteração — sem isso, as chamadas CUDA voltariam na hora e o tempo
    apareceria todo do lado dos dados.
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
    """Diz onde o tempo do treino foi gasto e qual é o próximo passo útil.

    A primeira época é deixada de fora da média: é ela que preenche o cache de
    features e onde o cuDNN ainda está medindo algoritmos de convolução, então
    ela é sistematicamente mais lenta e não representa o regime do treino.
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


# Quando a saída não é um terminal (o caso do run_pipeline.py, que lê por um
# pipe), cada refresh da barra vira uma LINHA no log — 794 por época, ~40 mil
# num treino de 50. O log do experimento fica ilegível e com megabytes de barra.
# Num terminal de verdade o `\r` sobrescreve no lugar e o comportamento é o de
# sempre.
_INTERVALO_BARRA = 0.1 if sys.stderr.isatty() else 30.0


def aviso_de_memoria(n_workers_treino: int, n_workers_dev: int) -> bool:
    """Estima a RAM do treino e avisa se não couber. Devolve True se avisou.

    Existe porque o modo de falha no Windows é silencioso e brutal: não há
    `MemoryError`: o sistema pagina, a interface congela e a única saída é
    segurar o botão de energia — perdendo o treino inteiro, já que não há
    retomada por checkpoint.
    """
    est = estimativa_ram_gb(n_workers_treino, n_workers_dev)
    total = memoria_total_gb()
    print(f"Workers: {n_workers_treino} treino + {n_workers_dev} dev  |  "
          f"RAM estimada no pico: ~{est['pico']:.1f} GB "
          f"({est['workers_treino']:.1f} treino + {est['workers_dev']:.1f} dev + "
          f"{est['principal']:.1f} principal + {est['sistema']:.1f} sistema)")
    if total is None:
        return False
    # A folga cobre o que o usuário tem aberto: navegador, editor, etc.
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
    """Grava um checkpoint de forma atômica (temporário + rename).

    Mesmo motivo do `save_history`: `torch.save` direto no destino deixa um
    `.pt` truncado se a máquina cair no meio da escrita — e cair no meio é
    exatamente o que acontece quando a RAM estoura e o usuário desliga no botão.
    Perder o `best.pt` significa perder todo o treino.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(destino.suffix + ".tmp")
    torch.save(payload, temporario)
    temporario.replace(destino)


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

    # `audio.duration` fixa o comprimento do sinal, então as features chegam
    # sempre com o mesmo shape. Nessa condição o cuDNN mede os algoritmos de
    # convolução disponíveis na primeira iteração e reusa a escolha no resto do
    # treino — o custo é pago uma vez e as épocas seguintes ficam mais rápidas.
    # Em troca, a escolha do algoritmo pode variar entre máquinas/execuções, o
    # que muda os últimos dígitos do resultado; `cudnn_benchmark: false` desliga.
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
    # persistent_workers evita recriar os processos a cada época. No Windows,
    # que usa `spawn`, cada criação custa a reimportação de torch+librosa
    # (~2,4 s medidos): com 4 workers e 50 épocas seriam ~8 minutos só de
    # inicialização. Só é seguro porque `set_epoch` grava num tensor em memória
    # compartilhada — com um int comum, os workers persistentes ficariam presos
    # à época em que nasceram e repetiriam o mesmo recorte/aumentação sempre.
    # Workers do dev, contados à parte. Antes o dev herdava `num_workers` e
    # `persistent_workers` do treino, e os dois conjuntos ficavam vivos ao mesmo
    # tempo: 8 processos de ~512 MB cada no Windows, 4 deles parados durante
    # todo o treino, acordando só nos ~50 s da validação. Num notebook de 16 GB
    # isso somava ~8,5 GB com o sistema e levava a máquina à paginação.
    #
    # O dev não precisa do mesmo paralelismo: ele é 100% cache (leitura de
    # memmap, sem decode de FLAC nem extração), exceto na primeira época. Por
    # isso o padrão é 2, e `dev_num_workers: 0` elimina os processos de uma vez.
    dev_workers = int(train_cfg.get("dev_num_workers", min(2, num_workers)))
    dev_workers = 0 if args.smoke else max(0, min(dev_workers, num_workers))

    # pin_memory usa memória não-paginável, de onde a cópia para a GPU é feita
    # por DMA e pode sobrepor-se ao cálculo (com `non_blocking=True` no `.to`).
    # Só faz sentido com CUDA; em CPU seria custo puro. É justamente a memória
    # que o SO não consegue liberar sob pressão, então vira `pin_memory: false`
    # quando a RAM é o gargalo.
    pin = bool(train_cfg.get("pin_memory", True)) and device.type == "cuda"
    aviso_de_memoria(num_workers, dev_workers)

    extras = {"persistent_workers": True} if num_workers > 0 else {}
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
                              num_workers=num_workers, generator=generator,
                              worker_init_fn=seed_worker, pin_memory=pin, **extras)
    # dev também recebe worker_init_fn: sem ele os workers da validação abrem
    # uma thread BLAS por núcleo cada um e disputam CPU entre si.
    # Sem `persistent_workers`: os processos do dev nascem e morrem a cada
    # validação, então não ocupam RAM durante o treino. Custa ~2,4 s por época
    # de recriação no Windows, contra 2 GB mantidos ociosos.
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
        # Varia a semente do recorte aleatório a cada época (no-op sem random_crop).
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        seen = 0
        skipped = 0
        cronometro = EpochTimer()
        # `pin` vem de cima (train.pin_memory): `non_blocking=True` só tem efeito
        # se a origem estiver realmente em memória paginada-fixa. Redefini-lo
        # aqui anularia `pin_memory: false`.
        barra = tqdm(train_loader, desc=f"Época {epoch}/{epochs}", leave=False,
                     mininterval=_INTERVALO_BARRA)
        for features, labels in cronometro.batches(barra):
            with cronometro.medindo("calculo"):
                features = {k: v.to(device, non_blocking=pin) for k, v in features.items()}
                labels = labels.to(device, non_blocking=pin)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    loss = criterion(model(features), labels)

                # Um batch com loss inf/NaN propagaria o estrago para os pesos e
                # para as estatísticas do BatchNorm; descartar é mais seguro que
                # treinar com ele. Se acontecer sempre, o aviso no fim denuncia.
                if not torch.isfinite(loss):
                    skipped += 1
                    continue

                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(optimizer)  # desfaz a escala antes da norma
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
        with cronometro.medindo("dev"):
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
        print(f"           {cronometro.resumo()}")
        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss,
                        **{f"dev_{k}": v for k, v in dev_metrics.items()},
                        **cronometro.as_dict()})
        # Gravado a cada época: se o processo cair na época 34 de 50, as curvas
        # do relatório sobrevivem. Não há retomada de treino no checkpoint.
        save_history(history, name)

        if scheduler is not None:
            scheduler.step(dev_metrics["eer"])
        save_checkpoint({"model_state": model.state_dict(), "config": config,
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
            save_checkpoint({"model_state": model.state_dict(), "config": config,
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

    print_time_report(history, num_workers)

    print(f"\nMelhor EER de validação: {best_eer * 100:.2f}%")
    if calibrate:
        print(f"Threshold calibrado no dev: {best_threshold:.4f} (salvo no checkpoint)")
    print(f"Melhor checkpoint: {best_ckpt}")
    print(f"Histórico:         {OUTPUT_DIR / f'{name}_history.json'}")
    print(f"Curvas:            {curves_path}")


if __name__ == "__main__":
    main()
