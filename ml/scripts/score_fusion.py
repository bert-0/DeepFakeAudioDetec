"""Fusão em nível de score entre modelos já treinados.

Diferente da *fusão de características* do Incremento 2 (que combina dois ramos
dentro de um mesmo modelo), aqui combinamos as **saídas** de modelos treinados
separadamente. É a prática padrão dos sistemas submetidos ao ASVspoof, e não
exige retreinar nada — só rodar a inferência de cada modelo uma vez.

A motivação é empírica: a análise por ataque mostrou que configurações
diferentes vencem em ataques diferentes (baixa resolução vai melhor no núcleo
duro; alta resolução, nos ataques semelhantes ao treino).

Regras de combinação implementadas:
  - mean : média simples dos scores
  - rank : média dos *postos* (normalized rank) — imune a diferenças de
           calibração entre modelos, que é a fraqueza da média simples
  - max  : score máximo (o sistema mais "desconfiado" decide)
  - min  : score mínimo

Uso:
    python scripts/score_fusion.py \\
        --model configs/baseline_v2.yaml  checkpoints/baseline_lfcc_cnn_v2.pt \\
        --model configs/baseline_v3a.yaml checkpoints/baseline_lfcc_cnn_v3a.pt \\
        --model configs/baseline_v4.yaml  checkpoints/baseline_lcnn_v4.pt
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

from src.config import load_config, seed_worker, output_name, resolve_device, set_seed  # noqa: E402
from src.data import build_dataset  # noqa: E402
from src.data.dataset import protocol_ids_and_systems  # noqa: E402
from src.features import FeatureExtractor  # noqa: E402
from src.metrics import compute_eer  # noqa: E402
from src.models import build_model  # noqa: E402
from src.scores import checkpoint_fingerprint, load_scores, scores_path  # noqa: E402

OUTPUT_DIR = Path("outputs")
RULES = ("mean", "rank", "max", "min")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fusão de scores entre modelos treinados")
    # Dois argumentos separados (em vez de "config:checkpoint") para não haver
    # ambiguidade com a letra de drive dos caminhos do Windows (C:\...).
    p.add_argument("--model", action="append", required=True, nargs=2,
                   metavar=("CONFIG", "CHECKPOINT"),
                   help="config e checkpoint de um modelo; repita a opção por modelo")
    p.add_argument("--partition", default="eval", choices=["train", "dev", "eval"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--name", default="fusion_scores", help="prefixo dos arquivos de saída")
    p.add_argument("--recompute", action="store_true",
                   help="refaz a inferência mesmo havendo scores salvos por evaluate.py")
    return p.parse_args()


@torch.no_grad()
def scores_of_model(config_path: str, ckpt_path: str, partition: str,
                    smoke: bool, device_arg: str | None, recompute: bool = False):
    """Scores de um modelo na partição: (labels, scores, system_ids, ids).

    Reaproveita o `.npz` deixado por `evaluate.py` quando ele descreve este mesmo
    checkpoint e protocolo — a fusão combina N modelos, e sem isso seriam N
    passadas completas de inferência só para reobter números já calculados.
    """
    config = load_config(config_path)
    set_seed(config["experiment"]["seed"])
    name = output_name(config, smoke)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    impressao = checkpoint_fingerprint(ckpt["model_state"])

    ds = None
    if smoke:
        extractor = FeatureExtractor(config["audio"], config["features"])
        ds = build_dataset(config, partition, extractor, smoke)
        ids, sistemas = list(ds.ids), list(ds.system_ids)
    else:
        ids, sistemas = protocol_ids_and_systems(config, partition)

    if not recompute:
        reuso, motivo = load_scores(scores_path(OUTPUT_DIR, name, partition),
                                    ids=ids, fingerprint=impressao, partition=partition)
        if reuso is not None:
            labels, scores, systems = reuso
            print(f"  scores {motivo} (inferência não repetida)")
            return labels, scores, systems, ids

    device = resolve_device(device_arg or config["train"]["device"])
    if ds is None:
        extractor = FeatureExtractor(config["audio"], config["features"])
        ds = build_dataset(config, partition, extractor, smoke)
    batch = config["smoke"]["batch_size"] if smoke else config["train"]["batch_size"]
    loader = DataLoader(ds, batch_size=batch, shuffle=False,
                        num_workers=0 if smoke else config["train"]["num_workers"],
                        worker_init_fn=seed_worker)

    model = build_model(config["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    labels, scores = [], []
    for features, y in loader:
        features = {k: v.to(device) for k, v in features.items()}
        scores.append(torch.softmax(model(features), dim=1)[:, 1].cpu().numpy())
        labels.append(y.numpy())
    systems = np.array(getattr(ds, "system_ids", ["-"] * len(ds)))
    return np.concatenate(labels), np.concatenate(scores), systems, list(ds.ids)


def to_ranks(scores: np.ndarray) -> np.ndarray:
    """Converte scores em postos normalizados em [0, 1].

    Torna a combinação imune a diferenças de calibração: só a *ordenação* de
    cada modelo importa, não a escala absoluta das suas probabilidades.

    Empates recebem o posto **médio**. Isso não é detalhe: o softmax satura em
    exatamente 1.0 para muitas amostras (medido: ~60 mil das 71 mil do conjunto
    de avaliação), e desempatar pela ordem do array inventaria uma ordenação que
    o modelo não produziu — alterando o EER em vários pontos percentuais.
    """
    from scipy.stats import rankdata

    return (rankdata(scores, method="average") - 1) / max(len(scores) - 1, 1)


def combine(all_scores: list[np.ndarray], rule: str) -> np.ndarray:
    stack = np.vstack(all_scores)
    if rule == "mean":
        return stack.mean(axis=0)
    if rule == "rank":
        return np.vstack([to_ranks(s) for s in all_scores]).mean(axis=0)
    if rule == "max":
        return stack.max(axis=0)
    if rule == "min":
        return stack.min(axis=0)
    raise ValueError(f"regra desconhecida: {rule}")


def eer_per_attack(labels, scores, systems) -> dict[str, float]:
    bona = labels == 0
    out = {}
    for atk in sorted({s for s, lab in zip(systems, labels) if lab == 1}):
        subset = bona | ((labels == 1) & (systems == atk))
        out[atk] = compute_eer(labels[subset], scores[subset])
    return out


def main() -> None:
    args = parse_args()
    specs = [(cfg, ckpt) for cfg, ckpt in args.model]
    if len(specs) < 2:
        raise SystemExit("Informe pelo menos dois modelos com --model.")

    labels_ref = None
    ids_ref = None
    systems = None
    per_model: dict[str, np.ndarray] = {}

    for cfg_path, ckpt_path in specs:
        tag = Path(cfg_path).stem
        print(f"Rodando {tag} ...", flush=True)
        labels, scores, sys_ids, ids = scores_of_model(
            cfg_path, ckpt_path, args.partition, args.smoke, args.device,
            recompute=args.recompute)
        if labels_ref is None:
            labels_ref, ids_ref, systems = labels, ids, sys_ids
        elif ids_ref != ids:
            raise SystemExit(
                f"{tag} avaliou uma lista de áudios diferente — os modelos "
                "precisam usar o mesmo protocolo e partição.")
        per_model[tag] = scores
        print(f"  EER individual: {compute_eer(labels, scores) * 100:.2f}%")

    print("\n=== FUSÃO ===")
    results = {"individual": {t: compute_eer(labels_ref, s) for t, s in per_model.items()}}
    best_rule, best_eer = None, float("inf")
    for rule in RULES:
        fused = combine(list(per_model.values()), rule)
        eer = compute_eer(labels_ref, fused)
        results[rule] = {"eer": eer, "per_attack": eer_per_attack(labels_ref, fused, systems)}
        flag = ""
        if eer < best_eer:
            best_eer, best_rule, flag = eer, rule, "  <-- melhor"
        print(f"  {rule:5s}: EER = {eer * 100:5.2f}%{flag}")

    melhor_isolado = min(results["individual"].values())
    print(f"\nMelhor modelo isolado : {melhor_isolado * 100:.2f}%")
    print(f"Melhor fusão ({best_rule:5s}) : {best_eer * 100:.2f}%  "
          f"({(best_eer - melhor_isolado) * 100:+.2f} pp)")

    print(f"\n=== EER POR ATAQUE (regra '{best_rule}') ===")
    pa = results[best_rule]["per_attack"]
    for atk in sorted(pa, key=lambda a: pa[a]):
        print(f"  {atk:>5s} {pa[atk] * 100:7.2f}%")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / f"{args.name}_{args.partition}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"models": [c for c, _ in specs], "results": results}, fh, indent=2)
    print(f"\nResultados (JSON): {out}")


if __name__ == "__main__":
    main()
