"""Análise de erros por tipo de ataque (TC1 §5.4).

O EER global esconde onde o modelo realmente falha: uma média de 20% pode ser
"20% em todos os ataques" ou "0% em doze ataques e 90% em um". Este script
calcula o EER **por algoritmo de síntese** (A07…A19 no conjunto de avaliação),
sempre comparando aquele ataque contra TODOS os áudios bonafide — que é o
protocolo padrão de reporte da ASVspoof.

Uso:
    python scripts/per_attack_eval.py --config configs/baseline_v3a.yaml \
        --checkpoint checkpoints/baseline_lfcc_cnn_v3a.pt
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

from src.config import load_config, output_name, resolve_device, set_seed  # noqa: E402
from src.data import build_dataset  # noqa: E402
from src.data.dataset import protocol_ids_and_systems  # noqa: E402
from src.features import FeatureExtractor  # noqa: E402
from src.metrics import compute_eer  # noqa: E402
from src.models import build_model  # noqa: E402
from src.scores import checkpoint_fingerprint, load_scores, scores_path  # noqa: E402

OUTPUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EER por tipo de ataque")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--partition", default="eval", choices=["train", "dev", "eval"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--recompute", action="store_true",
                   help="refaz a inferência mesmo havendo scores salvos por evaluate.py")
    return p.parse_args()


@torch.no_grad()
def collect_scores(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, scores = [], []
    for features, y in loader:
        features = {k: v.to(device) for k, v in features.items()}
        probs = torch.softmax(model(features), dim=1)[:, 1]
        scores.append(probs.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(scores)


def eer_per_attack(
    labels: np.ndarray, scores: np.ndarray, systems: np.ndarray
) -> dict[str, dict]:
    """EER de cada ataque, avaliado contra todos os bonafide."""
    bona = labels == 0
    results: dict[str, dict] = {}
    for attack in sorted({s for s, lab in zip(systems, labels) if lab == 1}):
        subset = bona | ((labels == 1) & (systems == attack))
        results[attack] = {
            "eer": compute_eer(labels[subset], scores[subset]),
            "n_spoof": int(((labels == 1) & (systems == attack)).sum()),
        }
    return results


def plot_per_attack(results: dict[str, dict], global_eer: float, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(results, key=lambda a: results[a]["eer"])
    eers = [results[a]["eer"] * 100 for a in order]

    fig, ax = plt.subplots(figsize=(max(7, len(order) * 0.6), 4.2))
    colors = ["tab:red" if e > global_eer * 100 else "tab:blue" for e in eers]
    bars = ax.bar(order, eers, color=colors)
    ax.axhline(global_eer * 100, color="black", linestyle="--", linewidth=1,
               label=f"EER global = {global_eer * 100:.2f}%")
    ax.set_ylabel("EER (%)")
    ax.set_xlabel("Algoritmo de síntese")
    ax.set_title("EER por tipo de ataque")
    ax.legend()
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
    set_seed(config["experiment"]["seed"])
    device = resolve_device(args.device or config["train"]["device"])
    name = output_name(config, args.smoke)

    extractor = FeatureExtractor(config["audio"], config["features"])
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    impressao = checkpoint_fingerprint(ckpt["model_state"])

    ds = None
    if args.smoke:
        ds = build_dataset(config, args.partition, extractor, args.smoke)
        ids, systems = ds.ids, np.array(ds.system_ids)
    else:
        ids, sistemas = protocol_ids_and_systems(config, args.partition)
        systems = np.array(sistemas)

    # `evaluate.py` já percorreu esta partição com este mesmo checkpoint. Repetir
    # a inferência daria exatamente os mesmos scores — no `eval` do LA, 71.237
    # áudios de novo. O arquivo só é aceito se o modelo e o protocolo baterem.
    reuso, motivo = (None, "recálculo pedido com --recompute")
    if not args.recompute:
        reuso, motivo = load_scores(scores_path(OUTPUT_DIR, name, args.partition),
                                    ids=ids, fingerprint=impressao,
                                    partition=args.partition)

    if reuso is not None:
        labels, scores, systems = reuso
        print(f"Scores {motivo} de {scores_path(OUTPUT_DIR, name, args.partition)} "
              "— inferência não repetida.\n")
    else:
        print(f"Rodando a inferência ({motivo}).")
        if ds is None:
            ds = build_dataset(config, args.partition, extractor, args.smoke)
        if not hasattr(ds, "system_ids"):
            raise SystemExit("O dataset não expõe system_ids — protocolo incompatível.")
        systems = np.array(ds.system_ids)
        batch_size = (config["smoke"]["batch_size"] if args.smoke
                      else config["train"]["batch_size"])
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=0 if args.smoke else config["train"]["num_workers"])
        model = build_model(config["model"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Modelo: {config['model']['name']} | partição: {args.partition} | "
              f"dispositivo: {device}\n")
        labels, scores = collect_scores(model, loader, device)

    global_eer = compute_eer(labels, scores)
    results = eer_per_attack(labels, scores, systems)

    order = sorted(results, key=lambda a: results[a]["eer"])
    print(f"{'ataque':>8s} {'EER':>9s} {'amostras':>10s}")
    for attack in order:
        r = results[attack]
        print(f"{attack:>8s} {r['eer'] * 100:8.2f}% {r['n_spoof']:10d}")
    print(f"\n{'GLOBAL':>8s} {global_eer * 100:8.2f}% {int((labels == 1).sum()):10d}")

    pior = order[-1]
    melhor = order[0]
    print(f"\nMelhor: {melhor} ({results[melhor]['eer'] * 100:.2f}%)  |  "
          f"Pior: {pior} ({results[pior]['eer'] * 100:.2f}%)")
    acima = [a for a in order if results[a]["eer"] > global_eer]
    print(f"{len(acima)} de {len(order)} ataques ficam acima do EER global: {', '.join(acima)}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    json_path = OUTPUT_DIR / f"{name}_{args.partition}_per_attack.json"
    plot_path = OUTPUT_DIR / f"{name}_{args.partition}_per_attack.png"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"global_eer": global_eer, "per_attack": results}, fh, indent=2)
    plot_per_attack(results, global_eer, plot_path)
    print(f"\nResultados (JSON): {json_path}")
    print(f"Gráfico:           {plot_path}")


if __name__ == "__main__":
    main()
