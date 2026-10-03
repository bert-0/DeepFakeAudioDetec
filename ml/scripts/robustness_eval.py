"""Avaliação de robustez (TC1 §5.5): EER sob ruído, ganho e degradações de canal.

Uso:
    python scripts/robustness_eval.py --config configs/baseline.yaml \
        --checkpoint checkpoints/baseline_lfcc_cnn.pt
    python scripts/robustness_eval.py --config configs/attention.yaml \
        --checkpoint checkpoints/attention_fusion.pt --smoke
    # conferência barata: 10 mil áudios estratificados por ataque
    python scripts/robustness_eval.py --config configs/fusion_v4.yaml \
        --checkpoint checkpoints/fusion_lcnn_v4.pt --amostra 10000

O EER sai dos log-odds (logit[1] - logit[0]), que não saturam. Cada linha mostra
também o EER pela probabilidade e quantos bonafide saturaram em 1,0. Os scores de
cada áudio vão para um `.npz` por condição (usado por `fusao_ao_vivo.py`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    load_config,
    output_name,
    resolve_device,
    seed_worker,
    set_seed,
)
from src.data import build_dataset  # noqa: E402
from src.features import FeatureExtractor  # noqa: E402
from src.metrics import (  # noqa: E402
    compute_eer,
    compute_metrics,
    format_metrics,
    probabilidade_e_logodds,
)
from src.models import build_model  # noqa: E402
from src.preprocess.augment import make_perturbation  # noqa: E402
from src.preprocess.channel import (  # noqa: E402
    ChannelChain,
    ChannelDegradation,
    medir_taxa_kbps,
)

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

# Condições de canal: o que uma chamada (Teams/Meet/Zoom) faz com o áudio.
# O nível do opus é o compression_level em [0,1]; como o Opus é VBR, a taxa é
# medida em execução. Níveis escolhidos para a faixa de voz do Teams (16-32 kbps).
CHANNEL_CONDITIONS = [
    ("opus_30kbps", [("opus", 0.90)]),        # rede boa
    ("opus_25kbps", [("opus", 0.92)]),        # típico
    ("opus_15kbps", [("opus", 0.96)]),        # rede apertada
    ("banda_estreita", [("band", 8000)]),     # telefonia: nada acima de 4 kHz
    ("banda_estreita_opus", [("band", 8000), ("opus", 0.92)]),  # o caso realista
    # Captura ao vivo (16 -> 48 -> 16 kHz): apaga só 7,6-8 kHz, e isso sozinho
    # inflou todos os scores no teste ao vivo (Seção 10.2.1 do resumo).
    ("captura_48k", [("captura", 48000)]),            # soxr (o de antes)
    ("captura_48k_fir", [("captura_fir", 48000)]),    # FIR longo (o atual)
]


def construir_canal(etapas, sample_rate):
    """Monta a degradação de canal de uma condição."""
    degradacoes = [ChannelDegradation(kind, level, sample_rate)
                   for kind, level in etapas]
    return degradacoes[0] if len(degradacoes) == 1 else ChannelChain(degradacoes)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Avaliação de robustez sob perturbações")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--partition", default="eval", choices=["train", "dev", "eval"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--sem-canal", action="store_true",
                   help="roda só as condições acústicas (ruído/ganho), pulando "
                        "as de canal (codec e banda estreita)")
    p.add_argument("--so", nargs="+", default=None, metavar="CONDICAO",
                   help="roda só estas condições (ex.: clean captura_48k)")
    p.add_argument("--amostra", type=int, default=None, metavar="N",
                   help="avalia só N áudios, estratificados por ataque (e bonafide), "
                        "com a semente do config — mesma amostra em toda condição")
    return p.parse_args()


def indices_estratificados(labels, systems, n: int, seed: int) -> list[int]:
    """N índices com a mesma proporção de cada (rótulo, ataque), sem perder ataque raro."""
    grupos: dict[tuple, list[int]] = {}
    for i, chave in enumerate(zip(labels, systems)):
        grupos.setdefault(chave, []).append(i)
    total = len(labels)
    if n >= total:
        return list(range(total))
    rng = np.random.default_rng(seed)
    escolhidos: list[int] = []
    for chave in sorted(grupos, key=str):
        membros = grupos[chave]
        k = max(1, round(n * len(membros) / total))
        escolhidos.extend(rng.choice(membros, size=min(k, len(membros)), replace=False).tolist())
    return sorted(escolhidos)


@torch.no_grad()
def run_inference(model, loader, device):
    """Devolve (labels, preds, P(spoof), log-odds)."""
    model.eval()
    labels, preds, scores, logodds = [], [], [], []
    for features, y in loader:
        features = {k: v.to(device) for k, v in features.items()}
        logits = model(features)
        probs, lo = probabilidade_e_logodds(logits)
        scores.append(probs)
        logodds.append(lo)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels.append(y.numpy())
    return (np.concatenate(labels), np.concatenate(preds),
            np.concatenate(scores), np.concatenate(logodds))


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


def relatar_taxas(sample_rate: int) -> None:
    """Mede a taxa real de cada nível do Opus (é VBR: o nível não garante a taxa)."""
    import numpy as np_

    rng = np_.random.default_rng(0)
    # Ruído rosa, parecido com o espectro da fala; um tom puro comprimiria demais.
    branco = rng.standard_normal(sample_rate * 4)
    espectro = np_.fft.rfft(branco)
    freqs = np_.fft.rfftfreq(len(branco), 1 / sample_rate)
    freqs[0] = freqs[1]
    referencia = np_.fft.irfft(espectro / np_.sqrt(freqs)).astype("float32")
    referencia = (0.3 * referencia / np_.abs(referencia).max()).astype("float32")

    print("Taxas do Opus medidas em ruído rosa de 4 s (o áudio real varia):")
    for label, etapas in CHANNEL_CONDITIONS:
        for kind, level in etapas:
            if kind == "opus":
                kbps = medir_taxa_kbps(referencia, sample_rate, level)
                print(f"  {label:22s} compression_level={level:<4} -> {kbps:6.1f} kbps")
    print()


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
    # Mesmo limiar de evaluate.py/infer.py. Com 0,5 fixo, o ruído pareceria um
    # colapso: os scores descem em bloco, mas a ordenação (e o EER) se mantém.
    threshold = ckpt.get("threshold") if config["train"].get("calibrate_threshold") else None
    if threshold is not None:
        print(f"Threshold aplicado: {threshold:.4f} (calibrado no treino)")

    batch_size = config["smoke"]["batch_size"] if args.smoke else config["train"]["batch_size"]
    print(f"Robustez | modelo: {config['model']['name']} | partição: {args.partition} | "
          f"dispositivo: {device}\n")

    sample_rate = int(config["audio"]["sample_rate"])
    condicoes: list[tuple[str, object]] = [
        (label, None if kind is None else make_perturbation(kind, level, seed=seed))
        for label, kind, level in CONDITIONS
    ]
    if not args.sem_canal:
        condicoes += [(label, construir_canal(etapas, sample_rate))
                      for label, etapas in CHANNEL_CONDITIONS]
        if not args.so or any(c.startswith("opus") or c.endswith("opus") for c in args.so):
            relatar_taxas(sample_rate)
    if args.so:
        conhecidas = {label for label, _ in condicoes}
        desconhecidas = set(args.so) - conhecidas
        if desconhecidas:
            raise SystemExit(f"condição desconhecida: {', '.join(sorted(desconhecidas))}. "
                             f"Disponíveis: {', '.join(sorted(conhecidas))}")
        condicoes = [(label, pert) for label, pert in condicoes if label in args.so]

    results: dict[str, dict] = {}
    por_audio: dict[str, dict] = {}
    indices = None
    for label, perturbation in condicoes:
        ds = build_dataset(config, args.partition, extractor, args.smoke, augmenter=perturbation)
        selecionados = list(range(len(ds)))
        if args.amostra:
            if indices is None:
                indices = indices_estratificados(ds.labels, ds.system_ids, args.amostra, seed)
                print(f"Amostra estratificada: {len(indices)} de {len(ds)} áudios\n")
            selecionados = indices
            base, ds = ds, Subset(ds, indices)
        else:
            base = ds
        # Sem cache: a perturbação muda as features. Como as perturbações são
        # serializáveis, os workers do config valem aqui também.
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=0 if args.smoke else config["train"]["num_workers"],
                            worker_init_fn=seed_worker)
        labels, preds, scores, logodds = run_inference(model, loader, device)
        metrics = compute_metrics(labels, preds, scores, threshold=threshold,
                                  eer_scores=logodds)
        # EER pela probabilidade (o que versões antigas reportavam), para conferir.
        metrics["eer_probabilidade"] = compute_eer(labels, scores)
        metrics["bonafide_saturados"] = int((scores[labels == 0] >= 1.0).sum())
        metrics["n_bonafide"] = int((labels == 0).sum())
        results[label] = metrics
        por_audio[label] = {
            "ids": np.array([base.ids[i] for i in selecionados]),
            "systems": np.array([base.system_ids[i] for i in selecionados]),
            "labels": labels, "scores": scores, "logodds": logodds,
            "threshold": np.nan if threshold is None else threshold,
        }
        print(f"{label:12s} -> {format_metrics(metrics)}  "
              f"[EER pela prob.={metrics['eer_probabilidade'] * 100:.2f}%  "
              f"bonafide em 1,0: {metrics['bonafide_saturados']}/{metrics['n_bonafide']}]")

    OUTPUT_DIR.mkdir(exist_ok=True)
    name = output_name(config, args.smoke)
    if args.amostra:
        name = f"{name}_amostra{args.amostra}"
    if args.so:
        name = f"{name}_" + "_".join(args.so)
    if ckpt.get("calibracao"):
        # Checkpoint recalibrado (calibrar_captura.py): o sufixo evita sobrescrever o original.
        name = f"{name}_limiar_captura"
    # A partição entra no nome: sem ela, uma execução em dev sobrescreve a de eval.
    json_path = OUTPUT_DIR / f"{name}_{args.partition}_robustness.json"
    plot_path = OUTPUT_DIR / f"{name}_{args.partition}_robustness.png"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    plot_robustness(results, plot_path)
    print(f"\nResultados (JSON): {json_path}")
    print(f"Gráfico:           {plot_path}")
    for label, dados in por_audio.items():
        npz_path = OUTPUT_DIR / f"{name}_{args.partition}_{label}_scores.npz"
        np.savez(npz_path, **dados)
        print(f"Scores ({label}):  {npz_path}")


if __name__ == "__main__":
    main()
