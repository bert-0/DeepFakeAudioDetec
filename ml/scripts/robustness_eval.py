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

from src.config import (  # noqa: E402
    load_config,
    output_name,
    resolve_device,
    seed_worker,
    set_seed,
)
from src.data import build_dataset  # noqa: E402
from src.features import FeatureExtractor  # noqa: E402
from src.metrics import compute_metrics, format_metrics  # noqa: E402
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

# Condições de CANAL: o que uma chamada de Teams/Meet/Zoom faz com o áudio.
# Separadas das acústicas acima porque respondem a outra pergunta — não "o
# modelo aguenta um ambiente ruidoso?", e sim "o modelo sobrevive ao meio de
# transmissão?". É a pergunta que decide se o monitor ao vivo (monitor.py) tem
# base para existir.
#
# `nivel` do opus é o compression_level em [0,1]; a taxa em kbps resultante
# depende do conteúdo (o Opus é VBR) e é medida e reportada em tempo de
# execução, em vez de anunciada.
# Os níveis foram calibrados para cair na faixa que o Teams usa de fato
# (16-32 kbps para voz), medida em ruído rosa de 4 s:
#   0.90 -> 30 kbps    0.92 -> 25 kbps    0.94 -> 19 kbps
#   0.96 -> 15 kbps    1.00 ->  6 kbps
# Varrer de 130 a 6 kbps testaria sobretudo faixas que uma chamada nunca usa.
CHANNEL_CONDITIONS = [
    ("opus_30kbps", [("opus", 0.90)]),        # rede boa
    ("opus_25kbps", [("opus", 0.92)]),        # típico
    ("opus_15kbps", [("opus", 0.96)]),        # rede apertada
    ("banda_estreita", [("band", 8000)]),     # telefonia: nada acima de 4 kHz
    ("banda_estreita_opus", [("band", 8000), ("opus", 0.92)]),  # o caso realista
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


def relatar_taxas(sample_rate: int) -> None:
    """Mostra a taxa que cada nível de compressão produz, medida.

    O Opus é VBR: o `compression_level` é um pedido, não uma garantia. Sem esta
    medição, o relatório citaria taxas nominais que o arquivo nunca teve.
    """
    import numpy as np_

    rng = np_.random.default_rng(0)
    # Ruído rosa: energia em todas as bandas, decrescente como a da fala. Um tom
    # puro comprimiria a quase nada e daria uma taxa irrealista.
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
    # Mesmo ponto de corte usado por evaluate.py/infer.py. Sem ele, accuracy e F1
    # sairiam no corte fixo de 0,5 e sugeririam um colapso que não existe: sob
    # ruído os scores descem em bloco sem perder a ordenação (o EER não muda).
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
        relatar_taxas(sample_rate)

    results: dict[str, dict] = {}
    for label, perturbation in condicoes:
        ds = build_dataset(config, args.partition, extractor, args.smoke, augmenter=perturbation)
        # As perturbações mudam as features, então o cache fica desligado e cada
        # condição recalcula tudo. Com `num_workers=0` isso era um único processo
        # extraindo 71.237 áudios seis vezes; as perturbações agora são
        # serializáveis, então os workers do config valem aqui também.
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=0 if args.smoke else config["train"]["num_workers"],
                            worker_init_fn=seed_worker)
        labels, preds, scores = run_inference(model, loader, device)
        metrics = compute_metrics(labels, preds, scores, threshold=threshold)
        results[label] = metrics
        print(f"{label:12s} -> {format_metrics(metrics)}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    name = output_name(config, args.smoke)
    # A partição entra no nome: sem ela, uma execução em dev sobrescreve a de eval.
    json_path = OUTPUT_DIR / f"{name}_{args.partition}_robustness.json"
    plot_path = OUTPUT_DIR / f"{name}_{args.partition}_robustness.png"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    plot_robustness(results, plot_path)
    print(f"\nResultados (JSON): {json_path}")
    print(f"Gráfico:           {plot_path}")


if __name__ == "__main__":
    main()
