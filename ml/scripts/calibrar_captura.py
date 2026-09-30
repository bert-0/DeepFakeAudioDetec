"""Recalibra o limiar de um checkpoint para a captura ao vivo.

**Por que.** A captura do monitor passa por 48 kHz e volta a 16 kHz, e a
conversão apaga a faixa de 7,6-8 kHz. Medido (`robustness_eval.py --so clean
captura_48k`, 10.002 áudios): a ordenação do `baseline_v2` perde ~5 pp (19,02%
-> 24,70%), mas o limiar calibrado em áudio limpo colapsa — os humanos acima
do limiar vão de 10% a 71%. Como a ordenação sobrevive, basta mover o limiar.

**Como, sem vazar o eval.** O limiar é o ponto de EER no **dev** (ataques
A01-A06, os do treino) passado pela mesma ida e volta — nunca no eval. O
efeito se mede depois no eval (A07-A19), com o `robustness_eval.py`.

**O que grava.** Uma cópia do checkpoint, com os MESMOS pesos e o limiar novo,
mais os metadados da calibração. O original não é tocado; o monitor usa a
cópia sem mudança nenhuma de código.

Uso:
    python scripts/calibrar_captura.py --config configs/baseline_v2.yaml \\
        --checkpoint checkpoints/baseline_lfcc_cnn_v2.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.robustness_eval import indices_estratificados, run_inference  # noqa: E402
from src.config import load_config, resolve_device, seed_worker, set_seed  # noqa: E402
from src.data import build_dataset  # noqa: E402
from src.features import FeatureExtractor  # noqa: E402
from src.metrics import compute_eer, compute_eer_with_threshold  # noqa: E402
from src.models import build_model  # noqa: E402
from src.preprocess.channel import ChannelDegradation  # noqa: E402

#: Acima desta fração de bonafide em probabilidade 1,0, não existe limiar em
#: probabilidade que os separe — o caso do fusion_v4 na captura (100%).
SATURACAO_MAXIMA = 0.05


def taxas_no_limiar(labels: np.ndarray, probs: np.ndarray, limiar: float) -> tuple[float, float]:
    """(fração de bonafide acima do limiar, fração de spoof abaixo)."""
    bona, spoof = probs[labels == 0], probs[labels == 1]
    return float((bona >= limiar).mean()), float((spoof < limiar).mean())


def caminho_saida(checkpoint: str | Path) -> Path:
    """O mesmo lugar onde o monitor e o infer procuram a cópia (src/limiares.py)."""
    from src.limiares import caminho_captura

    return caminho_captura(checkpoint)


def calibrar(labels, probs, logodds, limiar_original: float | None) -> dict:
    """Decide o limiar novo e diz se ele serve. Separado do I/O para testar."""
    bona = probs[labels == 0]
    saturados = float((bona >= 1.0).mean()) if bona.size else 0.0
    eer_prob, limiar = compute_eer_with_threshold(labels, probs)
    resultado = {
        "limiar": float(limiar),
        "eer_probabilidade": float(eer_prob),
        "eer_logodds": float(compute_eer(labels, logodds)),
        "bonafide_saturados": saturados,
        "serve": saturados <= SATURACAO_MAXIMA,
    }
    if limiar_original is not None:
        resultado["antes"] = taxas_no_limiar(labels, probs, limiar_original)
    resultado["depois"] = taxas_no_limiar(labels, probs, limiar)
    return resultado


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Recalibra o limiar para a captura ao vivo")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--amostra", type=int, default=10000, metavar="N",
                   help="áudios do dev, estratificados por ataque (0 = todos)")
    p.add_argument("--taxa", type=int, default=48000, help="taxa do dispositivo de captura")
    p.add_argument("--saida", default=None, help="padrão: <checkpoint>_captura.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    config = load_config(args.config)
    seed = config["experiment"]["seed"]
    set_seed(seed)
    device = resolve_device(args.device or config["train"]["device"])
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(config["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    limiar_original = ckpt.get("threshold")

    sr = int(config["audio"]["sample_rate"])
    extractor = FeatureExtractor(config["audio"], config["features"])
    ds = build_dataset(config, "dev", extractor, args.smoke,
                       augmenter=ChannelDegradation("captura", args.taxa, sr))
    if args.amostra and not args.smoke:
        ds = Subset(ds, indices_estratificados(ds.labels, ds.system_ids, args.amostra, seed))
    batch = config["smoke"]["batch_size"] if args.smoke else config["train"]["batch_size"]
    loader = DataLoader(ds, batch_size=batch, shuffle=False,
                        num_workers=0 if args.smoke else config["train"]["num_workers"],
                        worker_init_fn=seed_worker)
    print(f"Calibrando no dev passado pela captura {args.taxa // 1000} kHz "
          f"({len(ds)} áudios) | dispositivo: {device}")
    labels, _, probs, logodds = run_inference(model, loader, device)
    r = calibrar(labels, probs, logodds, limiar_original)

    print(f"\nEER no dev com captura: {r['eer_logodds'] * 100:.2f}% (log-odds) | "
          f"{r['eer_probabilidade'] * 100:.2f}% (probabilidade)")
    print(f"Bonafide em probabilidade 1,0: {r['bonafide_saturados'] * 100:.1f}%")
    if "antes" in r:
        fa, fr = r["antes"]
        print(f"Limiar original {limiar_original:.4f}: {fa * 100:5.1f}% dos humanos acima, "
              f"{fr * 100:5.1f}% dos sintéticos abaixo")
    fa, fr = r["depois"]
    print(f"Limiar novo     {r['limiar']:.4f}: {fa * 100:5.1f}% dos humanos acima, "
          f"{fr * 100:5.1f}% dos sintéticos abaixo")

    if not r["serve"]:
        print(f"\n[RECUSADO] {r['bonafide_saturados'] * 100:.0f}% dos humanos saturam em "
              "probabilidade 1,0: nenhum limiar em probabilidade os separa. Este modelo "
              "não serve ao monitor sob captura — a ordenação (log-odds) sobrevive, a "
              "probabilidade não. Nada foi gravado.")
        return 2

    saida = Path(args.saida) if args.saida else caminho_saida(args.checkpoint)
    payload = dict(ckpt)
    payload["threshold"] = r["limiar"]
    payload["calibracao"] = {
        "condicao": "captura", "taxa": args.taxa, "particao": "dev",
        "n_audios": int(len(labels)), "threshold_original": limiar_original,
        "eer_dev_logodds": r["eer_logodds"],
    }
    saida.parent.mkdir(parents=True, exist_ok=True)
    tmp = saida.with_suffix(saida.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(saida)
    print(f"\nGravado: {saida} (mesmos pesos, limiar novo). O original não foi tocado.")
    print("\nMeça o efeito no EVAL (ataques que a calibração não viu):")
    print(f"  python scripts/robustness_eval.py --config {args.config} --checkpoint "
          f"{saida.as_posix()} --amostra 10000 --so clean captura_48k")
    print("O monitor e o infer.py acham esta cópia sozinhos a partir do checkpoint "
          "original e escolhem o limiar pelo tipo de áudio (src/limiares.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
