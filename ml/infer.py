"""Inferência em um único arquivo de áudio (RF04/RF05/RF06/RF07 da APS).

Exemplo:
    python infer.py --config configs/baseline.yaml \
                    --checkpoint checkpoints/baseline_lfcc_cnn.pt \
                    --audio caminho/para/audio.wav
"""

from __future__ import annotations

import argparse

import torch

from src.config import load_config, resolve_device
from src.features import FeatureExtractor
from src.models import build_model
from src.preprocess import load_audio, preprocess_waveform

CLASS_NAMES = {0: "bonafide (autêntico)", 1: "spoof (deepfake)"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classifica um único áudio")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--audio", required=True, help="arquivo .wav/.mp3/.flac")
    p.add_argument("--device", default=None)
    p.add_argument("--threshold", type=float, default=None,
                   help="sobrescreve o threshold do checkpoint (padrão: o calibrado no dev)")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device or config["train"]["device"])

    extractor = FeatureExtractor(config["audio"], config["features"])
    wav = load_audio(args.audio, config["audio"]["sample_rate"])
    wav = preprocess_waveform(wav, config["audio"])
    features = {k: v.unsqueeze(0).to(device) for k, v in extractor(wav).items()}

    model = build_model(config["model"]).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    probs = torch.softmax(model(features), dim=1)[0]
    spoof_prob = float(probs[1])

    # Usa o mesmo ponto de corte calibrado no dev durante o treino; sem ele,
    # cai no argmax (equivalente a threshold 0,5).
    threshold = ckpt.get("threshold") if config["train"].get("calibrate_threshold") else None
    if args.threshold is not None:
        threshold = args.threshold
    pred = int(spoof_prob >= threshold) if threshold is not None else int(probs.argmax())
    confidence = float(probs[pred]) * 100

    print(f"\nArquivo: {args.audio}")
    print(f"Classificação: {CLASS_NAMES[pred]}")
    print(f"Probabilidade: {confidence:.1f}%")
    print(f"  (bonafide={probs[0] * 100:.1f}%  |  spoof={probs[1] * 100:.1f}%)")
    if threshold is not None:
        print(f"  threshold aplicado: {threshold:.4f}")


if __name__ == "__main__":
    main()
