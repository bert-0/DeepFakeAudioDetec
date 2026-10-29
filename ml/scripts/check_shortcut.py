"""Procura atalhos triviais na base (duração, duração sem silêncio, energia RMS).

Para cada grandeza calcula o EER de um classificador trivial e o compara com um
teste de permutação dos rótulos.

Uso:
    python scripts/check_shortcut.py --config configs/fusion_v4.yaml
    python scripts/check_shortcut.py --config configs/fusion_v4.yaml --n 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.data.dataset import parse_protocol  # noqa: E402
from src.metrics import compute_eer  # noqa: E402
from src.preprocess.audio import load_audio, trim_silence  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Procura atalhos triviais na base")
    p.add_argument("--config", required=True)
    p.add_argument("--particao", default="train", choices=["train", "dev", "eval"])
    p.add_argument("--n", type=int, default=3000,
                   help="quantos áudios amostrar (padrão: 3000)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def teste_permutacao(rotulos: np.ndarray, valores: np.ndarray,
                     n: int = 500, seed: int = 0) -> tuple[float, float]:
    """Distribuição nula do EER trivial, embaralhando os rótulos.

    Devolve (média do nulo, percentil 5); com classes desbalanceadas o acaso
    não dá exatamente 50%.
    """
    rng = np.random.default_rng(seed)
    nulos = np.empty(n)
    embaralhado = rotulos.copy()
    for i in range(n):
        rng.shuffle(embaralhado)
        nulos[i] = eer_trivial(embaralhado, valores)
    return float(nulos.mean()), float(np.percentile(nulos, 5))


def eer_trivial(rotulos: np.ndarray, valores: np.ndarray) -> float:
    """EER usando só `valores` como score, no melhor dos dois sentidos."""
    a = compute_eer(rotulos, valores)
    b = compute_eer(rotulos, -valores)
    return min(a, b)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    sr = int(config["audio"]["sample_rate"])
    top_db = config["audio"].get("top_db", 30)
    alvo_s = float(config["audio"]["duration"])

    protocolo = config["data"]["protocols"][args.particao]
    audio_dir = Path(config["data"]["audio_dir"][args.particao])
    itens = parse_protocol(Path(protocolo))
    if not itens:
        print(f"[ERRO] protocolo vazio ou ilegível: {protocolo}")
        return 1

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(itens))[:args.n]
    print(f"Base: {args.particao} | amostrando {len(idx)} de {len(itens)} áudios\n")

    rotulos, dur_bruta, dur_limpa, rms = [], [], [], []
    for k, i in enumerate(idx):
        nome, rotulo = itens[i]
        try:
            wav = load_audio(audio_dir / f"{nome}.flac", sr)
        except Exception:
            continue
        limpo = trim_silence(wav, top_db)
        rotulos.append(rotulo)
        dur_bruta.append(len(wav) / sr)
        dur_limpa.append(len(limpo) / sr)
        rms.append(float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))))
        if k and k % 1000 == 0:
            print(f"       ... {k}/{len(idx)}", flush=True)

    rotulos = np.array(rotulos)
    n_bona = int((rotulos == 0).sum())
    n_spoof = int((rotulos == 1).sum())
    print(f"\nAmostra: {n_bona} bonafide, {n_spoof} spoof\n")

    print(f"{'grandeza':26s} {'bonafide':>10s} {'spoof':>10s} {'EER':>8s} "
          f"{'acaso':>8s} {'p5':>7s}")
    print("-" * 78)
    achados = []
    for nome, vals in (("duração original (s)", np.array(dur_bruta)),
                       ("duração sem silêncio (s)", np.array(dur_limpa)),
                       ("energia RMS", np.array(rms))):
        mb, ms = vals[rotulos == 0].mean(), vals[rotulos == 1].mean()
        eer = eer_trivial(rotulos, vals) * 100
        nulo, p5 = teste_permutacao(rotulos, vals)
        nulo, p5 = nulo * 100, p5 * 100
        significativo = eer < p5
        marca = "  <-- SINAL REAL" if significativo else ""
        print(f"{nome:26s} {mb:10.3f} {ms:10.3f} {eer:7.2f}% {nulo:7.2f}% "
              f"{p5:6.2f}%{marca}")
        if significativo:
            achados.append((nome, eer, p5))

    # `fix_length` completa por repetição quem tem menos que `duration`; se a
    # proporção de repetidos diferir por classe, o próprio padding vira pista.
    curtos = np.array(dur_limpa) < alvo_s
    print(f"\nÁudios mais curtos que {alvo_s:.0f}s (recebem padding por repetição):")
    print(f"  bonafide {100 * curtos[rotulos == 0].mean():5.1f}%  |  "
          f"spoof {100 * curtos[rotulos == 1].mean():5.1f}%")

    print("\n" + "=" * 68)
    if achados:
        print("[AVISO] há sinal estatisticamente real em grandeza(s) trivial(is):")
        for nome, eer, p5 in achados:
            print(f"          {nome}: EER {eer:.2f}% (acaso não desce de {p5:.2f}%)")
        print("        Isso NÃO significa que o detector dependa disso — compare a\n"
              "        magnitude: um atalho de ~44% não explica um modelo de ~19%.\n"
              "        Mas precisa ser declarado na metodologia.")
        return 1
    print("[ OK ] nenhuma grandeza trivial separa as classes além do acaso.")
    print("       O desempenho do detector não pode ser explicado por elas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
