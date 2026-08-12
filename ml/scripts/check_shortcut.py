"""Procura atalhos triviais na base: o modelo aprende voz ou aprende artefato?

Um detector pode atingir EER baixo sem olhar para a voz, se bonafide e spoof
diferirem em alguma propriedade banal do arquivo. Na literatura do ASVspoof
isso é documentado: a duração do silêncio no início e no fim das gravações
difere entre as classes, e modelos treinados sem cuidado aprendem a contar
silêncio em vez de detectar síntese.

Este script mede três atalhos conhecidos **antes** de qualquer feature ou
modelo, direto no waveform:

  1. duração original do áudio
  2. duração após remover o silêncio (`trim_silence`)
  3. energia RMS

Para cada um, calcula o EER que um classificador **trivial** obteria usando
somente aquela grandeza. Se algum deles ficar longe de 50%, a base carrega esse
atalho — e parte do desempenho do detector pode vir dali, não da voz.

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


def eer_trivial(rotulos: np.ndarray, valores: np.ndarray) -> float:
    """EER usando só `valores` como score. Testa os dois sentidos.

    Um atalho pode correlacionar em qualquer direção (spoof mais longo ou mais
    curto), então o pior caso para a hipótese "não há atalho" é o melhor dos
    dois sentidos.
    """
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

    print(f"{'grandeza':28s} {'bonafide':>12s} {'spoof':>12s} {'EER trivial':>12s}")
    print("-" * 68)
    achados = []
    for nome, vals in (("duração original (s)", np.array(dur_bruta)),
                       ("duração sem silêncio (s)", np.array(dur_limpa)),
                       ("energia RMS", np.array(rms))):
        mb, ms = vals[rotulos == 0].mean(), vals[rotulos == 1].mean()
        eer = eer_trivial(rotulos, vals) * 100
        marca = "  <-- ATALHO" if eer < 40 else ""
        print(f"{nome:28s} {mb:12.3f} {ms:12.3f} {eer:11.2f}%{marca}")
        if eer < 40:
            achados.append((nome, eer))

    # `fix_length` completa por repetição quem tem menos que `duration`; se a
    # proporção de repetidos diferir por classe, o próprio padding vira pista.
    curtos = np.array(dur_limpa) < alvo_s
    print(f"\nÁudios mais curtos que {alvo_s:.0f}s (recebem padding por repetição):")
    print(f"  bonafide {100 * curtos[rotulos == 0].mean():5.1f}%  |  "
          f"spoof {100 * curtos[rotulos == 1].mean():5.1f}%")

    print("\n" + "=" * 68)
    if achados:
        print("[AVISO] a base carrega atalho(s) triviais:")
        for nome, eer in achados:
            print(f"          {nome}: EER {eer:.2f}% só com essa grandeza")
        print("        Parte do desempenho do detector pode vir daqui, e não da voz.")
        return 1
    print("[ OK ] nenhuma das grandezas triviais separa as classes "
          "(todas perto de 50%).")
    print("       O desempenho do detector não pode ser explicado por elas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
