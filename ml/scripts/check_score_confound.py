"""Mede se o score do detector depende de grandezas triviais (energia, duração).

Complementa o `check_shortcut.py`: usa a correlação de Spearman dentro de cada
classe, porque a global reflete só o efeito da classe nas duas grandezas.

Uso:
    python scripts/check_score_confound.py --config configs/fusion_v4.yaml \\
        --scores outputs/fusion_lcnn_v4_eval_scores.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.preprocess.audio import load_audio, trim_silence  # noqa: E402

#: Acima disto a grandeza explica parte relevante do score.
FORTE = 0.30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mede se o score do modelo depende de grandezas triviais")
    p.add_argument("--config", required=True)
    p.add_argument("--scores", required=True, help="arquivo .npz de evaluate.py")
    p.add_argument("--particao", default="eval", choices=["train", "dev", "eval"])
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def piso_de_ruido(n: int) -> float:
    """|rho| que o acaso produz com `n` pontos (~2 erros-padrão de Spearman)."""
    return 2.0 / np.sqrt(max(n - 1, 2))


def classificar(rho: float, piso: float) -> str:
    a = abs(rho)
    if a >= FORTE:
        return "DEPENDE"
    if a >= piso:
        return "acima do ruído"
    return "desprezível"


def main() -> int:
    from scipy.stats import spearmanr

    args = parse_args()
    config = load_config(args.config)
    sr = int(config["audio"]["sample_rate"])
    top_db = config["audio"].get("top_db", 30)
    audio_dir = Path(config["data"]["audio_dir"][args.particao])

    dados = np.load(args.scores, allow_pickle=False)
    ids, labels, scores = dados["ids"], dados["labels"], dados["scores"]
    print(f"Scores: {args.scores}  ({len(ids)} áudios)")

    rng = np.random.default_rng(args.seed)
    sel = rng.permutation(len(ids))[:args.n]

    rot, sco, rms, dur = [], [], [], []
    for k, i in enumerate(sel):
        try:
            wav = load_audio(audio_dir / f"{ids[i]}.flac", sr)
        except Exception:
            continue
        rot.append(int(labels[i]))
        sco.append(float(scores[i]))
        rms.append(float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))))
        dur.append(len(trim_silence(wav, top_db)) / sr)
        if k and k % 1000 == 0:
            print(f"       ... {k}/{len(sel)}", flush=True)

    rot = np.array(rot)
    sco = np.array(sco)
    grandezas = {"energia RMS": np.array(rms), "duração sem silêncio": np.array(dur)}
    print(f"\nAmostra: {(rot == 0).sum()} bonafide, {(rot == 1).sum()} spoof\n")

    piso = piso_de_ruido(min((rot == 0).sum(), (rot == 1).sum()))
    print(f"Piso de ruído desta amostra: |rho| = {piso:.3f} "
          f"(2 erros-padrão de Spearman)\n")
    print(f"{'grandeza':24s} {'global':>9s} {'bonafide':>10s} {'spoof':>9s}  veredito")
    print("-" * 72)
    dependencias = []
    for nome, vals in grandezas.items():
        # Grandeza constante: o spearmanr daria NaN e o veredito, "desprezível".
        if np.ptp(vals) == 0:
            print(f"{nome:24s} {'—':>9s} {'—':>10s} {'—':>9s}  constante, não avaliável")
            continue
        g = spearmanr(sco, vals).statistic
        b = spearmanr(sco[rot == 0], vals[rot == 0]).statistic
        s = spearmanr(sco[rot == 1], vals[rot == 1]).statistic
        pior = max(abs(b), abs(s))
        veredito = classificar(pior, piso)
        print(f"{nome:24s} {g:+9.3f} {b:+10.3f} {s:+9.3f}  {veredito}")
        if pior >= piso:
            dependencias.append((nome, pior, veredito))

    print("\n" + "=" * 72)
    print("A coluna 'global' confunde: ela sobe só porque a classe muda as duas\n"
          "coisas ao mesmo tempo. Decidem as duas colunas de dentro da classe.\n")
    if not dependencias:
        print("[ OK ] dentro de cada classe o score não acompanha nenhuma das\n"
              "       grandezas triviais. O detector não se apoia nelas — o que é\n"
              "       coerente com `peak_normalize` e a normalização por instância\n"
              "       das features, que removem nível e escala.")
        return 0
    print("[AVISO] o score acompanha grandeza(s) trivial(is) dentro da classe:")
    for nome, rho, veredito in dependencias:
        print(f"          {nome}: |rho| = {rho:.3f} (piso {piso:.3f}) — {veredito}")
    print("        Parte da decisão do modelo vem daí, e não do conteúdo\n"
          "        espectral. Precisa ser declarado.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
