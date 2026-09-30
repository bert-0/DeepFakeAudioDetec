"""O que o caminho do som fez com o espectro: sessão B contra sessão A.

Compara, nos mesmos áudios, a energia média por faixa de frequência (depois
da normalização por pico, como o modelo vê) e o piso de ruído das pausas.
Três assinaturas possíveis, cada uma com um culpado diferente:

- **faixa do topo apagada** (7,5-8 kHz muito abaixo) — reamostragem;
- **inclinação** (graves ou agudos reforçados de forma suave) — equalização,
  "aprimoramentos de áudio" do driver;
- **piso de ruído mais alto** nas pausas — ruído somado no caminho.

Uso:
    python scripts/comparar_espectro.py --pasta outputs/canal_real --a limpo --b controle
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

FAIXAS = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 3000),
          (3000, 4000), (4000, 5000), (5000, 6000), (6000, 7000),
          (7000, 7500), (7500, 7800), (7800, 8000)]


def _ids(sessao: Path) -> list[str]:
    proto = sessao / "protocolo_canal_real.txt"
    return [ln.split()[1] for ln in proto.read_text(encoding="utf-8").splitlines() if ln.strip()]


def espectro_e_piso(wav: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, float]:
    """PSD média (Welch) e piso: energia dos 10% de quadros mais baixos,
    em dB relativos ao pico — depois da normalização por pico."""
    from scipy.signal import welch

    x = np.asarray(wav, dtype=np.float64)
    pico = np.abs(x).max()
    if pico > 0:
        x = x / pico
    f, p = welch(x, sr, nperseg=1024)
    quadro = int(0.02 * sr)
    n = len(x) // quadro
    energias = (x[: n * quadro].reshape(n, quadro) ** 2).mean(axis=1)
    baixos = np.sort(energias)[: max(1, n // 10)]
    piso = 10 * np.log10(baixos.mean() + 1e-20)
    return f, p, piso


def comparar(pasta: Path, a: str, b: str):
    ids = sorted(set(_ids(pasta / a)) & set(_ids(pasta / b)))
    dif_faixas, pisos_a, pisos_b = [], [], []
    for aid in ids:
        wa, sr = sf.read(pasta / a / "capturado" / f"{aid}.flac", dtype="float32")
        wb, _ = sf.read(pasta / b / "capturado" / f"{aid}.flac", dtype="float32")
        f, pa, piso_a = espectro_e_piso(wa, sr)
        _, pb, piso_b = espectro_e_piso(wb, sr)
        linha = []
        for lo, hi in FAIXAS:
            m = (f >= lo) & (f < hi)
            linha.append(10 * np.log10((pb[m].mean() + 1e-20) / (pa[m].mean() + 1e-20)))
        dif_faixas.append(linha)
        pisos_a.append(piso_a)
        pisos_b.append(piso_b)
    return ids, np.array(dif_faixas), np.array(pisos_a), np.array(pisos_b)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Diferença de espectro entre duas sessões")
    p.add_argument("--pasta", default="outputs/canal_real")
    p.add_argument("--a", default="limpo")
    p.add_argument("--b", required=True)
    args = p.parse_args(argv)

    ids, dif, pa, pb = comparar(Path(args.pasta), args.a, args.b)
    if not ids:
        print("[ERRO] nenhum áudio em comum entre as sessões.")
        return 1
    print(f"{len(ids)} áudios em comum. Diferença {args.b} − {args.a}, em dB "
          "(mediana entre os áudios; 0 = igual):\n")
    for (lo, hi), col in zip(FAIXAS, dif.T):
        barra = "#" * min(40, int(abs(np.median(col)) * 2))
        print(f"  {lo:5d}-{hi:<5d} Hz  {np.median(col):+7.1f} dB  {barra}")
    print(f"\nPiso de ruído (10% de quadros mais baixos, dB do pico): "
          f"{args.a} {np.median(pa):6.1f} | {args.b} {np.median(pb):6.1f} "
          f"({np.median(pb - pa):+.1f} dB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
