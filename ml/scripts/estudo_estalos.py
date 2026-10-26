"""O que o caminho de captura põe na voz: o relógio dos estalos e o portão de ruído.

As gravações do microfone embutido "travam" ao ouvido: têm estalos em pares a
110 ms (RESUMO_TCC §10.2.16) e um pente de 9,09 Hz no espectro (§10.2.20). Este
estudo compara microfones (embutido, headset, celular) com os clones, que não
passam pelo microfone e servem de controle, em três medidas por pasta:
- a força do pente de 9,09 Hz (parte 4);
- o nível mediano do fundo nas pausas (quadros de 10 ms 25 dB abaixo da fala);
- os saltos de mais de 15 dB em 10 ms, a marca de um "portão" de redução de
  ruído, que corta o fundo nas pausas e solta de repente quando a voz volta.

A função `estalos` (picos de 1 ms acima de 7 kHz) fica para inspeção: na fala ela
confunde começo e fim de palavra com estalos, e por isso não entra no relatório.
Exploratório (RESUMO_TCC §10.2.33).

Uso (de dentro de ml/):
    python scripts/estudo_estalos.py --grupo embutido=data/vozes/ensaio2/real \\
        --grupo headset=data/vozes/outro_mic/headset --grupo clones=data/vozes/ensaio2/clone
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SR = 16000
QUADRO = 16            # 1 ms
CORTE_HZ = 7000.0
#: 12 deixava passar um falso a cada ~3 s de ruído branco (o quadro de 1 ms é curto).
FATOR = 20.0
VIZINHANCA = 200       # quadros (200 ms)
SEPARACAO = 5          # quadros
PAR_MS, TOLERANCIA_MS = 110.0, 3.0


def estalos(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """Instantes (s) dos estalos."""
    from scipy.ndimage import median_filter
    from scipy.signal import butter, sosfiltfilt

    alto = sosfiltfilt(butter(6, CORTE_HZ, "highpass", fs=sr, output="sos"), wav)
    n = len(alto) // QUADRO
    e = (alto[: n * QUADRO].reshape(n, QUADRO) ** 2).mean(axis=1) + 1e-14
    fundo = median_filter(e, size=VIZINHANCA, mode="nearest")
    candidatos = np.where(e > FATOR * fundo)[0]
    borda = int(0.01 * sr / QUADRO) + 1          # o filtro de ida e volta deixa transientes nas pontas
    candidatos = candidatos[(candidatos >= borda) & (candidatos < n - borda)]
    escolhidos: list[int] = []
    for q in candidatos:
        if escolhidos and q - escolhidos[-1] < SEPARACAO:
            if e[q] > e[escolhidos[-1]]:
                escolhidos[-1] = q
            continue
        escolhidos.append(int(q))
    return np.array(escolhidos) * QUADRO / sr


#: Pausa: quadro de 10 ms mais de 25 dB abaixo do percentil 90 do arquivo. Na fala,
#: consoantes como /s/ e /t/ também são picos de 1 ms acima de 7 kHz: só as pausas contam.
QUADRO_NIVEL_S, ABAIXO_DA_FALA_DB, SALTO_DB = 0.01, 25.0, 15.0


def niveis(x: np.ndarray, sr: int) -> np.ndarray:
    q = int(QUADRO_NIVEL_S * sr)
    n = len(x) // q
    return 10 * np.log10((x[:n * q].reshape(n, q) ** 2).mean(axis=1) + 1e-15)


def medir_arquivo(arq: Path) -> dict:
    """Na taxa original: pente de 9,09 Hz, nível do fundo nas pausas e saltos bruscos
    de nível (sinal de um "portão" de ruído). A contagem de estalos ficou de fora: o
    detector de picos confunde começo e fim de palavra com estalos (27/10/2026)."""
    from scripts.estudo_linhas import excesso, ler
    from scripts.estudo_pente import forca

    from scipy.ndimage import median_filter

    x, sr = ler(arq)
    e = niveis(x, sr)
    # Nível suavizado em 50 ms: um estalo não tira o próprio quadro da pausa.
    pausa_q = median_filter(e, size=5, mode="nearest") < np.percentile(e, 90) - ABAIXO_DA_FALA_DB
    pausa_s = float(pausa_q.sum() * QUADRO_NIVEL_S)
    return {"arquivo": arq.name, "duracao_s": len(x) / sr, "pausa_s": pausa_s,
            "pente_9hz": float(forca(excesso(x, sr), 9.0909, (4000.0, min(11500.0, sr / 2 - 100)))),
            "fundo_pausa_dbfs": float(np.median(e[pausa_q])) if pausa_q.any() else float("nan"),
            "saltos_por_s": float((np.abs(np.diff(e)) > SALTO_DB).sum() / (len(e) * QUADRO_NIVEL_S))}


def resumir(medidas: list[dict]) -> dict:
    pausa = sum(m["pausa_s"] for m in medidas)
    return {"arquivos": len(medidas), "pausa_s": pausa,
            "pente_9hz": float(np.median([m["pente_9hz"] for m in medidas])),
            "fundo_pausa_dbfs": float(np.nanmedian([m["fundo_pausa_dbfs"] for m in medidas])),
            "saltos_por_s": float(np.mean([m["saltos_por_s"] for m in medidas]))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Estalos curtos nas gravações, por microfone")
    p.add_argument("--grupo", action="append", required=True, metavar="NOME=PASTA")
    p.add_argument("--json", default="outputs/estudo_estalos.json")
    args = p.parse_args(argv)

    saida = {}
    print(f"{'grupo':14s} {'arqs':>4s} {'pausa s':>8s} {'pente 9Hz':>9s} {'fundo dBFS':>10s} {'saltos/s':>8s}")
    for item in args.grupo:
        nome, pasta = item.split("=", 1)
        medidas = [medir_arquivo(a) for a in sorted(Path(pasta).glob("*.wav"))]
        if not medidas:
            p.error(f"{pasta}: nenhum .wav")
        r = resumir(medidas)
        saida[nome] = {"resumo": r, "arquivos": medidas}
        print(f"{nome:14s} {r['arquivos']:4d} {r['pausa_s']:8.1f} {r['pente_9hz']:+9.2f} "
              f"{r['fundo_pausa_dbfs']:10.1f} {r['saltos_por_s']:8.2f}")
    print("\npente: força na base de 9,09 Hz (estalos a cada 110 ms); fundo: nível mediano das pausas; "
          "saltos: mudanças de mais de 15 dB em 10 ms (portão de ruído)")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nResultado: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
