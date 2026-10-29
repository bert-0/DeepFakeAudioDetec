"""O relógio do vocoder no espectro: um pente de linhas na taxa de quadros.

Um vocoder neural monta o som quadro a quadro (HiFi-GAN do XTTS: 256 amostras
a 24 kHz, 93,75 quadros/s; HiFT do Chatterbox: 8x5x3 e iSTFT de passo 4, 480
amostras, 50 quadros/s). Se cada quadro deixa a mesma marca, o espectro médio
ganha linhas finas nos múltiplos da taxa de quadros, mais visíveis na banda
alta, onde a voz tem pouca energia. A voz humana não tem relógio fixo.

Força do pente numa base: média do excesso (dB acima do contorno, 1 Hz) nos
múltiplos da base, menos a média nos pontos do meio entre eles. Medida por
arquivo; a separação clone x real é a AUC (sem limiar). Além das bases da
arquitetura, uma varredura de 40 a 200 Hz pega o pente mais forte sem saber o
gerador. Estudo exploratório (RESUMO_TCC §10.2.20): não muda nenhum placar.

Uso (de dentro de ml/):
    python scripts/estudo_pente.py --base xtts=93.75 --base chatterbox=50 \\
        --base microfone=9.0909 --real L01_real --real L02_real \\
        --comparar L01_chatterbox:L01_xtts --comparar L02_chatterbox:L02_xtts \\
        --grupo L01_real=data/vozes/ensaio2/real --grupo L02_real=data/vozes/L02/real \\
        --grupo L01_xtts=data/vozes/ensaio2/clone --grupo L02_xtts=data/vozes/L02/clone \\
        --grupo L01_chatterbox=data/vozes/ensaio2/clone_chatterbox \\
        --grupo L02_chatterbox=data/vozes/L02/clone_chatterbox
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.estudo_linhas import GRADE, TOLERANCIA_HZ, excesso, ler  # noqa: E402

#: Banda do pente: acima dos harmônicos fortes da voz, abaixo dos 12 kHz dos clones.
FAIXA_NATIVA = (4000.0, 11500.0)
#: No caminho do modelo (16 kHz), até onde o FIR deixa passar.
FAIXA_MODELO = (4000.0, 7600.0)
BASES_VARRIDAS = np.arange(40.0, 200.001, 0.25)


def forca(ex: np.ndarray, base: float, faixa=FAIXA_NATIVA, grade=GRADE) -> float:
    """Excesso médio nos múltiplos da base menos o dos pontos do meio, em dB."""
    k = np.arange(np.ceil(faixa[0] / base), np.floor(faixa[1] / base) + 1)
    if len(k) < 3:
        return float("nan")

    def media(centros):
        valores = []
        for c in centros:
            i = int(round(c - grade[0]))
            trecho = ex[max(0, i - TOLERANCIA_HZ):i + TOLERANCIA_HZ + 1]
            if trecho.size and not np.all(np.isnan(trecho)):
                valores.append(np.nanmax(trecho))
        return float(np.mean(valores)) if valores else float("nan")

    return media(k * base) - media((k + 0.5) * base)


def varrido(ex: np.ndarray, faixa=FAIXA_NATIVA, grade=GRADE) -> tuple[float, float]:
    """(base, força) do pente mais forte entre 40 e 200 Hz. Mesma conta de `forca`,
    com o máximo a ±2 Hz calculado uma vez (a banda não pode ter NaN)."""
    from scipy.ndimage import maximum_filter1d

    mx = maximum_filter1d(ex, size=2 * TOLERANCIA_HZ + 1, mode="nearest")
    forcas = np.full(len(BASES_VARRIDAS), np.nan)
    for j, base in enumerate(BASES_VARRIDAS):
        k = np.arange(np.ceil(faixa[0] / base), np.floor(faixa[1] / base) + 1)
        if len(k) >= 3:
            forcas[j] = (mx[np.rint(k * base - grade[0]).astype(int)].mean()
                         - mx[np.rint((k + 0.5) * base - grade[0]).astype(int)].mean())
    i = int(np.nanargmax(forcas))
    return float(BASES_VARRIDAS[i]), float(forcas[i])


def auc(positivos, negativos) -> float:
    """P(positivo > negativo), empates contam meio."""
    p, n = np.asarray(positivos, float)[:, None], np.asarray(negativos, float)[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def excesso_no_modelo(arq: Path) -> np.ndarray:
    """Excesso a 1 Hz pelo caminho do modelo (16 kHz, FIR a partir de 48 kHz), na GRADE."""
    from scripts.estudo_linhas import espectros_no_modelo

    f1, ex1, _, _ = espectros_no_modelo([arq])[0]
    return np.interp(GRADE, f1, ex1, right=np.nan)


def medir(arquivos: list[Path], bases: dict[str, float]) -> dict:
    """Por arquivo: força em cada base e o pente varrido, no nativo e no caminho do modelo."""
    saida = {"arquivos": [a.name for a in arquivos]}
    for caminho, faixa, obter in (("nativo", FAIXA_NATIVA, lambda a: excesso(*ler(a))),
                                  ("modelo", FAIXA_MODELO, excesso_no_modelo)):
        exs = [obter(a) for a in arquivos]
        r = {nome: [forca(ex, b, faixa) for ex in exs] for nome, b in bases.items()}
        varridos = [varrido(ex, faixa) for ex in exs]
        r["varrido_base"] = [b for b, _ in varridos]
        r["varrido"] = [f for _, f in varridos]
        saida[caminho] = r
    return saida


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Pente de linhas na taxa de quadros do vocoder")
    p.add_argument("--grupo", action="append", required=True, metavar="NOME=PASTA")
    p.add_argument("--real", action="append", required=True, metavar="NOME",
                   help="grupos de gravações reais (a referência da AUC)")
    p.add_argument("--base", action="append", default=[], metavar="NOME=HZ",
                   help="taxa de quadros tirada da arquitetura do gerador")
    p.add_argument("--comparar", action="append", default=[], metavar="A:B",
                   help="AUC do grupo A contra o B (por exemplo, um gerador contra o outro)")
    p.add_argument("--json", default="outputs/estudo_pente.json")
    args = p.parse_args(argv)

    grupos = {}
    for item in args.grupo:
        nome, pasta = item.split("=", 1)
        grupos[nome] = sorted(Path(pasta).glob("*.wav"))
    pares = [c.split(":", 1) for c in args.comparar]
    faltam = [r for r in args.real + [g for par in pares for g in par] if r not in grupos]
    if faltam:
        p.error(f"grupo citado sem --grupo correspondente: {', '.join(faltam)}")
    bases = {}
    for item in args.base:
        nome, hz = item.split("=", 1)
        bases[nome] = float(hz)

    medidas = {nome: medir(arqs, bases) for nome, arqs in grupos.items()}
    colunas = list(bases) + ["varrido"]
    resumo = {}
    for caminho, faixa in (("nativo", FAIXA_NATIVA), ("modelo", FAIXA_MODELO)):
        print(f"\nCaminho {caminho} ({faixa[0]:.0f}-{faixa[1]:.0f} Hz): força mediana do pente em dB "
              f"[AUC contra os reais]")
        print(f"{'grupo':16s} " + " ".join(f"{c:>20s}" for c in colunas))
        resumo[caminho] = {}
        for nome in grupos:
            linha, resumo[caminho][nome] = [], {}
            for c in colunas:
                valores = medidas[nome][caminho][c]
                reais = [v for r in args.real for v in medidas[r][caminho][c]]
                a = auc(valores, reais)
                resumo[caminho][nome][c] = {"mediana": float(np.nanmedian(valores)), "auc": a}
                linha.append(f"{np.nanmedian(valores):+6.2f} [{a:4.2f}]")
            print(f"{nome:16s} " + " ".join(f"{t:>20s}" for t in linha))
        print("Base do pente varrido, por arquivo (Hz):")
        for nome in grupos:
            print(f"  {nome:16s} " + " ".join(f"{b:g}" for b in medidas[nome][caminho]["varrido_base"]))
        if pares:
            resumo[caminho]["comparacoes"] = {}
            print("AUC de um grupo contra outro:")
            for a, b in pares:
                aucs = {c: auc(medidas[a][caminho][c], medidas[b][caminho][c]) for c in colunas}
                resumo[caminho]["comparacoes"][f"{a}:{b}"] = aucs
                print(f"  {a} x {b}: " + ", ".join(f"{c} {v:.2f}" for c, v in aucs.items()))
    print("\nAUC dos reais contra os próprios reais fica perto de 0,5 por construção.")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({"bases": bases, "medidas": medidas, "resumo": resumo},
                                          indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resultado: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
