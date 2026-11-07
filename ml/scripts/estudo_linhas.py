"""Linhas fixas no espectro: tons que o gerador (ou o microfone) deixa no áudio.

A fala muda de altura o tempo todo, então no espectro médio de vários
arquivos os harmônicos se espalham. Um tom de frequência fixa (resto de
reamostragem do vocoder, ruído elétrico do microfone) continua uma linha
estreita. Para cada arquivo: espectro de Welch com 1 Hz de resolução, em dB,
menos o contorno (mediana em 41 Hz). As linhas são escolhidas num grupo
(por exemplo, os clones de um locutor) e contadas em todos os outros, para não
escolher e conferir nos mesmos arquivos. Abaixo de 8 kHz, mede também quanto
da linha sobra pelo caminho do modelo (16 kHz) na janela de 25 ms do detector.
Estudo exploratório (RESUMO_TCC §10.2.19): não muda nenhum placar.

Uso (de dentro de ml/):
    python scripts/estudo_linhas.py --escolher L01_xtts --escolher L01_chatterbox \\
        --escolher L01_real \\
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

#: Grade comum de 1 Hz. Os clones vêm a 24 kHz: acima de 12 kHz não há o que comparar.
GRADE = np.arange(0, 12001, dtype=float)
CONTORNO_HZ = 41
#: Abaixo disso as linhas se confundem com a altura da voz (F0 e primeiros harmônicos).
MINIMO_HZ = 300
#: Uma linha "aparece" num arquivo se passa 3 dB do contorno a até 2 Hz do lugar.
PRESENCA_DB = 3.0
TOLERANCIA_HZ = 2
#: Linhas escolhidas a menos de 15 Hz uma da outra contam como a mesma.
SEPARACAO_HZ = 15


def excesso(wav: np.ndarray, sr: int) -> np.ndarray:
    """dB acima do contorno, na GRADE (NaN acima da metade da taxa)."""
    from scipy.ndimage import median_filter
    from scipy.signal import welch

    f, p = welch(wav, sr, nperseg=sr, noverlap=sr // 2)
    db = 10 * np.log10(p + 1e-30)
    ex = db - median_filter(db, size=CONTORNO_HZ, mode="nearest")
    return np.interp(GRADE, f, ex, right=np.nan)


def ler(arq: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    wav, sr = sf.read(arq, dtype="float64")
    return (wav.mean(axis=1) if wav.ndim > 1 else wav), sr


def matriz(arquivos: list[Path]) -> np.ndarray:
    """(arquivos, GRADE): o excesso de cada arquivo, na taxa original dele."""
    return np.array([excesso(*ler(a)) for a in arquivos])


def escolher_linhas(m: np.ndarray, n: int = 8, minimo_hz: float = MINIMO_HZ) -> list[tuple[float, float]]:
    """As n linhas mais fortes na mediana dos arquivos: (Hz, dB)."""
    mediana = np.nan_to_num(np.nanmedian(m, axis=0), nan=-99.0)
    ordem = np.argsort(-mediana)
    saida: list[tuple[float, float]] = []
    for i in ordem:
        if GRADE[i] < minimo_hz:
            continue
        if all(abs(GRADE[i] - f) > SEPARACAO_HZ for f, _ in saida):
            saida.append((float(GRADE[i]), float(mediana[i])))
        if len(saida) == n:
            break
    return saida


def presenca(m: np.ndarray, hz: float) -> int:
    """Em quantos arquivos a linha passa PRESENCA_DB do contorno."""
    i = int(round(hz))
    trecho = m[:, max(0, i - TOLERANCIA_HZ):i + TOLERANCIA_HZ + 1]
    return int((np.nan_to_num(np.nanmax(trecho, axis=1), nan=-99.0) > PRESENCA_DB).sum())


#: Janela do detector (configs/fusion_v4*.yaml): 400 amostras a 16 kHz, FFT de 512.
JANELA_MODELO, NFFT_MODELO = 400, 512
#: Controle: o mesmo cálculo a 100 Hz de cada lado da linha, onde só há ruído.
CONTROLE_HZ = 100


def espectros_no_modelo(arquivos: list[Path]) -> list[tuple]:
    """Excesso de cada arquivo pelo caminho do modelo (16 kHz, FIR a partir de 48 kHz),
    em 1 s e na janela de 25 ms do detector: [(f1, ex1, f25, ex25), ...]."""
    from scipy.ndimage import median_filter
    from scipy.signal import welch

    from src.capture import FileSource

    saida = []
    for arq in arquivos:
        with FileSource(arq, 16000, taxa_comum=48000) as fonte:
            x = fonte._wav.astype(np.float64)
        par = []
        for nper, nfft, contorno in ((16000, 16000, CONTORNO_HZ), (JANELA_MODELO, NFFT_MODELO, 9)):
            f, p = welch(x, 16000, nperseg=nper, nfft=nfft, noverlap=nper // 2)
            db = 10 * np.log10(p + 1e-30)
            par += [f, db - median_filter(db, size=contorno, mode="nearest")]
        saida.append(tuple(par))
    return saida


def no_modelo(espectros: list[tuple], hz: float) -> dict:
    """Mediana do excesso na linha e no controle, em 1 Hz (máximo a ±2 Hz) e em 25 ms
    (ponto mais próximo: a linha cabe inteira num ponto de 31 Hz)."""
    def em(f1, ex1, f25, ex25, alvo):
        i = int(np.argmin(np.abs(f1 - alvo)))
        return ex1[max(0, i - TOLERANCIA_HZ):i + TOLERANCIA_HZ + 1].max(), np.interp(alvo, f25, ex25)

    linha = np.array([em(*e, hz) for e in espectros])
    controle = np.array([em(*e, hz + d) for e in espectros for d in (-CONTROLE_HZ, CONTROLE_HZ)])
    return {"1hz": float(np.median(linha[:, 0])), "25ms": float(np.median(linha[:, 1])),
            "controle_1hz": float(np.median(controle[:, 0])), "controle_25ms": float(np.median(controle[:, 1]))}


def estudar(grupos: dict[str, list[Path]], escolher: list[str], n: int = 8) -> dict:
    matrizes = {nome: matriz(arqs) for nome, arqs in grupos.items()}
    no_caminho = {nome: espectros_no_modelo(arqs) for nome, arqs in grupos.items()}
    saida = {}
    for origem in escolher:
        linhas = []
        for hz, db in escolher_linhas(matrizes[origem], n):
            linha = {"hz": hz, "db": db,
                     "presenca": {g: [presenca(m, hz), len(m)] for g, m in matrizes.items()}}
            if hz + CONTROLE_HZ < 8000:
                linha["modelo"] = {g: no_modelo(e, hz) for g, e in no_caminho.items()}
            linhas.append(linha)
        saida[origem] = linhas
    return saida


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Linhas fixas no espectro, escolhidas num grupo e contadas nos outros")
    p.add_argument("--grupo", action="append", required=True, metavar="NOME=PASTA")
    p.add_argument("--escolher", action="append", required=True, metavar="NOME",
                   help="grupo onde as linhas são escolhidas (pode repetir)")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--json", default="outputs/estudo_linhas.json")
    args = p.parse_args(argv)

    grupos = {}
    for item in args.grupo:
        nome, pasta = item.split("=", 1)
        grupos[nome] = sorted(Path(pasta).glob("*.wav"))
    faltam = [e for e in args.escolher if e not in grupos]
    if faltam:
        p.error(f"--escolher sem --grupo correspondente: {', '.join(faltam)}")
    resultado = estudar(grupos, args.escolher, args.n)
    nomes = list(grupos)
    for origem, linhas in resultado.items():
        print(f"\nLinhas escolhidas em {origem}; arquivos com a linha (+{PRESENCA_DB:.0f} dB) em cada grupo:")
        print(f"{'Hz':>7s} {'dB':>5s} " + " ".join(f"{g:>15s}" for g in nomes))
        for linha in linhas:
            print(f"{linha['hz']:7.0f} {linha['db']:+5.1f} " + " ".join(
                f"{'%d/%d' % tuple(linha['presenca'][g]):>15s}" for g in nomes))
        print(f"No caminho do modelo (16 kHz), excesso mediano em dB: 1 Hz | 25 ms "
              f"(controle a ±{CONTROLE_HZ} Hz: 1 Hz | 25 ms)")
        for linha in linhas:
            if "modelo" in linha:
                print(f"{linha['hz']:7.0f} {'':5s} " + " ".join(
                    "{:>15s}".format("%+.1f|%+.1f (%+.1f|%+.1f)" % (
                        linha["modelo"][g]["1hz"], linha["modelo"][g]["25ms"],
                        linha["modelo"][g]["controle_1hz"], linha["modelo"][g]["controle_25ms"]))
                    for g in nomes))
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultado: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
