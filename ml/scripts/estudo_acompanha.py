"""Parte 7: a linha fixa acompanha a fala?

Hipótese escrita antes do teste (RESUMO_TCC §10.2.21): a linha do gerador faz
parte do som sintetizado e aparece na fala, sumindo nas pausas; a do ambiente
(ventoinha, rede elétrica, microfone) é constante e fica mais visível nas
pausas, quando a voz não a encobre. Se valer, separa "tom de máquina que fala"
de "zumbido da sala", o risco apontado na parte 5.

Para cada arquivo e cada linha: espectrograma com janela de 0,25 s (4 Hz) e
passo de 50 ms; o excesso da linha é a potência nela (máximo a ±1 ponto) sobre
a mediana dos vizinhos a 16-48 Hz, em dB. Quadros de fala: energia (100-7600
Hz) acima do percentil 60 do arquivo; pausas: abaixo do percentil 20.
Diferença = excesso mediano na fala - nas pausas.

Critério combinado antes: uma linha do gerador "acompanha a fala" se a
diferença for positiva em pelo menos 70% dos arquivos do grupo; uma do ambiente,
se for negativa em pelo menos 70%. Linhas testadas (escolhidas nas partes 3 e 5,
antes deste teste): XTTS 6.078, 7.014 e 8.139 Hz; Chatterbox 6.550, 7.450 e 7.850
Hz; microfone 527 e 655 Hz; ASVspoof 5: A14 4.000, 7.562 e 7.875 Hz, A13 2.756,
5.512 e 7.732 Hz, A07 7.000 Hz.
Estudo exploratório (RESUMO_TCC §10.2.23): não muda nenhum placar.

Uso (de dentro de ml/):
    python scripts/estudo_acompanha.py \\
        --grupo L01_xtts=data/vozes/ensaio2/clone --linha L01_xtts=6078,7014,8139 ...
    python scripts/estudo_acompanha.py --protocolo P --pasta F --classes A14 --n 100 \\
        --linha A14=4000,7562,7875
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

JANELA_S, PASSO_S = 0.25, 0.05
VIZINHOS_HZ = (16.0, 48.0)
FALA_PERCENTIL, PAUSA_PERCENTIL = 60, 20
CRITERIO = 0.70


def perfil(wav: np.ndarray, sr: int, hz: float) -> tuple[np.ndarray, np.ndarray]:
    """(excesso da linha em dB, energia do quadro em dB), um valor por quadro."""
    from scipy.signal import stft

    nper = int(JANELA_S * sr)
    f, _, z = stft(wav, sr, nperseg=nper, noverlap=nper - int(PASSO_S * sr), boundary=None, padded=False)
    p = np.abs(z) ** 2
    i = int(np.argmin(np.abs(f - hz)))
    linha = p[max(0, i - 1):i + 2].max(axis=0)
    dist = np.abs(f - hz)
    vizinhos = np.median(p[(dist >= VIZINHOS_HZ[0]) & (dist <= VIZINHOS_HZ[1])], axis=0)
    excesso = 10 * np.log10((linha + 1e-20) / (vizinhos + 1e-20))
    energia = 10 * np.log10(p[(f >= 100) & (f <= 7600)].sum(axis=0) + 1e-20)
    return excesso, energia


def acompanha(wav: np.ndarray, sr: int, hz: float) -> dict | None:
    """Excesso mediano na fala e nas pausas; None se faltar quadro de algum lado."""
    excesso, energia = perfil(wav, sr, hz)
    if len(energia) < 10:
        return None
    fala = excesso[energia >= np.percentile(energia, FALA_PERCENTIL)]
    pausa = excesso[energia <= np.percentile(energia, PAUSA_PERCENTIL)]
    if len(fala) < 3 or len(pausa) < 3:
        return None
    return {"fala": float(np.median(fala)), "pausa": float(np.median(pausa)),
            "diferenca": float(np.median(fala) - np.median(pausa))}


def estudar(grupos: dict[str, list[Path]], linhas: dict[str, list[float]]) -> dict:
    import soundfile as sf

    saida = {}
    for nome, arquivos in grupos.items():
        r = {}
        for hz in linhas.get(nome, []):
            medidas = []
            for arq in arquivos:
                wav, sr = sf.read(arq, dtype="float64")
                wav = wav.mean(axis=1) if wav.ndim > 1 else wav
                if hz < sr / 2 - VIZINHOS_HZ[1]:
                    m = acompanha(wav, sr, hz)
                    if m:
                        medidas.append(m)
            if medidas:
                dif = np.array([m["diferenca"] for m in medidas])
                r[f"{hz:g}"] = {"n": len(medidas), "fala": float(np.median([m["fala"] for m in medidas])),
                                "pausa": float(np.median([m["pausa"] for m in medidas])),
                                "diferenca": float(np.median(dif)),
                                "positiva": int((dif > 0).sum()), "negativa": int((dif < 0).sum())}
        saida[nome] = r
    return saida


def veredito(r: dict) -> str:
    if r["positiva"] >= CRITERIO * r["n"]:
        return "acompanha a fala"
    if r["negativa"] >= CRITERIO * r["n"]:
        return "fica nas pausas"
    return "indefinida"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="A linha fixa acompanha a fala? (parte 7)")
    p.add_argument("--grupo", action="append", default=[], metavar="NOME=PASTA")
    p.add_argument("--protocolo")
    p.add_argument("--pasta")
    p.add_argument("--classes", default="", help="classes do protocolo, separadas por vírgula")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--linha", action="append", required=True, metavar="GRUPO=HZ,HZ",
                   help="linhas a medir em cada grupo (escolhidas antes, nas partes 3 e 5)")
    p.add_argument("--json", default="outputs/estudo_acompanha.json")
    args = p.parse_args(argv)

    grupos: dict[str, list[Path]] = {}
    for item in args.grupo:
        nome, pasta = item.split("=", 1)
        grupos[nome] = sorted(Path(pasta).glob("*.wav"))
    if args.protocolo:
        from scripts.estudo_bases import ler_protocolo

        classes = ler_protocolo(Path(args.protocolo))
        rng = np.random.default_rng(args.semente)
        for c in [c for c in args.classes.split(",") if c]:
            ids = classes[c]
            grupos[c] = [Path(args.pasta) / f"{ids[i]}.flac" for i in rng.permutation(len(ids))[:args.n]]
    linhas = {}
    for item in args.linha:
        nome, valores = item.split("=", 1)
        linhas[nome] = [float(v) for v in valores.split(",")]
    faltam = [n for n in linhas if n not in grupos]
    if faltam:
        p.error(f"--linha sem grupo: {', '.join(faltam)}")

    r = estudar(grupos, linhas)
    print(f"{'grupo':16s} {'linha':>7s} {'n':>4s} {'fala dB':>8s} {'pausa dB':>9s} {'dif.':>6s} "
          f"{'dif.>0':>7s}  veredito (critério: {CRITERIO:.0%} dos arquivos)")
    for nome, linhas_r in r.items():
        for hz, x in linhas_r.items():
            print(f"{nome:16s} {hz:>7s} {x['n']:4d} {x['fala']:+8.2f} {x['pausa']:+9.2f} "
                  f"{x['diferenca']:+6.2f} {x['positiva']:>3d}/{x['n']:<3d}  {veredito(x)}")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultado: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
