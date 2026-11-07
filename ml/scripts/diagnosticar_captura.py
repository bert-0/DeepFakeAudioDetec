"""Mede como uma gravação se deformou no tempo em relação ao que foi tocado.

Localiza cada pedaço da referência na gravação e separa três casos: atraso
constante, deslocamento em rampa (taxa de amostragem errada) e degraus
(amostras perdidas ou inseridas).

Uso:
    python scripts/diagnosticar_captura.py --pasta outputs/canal_real \\
        --gravacao outputs/canal_real/controle.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture.alinhamento import TAXA_ENVELOPE_HZ, envelope  # noqa: E402
from src.preprocess import load_audio  # noqa: E402

PEDACO_S = 3.0

#: Fatores de escala testados: cobre 44,1<->48 kHz (8,8%) com folga.
FATORES = np.linspace(0.85, 1.15, 601)


def correlacao_normalizada(pedaco: np.ndarray, sinal: np.ndarray) -> tuple[int, float]:
    """Melhor posição de `pedaco` em `sinal`, com Pearson normalizado em cada deslocamento."""
    from scipy.signal import correlate

    m = len(pedaco)
    if len(sinal) < m or m < 2:
        return 0, 0.0
    r = np.asarray(pedaco, dtype=np.float64)
    r = r - r.mean()
    nr = np.linalg.norm(r)
    x = np.asarray(sinal, dtype=np.float64)
    num = correlate(x, r, mode="valid", method="fft")
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    soma = c1[m:] - c1[:-m]
    soma2 = c2[m:] - c2[:-m]
    var = np.maximum(soma2 - soma * soma / m, 0.0)
    den = np.sqrt(var) * nr
    coef = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    i = int(np.argmax(coef))
    return i, float(coef[i])


def reescalar(env: np.ndarray, fator: float) -> np.ndarray:
    """Desfaz um esticamento de `fator` (gravação `fator` vezes mais longa)."""
    n = max(2, int(round(len(env) / fator)))
    return np.interp(np.arange(n) * fator, np.arange(len(env)), env)


def estimar_fator(env_r: np.ndarray, env_c: np.ndarray) -> tuple[float, float, float]:
    """(melhor fator, correlação nele, correlação em 1,0) da referência inteira."""
    melhor = (1.0, -1.0)
    em_um = correlacao_normalizada(env_r, env_c)[1]
    for f in FATORES:
        c = correlacao_normalizada(env_r, reescalar(env_c, f))[1]
        if c > melhor[1]:
            melhor = (float(f), c)
    return melhor[0], melhor[1], em_um


def deslocamentos(env_r: np.ndarray, env_c: np.ndarray,
                  pedaco_s: float = PEDACO_S) -> list[tuple[float, float, float]]:
    """(instante na referência, deslocamento em s, correlação) por pedaço.

    Cada pedaço é procurado na gravação inteira, para não perder buracos grandes.
    """
    n = int(pedaco_s * TAXA_ENVELOPE_HZ)
    saida = []
    for ini in range(0, len(env_r) - n + 1, n):
        pedaco = env_r[ini:ini + n]
        if pedaco.std() == 0:
            continue
        pos, corr = correlacao_normalizada(pedaco, env_c)
        saida.append((ini / TAXA_ENVELOPE_HZ, (pos - ini) / TAXA_ENVELOPE_HZ, corr))
    return saida


def _n_saltos(pontos) -> tuple[int, float]:
    d = np.array([dd for _, dd, c in pontos if c >= 0.5])
    if len(d) < 2:
        return 0, 0.0
    # Mediana de 3 descarta casamentos errados isolados; um degrau real persiste.
    if len(d) >= 3:
        from scipy.signal import medfilt

        d = medfilt(d, 3)
        d[0], d[-1] = d[1], d[-2]
    saltos = np.abs(np.diff(d))
    return int((saltos > 0.05).sum()), float(saltos.max())


def analisar(referencia: np.ndarray, captura: np.ndarray, sr: int):
    """Estima escala e deslocamentos; reescala só se isso deixar menos saltos.

    Buracos também mudam a duração total e enganam o ajuste de escala.
    """
    env_r, env_c = envelope(referencia, sr), envelope(captura, sr)
    fator, corr_fator, corr_um = estimar_fator(env_r, env_c)
    sem = deslocamentos(env_r, env_c)
    usar, pontos = 1.0, sem
    if abs(fator - 1.0) > 0.003:
        com = deslocamentos(env_r, reescalar(env_c, fator))
        if _n_saltos(com)[0] < _n_saltos(sem)[0]:
            usar, pontos = fator, com
    return fator, corr_fator, corr_um, usar, pontos


def diagnosticar(fator, corr_fator, corr_um, usar, pontos,
                 duracao_ref: float, duracao_cap: float) -> list[str]:
    linhas = [f"Duração: referência {duracao_ref:.1f} s | gravação {duracao_cap:.1f} s",
              f"Escala: correlação {corr_um:.2f} sem correção | {corr_fator:.2f} com "
              f"fator {fator:.4f}"]
    bons = [(t, d) for t, d, c in pontos if c >= 0.5]
    linhas.append(f"Pedaços de {PEDACO_S:.0f} s bem localizados"
                  f"{' (após corrigir a escala)' if usar != 1.0 else ''}: "
                  f"{len(bons)}/{len(pontos)} (correlação ≥ 0,5)")
    # Decide pelos pedaços: se a gravação começou atrasada, a correlação global
    # despenca mesmo com os pedaços bem localizados.
    if len(bons) < max(3, len(pontos) // 2):
        linhas.append("DIAGNÓSTICO: quase nada da referência aparece na gravação. O que foi "
                      "gravado não é a playlist (dispositivo errado, volume zero, outro som "
                      "por cima) — ou está deformado demais para qualquer alinhamento.")
        return linhas
    n_saltos, maior = _n_saltos(pontos)
    if usar != 1.0:
        linhas.append(
            f"DIAGNÓSTICO: TAXA DE AMOSTRAGEM. A gravação corre {fator:.4f}x o tempo da "
            f"referência. Compare com 48000/44100 = {48000/44100:.4f} e 44100/48000 = "
            f"{44100/48000:.4f}: algum ponto do caminho usou a taxa errada. Isso também "
            "muda o tom da voz — o modelo vê outro sinal.")
    if n_saltos:
        linhas.append(
            f"DIAGNÓSTICO: DEGRAUS. O deslocamento salta {n_saltos} vez(es) (maior salto "
            f"{maior:.2f} s): amostras perdidas ou inseridas no caminho — "
            "descontinuidades da captura ou engasgos da reprodução.")
    mediano = float(np.median([b[1] for b in bons]))
    if mediano < -0.05:
        linhas.append(
            f"DIAGNÓSTICO: A GRAVAÇÃO COMEÇOU {-mediano:.1f} s DEPOIS da reprodução. O "
            "início da playlist ficou de fora. O `alinhar` lida com isso (perde só os "
            "trechos do começo); da próxima vez, dê Enter no `tocar` só quando o monitor "
            "já mostrar a tabela.")
    if usar == 1.0 and not n_saltos and mediano >= -0.05:
        linhas.append(f"DIAGNÓSTICO: deslocamento praticamente constante ({mediano:+.2f} s) "
                      "— só atraso. O alinhamento deveria funcionar; se não funcionou, o "
                      "problema é outro.")
    return linhas


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Diagnóstico de deformação temporal da gravação")
    p.add_argument("--pasta", default="outputs/canal_real")
    p.add_argument("--gravacao", required=True)
    args = p.parse_args(argv)

    referencia, sr = sf.read(Path(args.pasta) / "referencia.wav", dtype="float32")
    captura = load_audio(args.gravacao, sr)
    fator, corr_fator, corr_um, usar, pontos = analisar(referencia, captura, sr)

    print(f"{'t ref':>7s} {'desloc.':>9s} {'corr':>6s}")
    for t, d, c in pontos:
        print(f"{t:6.0f}s {d:+8.2f}s {c:6.2f}{'' if c >= 0.5 else '   (fraco)'}")
    print()
    for linha in diagnosticar(fator, corr_fator, corr_um, usar, pontos,
                              len(referencia) / sr, len(captura) / sr):
        print(linha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
