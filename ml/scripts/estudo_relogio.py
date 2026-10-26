"""O "relógio" do gerador: picos de modulação em ritmos fixos.

Os sintetizadores atuais montam o som em passos fixos (tokens, quadros do
vocoder); a voz humana não tem relógio. Este estudo mede o espectro de
modulação: com que ritmos a energia de cada faixa de frequência sobe e desce,
nos trechos com fala, a 16 kHz (o domínio do detector). Um pico estreito num
ritmo fixo, nos clones e não nas gravações reais, é uma marca do gerador.
Estudo exploratório (RESUMO_TCC §10.2.18): não muda nenhum placar.

Uso (de dentro de ml/):
    python scripts/estudo_relogio.py \\
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

SR = 16000
#: 2 ms entre quadros: o espectro de modulação vai até 250 Hz.
PASSO = 32
JANELA = 256
FAIXAS_HZ = ((0, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000),
             (4000, 5000), (5000, 6000), (6000, 8000))
#: Trechos de fala contínua de pelo menos 1 s: resolução de 1 Hz na modulação.
TRECHO_S = 1.0


def energias_por_faixa(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """Log-energia de cada faixa a cada 2 ms: (faixas, quadros)."""
    from numpy.lib.stride_tricks import sliding_window_view

    quadros = sliding_window_view(wav, JANELA)[::PASSO] * np.hanning(JANELA)
    espectro = np.abs(np.fft.rfft(quadros, axis=1)) ** 2
    f = np.fft.rfftfreq(JANELA, 1 / sr)
    linhas = [espectro[:, (f >= a) & (f < b)].sum(axis=1) for a, b in FAIXAS_HZ]
    return np.log(np.array(linhas) + 1e-10)


def modulacao(wav: np.ndarray, trechos: list[tuple[int, int]], sr: int = SR):
    """Espectro de modulação médio (sobre faixas e trechos), em 1 Hz por ponto.

    Cada trecho de fala é cortado em pedaços de 1 s; cada trajetória de
    log-energia perde a média e passa por uma janela de Hann antes da FFT."""
    taxa_quadros = sr / PASSO
    n = int(TRECHO_S * taxa_quadros)
    energias = energias_por_faixa(wav, sr)
    acumulado, contagem = np.zeros(n // 2 + 1), 0
    for a, b in trechos:
        qa, qb = a // PASSO, max(a // PASSO, (b - JANELA) // PASSO)
        for ini in range(qa, qb - n + 1, n):
            pedaco = energias[:, ini:ini + n]
            pedaco = (pedaco - pedaco.mean(axis=1, keepdims=True)) * np.hanning(n)
            acumulado += (np.abs(np.fft.rfft(pedaco, axis=1)) ** 2).mean(axis=0)
            contagem += 1
    freqs = np.fft.rfftfreq(n, 1 / taxa_quadros)
    return freqs, (acumulado / contagem if contagem else acumulado), contagem


def picos(freqs: np.ndarray, espectro: np.ndarray, minimo_hz: float = 10.0,
          proeminencia_db: float = 3.0) -> list[tuple[float, float]]:
    """Picos estreitos acima do contorno suave (mediana em 9 Hz), em dB."""
    from scipy.ndimage import median_filter

    db = 10 * np.log10(espectro + 1e-20)
    contorno = median_filter(db, size=9, mode="nearest")
    excesso = db - contorno
    saida = []
    for i in range(1, len(db) - 1):
        if (freqs[i] >= minimo_hz and excesso[i] >= proeminencia_db
                and db[i] >= db[i - 1] and db[i] >= db[i + 1]):
            saida.append((float(freqs[i]), float(excesso[i])))
    return sorted(saida, key=lambda p: -p[1])


def pente(freqs: np.ndarray, espectro: np.ndarray, faixa=(12.0, 160.0)) -> dict:
    """Ondulação regular (pente) do espectro de modulação: espaçamento pela
    autocorrelação do excesso sobre o contorno (mediana em 21 Hz)."""
    from scipy.ndimage import median_filter

    db = 10 * np.log10(espectro + 1e-20)
    ex = db - median_filter(db, size=21, mode="nearest")
    e = ex[(freqs >= faixa[0]) & (freqs <= faixa[1])]
    e = e - e.mean()
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    ac = ac / ac[0]
    lag = int(np.argmax(ac[4:30])) + 4
    passo = float(freqs[1] - freqs[0])
    return {"espacamento_hz": lag * passo, "autocorrelacao": float(ac[lag]), "ondulacao_db": float(e.std())}


def estudar(grupos: dict[str, list[Path]]) -> dict:
    from src.capture import FileSource
    from src.capture.checagens import trechos_de_fala

    resultado = {}
    for nome, arquivos in grupos.items():
        soma, total, freqs = None, 0, None
        for arq in arquivos:
            with FileSource(arq, SR, taxa_comum=48000) as fonte:
                wav = fonte._wav.astype(np.float64)
            freqs, esp, n = modulacao(wav, trechos_de_fala(wav.astype(np.float32), SR))
            if n:
                soma = esp * n if soma is None else soma + esp * n
                total += n
        media = soma / total if total else None
        resultado[nome] = {"pedacos": total, "freqs": freqs.tolist() if freqs is not None else [],
                           "espectro": media.tolist() if media is not None else [],
                           "picos": picos(freqs, media)[:8] if media is not None else [],
                           "pente": pente(freqs, media) if media is not None else None}
    return resultado


def grafico(resultado: dict, destino: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))
    for nome, r in resultado.items():
        if r["espectro"]:
            estilo = "-" if "real" in nome else ("--" if "xtts" in nome else ":")
            ax.plot(r["freqs"], 10 * np.log10(np.array(r["espectro"]) + 1e-20), estilo,
                    lw=1.2, label=f"{nome} ({r['pedacos']} s)")
    ax.set_xlim(1, 250)
    ax.set_xscale("log")
    ax.set_xlabel("ritmo da modulação (Hz)")
    ax.set_ylabel("dB")
    ax.set_title("Espectro de modulação na fala (média das faixas, 16 kHz)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destino, dpi=80)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Picos de modulação em ritmos fixos (relógio do gerador)")
    p.add_argument("--grupo", action="append", required=True, metavar="NOME=PASTA")
    p.add_argument("--json", default="outputs/estudo_relogio.json")
    p.add_argument("--grafico", default="outputs/estudo_relogio.png")
    args = p.parse_args(argv)

    grupos = {}
    for item in args.grupo:
        nome, pasta = item.split("=", 1)
        grupos[nome] = sorted(Path(pasta).glob("*.wav"))
    resultado = estudar(grupos)
    for nome, r in resultado.items():
        texto = ", ".join(f"{f:.0f} Hz (+{e:.1f} dB)" for f, e in r["picos"][:6]) or "nenhum"
        print(f"{nome:16s} {r['pedacos']:4d} s de fala | picos acima do contorno: {texto}")
        if r["pente"]:
            pt = r["pente"]
            print(f"{'':16s}      pente: espaçamento {pt['espacamento_hz']:.0f} Hz, autocorrelação "
                  f"{pt['autocorrelacao']:+.2f}, ondulação {pt['ondulacao_db']:.2f} dB")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(resultado, ensure_ascii=False), encoding="utf-8")
    grafico(resultado, Path(args.grafico))
    print(f"\nResultado: {args.json} | gráfico: {args.grafico}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
