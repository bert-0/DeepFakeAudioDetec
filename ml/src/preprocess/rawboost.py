"""RawBoost (Tak et al., 2022): degradação aleatória no áudio bruto.

Reimplementado a partir do código dos autores (TakHemlata/RawBoost-antispoofing),
com os mesmos padrões, mas com o `rng` da amostra no lugar do gerador global do
numpy (o global se repetiria igual em cada worker do DataLoader) e a filtragem
por FFT no lugar do `lfilter` (mesmo resultado; o `lfilter` com ~500
coeficientes em 6 s de áudio, cinco vezes por amostra, travaria o carregamento).

- `lnl`: ruído convolutivo linear e não linear. Soma de x, x², …, x^ordem, cada
  potência por um filtro de notches aleatórios; da 2ª ordem em diante, com ganho
  5 a 20 dB mais baixo.
- `isd`: ruído impulsivo dependente do sinal. Até 10% das amostras ganham um
  desvio proporcional ao próprio valor.
- `ssi`: ruído aditivo estacionário. Ruído branco colorido por um filtro de
  notches, numa SNR sorteada entre 10 e 40 dB.

Combinações em série, na ordem lnl → isd → ssi. Os autores usaram `lnl+isd`
na ASVspoof 2021 LA e `ssi` na DF (`--algo` 5 e 3 nos scripts de treino do
wav2vec 2.0); a combinação em paralelo (`--algo` 8) não foi reimplementada.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

ALGORITMOS = ("lnl", "isd", "ssi")


@dataclass(frozen=True)
class Parametros:
    """Os padrões do `main.py` dos autores; o nome original vem no comentário."""
    n_bandas: int = 5           # nBands: notches por filtro
    f_min: float = 20.0         # minF: centro do notch, Hz
    f_max: float = 8000.0       # maxF
    banda_min: float = 100.0    # minBW: largura do notch, Hz
    banda_max: float = 1000.0   # maxBW
    coef_min: int = 10          # minCoeff: coeficientes de cada notch
    coef_max: int = 100         # maxCoeff
    ganho_min: float = 0.0      # minG: ganho do termo linear, dB
    ganho_max: float = 0.0      # maxG
    desvio_min: float = 5.0     # minBiasLinNonLin: quanto os termos não lineares ficam abaixo
    desvio_max: float = 20.0    # maxBiasLinNonLin
    ordem: int = 5              # N_f: 1 = só o termo linear
    p_max: float = 10.0         # P: % máxima de amostras no isd
    g_sd: float = 2.0           # g_sd: ganho do desvio no isd
    snr_min: float = 10.0       # SNRmin, dB
    snr_max: float = 40.0       # SNRmax


def parametros(sobrescrever: dict | None = None) -> Parametros:
    """Os padrões, com os campos de `sobrescrever` trocados; nome desconhecido é erro."""
    sobrescrever = dict(sobrescrever or {})
    validos = {f.name for f in fields(Parametros)}
    desconhecidos = set(sobrescrever) - validos
    if desconhecidos:
        raise ValueError(f"rawboost: parâmetros desconhecidos {sorted(desconhecidos)}; "
                         f"use {sorted(validos)}")
    return Parametros(**sobrescrever)


def variante(nome: str) -> tuple[str, ...]:
    """'lnl+isd' -> ('lnl', 'isd'), na ordem em que os algoritmos são aplicados."""
    partes = tuple(str(nome).split("+"))
    if not partes or any(p not in ALGORITMOS for p in partes) or len(set(partes)) != len(partes):
        raise ValueError(f"rawboost: variante {nome!r} inválida; combine {ALGORITMOS} com '+'")
    return tuple(sorted(partes, key=ALGORITMOS.index))


def _faixa(rng: np.random.Generator, a: float, b: float) -> float:
    # Nas ordens não lineares a faixa de ganho vem invertida (-5 a -20 dB); o
    # np.random.uniform dos autores aceita, o Generator não.
    return float(rng.uniform(min(a, b), max(a, b)))


def _normalizar(y: np.ndarray, sempre: bool) -> np.ndarray:
    pico = float(np.max(np.abs(y))) if y.size else 0.0
    if pico > 0 and (sempre or pico > 1):
        return y / pico
    return y


def coeficientes_notch(rng: np.random.Generator, p: Parametros, sr: int,
                       ganho_min: float, ganho_max: float) -> np.ndarray:
    """Cascata de `n_bandas` notches (firwin com duas frequências é rejeita-faixa),
    com o pico da resposta levado a um ganho sorteado em dB."""
    from scipy.signal import firwin, freqz

    b = np.ones(1)
    for _ in range(p.n_bandas):
        fc = _faixa(rng, p.f_min, p.f_max)
        largura = _faixa(rng, p.banda_min, p.banda_max)
        c = int(_faixa(rng, p.coef_min, p.coef_max))
        c += 1 - c % 2                      # rejeita-faixa precisa de nº ímpar
        f1, f2 = fc - largura / 2, fc + largura / 2
        f1 = f1 if f1 > 0 else 1e-3
        f2 = f2 if f2 < sr / 2 else sr / 2 - 1e-3
        b = np.convolve(firwin(c, [f1, f2], window="hamming", fs=sr), b)
    g = _faixa(rng, ganho_min, ganho_max)
    _, h = freqz(b, 1, fs=sr)
    return 10 ** (g / 20) * b / np.max(np.abs(h))


def filtrar(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """O `filterFIR` dos autores (zeros no fim e corte de metade do filtro em cada
    ponta), com a convolução por FFT."""
    from scipy.signal import oaconvolve

    n = len(b) + 1
    y = oaconvolve(np.pad(x, (0, n)), b)[: len(x) + n]
    return y[int(n / 2): int(len(y) - n / 2)]


def lnl(x: np.ndarray, rng: np.random.Generator, p: Parametros, sr: int) -> np.ndarray:
    y = np.zeros(len(x))
    gmin, gmax = p.ganho_min, p.ganho_max
    for i in range(p.ordem):
        if i == 1:
            # Fica assim para todas as ordens não lineares, como no original.
            gmin, gmax = gmin - p.desvio_min, gmax - p.desvio_max
        y = y + filtrar(np.power(x, i + 1), coeficientes_notch(rng, p, sr, gmin, gmax))
    return _normalizar(y - y.mean(), sempre=False)


def isd(x: np.ndarray, rng: np.random.Generator, p: Parametros) -> np.ndarray:
    beta = _faixa(rng, 0.0, p.p_max)
    n = int(len(x) * beta / 100)
    idx = rng.permutation(len(x))[:n]
    f_r = (2 * rng.random(n) - 1) * (2 * rng.random(n) - 1)
    y = np.array(x, dtype=np.float64)
    y[idx] = x[idx] + p.g_sd * x[idx] * f_r
    return _normalizar(y, sempre=False)


def ssi(x: np.ndarray, rng: np.random.Generator, p: Parametros, sr: int) -> np.ndarray:
    ruido = rng.standard_normal(len(x))
    ruido = _normalizar(filtrar(ruido, coeficientes_notch(rng, p, sr, p.ganho_min, p.ganho_max)),
                        sempre=True)
    snr = _faixa(rng, p.snr_min, p.snr_max)
    ruido = ruido / np.linalg.norm(ruido) * np.linalg.norm(x) / 10 ** (0.05 * snr)
    return x + ruido


def aplicar(x: np.ndarray, nome: str, rng: np.random.Generator,
            p: Parametros | None = None, sr: int = 16000) -> np.ndarray:
    """Aplica a variante (`lnl`, `isd`, `ssi` ou combinações com '+') em série."""
    p = p or Parametros()
    y = np.asarray(x, dtype=np.float64)
    for algoritmo in variante(nome):
        if algoritmo == "lnl":
            y = lnl(y, rng, p, sr)
        elif algoritmo == "isd":
            y = isd(y, rng, p)
        else:
            y = ssi(y, rng, p, sr)
    return y.astype(np.float32)
