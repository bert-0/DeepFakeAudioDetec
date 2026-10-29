"""Alinha a gravação de uma chamada real aos áudios que foram tocados nela.

Correlaciona o envelope de energia da gravação com o da referência: primeiro um
atraso global, depois um ajuste por trecho para absorver a deriva de relógio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

#: Resolução de 10 ms, bem menor que a folga de silêncio do recorte.
TAXA_ENVELOPE_HZ = 100

#: Abaixo disto o trecho é descartado em vez de sair com rótulo errado.
#: Medido: pior caso 0,798 com opus + 8 kHz, 0,118 com ruído puro.
CORRELACAO_MINIMA = 0.5

#: Quanto o ajuste por trecho pode andar em torno da posição prevista.
BUSCA_LOCAL_S = 0.40

#: Refinamento pela forma de onda: o envelope erra até 5 ms, meio passo do STFT,
#: o que já mudava scores em até 0,17 no mesmo áudio.
BUSCA_FINA_S = 0.02

#: Abaixo disto fica a posição do envelope; o refinamento nunca descarta trecho.
CORRELACAO_FINA_MINIMA = 0.5


@dataclass
class Trecho:
    """Um áudio da playlist: posição na referência e rótulo."""
    id: str
    rotulo: str          # "bonafide" | "spoof"
    sistema: str         # A07…A19, ou "-"
    inicio: int          # amostra inicial NA REFERÊNCIA
    n: int               # nº de amostras


@dataclass
class Encaixe:
    """Resultado do alinhamento de um trecho."""
    trecho: Trecho
    inicio_capturado: int    # amostra inicial NA GRAVAÇÃO
    correlacao: float
    refinado: bool = False   # posição ajustada pela forma de onda

    @property
    def confiavel(self) -> bool:
        return self.correlacao >= CORRELACAO_MINIMA


def envelope(wav: np.ndarray, sample_rate: int,
             taxa_hz: int = TAXA_ENVELOPE_HZ) -> np.ndarray:
    """Energia RMS em janelas curtas, sem sobreposição."""
    passo = max(1, int(round(sample_rate / taxa_hz)))
    n = len(wav) // passo
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    blocos = np.asarray(wav[: n * passo], dtype=np.float64).reshape(n, passo)
    return np.sqrt((blocos ** 2).mean(axis=1))


def _normalizar(x: np.ndarray) -> np.ndarray:
    """Média zero e norma 1, para anular o efeito do AGC."""
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    norma = np.linalg.norm(x)
    return x / norma if norma > 0 else x


def correlacao_maxima(referencia: np.ndarray,
                      captura: np.ndarray) -> tuple[int, float]:
    """Desloca `referencia` sobre `captura` e devolve (atraso, correlação).

    O atraso é em amostras do envelope e nunca é negativo.
    """
    from scipy.signal import correlate

    if referencia.size == 0 or captura.size < referencia.size:
        return 0, 0.0
    r, c = _normalizar(referencia), _normalizar(captura)
    bruto = correlate(c, r, mode="valid", method="fft")
    i = int(np.argmax(bruto))
    return i, float(bruto[i])


def atraso_global(env_ref: np.ndarray, env_cap: np.ndarray) -> int:
    """Atraso da gravação em relação à referência, em amostras do envelope.

    Pode ser negativo, se a reprodução começou antes da gravação.
    """
    from scipy.signal import correlate

    if env_ref.size == 0 or env_cap.size == 0:
        return 0
    bruto = correlate(_normalizar(env_cap), _normalizar(env_ref), mode="full", method="fft")
    return int(np.argmax(bruto)) - (len(env_ref) - 1)


def refinar_amostra(segmento: np.ndarray, captura: np.ndarray, inicio: int,
                    busca: int) -> tuple[int, float]:
    """Ajusta `inicio` à amostra, correlacionando a forma de onda em ±`busca`.

    Devolve (início refinado, correlação normalizada no ponto escolhido).
    """
    from scipy.signal import correlate

    lo = max(0, inicio - busca)
    hi = min(len(captura), inicio + len(segmento) + busca)
    regiao = np.asarray(captura[lo:hi], dtype=np.float64)
    seg = np.asarray(segmento, dtype=np.float64)
    if len(regiao) < len(seg) or not np.any(seg):
        return inicio, 0.0
    bruto = correlate(regiao, seg, mode="valid", method="fft")
    k = int(np.argmax(bruto))
    janela = regiao[k:k + len(seg)]
    denom = np.linalg.norm(seg) * np.linalg.norm(janela)
    return lo + k, float(bruto[k] / denom) if denom > 0 else 0.0


def alinhar(trechos: list[Trecho], referencia: np.ndarray, captura: np.ndarray,
            sample_rate: int, busca_s: float = BUSCA_LOCAL_S) -> list[Encaixe]:
    """Localiza cada trecho da playlist dentro da gravação.

    O ajuste é sequencial (a correção de um trecho vira o palpite do seguinte)
    porque a deriva de relógio é cumulativa.
    """
    env_ref = envelope(referencia, sample_rate)
    env_cap = envelope(captura, sample_rate)
    atraso_env = atraso_global(env_ref, env_cap)
    por_amostra = sample_rate / TAXA_ENVELOPE_HZ

    encaixes: list[Encaixe] = []
    deriva = 0            # correção acumulada, em amostras do envelope
    busca = max(1, int(round(busca_s * TAXA_ENVELOPE_HZ)))
    for t in trechos:
        ini_ref = int(round(t.inicio / por_amostra))
        n_env = max(1, int(round(t.n / por_amostra)))
        previsto = atraso_env + ini_ref + deriva
        lo = max(0, previsto - busca)
        hi = min(len(env_cap), previsto + n_env + busca)
        janela = env_cap[lo:hi]
        alvo = env_ref[ini_ref:ini_ref + n_env]
        desloc, corr = correlacao_maxima(alvo, janela)
        inicio_env = lo + desloc
        inicio = int(round(inicio_env * por_amostra))
        refinado = False
        if corr >= CORRELACAO_MINIMA:
            deriva = inicio_env - (atraso_env + ini_ref)
            fino, corr_fina = refinar_amostra(
                referencia[t.inicio:t.inicio + t.n], captura, inicio,
                int(round(BUSCA_FINA_S * sample_rate)))
            if corr_fina >= CORRELACAO_FINA_MINIMA:
                inicio, refinado = fino, True
        encaixes.append(Encaixe(t, inicio, corr, refinado))
    return encaixes


def recortar(captura: np.ndarray, encaixes: list[Encaixe]) -> list[tuple[Trecho, np.ndarray]]:
    """Devolve (trecho, áudio) só dos encaixes confiáveis."""
    saida = []
    for e in encaixes:
        if not e.confiavel:
            continue
        pedaco = captura[e.inicio_capturado: e.inicio_capturado + e.trecho.n]
        if len(pedaco) < e.trecho.n // 2:     # cortado pelo fim da gravação
            continue
        saida.append((e.trecho, np.asarray(pedaco, dtype=np.float32)))
    return saida


def montar_referencia(audios: list[tuple[Trecho, np.ndarray]], sample_rate: int,
                      gap_s: float) -> tuple[np.ndarray, list[Trecho]]:
    """Concatena os áudios com silêncio entre eles e devolve o mapa de posições.

    O silêncio dá folga ao recorte e separa os áudios para a supressão de ruído.
    """
    gap = np.zeros(int(round(gap_s * sample_rate)), dtype=np.float32)
    partes, mapa, cursor = [gap], [], len(gap)
    for t, wav in audios:
        partes.append(np.asarray(wav, dtype=np.float32))
        mapa.append(Trecho(t.id, t.rotulo, t.sistema, cursor, len(wav)))
        cursor += len(wav) + len(gap)
        partes.append(gap)
    return np.concatenate(partes), mapa


def salvar_mapa(mapa: list[Trecho], destino: Path, sample_rate: int) -> None:
    destino.write_text(json.dumps(
        {"sample_rate": sample_rate, "trechos": [asdict(t) for t in mapa]},
        indent=2, ensure_ascii=False), encoding="utf-8")


def carregar_mapa(origem: Path) -> tuple[list[Trecho], int]:
    dados = json.loads(Path(origem).read_text(encoding="utf-8"))
    return [Trecho(**t) for t in dados["trechos"]], dados["sample_rate"]


def linha_de_protocolo(t: Trecho) -> str:
    """Formato ASVspoof: `SPEAKER FILE - SYSTEM KEY`, para o evaluate.py ler."""
    return f"LA_XXXX {t.id} - {t.sistema} {t.rotulo}"
