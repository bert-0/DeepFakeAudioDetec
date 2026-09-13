"""Alinha uma gravação de chamada real de volta aos áudios que foram tocados.

**O problema.** A camada 2 consiste em tocar áudios de rótulo conhecido dentro
de uma chamada real (Teams, Meet) e capturar o que sai do outro lado. Só que a
gravação chega como um bloco único de vários minutos: sem saber onde cada áudio
começa e termina, não há como atribuir rótulo nenhum e a medida não existe.

**Por que não marcar com um bipe.** A ideia óbvia — um tom entre os áudios —
falha justamente no cenário que interessa: a supressão de ruído do Teams é
treinada para remover o que *não* é fala, e um seno puro é o exemplo canônico
disso. O marcador some no caminho.

**O que se usa aqui.** Correlação cruzada do *envelope de energia* contra a
referência tocada. O envelope sobrevive ao codec, à supressão de ruído e ao
ganho automático, porque nenhum deles reordena o áudio no tempo — eles mudam o
espectro e a amplitude, não quando a fala acontece. A normalização antes da
correlação tira o efeito do AGC.

O alinhamento é feito em duas etapas: um deslocamento global (a latência da
chamada, tipicamente centenas de ms) e depois um ajuste por trecho, que absorve
a deriva de relógio entre as duas placas de som.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

#: Taxa do envelope. 100 Hz dá resolução de 10 ms — bem abaixo do erro que
#: importa aqui (o recorte tem folga de `gap_s` de silêncio dos dois lados).
TAXA_ENVELOPE_HZ = 100

#: Correlação normalizada abaixo disto significa que não se achou o trecho.
#: Emitir um recorte assim mesmo produziria áudio com rótulo errado — e um EER
#: que parece resultado mas não é. Prefere-se descartar e dizer quantos caíram.
#:
#: Medido em `tests/test_alinhamento.py`, pior caso de cada cenário:
#:
#:     canal limpo                       0,798
#:     opus + banda estreita de 8 kHz    0,798   <- o canal que interessa
#:     ruído puro (microfone errado)     0,118
#:
#: O corte fica entre os dois grupos com folga de 1,6x para baixo e 4,2x para
#: cima. O codec praticamente não mexe no envelope, que é o ponto do método.
CORRELACAO_MINIMA = 0.5

#: Quanto o ajuste por trecho pode andar em torno da posição prevista.
BUSCA_LOCAL_S = 0.40


@dataclass
class Trecho:
    """Um áudio dentro da playlist: onde ele está na referência e o rótulo."""
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

    @property
    def confiavel(self) -> bool:
        return self.correlacao >= CORRELACAO_MINIMA


def envelope(wav: np.ndarray, sample_rate: int,
             taxa_hz: int = TAXA_ENVELOPE_HZ) -> np.ndarray:
    """Energia RMS em janelas curtas, sem sobreposição.

    É o que sobrevive ao canal: codec, supressão de ruído e AGC mexem no
    espectro e na amplitude, não em *quando* a fala acontece.
    """
    passo = max(1, int(round(sample_rate / taxa_hz)))
    n = len(wav) // passo
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    blocos = np.asarray(wav[: n * passo], dtype=np.float64).reshape(n, passo)
    return np.sqrt((blocos ** 2).mean(axis=1))


def _normalizar(x: np.ndarray) -> np.ndarray:
    """Média zero e norma 1 — tira o ganho, que o AGC mexe o tempo todo."""
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    norma = np.linalg.norm(x)
    return x / norma if norma > 0 else x


def correlacao_maxima(referencia: np.ndarray,
                      captura: np.ndarray) -> tuple[int, float]:
    """Desloca `referencia` sobre `captura` e devolve (atraso, correlação).

    O atraso é em amostras do envelope e nunca é negativo: a gravação começa
    antes da reprodução, por construção do procedimento.
    """
    from scipy.signal import correlate

    if referencia.size == 0 or captura.size < referencia.size:
        return 0, 0.0
    r, c = _normalizar(referencia), _normalizar(captura)
    bruto = correlate(c, r, mode="valid", method="fft")
    i = int(np.argmax(bruto))
    return i, float(bruto[i])


def alinhar(trechos: list[Trecho], referencia: np.ndarray, captura: np.ndarray,
            sample_rate: int, busca_s: float = BUSCA_LOCAL_S) -> list[Encaixe]:
    """Localiza cada trecho da playlist dentro da gravação.

    Duas etapas: o atraso global da chamada, e depois um ajuste por trecho que
    absorve a deriva de relógio entre as duas placas de som. O ajuste é
    *sequencial* — a correção de um trecho é o palpite inicial do seguinte —
    porque a deriva é cumulativa, não aleatória.
    """
    env_ref = envelope(referencia, sample_rate)
    env_cap = envelope(captura, sample_rate)
    atraso_env, _ = correlacao_maxima(env_ref, env_cap)
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
        if corr >= CORRELACAO_MINIMA:
            deriva = inicio_env - (atraso_env + ini_ref)
        encaixes.append(Encaixe(t, int(round(inicio_env * por_amostra)), corr))
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

    O silêncio existe para dar folga ao recorte e para que a supressão de ruído
    não trate a emenda entre dois áudios como um único fluxo contínuo.
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
