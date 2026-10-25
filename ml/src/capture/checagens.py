"""Checagens de uma gravação logo depois de gravar (aba Gravar, Incremento 4).

Cada checagem diz se a frase deve ser regravada. Os limites são os mesmos para
todos os locutores (docs/etica/PROTOCOLO_GRAVACAO.md) e vêm do que o modelo faz
com o áudio: corte de silêncio a 30 dB do pico e janela de 4 s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qualidade import FRACAO_MINIMA, fracao_energia_alta

#: Amostra no limite do conversor. Um pico isolado ali é possível; uma sequência
#: de amostras iguais no limite é corte.
LIMITE_SATURACAO = 0.999
SEGUIDAS_SATURADAS = 3

#: O modelo decide em janelas de 4 s (RESUMO_TCC §9.1).
FALA_MINIMA_S = 4.0
#: Silêncio mínimo antes e depois da fala. Menos que isso indica frase cortada,
#: e deixa poucos quadros de pausa para medir o piso de ruído.
BORDA_S = 0.5

#: O `trim_silence` corta só o que fica 30 dB abaixo do pico; ruído acima disso
#: chega ao modelo, e ruído nas pausas desloca o score (RESUMO_TCC §10.2.1).
TOP_DB = 30.0
MARGEM_RUIDO_DB = 5.0
#: Medido nas frases do microfone embutido (24/10/2026): a +6 dB o ruído já entra
#: como fala; a +10 dB a fala começa depois do aviso e dura 5 a 7 s.
FALA_ACIMA_DO_PISO_DB = 10.0
#: Tecla e mão no notebook chegam ao microfone embutido pela carcaça: pancadas de
#: ~150 ms com metade da energia abaixo de 150 Hz, que a checagem tomava por fala
#: nas bordas. A voz fica quase toda acima disso (24/10/2026).
FALA_PASSA_ALTAS_HZ = 150.0
TRECHO_MINIMO_S = 0.25

FAIXAS_HZ = ((0, 4000), (4000, 7600), (7600, 8000))


@dataclass(frozen=True)
class Checagem:
    nome: str
    ok: bool
    valor: str
    motivo: str | None = None


def _maior_sequencia(mascara: np.ndarray) -> int:
    if not mascara.any():
        return 0
    bordas = np.diff(np.concatenate([[0], mascara.astype(np.int8), [0]]))
    return int((np.flatnonzero(bordas == -1) - np.flatnonzero(bordas == 1)).max())


def _db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-12)))


def _sem_graves(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    """Passa-altas só para achar a fala; as outras checagens usam o sinal inteiro."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, FALA_PASSA_ALTAS_HZ, btype="highpass", fs=sample_rate, output="sos")
    if wav.size <= 3 * (2 * len(sos) + 1):
        return wav
    return sosfiltfilt(sos, wav).astype(np.float32)


def trechos_de_fala(wav: np.ndarray, sample_rate: int,
                    top_db: float = TOP_DB) -> list[tuple[int, int]]:
    """Trechos com fala, em amostras: a regra da checagem de fala.

    Fala é o que fica a menos de `top_db` do pico e pelo menos
    FALA_ACIMA_DO_PISO_DB acima do ruído: com o ruído a menos de 30 dB da fala
    (microfone embutido, 16–21 dB), o corte só pelo pico marcaria a gravação
    inteira como fala. Sem graves e sem trechos curtos: pancadas de teclado.
    """
    import librosa

    voz = _sem_graves(np.asarray(wav, dtype=np.float32).reshape(-1), sample_rate)
    rms_voz = librosa.feature.rms(y=voz)[0]
    if rms_voz.size == 0 or rms_voz.max() <= 0:
        return []
    limite = max(float(rms_voz.max()) * 10 ** (-top_db / 20),
                 float(np.percentile(rms_voz, 10)) * 10 ** (FALA_ACIMA_DO_PISO_DB / 20))
    corte_fala = max(1.0, _db(float(rms_voz.max())) - _db(limite))
    return [(int(i), int(f)) for i, f in librosa.effects.split(voz, top_db=corte_fala)
            if f - i >= TRECHO_MINIMO_S * sample_rate]


def checar(wav: np.ndarray, sample_rate: int, top_db: float = TOP_DB,
           piso_minimo: float | None = None) -> list[Checagem]:
    """As quatro checagens do protocolo: saturação, fala, banda e piso de ruído.

    `piso_minimo`: distância mínima entre fala e ruído, em dB (padrão: o corte de
    silêncio mais a margem, 35 dB). O teste com a voz própria usa 15, a margem do
    microfone embutido na sala (RESUMO_TCC §10.2.12).
    """
    import librosa

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    pico = float(np.abs(wav).max()) if wav.size else 0.0
    saida = []

    seguidas = _maior_sequencia(np.abs(wav) >= LIMITE_SATURACAO)
    saturou = seguidas >= SEGUIDAS_SATURADAS
    saida.append(Checagem("saturação", not saturou, f"pico {_db(pico):.1f} dBFS",
                          f"{seguidas} amostras seguidas no limite: baixe o ganho"
                          if saturou else None))

    if pico == 0.0:
        return saida + [Checagem("fala", False, "0,0 s", "nenhum som gravado"),
                        Checagem("banda", False, "—", "nenhum som gravado"),
                        Checagem("piso de ruído", False, "—", "nenhum som gravado")]

    # Fala pelos quadros mais fortes, ruído pelos mais fracos: não depende de
    # achar as pausas, que somem quando o ruído passa do corte de silêncio.
    rms = librosa.feature.rms(y=wav)[0]
    distancia = _db(np.percentile(rms, 95)) - _db(np.percentile(rms, 10))

    trechos = trechos_de_fala(wav, sample_rate, top_db)
    fala_s = float(sum(f - i for i, f in trechos)) / sample_rate
    borda = int(BORDA_S * sample_rate)
    cortada = len(trechos) > 0 and (trechos[0][0] < borda or trechos[-1][1] > len(wav) - borda)
    if fala_s < FALA_MINIMA_S:
        motivo = f"menos de {FALA_MINIMA_S:.0f} s de fala: a frase não ocupa uma janela"
    elif cortada:
        motivo = "a fala encosta no início ou no fim: deixe um silêncio antes e depois"
    else:
        motivo = None
    saida.append(Checagem("fala", motivo is None, f"{fala_s:.1f} s", motivo))

    alta = fracao_energia_alta(wav, sample_rate)
    saida.append(Checagem("banda", alta >= FRACAO_MINIMA, f"{100 * alta:.0f}% acima de 4 kHz",
                          None if alta >= FRACAO_MINIMA else
                          "nada acima de 4 kHz: o caminho perdeu a banda alta"))

    minimo = top_db + MARGEM_RUIDO_DB if piso_minimo is None else float(piso_minimo)
    motivo = f"ruído a menos de {minimo:.0f} dB da fala"
    if minimo > top_db:
        motivo += ": o corte de silêncio do modelo não o remove"
    saida.append(Checagem("piso de ruído", distancia >= minimo, f"{distancia:.0f} dB abaixo da fala",
                          None if distancia >= minimo else motivo))
    return saida


def energia_por_faixa(wav: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Fração da energia em 0–4, 4–7,6 e 7,6–8 kHz, e acima de 8 kHz se houver."""
    from scipy.signal import welch

    wav = np.asarray(wav, dtype=np.float64).reshape(-1)
    f, p = welch(wav, sample_rate, nperseg=min(2048, max(wav.size, 8)))
    total = p.sum()
    faixas = list(FAIXAS_HZ) + ([(8000, sample_rate / 2)] if sample_rate > 16000 else [])
    saida = {}
    for baixo, alto in faixas:
        nome = f"{baixo / 1000:g}–{alto / 1000:g} kHz".replace(".", ",")
        dentro = (f >= baixo) & ((f < alto) if alto < sample_rate / 2 else (f <= alto))
        saida[nome] = float(p[dentro].sum() / total) if total > 0 else 0.0
    return saida
