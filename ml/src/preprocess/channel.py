"""Degradações de canal de chamada (Opus, banda estreita, captura ao vivo).

Fica fora do `augment.py` de propósito: só o `robustness_eval.py` importa este
módulo, então medir robustez não mexe no caminho de treino.
"""

from __future__ import annotations

import io

import numpy as np

#: Taxas que o Opus aceita internamente. Pedir outra faz o libsndfile recusar.
TAXAS_OPUS = (8000, 12000, 16000, 24000, 48000)


def opus_roundtrip(wav: np.ndarray, sample_rate: int,
                   compression_level: float = 0.9) -> np.ndarray:
    """Codifica em Opus e decodifica de volta, em memória (libsndfile do `soundfile`).

    `compression_level` vai de 0 (melhor) a 1 (pior): ~260 a ~7 kbps em ruído
    branco.
    """
    import soundfile as sf

    if sample_rate not in TAXAS_OPUS:
        raise ValueError(
            f"o Opus não aceita {sample_rate} Hz; use uma de {TAXAS_OPUS}")
    buffer = io.BytesIO()
    with sf.SoundFile(buffer, "w", samplerate=sample_rate, channels=1,
                      format="OGG", subtype="OPUS",
                      compression_level=float(compression_level)) as arquivo:
        arquivo.write(wav)
    buffer.seek(0)
    saida, _ = sf.read(buffer, dtype="float32")
    return saida.astype(np.float32)


def limitar_banda(wav: np.ndarray, sample_rate: int,
                  taxa_intermediaria: int) -> np.ndarray:
    """Reamostra para `taxa_intermediaria` e volta, perdendo o que passa da Nyquist.

    Com 8000 Hz simula telefonia de banda estreita.
    """
    import librosa

    if taxa_intermediaria >= sample_rate:
        return wav.astype(np.float32)
    reduzido = librosa.resample(wav.astype(np.float32), orig_sr=sample_rate,
                                target_sr=taxa_intermediaria)
    voltou = librosa.resample(reduzido, orig_sr=taxa_intermediaria,
                              target_sr=sample_rate)
    return voltou.astype(np.float32)


def ida_e_volta_captura(wav: np.ndarray, sample_rate: int,
                        taxa_dispositivo: int = 48000) -> np.ndarray:
    """Sobe à taxa do dispositivo e volta, como a captura ao vivo.

    Mesmo `soxr` do `monitor.py`. O antialiasing apaga 7,6-8 kHz, faixa de que
    o modelo depende (Seção 10.2.1).
    """
    import soxr

    x = np.asarray(wav, dtype=np.float32)
    return soxr.resample(soxr.resample(x, sample_rate, taxa_dispositivo),
                         taxa_dispositivo, sample_rate).astype(np.float32)


def _ajustar_comprimento(saida: np.ndarray, n: int) -> np.ndarray:
    """Corta ou completa com zeros até `n` amostras, sem mudar o shape das features.

    Hoje o Opus preserva o comprimento; isto é só proteção.
    """
    if saida.size == n:
        return saida
    if saida.size > n:
        return saida[:n]
    return np.pad(saida, (0, n - saida.size))


class ChannelDegradation:
    """Degradação de canal com a assinatura `(wav, rng=None) -> wav`.

    Classe, e não lambda, pelo mesmo motivo do `Perturbation`; `rng` é ignorado.
    `kind`: 'opus' (level = compression_level em [0, 1]), 'band' (taxa
    intermediária em Hz) ou 'captura' (taxa do dispositivo em Hz).
    """

    def __init__(self, kind: str, level: float, sample_rate: int):
        if kind not in ("opus", "band", "captura"):
            raise ValueError(f"degradação de canal desconhecida: {kind!r}")
        self.kind = kind
        self.level = level
        self.sample_rate = int(sample_rate)

    def __call__(self, wav: np.ndarray,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        n = wav.size
        if self.kind == "opus":
            saida = opus_roundtrip(wav, self.sample_rate, self.level)
        elif self.kind == "captura":
            saida = ida_e_volta_captura(wav, self.sample_rate, int(self.level))
        else:
            saida = limitar_banda(wav, self.sample_rate, int(self.level))
        return _ajustar_comprimento(saida, n)


class ChannelChain:
    """Aplica degradações em sequência (ex.: banda estreita e depois codec)."""

    def __init__(self, etapas: list[ChannelDegradation]):
        if not etapas:
            raise ValueError("a cadeia precisa de pelo menos uma etapa")
        self.etapas = list(etapas)

    def __call__(self, wav: np.ndarray,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        for etapa in self.etapas:
            wav = etapa(wav, rng)
        return wav


def medir_taxa_kbps(wav: np.ndarray, sample_rate: int,
                    compression_level: float) -> float:
    """Taxa de bits que o Opus produz neste sinal, em kbps (é VBR)."""
    import soundfile as sf

    buffer = io.BytesIO()
    with sf.SoundFile(buffer, "w", samplerate=sample_rate, channels=1,
                      format="OGG", subtype="OPUS",
                      compression_level=float(compression_level)) as arquivo:
        arquivo.write(wav)
    segundos = wav.size / sample_rate
    return buffer.getbuffer().nbytes * 8 / segundos / 1000
