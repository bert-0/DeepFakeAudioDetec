"""Degradações de canal: o que uma chamada faz com o áudio.

Módulo **separado** do `augment.py` de propósito. Aquele é importado pelo
`train.py` (`Augmenter`); este só pelo `robustness_eval.py`. Manter a fronteira
garante que medir robustez não possa alterar o caminho de treino, de avaliação
ou de fusão — nem por acidente.

O detector foi treinado no ASVspoof: áudio limpo, 16 kHz, sem compressão. Numa
chamada de Teams/Meet/Zoom o sinal passa por codec Opus a dezenas de kbps e, em
rede ruim, por banda estreita. Com `n_filter: 70` metade do banco de filtros do
LFCC olha acima de 4 kHz — justamente a região que o canal degrada primeiro.

Medido aqui, com ruído rosa (energia em todas as bandas), Opus a ~26 kbps:

    0- 1000 Hz   -0,9 dB
 1000- 2000 Hz   -1,8 dB
 2000- 4000 Hz   -2,6 dB
 4000- 6000 Hz   -3,7 dB
 6000- 8000 Hz   -4,4 dB

Ou seja: o codec não corta a alta frequência, atenua progressivamente. É bem
mais suave que um passa-baixa, mas o LFCC é uma DCT dos *logs* das energias do
banco, então alguns dB nas bandas altas deslocam os coeficientes de qualquer
forma. O efeito no EER só se conhece medindo.
"""

from __future__ import annotations

import io

import numpy as np

#: Taxas que o Opus aceita internamente. Pedir outra faz o libsndfile recusar.
TAXAS_OPUS = (8000, 12000, 16000, 24000, 48000)


def opus_roundtrip(wav: np.ndarray, sample_rate: int,
                   compression_level: float = 0.9) -> np.ndarray:
    """Codifica em Opus e decodifica de volta, tudo em memória.

    Usa o libsndfile que já vem com o `soundfile` — sem ffmpeg e sem dependência
    nova. `compression_level` vai de 0 (maior qualidade) a 1 (menor); medido em
    ruído branco de 4 s, isso varia de ~260 kbps a ~7 kbps. Como o Opus é VBR, a
    taxa resultante depende do conteúdo, então o script de robustez **mede e
    reporta** a taxa obtida em vez de fingir precisão que não existe.
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
    """Reamostra para `taxa_intermediaria` e volta — perde tudo acima da nova Nyquist.

    Com 8000 Hz simula a telefonia de banda estreita, para onde uma chamada cai
    quando a rede aperta: nada acima de 4 kHz sobrevive, ou seja, metade do banco
    de filtros passa a ver silêncio.
    """
    import librosa

    if taxa_intermediaria >= sample_rate:
        return wav.astype(np.float32)
    reduzido = librosa.resample(wav.astype(np.float32), orig_sr=sample_rate,
                                target_sr=taxa_intermediaria)
    voltou = librosa.resample(reduzido, orig_sr=taxa_intermediaria,
                              target_sr=sample_rate)
    return voltou.astype(np.float32)


def _ajustar_comprimento(saida: np.ndarray, n: int) -> np.ndarray:
    """Garante o mesmo nº de amostras da entrada.

    A perturbação é aplicada **depois** do `preprocess_waveform`, quando o sinal
    já tem o comprimento fixo que define o shape das features. Um codec que
    devolvesse alguns quadros a mais ou a menos mudaria esse shape e derrubaria
    a inferência no meio de uma avaliação de horas. Medido, o Opus preserva o
    comprimento; esta função existe para o caso em que deixe de preservar.
    """
    if saida.size == n:
        return saida
    if saida.size > n:
        return saida[:n]
    return np.pad(saida, (0, n - saida.size))


class ChannelDegradation:
    """Degradação de canal com a mesma assinatura do `Perturbation`.

    `(wav, rng=None) -> wav`, para que o dataset use os dois de forma
    intercambiável. **É uma classe e não uma closure** pelo mesmo motivo já
    documentado no `augment.py`: o dataset inteiro é serializado para os workers
    do DataLoader, e no Windows (`spawn`) isso acontece sempre. Uma `lambda`
    prenderia a avaliação a `num_workers=0`.

    Não usa `rng`: o canal é determinístico. O parâmetro existe só para casar
    com a interface.

    kind:
      - 'opus'  — level = `compression_level` em [0, 1]
      - 'band'  — level = taxa intermediária em Hz (8000 = banda estreita)
    """

    def __init__(self, kind: str, level: float, sample_rate: int):
        if kind not in ("opus", "band"):
            raise ValueError(f"degradação de canal desconhecida: {kind!r}")
        self.kind = kind
        self.level = level
        self.sample_rate = int(sample_rate)

    def __call__(self, wav: np.ndarray,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        n = wav.size
        if self.kind == "opus":
            saida = opus_roundtrip(wav, self.sample_rate, self.level)
        else:
            saida = limitar_banda(wav, self.sample_rate, int(self.level))
        return _ajustar_comprimento(saida, n)


class ChannelChain:
    """Aplica degradações em sequência — uma chamada real empilha mais de uma.

    O caso que importa é banda estreita **e** codec: quando a rede aperta, o
    Teams cai para banda estreita e continua comprimindo. Aplicar só um dos dois
    subestimaria a degradação.
    """

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
    """Taxa de bits que o Opus produz neste sinal, em kbps.

    O Opus é VBR: a taxa depende do conteúdo. Reportar a taxa medida no áudio
    real é mais honesto do que anunciar um número nominal.
    """
    import soundfile as sf

    buffer = io.BytesIO()
    with sf.SoundFile(buffer, "w", samplerate=sample_rate, channels=1,
                      format="OGG", subtype="OPUS",
                      compression_level=float(compression_level)) as arquivo:
        arquivo.write(wav)
    segundos = wav.size / sample_rate
    return buffer.getbuffer().nbytes * 8 / segundos / 1000
