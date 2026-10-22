"""Codecs para o aumento de dados: codifica e decodifica em memória (libsndfile).

Só os que o `soundfile` já traz e que rodam em milissegundos no DataLoader: MP3,
Vorbis e Opus em banda larga; G.711 (µ-law, A-law) e GSM 06.10 a 8 kHz, a banda
de telefonia. AMR, Speex, AAC e Encodec, que o eval da ASVspoof 5 usa (C2–C4,
C6, C7, C9, C10), ficam de fora: exigiriam o ffmpeg num processo por amostra ou
um modelo neural por amostra. No eval, esses ficam como codecs que o treino
nunca viu.
"""

from __future__ import annotations

import io

import numpy as np

#: nome: (formato, subtipo, taxa de codificação ou None, faixa de compression_level
#: ou None). Taxas medidas num áudio de 6 s do recorte, a 16 kHz (RESUMO_TCC §10.7).
CODECS = {
    "mp3": ("MP3", "MPEG_LAYER_III", None, (0.0, 0.99)),   # 77 a 21 kbps; 1,0 dá erro
    "vorbis": ("OGG", "VORBIS", None, (0.0, 0.99)),        # 77 a 30 kbps
    "opus": ("OGG", "OPUS", None, (0.9, 0.99)),            # 30 a 9 kbps, a faixa de voz
    "ulaw": ("WAV", "ULAW", 8000, None),                   # G.711, 64 kbps
    "alaw": ("WAV", "ALAW", 8000, None),                   # G.711, 64 kbps
    "gsm": ("WAV", "GSM610", 8000, None),                  # GSM 06.10, 13 kbps
}


def faixa_padrao(nome: str) -> tuple[float, float] | None:
    if nome not in CODECS:
        raise ValueError(f"codec desconhecido: {nome!r} (use um de {sorted(CODECS)})")
    return CODECS[nome][3]


def codificar(wav: np.ndarray, sample_rate: int, nome: str,
              nivel: float | None = None) -> np.ndarray:
    """Ida e volta pelo codec. `nivel` é o compression_level (0 = melhor); os
    codecs de telefonia não têm nível e passam por 8 kHz com o FIR do projeto."""
    import soundfile as sf

    from .reamostragem import para_taxa

    faixa_padrao(nome)                           # nome desconhecido levanta aqui
    formato, subtipo, taxa, faixa = CODECS[nome]
    n = len(wav)
    # Os codecs inteiros saturam acima de 1; o ssi do RawBoost pode passar disso.
    x = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    taxa_cod = taxa if taxa and taxa < sample_rate else sample_rate
    if taxa_cod != sample_rate:
        x = para_taxa(x, sample_rate, taxa_cod)
    kw = {"compression_level": float(nivel if nivel is not None else faixa[0])} if faixa else {}
    buffer = io.BytesIO()
    with sf.SoundFile(buffer, "w", samplerate=taxa_cod, channels=1, format=formato,
                      subtype=subtipo, **kw) as arquivo:
        arquivo.write(x)
    buffer.seek(0)
    y, _ = sf.read(buffer, dtype="float32")
    if taxa_cod != sample_rate:
        y = para_taxa(y, taxa_cod, sample_rate)
    if len(y) >= n:
        return y[:n].astype(np.float32)
    return np.pad(y, (0, n - len(y))).astype(np.float32)
