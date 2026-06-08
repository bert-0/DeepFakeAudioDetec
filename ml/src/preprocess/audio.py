"""PreProcessador — leitura e padronização do sinal de áudio (RF02).

Etapas: carregar -> mono -> resample -> remover silêncio -> normalizar ->
ajustar para um comprimento fixo (recorte ou padding).
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Carrega um arquivo de áudio como mono, reamostrado para `sample_rate`."""
    wav, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return wav.astype(np.float32)


def trim_silence(wav: np.ndarray, top_db: float) -> np.ndarray:
    """Remove silêncio no início/fim do sinal."""
    trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
    # Se o trim zerar o sinal (áudio muito baixo), mantém o original.
    return trimmed if trimmed.size > 0 else wav


def peak_normalize(wav: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normaliza a amplitude pelo pico (faixa aproximada [-1, 1])."""
    peak = np.max(np.abs(wav))
    return wav / (peak + eps)


def fix_length(wav: np.ndarray, n_samples: int) -> np.ndarray:
    """Ajusta o sinal para exatamente `n_samples` (padding por repetição ou recorte).

    O padding repete o próprio sinal (em vez de zeros) para não introduzir
    longos trechos de silêncio que distorceriam as features.
    """
    if wav.size == n_samples:
        return wav
    if wav.size > n_samples:
        return wav[:n_samples]
    # wav.size < n_samples -> repete até cobrir o comprimento
    repeats = int(np.ceil(n_samples / wav.size))
    return np.tile(wav, repeats)[:n_samples]


def preprocess_waveform(wav: np.ndarray, audio_cfg: dict) -> np.ndarray:
    """Aplica o pré-processamento completo a um waveform já carregado."""
    if audio_cfg.get("trim_silence", False):
        wav = trim_silence(wav, audio_cfg.get("top_db", 30))
    if audio_cfg.get("peak_normalize", False):
        wav = peak_normalize(wav)
    n_samples = int(audio_cfg["sample_rate"] * audio_cfg["duration"])
    wav = fix_length(wav, n_samples)
    return wav.astype(np.float32)
