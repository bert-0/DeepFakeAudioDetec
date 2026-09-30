"""PreProcessador — leitura e padronização do sinal de áudio (RF02).

Etapas: carregar -> mono -> resample -> remover silêncio -> normalizar ->
ajustar para um comprimento fixo (recorte ou padding).
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


class AudioLoadError(RuntimeError):
    """Falha ao ler um arquivo de áudio, com o caminho embutido na mensagem."""


#: Folga ao comparar a duração decodificada com a do cabeçalho: reamostragem e
#: arredondamento mudam o comprimento em alguns quadros.
TOLERANCIA_S = 0.01
TOLERANCIA_RELATIVA = 0.01


def _duracao_do_cabecalho(path: str | Path) -> float | None:
    """Duração que o cabeçalho declara, em segundos, ou `None` se ilegível.

    O cabeçalho sobrevive à truncagem, então diz quanto o arquivo deveria ter.
    """
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames / info.samplerate if info.samplerate else None
    except Exception:
        return None   # formato que o libsndfile não abre; nada a comparar


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Carrega o áudio como mono em `sample_rate`; erros levam o caminho no texto.

    A duração lida é conferida com a do cabeçalho, porque no Windows o
    `audioread` devolve áudio parcial de um arquivo truncado sem reclamar.
    """
    try:
        wav, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    except FileNotFoundError:
        raise  # já traz o caminho
    except Exception as erro:
        detalhe = str(erro) or type(erro).__name__
        raise AudioLoadError(
            f"falha ao ler o áudio {path}: {detalhe}. "
            "Arquivo possivelmente truncado ou corrompido — rode "
            "`python scripts/check_data.py --config <cfg> --deep` para "
            "localizar todos os arquivos ilegíveis da base."
        ) from erro

    esperado = _duracao_do_cabecalho(path)
    if esperado is not None:
        obtido = len(wav) / sample_rate
        folga = max(TOLERANCIA_S, esperado * TOLERANCIA_RELATIVA)
        if obtido < esperado - folga:
            raise AudioLoadError(
                f"falha ao ler o áudio {path}: o cabeçalho declara "
                f"{esperado:.3f}s mas só {obtido:.3f}s foram decodificados. "
                "Arquivo possivelmente truncado ou corrompido — rode "
                "`python scripts/check_data.py --config <cfg> --deep` para "
                "localizar todos os arquivos ilegíveis da base."
            )
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


def fix_length(
    wav: np.ndarray,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Ajusta para `n_samples`: recorta se longo, repete o sinal (não zeros) se curto.

    Com `rng` o recorte cai numa posição aleatória (treino); sem ele, pega o início.
    """
    if wav.size == n_samples:
        return wav
    if wav.size > n_samples:
        if rng is None:
            return wav[:n_samples]
        start = int(rng.integers(0, wav.size - n_samples + 1))
        return wav[start:start + n_samples]
    repeats = int(np.ceil(n_samples / wav.size))
    return np.tile(wav, repeats)[:n_samples]


def preprocess_waveform(
    wav: np.ndarray,
    audio_cfg: dict,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Aplica o pré-processamento completo a um waveform já carregado.

    `rng` liga o recorte aleatório; passe só no treino.
    """
    if audio_cfg.get("trim_silence", False):
        wav = trim_silence(wav, audio_cfg.get("top_db", 30))
    if audio_cfg.get("peak_normalize", False):
        wav = peak_normalize(wav)
    n_samples = int(audio_cfg["sample_rate"] * audio_cfg["duration"])
    wav = fix_length(wav, n_samples, rng=rng)
    return wav.astype(np.float32)
