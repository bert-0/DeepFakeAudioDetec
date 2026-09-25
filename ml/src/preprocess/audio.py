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


#: Folga ao comparar a duração decodificada com a declarada no cabeçalho.
#: Reamostragem e arredondamento mudam o comprimento em alguns quadros, nunca
#: em fração perceptível do sinal.
TOLERANCIA_S = 0.01
TOLERANCIA_RELATIVA = 0.01


def _duracao_do_cabecalho(path: str | Path) -> float | None:
    """Duração que o cabeçalho declara, em segundos, ou `None` se ilegível.

    O cabeçalho **sobrevive à truncagem**: um FLAC cortado pela metade continua
    anunciando a duração original. É justamente isso que o torna útil aqui —
    ele diz o que o arquivo deveria ter, para comparar com o que saiu.
    """
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames / info.samplerate if info.samplerate else None
    except Exception:
        return None   # formato que o libsndfile não abre; nada a comparar


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Carrega um arquivo de áudio como mono, reamostrado para `sample_rate`.

    Um único `.flac` corrompido entre os 121.461 da base derruba um treino de
    horas — e sem este tratamento a mensagem não diz **qual**. O `librosa.load`,
    ao falhar no soundfile, tenta o backend `audioread`; sem ffmpeg instalado
    isso termina em `NoBackendError` com mensagem **vazia**, descartando o erro
    real do libsndfile (`flac decoder lost sync`, `Internal psf_fseek() failed`).

    Aqui o caminho vai para a mensagem e a exceção original fica encadeada,
    acessível por `__cause__`.

    **Nem todo decodificador falha num arquivo truncado.** No Linux o libsndfile
    recusa e o erro sobe. No Windows o `audioread`, com os backends que costumam
    vir instalados, decodifica o pedaço que existe e devolve áudio parcial sem
    reclamar — e aí o treino consome meio enunciado como se fosse inteiro. Foi
    medido: os testes de truncagem passam no Linux e falhavam no Windows.

    Por isso a leitura não confia no decodificador: a duração obtida é conferida
    contra a que o cabeçalho declara. A verificação é a mesma nos dois sistemas,
    qualquer que seja o backend que o librosa tenha escolhido.
    """
    try:
        wav, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    except FileNotFoundError:
        raise  # já traz o caminho e é inequívoco
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
    """Ajusta o sinal para exatamente `n_samples` (padding por repetição ou recorte).

    O padding repete o próprio sinal (em vez de zeros) para não introduzir
    longos trechos de silêncio que distorceriam as features.

    Se `rng` for informado e o sinal for mais longo que `n_samples`, o recorte é
    feito em uma posição **aleatória** (random crop). Isso é usado apenas no
    treino: cada época vê um trecho diferente do mesmo áudio, o que aumenta a
    diversidade dos dados e reduz o overfitting. Sem `rng` o recorte é
    determinístico (início do sinal), garantindo avaliação reprodutível.
    """
    if wav.size == n_samples:
        return wav
    if wav.size > n_samples:
        if rng is None:
            return wav[:n_samples]
        start = int(rng.integers(0, wav.size - n_samples + 1))
        return wav[start:start + n_samples]
    # wav.size < n_samples -> repete até cobrir o comprimento
    repeats = int(np.ceil(n_samples / wav.size))
    return np.tile(wav, repeats)[:n_samples]


def preprocess_waveform(
    wav: np.ndarray,
    audio_cfg: dict,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Aplica o pré-processamento completo a um waveform já carregado.

    `rng` habilita o recorte aleatório (ver `fix_length`); deve ser passado
    apenas no conjunto de treino.
    """
    if audio_cfg.get("trim_silence", False):
        wav = trim_silence(wav, audio_cfg.get("top_db", 30))
    if audio_cfg.get("peak_normalize", False):
        wav = peak_normalize(wav)
    n_samples = int(audio_cfg["sample_rate"] * audio_cfg["duration"])
    wav = fix_length(wav, n_samples, rng=rng)
    return wav.astype(np.float32)
