"""Fontes de áudio para o monitor ao vivo.

`WasapiLoopbackSource` captura a saída do sistema no Windows; `FileSource` lê um
arquivo simulando tempo real, para testar o resto do caminho em qualquer máquina.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class CaptureError(RuntimeError):
    """Falha ao abrir ou ler a fonte de áudio."""


class AudioSource(ABC):
    """Fonte de áudio mono em `sample_rate`, entregue em blocos de float32."""

    #: taxa de amostragem das amostras devolvidas por `blocos()`
    sample_rate: int

    @abstractmethod
    def blocos(self):
        """Gera arrays float32 mono. O tamanho de cada bloco é livre."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.fechar()

    def fechar(self) -> None:
        """Libera o dispositivo. Sem efeito por padrão."""


class FileSource(AudioSource):
    """Lê um arquivo em blocos, como se estivesse chegando ao vivo.

    Usada nos testes e para reprocessar exatamente o áudio de uma captura.
    """

    def __init__(self, caminho: str | Path, sample_rate: int,
                 bloco: int = 1600):
        import librosa

        self.sample_rate = sample_rate
        self.caminho = Path(caminho)
        self._bloco = int(bloco)
        try:
            self._wav, _ = librosa.load(str(caminho), sr=sample_rate, mono=True)
        except FileNotFoundError:
            raise
        except Exception as erro:
            raise CaptureError(f"não foi possível ler {caminho}: {erro}") from erro
        self._wav = self._wav.astype(np.float32)

    def blocos(self):
        for i in range(0, len(self._wav), self._bloco):
            yield self._wav[i:i + self._bloco]


#: Buffer do WASAPI. O padrão (~10 ms) perdia amostras sempre que o GIL segurava
#: a thread de captura; 1 s tolera as pausas sem aumentar a latência de leitura.
BUFFER_CAPTURA_S = 1.0


class WasapiLoopbackSource(AudioSource):
    """Captura a saída de áudio do sistema no Windows (loopback WASAPI).

    Pega o mix final da chamada, sem separar participantes; converte para mono
    e reamostra para a taxa do modelo.
    """

    def __init__(self, sample_rate: int, nome_dispositivo: str | None = None,
                 bloco_ms: int = 100):
        self.sample_rate = sample_rate
        self.nome_dispositivo = nome_dispositivo
        self._bloco_ms = int(bloco_ms)
        self._mic = None
        self._taxa_dispositivo = None
        #: Quantas vezes o WASAPI avisou de amostras perdidas nesta captura.
        self.descontinuidades = 0
        self._abrir()

    def _abrir(self) -> None:
        try:
            import soundcard
        except ImportError as erro:
            raise CaptureError(
                "a captura ao vivo precisa do pacote `soundcard`: "
                "pip install soundcard. Ele existe só para Windows/macOS/Linux "
                "com PulseAudio; no Windows é o que dá acesso ao loopback WASAPI."
            ) from erro

        try:
            if self.nome_dispositivo:
                alvo = self.nome_dispositivo
            else:
                alvo = soundcard.default_speaker().name
            # include_loopback=True transforma a SAÍDA num dispositivo de entrada.
            self._mic = soundcard.get_microphone(alvo, include_loopback=True)
        except Exception as erro:
            raise CaptureError(
                f"não foi possível abrir o loopback de '{self.nome_dispositivo or 'saída padrão'}': "
                f"{erro}. Liste os dispositivos com `python monitor.py --listar-dispositivos`."
            ) from erro
        # Pedir 16 kHz direto ao driver falha em muitos dispositivos.
        self._taxa_dispositivo = 48000

    def blocos(self):
        """Blocos já em mono e na taxa do modelo, na ordem em que foram gravados.

        A gravação roda numa thread própria para o processamento não fazer o buffer
        transbordar; a reamostragem é contínua (FIR longo, preserva até ~7,9 kHz).
        """
        import queue
        import threading
        import warnings

        fila: queue.Queue = queue.Queue()
        self._parar = threading.Event()
        quadros = int(self._taxa_dispositivo * self._bloco_ms / 1000)
        conversor = _conversor(self._taxa_dispositivo, self.sample_rate)

        def gravar():
            _iniciar_com()
            try:
                buffer = int(self._taxa_dispositivo * BUFFER_CAPTURA_S)
                with self._mic.recorder(samplerate=self._taxa_dispositivo,
                                        blocksize=buffer) as gravador:
                    while not self._parar.is_set():
                        with warnings.catch_warnings(record=True) as avisos:
                            warnings.simplefilter("always")
                            dados = gravador.record(numframes=quadros)
                        self.descontinuidades += sum(
                            1 for a in avisos if "discontinuity" in str(a.message))
                        fila.put(dados)
            except Exception as erro:          # repassado ao consumidor
                fila.put(erro)
            finally:
                fila.put(None)

        self.descontinuidades = 0
        self._thread = threading.Thread(target=gravar, name="captura-loopback",
                                        daemon=True)
        self._thread.start()
        while True:
            try:
                # Sem timeout, o get() bloqueia o Ctrl+C no Windows.
                dados = fila.get(timeout=0.5)
            except queue.Empty:
                continue
            if dados is None:
                return
            if isinstance(dados, Exception):
                raise CaptureError(f"a captura parou: {dados}") from dados
            mono = dados.mean(axis=1) if dados.ndim > 1 else dados
            yield conversor(np.ascontiguousarray(mono, dtype=np.float32))

    def fechar(self) -> None:
        parar = getattr(self, "_parar", None)
        if parar is not None:
            parar.set()
        self._mic = None


def _iniciar_com() -> None:
    """Inicializa o COM na thread de captura (Windows); repetir é inofensivo."""
    import sys

    if sys.platform == "win32":
        import ctypes

        ctypes.windll.ole32.CoInitializeEx(None, 0)   # COINIT_MULTITHREADED


def _conversor(origem: int, destino: int):
    """Conversor contínuo: FIR longo quando a razão é inteira (48 -> 16 kHz), soxr
    nos outros casos. O soxr apaga 7,6-8 kHz; o FIR mantém até ~7,9 kHz."""
    if origem % destino == 0:
        from ..preprocess.reamostragem import DecimadorFIR

        return DecimadorFIR(origem, destino)
    import soxr

    fluxo = soxr.ResampleStream(origem, destino, 1, dtype="float32")
    return fluxo.resample_chunk


def resample_mono(wav: np.ndarray, origem: int, destino: int) -> np.ndarray:
    """Reamostra um sinal mono. Sem efeito quando as taxas coincidem.

    Isolada aqui porque a reamostragem 48 -> 16 kHz faz parte do desvio de domínio.
    """
    if origem == destino or wav.size == 0:
        return wav.astype(np.float32)
    import librosa

    return librosa.resample(wav.astype(np.float32), orig_sr=origem,
                            target_sr=destino).astype(np.float32)


def abrir_fonte(origem: str | Path | None, sample_rate: int,
                dispositivo: str | None = None) -> AudioSource:
    """Escolhe a fonte: um arquivo quando `origem` é dado, o sistema quando não."""
    if origem is not None:
        return FileSource(origem, sample_rate)
    return WasapiLoopbackSource(sample_rate, nome_dispositivo=dispositivo)
