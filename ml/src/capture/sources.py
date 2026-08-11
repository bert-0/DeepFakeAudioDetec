"""Fontes de áudio: de onde vêm as amostras a analisar.

Duas implementações com a mesma interface:

- `WasapiLoopbackSource` — captura a saída do sistema no Windows. É a que serve
  ao caso real (uma chamada em andamento) e a única que depende de biblioteca
  externa e de sistema operacional.
- `FileSource` — lê um `.wav`/`.flac` fingindo ser tempo real. Existe para que
  todo o resto do caminho (janela, features, modelo, agregação) seja testável
  em qualquer máquina, inclusive no Linux da integração contínua.

A separação importa: sem ela, nada do monitor ao vivo teria teste automatizado.
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

    Usada nos testes e no modo de repetição: permite reprocessar exatamente o
    mesmo áudio que uma captura gravou, o que é o que torna um resultado ao vivo
    reproduzível.
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


class WasapiLoopbackSource(AudioSource):
    """Captura a saída de áudio do sistema no Windows (loopback WASAPI).

    Pega o *mix* final: a voz de todos os participantes da chamada mais qualquer
    outro som que esteja tocando. Não há separação por participante — isso o
    Teams só entregaria através do bot de mídia, que é exatamente o caminho que
    este módulo evita. A consequência precisa aparecer na interface: o veredito é
    sobre o trecho de áudio, não sobre uma pessoa.

    O dispositivo entrega tipicamente 48 kHz estéreo; a conversão para mono e a
    reamostragem para a taxa do modelo acontecem aqui.
    """

    def __init__(self, sample_rate: int, nome_dispositivo: str | None = None,
                 bloco_ms: int = 100):
        self.sample_rate = sample_rate
        self.nome_dispositivo = nome_dispositivo
        self._bloco_ms = int(bloco_ms)
        self._mic = None
        self._taxa_dispositivo = None
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
        # A placa costuma operar em 48 kHz; gravamos nela e reamostramos depois,
        # porque pedir 16 kHz direto ao driver falha em muitos dispositivos.
        self._taxa_dispositivo = 48000

    def blocos(self):
        quadros = int(self._taxa_dispositivo * self._bloco_ms / 1000)
        with self._mic.recorder(samplerate=self._taxa_dispositivo) as gravador:
            while True:
                dados = gravador.record(numframes=quadros)
                mono = dados.mean(axis=1) if dados.ndim > 1 else dados
                yield resample_mono(mono.astype(np.float32),
                                    self._taxa_dispositivo, self.sample_rate)

    def fechar(self) -> None:
        self._mic = None


def resample_mono(wav: np.ndarray, origem: int, destino: int) -> np.ndarray:
    """Reamostra um sinal mono. Sem efeito quando as taxas coincidem.

    A reamostragem 48 kHz -> 16 kHz **faz parte do desvio de domínio**: o modelo
    foi treinado em áudio que já nasceu a 16 kHz. Fica isolada aqui para poder
    ser trocada e medida.
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
