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


#: Buffer do WASAPI pedido ao `soundcard`. Sem isso ele usa UM período do
#: dispositivo (~10 ms): se a thread de captura ficar parada mais que isso — e
#: o GIL do Python a deixa parada enquanto a janela anterior vira features — o
#: áudio que chega é descartado e marcado como descontinuidade. Medido no
#: primeiro controle real: 58 perdas em 3 minutos, ~1 a cada 3 s, o bastante
#: para quase todo áudio da playlist levar um clique. Um segundo de buffer
#: tolera qualquer pausa realista, e não aumenta a latência de leitura: o
#: `record()` devolve assim que os quadros pedidos chegam.
BUFFER_CAPTURA_S = 1.0


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
        # A placa costuma operar em 48 kHz; gravamos nela e reamostramos depois,
        # porque pedir 16 kHz direto ao driver falha em muitos dispositivos.
        self._taxa_dispositivo = 48000

    def blocos(self):
        """Blocos já em mono e na taxa do modelo, na ordem em que foram gravados.

        A gravação roda numa **thread própria** e entrega por uma fila. Antes ela
        dividia o laço com o processamento: enquanto a janela passava pelo
        modelo, ninguém lia o dispositivo, o buffer do WASAPI transbordava e
        amostras se perdiam — o `soundcard` avisava com "data discontinuity in
        recording", dezenas de vezes numa sessão de 3 minutos. Cada perda é um
        salto na forma de onda, um clique de banda larga que o modelo não viu no
        treino.

        A reamostragem é **contínua** (`soxr.ResampleStream`): reamostrar cada
        bloco de 100 ms isoladamente cria uma borda a cada bloco.
        """
        import queue
        import threading
        import warnings

        import soxr

        fila: queue.Queue = queue.Queue()
        self._parar = threading.Event()
        quadros = int(self._taxa_dispositivo * self._bloco_ms / 1000)
        conversor = soxr.ResampleStream(self._taxa_dispositivo, self.sample_rate,
                                        1, dtype="float32")

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
            except Exception as erro:          # entregue ao consumidor, não engolido
                fila.put(erro)
            finally:
                fila.put(None)

        self.descontinuidades = 0
        self._thread = threading.Thread(target=gravar, name="captura-loopback",
                                        daemon=True)
        self._thread.start()
        while True:
            try:
                # Com timeout: um get() sem prazo bloqueia o Ctrl+C no Windows.
                dados = fila.get(timeout=0.5)
            except queue.Empty:
                continue
            if dados is None:
                return
            if isinstance(dados, Exception):
                raise CaptureError(f"a captura parou: {dados}") from dados
            mono = dados.mean(axis=1) if dados.ndim > 1 else dados
            yield conversor.resample_chunk(np.ascontiguousarray(mono, dtype=np.float32))

    def fechar(self) -> None:
        parar = getattr(self, "_parar", None)
        if parar is not None:
            parar.set()
        self._mic = None


def _iniciar_com() -> None:
    """Inicializa o COM na thread de captura (Windows).

    O `soundcard` inicializa o COM na thread que o importa. Uma thread nova
    costuma herdar o apartamento multithread implicitamente, mas isso não é
    garantido; chamar de novo é inofensivo (devolve S_FALSE).
    """
    import sys

    if sys.platform == "win32":
        import ctypes

        ctypes.windll.ole32.CoInitializeEx(None, 0)   # COINIT_MULTITHREADED


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
