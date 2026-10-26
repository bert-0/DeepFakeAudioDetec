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
                 bloco: int = 1600, taxa_comum: int | None = None):
        """`taxa_comum`: passa o arquivo por ela antes da taxa do modelo, para
        arquivos de taxas diferentes (clone de 24 kHz, gravação de 48 kHz) chegarem
        ao modelo pela mesma conversão final (RESUMO_TCC §10.2.13)."""
        import librosa

        self.sample_rate = sample_rate
        self.caminho = Path(caminho)
        self._bloco = int(bloco)
        try:
            from ..preprocess.reamostragem import para_taxa

            wav, taxa = librosa.load(str(caminho), sr=None, mono=True)
            if taxa_comum and taxa != taxa_comum:
                wav, taxa = para_taxa(wav, taxa, taxa_comum), taxa_comum
            self._wav = para_taxa(wav, taxa, sample_rate)
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


class MicrofoneSource(WasapiLoopbackSource):
    """Captura um microfone (o padrão do sistema, se nenhum for dado).

    Mesmo caminho do loopback: 48 kHz, thread própria e FIR para 16 kHz. Os
    aprimoramentos do microfone no Windows também deslocam o score.
    """

    def _abrir(self) -> None:
        try:
            import soundcard
        except ImportError as erro:
            raise CaptureError("a captura do microfone precisa do pacote `soundcard`: "
                               "pip install soundcard") from erro
        try:
            alvo = self.nome_dispositivo or soundcard.default_microphone().name
            self._mic = soundcard.get_microphone(alvo, include_loopback=False)
        except Exception as erro:
            raise CaptureError(
                f"não foi possível abrir o microfone '{self.nome_dispositivo or 'padrão'}': "
                f"{erro}. Liste os dispositivos com `python monitor.py --listar-dispositivos`."
            ) from erro
        self._taxa_dispositivo = 48000


#: Múltiplos inteiros de 16 kHz, para o FIR; 44,1 kHz cairia no soxr. Há
#: microfones USB que só abrem a 96 kHz.
TAXAS_MICROFONE = (48000, 96000)


class MicrofoneWasapiSource(AudioSource):
    """Microfone por WASAPI em modo exclusivo (sounddevice), a 48 ou 96 kHz, com o FIR.

    No modo exclusivo o áudio não passa pelo mixer do Windows: sem os
    aprimoramentos do microfone e sem conversão de taxa. Se o driver recusar,
    abre em modo compartilhado e deixa o motivo em `aviso`. Com `sample_rate`
    None, entrega o áudio bruto, na taxa do dispositivo.
    """

    def __init__(self, sample_rate: int | None, nome_dispositivo: str | None = None,
                 bloco_ms: int | None = None, exclusivo: bool = True):
        self.nome_dispositivo = nome_dispositivo
        # None: os pacotes no tamanho do driver (480 no Realtek). Com blocos fixos de
        # 100 ms, as gravações tinham uma emenda áspera a cada 1,1 s, de causa ainda
        # não explicada (RESUMO_TCC §10.2.16).
        self._bloco_ms = None if bloco_ms is None else int(bloco_ms)
        #: Tamanho dos pacotes que o driver entregou (diagnóstico).
        self.quadros_por_pacote: int | None = None
        self._exclusivo_pedido = exclusivo
        #: "exclusivo" ou "compartilhado", conforme o driver aceitou.
        self.modo: str | None = None
        self.aviso: str | None = None
        #: Quantas vezes o PortAudio avisou que o buffer de entrada transbordou.
        self.descontinuidades = 0
        self._abrir()
        self.sample_rate = sample_rate or self.taxa_dispositivo

    def _abrir(self) -> None:
        sd = _importar_sounddevice()
        self._sd = sd
        self._indice, self.nome = _microfone_wasapi(sd, self.nome_dispositivo)
        erros = []
        for exclusivo in ([True] if self._exclusivo_pedido else []) + [False]:
            for taxa in TAXAS_MICROFONE:
                # Alguns drivers só aceitam estéreo no modo exclusivo.
                for canais in (1, 2):
                    extra = sd.WasapiSettings(exclusive=exclusivo)
                    try:
                        sd.check_input_settings(device=self._indice, channels=canais,
                                                dtype="float32", extra_settings=extra,
                                                samplerate=taxa)
                    except Exception as erro:
                        modo = "exclusivo" if exclusivo else "compartilhado"
                        erros.append(f"{modo}, {taxa // 1000} kHz, {canais} canal(is): {erro}")
                        continue
                    self.modo = "exclusivo" if exclusivo else "compartilhado"
                    self.taxa_dispositivo, self._canais, self._extra = taxa, canais, extra
                    if self._exclusivo_pedido and not exclusivo:
                        self.aviso = ("o driver recusou o modo exclusivo; gravando em modo "
                                      "compartilhado, em que os aprimoramentos do Windows "
                                      "podem alterar o áudio")
                    return
        raise CaptureError(
            f"o microfone '{self.nome}' não abre a 48 nem a 96 kHz ({'; '.join(erros)}). "
            "Configure o formato dele para 48000 Hz em Som > Gravação > "
            "Propriedades > Avançado.")

    def blocos(self):
        """Blocos já em mono e na taxa do modelo, na ordem em que foram gravados.

        O PortAudio chama o callback na thread dele; a fila separa a captura do
        processamento, como no loopback.
        """
        import queue
        import threading

        fila: queue.Queue = queue.Queue()
        self._parar = threading.Event()
        self.descontinuidades = 0
        quadros = (0 if self._bloco_ms is None
                   else int(self.taxa_dispositivo * self._bloco_ms / 1000))
        # Na taxa do dispositivo, o áudio sai bruto: o FIR 1:1 cortaria em 7,9 kHz.
        conversor = (_conversor(self.taxa_dispositivo, self.sample_rate)
                     if self.sample_rate != self.taxa_dispositivo else (lambda x: x))

        def callback(dados, n_quadros, tempo, status):
            if getattr(status, "input_overflow", False):
                self.descontinuidades += 1
            if self.quadros_por_pacote is None:
                self.quadros_por_pacote = int(n_quadros)
            fila.put(dados.copy())

        try:
            fluxo = self._sd.InputStream(
                device=self._indice, channels=self._canais, dtype="float32",
                samplerate=self.taxa_dispositivo, blocksize=quadros,
                extra_settings=self._extra, callback=callback)
        except Exception as erro:
            raise CaptureError(f"não foi possível abrir o microfone '{self.nome}': {erro}") from erro
        with fluxo:
            while not self._parar.is_set():
                try:
                    dados = fila.get(timeout=0.5)
                except queue.Empty:
                    if not fluxo.active:
                        raise CaptureError(f"a captura do microfone '{self.nome}' parou") from None
                    continue
                mono = dados.mean(axis=1) if dados.ndim > 1 else dados
                yield conversor(np.ascontiguousarray(mono, dtype=np.float32))

    def fechar(self) -> None:
        parar = getattr(self, "_parar", None)
        if parar is not None:
            parar.set()


def abrir_microfone(sample_rate: int, nome_dispositivo: str | None = None) -> AudioSource:
    """Microfone por WASAPI exclusivo; sem o `sounddevice`, o caminho do `soundcard`
    (modo compartilhado), com o aviso em `aviso`."""
    try:
        import sounddevice  # noqa: F401
    except (ImportError, OSError):
        fonte = MicrofoneSource(sample_rate, nome_dispositivo=nome_dispositivo)
        fonte.modo = "compartilhado"
        fonte.aviso = ("sem o pacote sounddevice o microfone abre em modo compartilhado, "
                       "sujeito aos aprimoramentos do Windows: pip install sounddevice")
        return fonte
    return MicrofoneWasapiSource(sample_rate, nome_dispositivo)


def _importar_sounddevice():
    try:
        import sounddevice
    except (ImportError, OSError) as erro:
        raise CaptureError("o microfone por WASAPI precisa do pacote `sounddevice`: "
                           "pip install sounddevice") from erro
    return sounddevice


def _entradas_wasapi(sd) -> tuple[list[tuple[int, str]], tuple[int, str] | None]:
    """Microfones do WASAPI (índice, nome) e o padrão do Windows, se houver."""
    apis = sd.query_hostapis()
    wasapi = next((i for i, a in enumerate(apis) if "WASAPI" in a["name"]), None)
    if wasapi is None:
        raise CaptureError("o WASAPI não está disponível (ele só existe no Windows)")
    dispositivos = list(sd.query_devices())
    entradas = [(i, d["name"]) for i, d in enumerate(dispositivos)
                if d["hostapi"] == wasapi and d["max_input_channels"] > 0]
    padrao = apis[wasapi]["default_input_device"]
    if padrao is None or padrao < 0:
        return entradas, None
    return entradas, (padrao, dispositivos[padrao]["name"])


def _microfone_wasapi(sd, nome: str | None) -> tuple[int, str]:
    """Índice e nome do microfone no WASAPI: o que contém `nome`, ou o padrão."""
    entradas, padrao = _entradas_wasapi(sd)
    if nome:
        achados = [(i, n) for i, n in entradas if nome.lower() in n.lower()]
        if not achados:
            raise CaptureError(f"nenhum microfone WASAPI com '{nome}' no nome. Disponíveis: "
                               + "; ".join(n for _, n in entradas))
        return achados[0]
    if padrao is None:
        raise CaptureError("o Windows não tem microfone padrão no WASAPI")
    return padrao


def listar_dispositivos() -> dict:
    """Nomes das saídas (para o loopback) e dos microfones, com os padrões."""
    import soundcard

    _iniciar_com()
    d = {
        "sistema": [x.name for x in soundcard.all_speakers()],
        "microfone": [x.name for x in soundcard.all_microphones()],
        "padrao": {"sistema": soundcard.default_speaker().name,
                   "microfone": soundcard.default_microphone().name},
    }
    # Com o sounddevice, o microfone abre pelo WASAPI, que tem os próprios nomes.
    try:
        entradas, padrao = _entradas_wasapi(_importar_sounddevice())
    except CaptureError:
        return d
    d["microfone"] = [n for _, n in entradas]
    d["padrao"]["microfone"] = padrao[1] if padrao else None
    return d


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
