"""Microfone por WASAPI exclusivo (src/capture/sources.py, MicrofoneWasapiSource).

O `sounddevice` real só abre o hardware no Windows; aqui um falso entrega o sinal
pelo callback numa thread própria, como o PortAudio, e recusa as configurações
pedidas no teste.
"""

import sys
import threading
import time
import types

import numpy as np
import pytest

from src.capture.sources import CaptureError, MicrofoneWasapiSource, abrir_microfone


class _FluxoFalso:
    def __init__(self, sd, callback, channels, blocksize, **kw):
        # blocksize 0: o driver escolhe; o falso usa o pacote dele (528, como o Realtek).
        self.sd, self.callback, self.canais = sd, callback, channels
        self.pedido, self.bloco = blocksize, blocksize or sd.pacote
        self.kw = kw
        self.active = False
        sd.fluxos.append(self)

    def __enter__(self):
        self.active = True
        self._parar = threading.Event()
        self._thread = threading.Thread(target=self._rodar, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._parar.set()
        self._thread.join(timeout=2)
        self.active = False
        return False

    def _rodar(self):
        sinal, pos, n = self.sd.sinal, 0, 0
        while not self._parar.is_set() and pos < len(sinal):
            bloco = sinal[pos:pos + self.bloco]
            pos += self.bloco
            n += 1
            dados = np.repeat(bloco[:, None], self.canais, axis=1).astype(np.float32)
            status = types.SimpleNamespace(input_overflow=n in self.sd.transbordar_em)
            self.callback(dados, len(bloco), None, status)
        if self.sd.parar_no_fim:
            self.active = False


def _sounddevice_falso(monkeypatch, sinal=None, recusar=(), parar_no_fim=False,
                       transbordar_em=(), taxas=(48000,), pacote=528):
    """`recusar`: pares (exclusivo, canais) que o driver não aceita; `taxas`: as
    taxas que ele aceita; `pacote`: o tamanho que o driver entrega com blocksize 0."""
    sd = types.SimpleNamespace(
        sinal=np.zeros(0) if sinal is None else sinal, fluxos=[], parar_no_fim=parar_no_fim,
        transbordar_em=set(transbordar_em), pacote=pacote)
    sd.query_hostapis = lambda: [
        {"name": "MME", "default_input_device": 0},
        {"name": "Windows WASAPI", "default_input_device": 2}]
    sd.query_devices = lambda: [
        {"name": "Microfone (MME)", "hostapi": 0, "max_input_channels": 2},
        {"name": "Alto-falantes (Realtek)", "hostapi": 1, "max_input_channels": 0},
        {"name": "Microfone (Realtek Audio)", "hostapi": 1, "max_input_channels": 2},
        {"name": "Headset (USB Audio)", "hostapi": 1, "max_input_channels": 1}]
    sd.WasapiSettings = lambda exclusive=False: types.SimpleNamespace(exclusive=exclusive)

    def check_input_settings(device, channels, dtype, extra_settings, samplerate):
        assert dtype == "float32"
        if samplerate not in taxas or (extra_settings.exclusive, channels) in recusar:
            raise RuntimeError("Invalid sample rate")

    sd.check_input_settings = check_input_settings
    sd.InputStream = lambda **kw: _FluxoFalso(sd, **kw)
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return sd


def _coletar(fonte, n_blocos, atraso_s=0.0):
    saida = []
    for i, bloco in enumerate(fonte.blocos()):
        saida.append(bloco)
        time.sleep(atraso_s)
        if i + 1 >= n_blocos:
            break
    fonte.fechar()
    return np.concatenate(saida)


def test_exclusivo_entrega_o_sinal_reamostrado_pelo_fir(monkeypatch):
    from src.preprocess.reamostragem import DecimadorFIR

    sinal = np.random.default_rng(0).standard_normal(48000 * 3) * 0.1
    sd = _sounddevice_falso(monkeypatch, sinal)
    fonte = MicrofoneWasapiSource(16000)
    assert (fonte.modo, fonte.aviso, fonte.nome) == ("exclusivo", None, "Microfone (Realtek Audio)")
    capturado = _coletar(fonte, 20, atraso_s=0.01)
    referencia = DecimadorFIR()(sinal[: 20 * 4800])
    np.testing.assert_allclose(capturado, referencia[: len(capturado)], atol=1e-4)
    assert sd.fluxos[0].kw["extra_settings"].exclusive and sd.fluxos[0].canais == 1
    assert sd.fluxos[0].kw["samplerate"] == 48000


def test_driver_que_so_aceita_estereo_no_exclusivo(monkeypatch):
    sinal = np.random.default_rng(1).standard_normal(48000) * 0.1
    sd = _sounddevice_falso(monkeypatch, sinal, recusar={(True, 1)})
    fonte = MicrofoneWasapiSource(16000)
    assert fonte.modo == "exclusivo" and fonte._canais == 2
    assert len(_coletar(fonte, 3)) == 3 * 528 // 3, "3 pacotes do driver, de 48 para 16 kHz"
    assert sd.fluxos[0].canais == 2


def test_exclusivo_recusado_cai_no_compartilhado_com_aviso(monkeypatch):
    _sounddevice_falso(monkeypatch, recusar={(True, 1), (True, 2)})
    fonte = MicrofoneWasapiSource(16000)
    assert fonte.modo == "compartilhado"
    assert "recusou o modo exclusivo" in fonte.aviso


def test_sem_48_nem_96_khz_explica_como_configurar(monkeypatch):
    _sounddevice_falso(monkeypatch, taxas=(44100,))
    with pytest.raises(CaptureError, match="48 nem a 96 kHz.*48000 Hz"):
        MicrofoneWasapiSource(16000)


def test_microfone_que_so_abre_a_96_khz(monkeypatch):
    """Como o "USB Digital Audio" medido: só 96 kHz, 1 canal. 96 = 6 x 16, o FIR serve."""
    from src.preprocess.reamostragem import DecimadorFIR

    sinal = np.random.default_rng(5).standard_normal(96000 * 2) * 0.1
    sd = _sounddevice_falso(monkeypatch, sinal, taxas=(96000,))
    fonte = MicrofoneWasapiSource(16000)
    assert (fonte.modo, fonte.taxa_dispositivo) == ("exclusivo", 96000)
    capturado = _coletar(fonte, 10)
    referencia = DecimadorFIR(96000, 16000)(sinal[: 10 * 9600])
    np.testing.assert_allclose(capturado, referencia[: len(capturado)], atol=1e-4)
    assert sd.fluxos[0].kw["samplerate"] == 96000


def test_sem_taxa_pedida_entrega_o_audio_bruto(monkeypatch):
    sinal = np.random.default_rng(6).standard_normal(96000) * 0.1
    _sounddevice_falso(monkeypatch, sinal, taxas=(96000,))
    fonte = MicrofoneWasapiSource(None)
    assert fonte.sample_rate == 96000
    capturado = _coletar(fonte, 5)
    np.testing.assert_allclose(capturado, sinal[: len(capturado)], atol=1e-6)


def test_escolhe_o_microfone_pelo_nome_so_no_wasapi(monkeypatch):
    _sounddevice_falso(monkeypatch)
    assert MicrofoneWasapiSource(16000, nome_dispositivo="headset").nome == "Headset (USB Audio)"
    with pytest.raises(CaptureError, match="Disponíveis: Microfone \\(Realtek Audio\\); Headset"):
        MicrofoneWasapiSource(16000, nome_dispositivo="MME")


def test_conta_transbordamentos_do_buffer(monkeypatch):
    sinal = np.random.default_rng(2).standard_normal(48000) * 0.1
    _sounddevice_falso(monkeypatch, sinal, transbordar_em={2, 4})
    fonte = MicrofoneWasapiSource(16000)
    _coletar(fonte, 6)
    assert fonte.descontinuidades == 2


def test_fluxo_que_para_vira_erro_de_captura(monkeypatch):
    sinal = np.random.default_rng(3).standard_normal(4800 * 2) * 0.1
    _sounddevice_falso(monkeypatch, sinal, parar_no_fim=True)
    fonte = MicrofoneWasapiSource(16000)
    with pytest.raises(CaptureError, match="parou"):
        _coletar(fonte, 100)


def test_sem_sounddevice_usa_o_soundcard_com_aviso(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)    # import falha
    mic = types.SimpleNamespace(recorder=None)
    monkeypatch.setitem(sys.modules, "soundcard", types.SimpleNamespace(
        default_microphone=lambda: types.SimpleNamespace(name="mic"),
        get_microphone=lambda nome, include_loopback: mic))
    fonte = abrir_microfone(16000)
    assert type(fonte).__name__ == "MicrofoneSource"
    assert fonte.modo == "compartilhado" and "pip install sounddevice" in fonte.aviso


def test_com_sounddevice_abre_pelo_wasapi(monkeypatch):
    _sounddevice_falso(monkeypatch)
    fonte = abrir_microfone(16000)
    assert isinstance(fonte, MicrofoneWasapiSource) and fonte.modo == "exclusivo"


def test_pacotes_no_tamanho_do_driver(monkeypatch):
    """25/10/2026: blocos fixos de 100 ms faziam o PortAudio emendar pacotes de 528
    amostras, e as gravações tinham uma descontinuidade a cada 1,1 s (o encaixe de
    4.800 com 528 se repete a cada 52.800 amostras). Agora o driver escolhe."""
    sinal = np.random.default_rng(7).standard_normal(48000) * 0.1
    sd = _sounddevice_falso(monkeypatch, sinal)
    fonte = MicrofoneWasapiSource(None)
    capturado = _coletar(fonte, 20)
    assert sd.fluxos[0].pedido == 0
    assert fonte.quadros_por_pacote == 528
    np.testing.assert_array_equal(capturado, sinal[:len(capturado)].astype(np.float32))


@pytest.mark.parametrize("pacote", [441, 528, 1000, 4800])
def test_qualquer_tamanho_de_pacote_da_o_mesmo_sinal_a_16_khz(monkeypatch, pacote):
    """O FIR guarda a fase entre os pacotes: o tamanho que o driver escolher não muda o sinal."""
    from src.preprocess.reamostragem import DecimadorFIR

    sinal = np.random.default_rng(8).standard_normal(48000 * 2) * 0.1
    _sounddevice_falso(monkeypatch, sinal, pacote=pacote)
    capturado = _coletar(MicrofoneWasapiSource(16000), 48000 // pacote)
    referencia = DecimadorFIR()(sinal)
    np.testing.assert_allclose(capturado, referencia[: len(capturado)], atol=1e-4)


def test_bloco_fixo_ainda_pode_ser_pedido(monkeypatch):
    sd = _sounddevice_falso(monkeypatch, np.zeros(48000))
    _coletar(MicrofoneWasapiSource(None, bloco_ms=100), 2)
    assert sd.fluxos[0].pedido == 4800
