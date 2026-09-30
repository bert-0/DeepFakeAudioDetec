"""Captura do loopback numa thread própria (src/capture/sources.py).

O `soundcard` real só existe no Windows com placa de som; aqui ele é trocado por
um falso que grava uma senoide contínua, avisa de descontinuidade quando
mandado e demora quanto se quiser — o suficiente para testar a ordem dos
blocos, a contagem de perdas e a reamostragem sem bordas.
"""

import sys
import threading
import time
import types
import warnings

import numpy as np
import pytest

from src.capture.sources import CaptureError, WasapiLoopbackSource


class _GravadorFalso:
    def __init__(self, sinal, avisar_em, falhar_em=None):
        self.sinal, self.pos = sinal, 0
        self.avisar_em, self.falhar_em, self.n = set(avisar_em), falhar_em, 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def record(self, numframes):
        self.n += 1
        if self.falhar_em is not None and self.n == self.falhar_em:
            raise RuntimeError("dispositivo removido")
        if self.n in self.avisar_em:
            warnings.warn("data discontinuity in recording", RuntimeWarning)
        bloco = self.sinal[self.pos:self.pos + numframes]
        self.pos += numframes
        if len(bloco) < numframes:
            bloco = np.concatenate([bloco, np.zeros(numframes - len(bloco))])
        return np.stack([bloco, bloco], axis=1).astype(np.float32)   # estéreo


def _soundcard_falso(monkeypatch, gravador):
    def recorder(samplerate, blocksize=None):
        gravador.blocksize = blocksize
        return gravador

    mic = types.SimpleNamespace(recorder=recorder)
    falso = types.SimpleNamespace(
        default_speaker=lambda: types.SimpleNamespace(name="alto-falante"),
        get_microphone=lambda nome, include_loopback: mic)
    monkeypatch.setitem(sys.modules, "soundcard", falso)


def _coletar(fonte, n_blocos, atraso_s=0.0):
    saida = []
    for i, bloco in enumerate(fonte.blocos()):
        saida.append(bloco)
        time.sleep(atraso_s)            # o "modelo" demorando
        if i + 1 >= n_blocos:
            break
    fonte.fechar()
    return np.concatenate(saida)


def test_consumidor_lento_nao_perde_nem_reordena(monkeypatch):
    """Com o "modelo" demorando entre blocos, o que sai tem de ser exatamente a
    reamostragem do sinal inteiro, sem buraco e sem troca de ordem."""
    import soxr

    rng = np.random.default_rng(0)
    sinal = rng.standard_normal(48000 * 3) * 0.1
    _soundcard_falso(monkeypatch, _GravadorFalso(sinal, avisar_em=[]))
    fonte = WasapiLoopbackSource(16000, bloco_ms=100)
    capturado = _coletar(fonte, 20, atraso_s=0.02)
    referencia = soxr.resample(sinal[: 20 * 4800].astype(np.float32), 48000, 16000)
    n = len(capturado) - 200        # o fim do fluxo ainda não foi descarregado
    assert fonte.descontinuidades == 0
    np.testing.assert_allclose(capturado[:n], referencia[:n], atol=1e-4)


def test_reamostragem_continua_sem_bordas_entre_blocos(monkeypatch):
    """Bloco a bloco com estado: nenhuma descontinuidade a cada 100 ms."""
    t = np.arange(48000 * 2) / 48000
    sinal = 0.5 * np.sin(2 * np.pi * 300 * t)
    _soundcard_falso(monkeypatch, _GravadorFalso(sinal, avisar_em=[]))
    capturado = _coletar(WasapiLoopbackSource(16000, bloco_ms=100), 15)
    salto = np.abs(np.diff(capturado[2000:]))
    # Senoide de 300 Hz a 16 kHz: a maior variação entre amostras é ~0,059.
    assert salto.max() < 0.07


def test_descontinuidades_sao_contadas_e_nao_poluem_a_tela(monkeypatch, recwarn):
    sinal = np.zeros(48000 * 2)
    _soundcard_falso(monkeypatch, _GravadorFalso(sinal, avisar_em=[2, 5, 6]))
    fonte = WasapiLoopbackSource(16000, bloco_ms=100)
    _coletar(fonte, 10)
    assert fonte.descontinuidades == 3
    assert not [w for w in recwarn if "discontinuity" in str(w.message)]


def test_falha_do_dispositivo_chega_ao_consumidor(monkeypatch):
    _soundcard_falso(monkeypatch, _GravadorFalso(np.zeros(48000), [], falhar_em=3))
    fonte = WasapiLoopbackSource(16000, bloco_ms=100)
    with pytest.raises(CaptureError, match="dispositivo removido"):
        _coletar(fonte, 50)


def test_fechar_encerra_a_thread(monkeypatch):
    _soundcard_falso(monkeypatch, _GravadorFalso(np.zeros(48000 * 60), []))
    fonte = WasapiLoopbackSource(16000, bloco_ms=100)
    _coletar(fonte, 3)
    fonte._thread.join(timeout=2)
    assert not fonte._thread.is_alive()
    assert threading.active_count() >= 1


def test_buffer_do_wasapi_e_pedido_grande(monkeypatch):
    """O padrão do soundcard é um período (~10 ms): qualquer pausa da thread
    perdia áudio. O pedido tem de cobrir pausas de centenas de ms."""
    from src.capture.sources import BUFFER_CAPTURA_S

    gravador = _GravadorFalso(np.zeros(48000), [])
    _soundcard_falso(monkeypatch, gravador)
    _coletar(WasapiLoopbackSource(16000, bloco_ms=100), 2)
    assert gravador.blocksize == int(48000 * BUFFER_CAPTURA_S)
    assert BUFFER_CAPTURA_S >= 0.5
