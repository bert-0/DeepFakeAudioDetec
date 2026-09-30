"""Conversão 16 <-> 48 kHz com FIR longo (src/preprocess/reamostragem.py)."""

import numpy as np
from scipy.signal import welch

from src.preprocess.channel import ChannelDegradation
from src.preprocess.reamostragem import DecimadorFIR, descer, ida_e_volta, subir

SR = 16000


def _db(x, y, lo, hi):
    f, px = welch(x, SR, nperseg=4096)
    _, py = welch(y[: len(x)], SR, nperseg=4096)
    m = (f >= lo) & (f < hi)
    return 10 * np.log10(py[m].mean() / px[m].mean())


def _ruido(segundos=10, semente=0):
    return np.random.default_rng(semente).standard_normal(SR * segundos).astype(np.float32) * 0.1


def test_ida_e_volta_preserva_o_topo_da_banda():
    """Onde o soxr perdia 8 dB (7,6-7,8 kHz), o FIR não perde nada."""
    x = _ruido()
    y = ida_e_volta(x, SR)
    assert len(y) == len(x)
    assert abs(_db(x, y, 1000, 7600)) < 0.2
    assert abs(_db(x, y, 7600, 7800)) < 0.5
    assert _db(x, y, 7800, 7950) > -4


def test_ganho_unitario_e_alinhado():
    x = _ruido(3)
    y = descer(subir(x, SR), para=SR)[: len(x)]
    assert np.corrcoef(x[500:-500], y[500:-500])[0, 1] > 0.98


def test_decimador_continuo_igual_ao_de_uma_vez():
    """Blocos de qualquer tamanho dão o mesmo sinal, a menos do atraso fixo."""
    x48 = np.random.default_rng(1).standard_normal(48000 * 3)
    d = DecimadorFIR()
    continuo = np.concatenate([d(b) for b in np.array_split(x48, 37)])
    atraso = (len(d.h) - 1) // 2 // 3
    np.testing.assert_allclose(continuo[atraso:atraso + 40000], descer(x48)[:40000], atol=1e-5)


def test_canal_captura_fir_na_simulacao():
    x = _ruido(4)
    y = ChannelDegradation("captura_fir", 48000, SR)(x)
    assert y.size == x.size
    assert abs(_db(x, y, 7600, 7800)) < 0.5
    antigo = ChannelDegradation("captura", 48000, SR)(x)
    assert _db(x, antigo, 7800, 7950) < -20, "o soxr apaga essa faixa; o FIR não"
