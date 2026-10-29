"""RawBoost (src/preprocess/rawboost.py) e o tipo `rawboost` do gancho de degradação.

A referência é o código dos autores: as funções aqui são conferidas contra o
que ele faz (a filtragem contra o `lfilter`, a SNR do ssi, os limites do isd),
com sinais sintéticos.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import freqz, lfilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import rawboost as rb  # noqa: E402
from src.preprocess.degradacao import Degradacao  # noqa: E402

SR = 16000


def _fala(segundos=3.0, amplitude=0.1, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(segundos * SR)) / SR
    return (amplitude * (np.sin(2 * np.pi * 180 * t) + 0.3 * rng.standard_normal(t.size))).astype(np.float32)


@pytest.mark.parametrize("n_coef", [51, 52])
def test_filtragem_por_fft_igual_ao_lfilter_dos_autores(n_coef):
    x = _fala().astype(np.float64)
    b = np.random.default_rng(1).standard_normal(n_coef)
    n = len(b) + 1
    referencia = lfilter(b, 1, np.pad(x, (0, n)))
    referencia = referencia[int(n / 2): int(len(referencia) - n / 2)]
    y = rb.filtrar(x, b)
    assert len(y) == len(x)
    np.testing.assert_allclose(y, referencia, atol=1e-9)


def test_filtro_de_notches_com_pico_no_ganho_sorteado():
    p = rb.Parametros(n_bandas=1, f_min=2000, f_max=2001, banda_min=400, banda_max=401,
                      coef_min=101, coef_max=102)
    b = rb.coeficientes_notch(np.random.default_rng(0), p, SR, -6.0, -6.0)
    f, h = freqz(b, 1, fs=SR, worN=4096)
    db = 20 * np.log10(np.abs(h) + 1e-12)
    # pico levado a -6 dB; o código mede com os 512 pontos do freqz, como o original
    assert abs(db.max() - (-6.0)) < 0.01
    assert db[np.argmin(np.abs(f - 2000))] < db.max() - 20  # o notch em 2 kHz


def test_ganho_das_ordens_nao_lineares_vem_invertido_e_funciona():
    """-5 a -20 dB: o Generator recusaria uniform(-5, -20) sem a troca."""
    y = rb.lnl(_fala().astype(np.float64), np.random.default_rng(0), rb.Parametros(), SR)
    assert y.shape == (3 * SR,) and np.isfinite(y).all()
    assert abs(y.mean()) < 1e-9 and np.abs(y).max() <= 1.0


def test_isd_mexe_em_ate_p_por_cento_e_proporcional_ao_sinal():
    x = _fala().astype(np.float64)                         # pico < 1: sem normalizar
    p = rb.Parametros()
    for seed in range(5):
        y = rb.isd(x, np.random.default_rng(seed), p)
        mudou = y != x
        assert mudou.mean() <= p.p_max / 100
        assert np.all(np.abs(y - x) <= p.g_sd * np.abs(x) + 1e-12)


def test_ssi_cai_na_snr_sorteada():
    x = _fala().astype(np.float64)
    for seed in range(5):
        y = rb.ssi(x, np.random.default_rng(seed), rb.Parametros(), SR)
        snr = 10 * np.log10(np.sum(x ** 2) / np.sum((y - x) ** 2))
        assert 10 - 1e-6 <= snr <= 40 + 1e-6
    fixa = rb.parametros({"snr_min": 20, "snr_max": 20})
    y = rb.ssi(x, np.random.default_rng(0), fixa, SR)
    assert abs(10 * np.log10(np.sum(x ** 2) / np.sum((y - x) ** 2)) - 20) < 1e-6


def test_combinacao_em_serie_na_ordem_dos_autores():
    x = _fala()
    a = rb.aplicar(x, "isd+lnl", np.random.default_rng(3))   # a ordem do nome não importa
    rng = np.random.default_rng(3)
    p = rb.Parametros()
    b = rb.isd(rb.lnl(x.astype(np.float64), rng, p, SR), rng, p).astype(np.float32)
    np.testing.assert_array_equal(a, b)
    assert rb.variante("ssi+isd+lnl") == ("lnl", "isd", "ssi")


@pytest.mark.parametrize("nome", ["mp3", "lnl+lnl", "", "lnl+"])
def test_variante_invalida(nome):
    with pytest.raises(ValueError, match="variante"):
        rb.variante(nome)


def test_parametro_desconhecido_e_recusado():
    with pytest.raises(ValueError, match="desconhecidos"):
        rb.parametros({"snr": 10})


def test_usa_so_o_gerador_da_amostra():
    estado = np.random.get_state()
    x = _fala()
    a = rb.aplicar(x, "lnl+isd+ssi", np.random.default_rng(9))
    b = rb.aplicar(x, "lnl+isd+ssi", np.random.default_rng(9))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, rb.aplicar(x, "lnl+isd+ssi", np.random.default_rng(10)))
    novo = np.random.get_state()
    assert estado[0] == novo[0] and np.array_equal(estado[1], novo[1]) and estado[2:] == novo[2:]


def test_tipo_rawboost_no_gancho():
    x = _fala()
    d = Degradacao({"enabled": True, "etapas": [
        {"tipo": "rawboost", "prob": 1.0, "opcoes": ["lnl+isd", "ssi"],
         "parametros": {"snr_min": 30, "snr_max": 30}}]}, SR)
    y = d(x, np.random.default_rng(0))
    assert y.shape == x.shape and y.dtype == np.float32 and not np.array_equal(x, y)
    assert d.etapas[0].parametros_rawboost.snr_min == 30
    assert {d.etapas[0].parametro(np.random.default_rng(s)) for s in range(20)} == {"lnl+isd", "ssi"}


@pytest.mark.parametrize("etapa, erro", [
    ({"tipo": "rawboost", "faixa": ["lnl", "ssi"]}, "não faixa"),
    ({"tipo": "rawboost", "nivel": "mp3"}, "variante"),
    ({"tipo": "rawboost", "nivel": "lnl", "parametros": {"x": 1}}, "desconhecidos"),
    ({"tipo": "ruido", "nivel": 20, "parametros": {"snr_min": 1}}, "só vale para rawboost"),
])
def test_config_do_rawboost_invalida(etapa, erro):
    with pytest.raises(ValueError, match=erro):
        Degradacao({"enabled": True, "etapas": [etapa]}, SR)
