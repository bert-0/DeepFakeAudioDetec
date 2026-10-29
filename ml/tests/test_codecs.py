"""Codecs do aumento de dados (src/preprocess/codecs.py) e o tipo `codec` do gancho.

Sinal sintético com energia até 8 kHz, para ver os codecs de telefonia
cortarem a banda alta e os de banda larga não.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import codecs  # noqa: E402
from src.preprocess.degradacao import Degradacao  # noqa: E402

SR = 16000


def _sinal(segundos=2.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(segundos * SR)) / SR
    tons = sum(np.sin(2 * np.pi * f * t) for f in (300, 1200, 3000, 5500, 7000))
    return (0.05 * tons + 0.02 * rng.standard_normal(t.size)).astype(np.float32)


def _alta(x):
    f, p = welch(x, SR, nperseg=1024)
    return p[f > 4200].sum() / p.sum()


@pytest.mark.parametrize("nome", sorted(codecs.CODECS))
def test_ida_e_volta_mantem_comprimento_e_altera_o_sinal(nome):
    x = _sinal()
    y = codecs.codificar(x, SR, nome)
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all()
    assert not np.array_equal(x, y)
    # alinhado: o libsndfile compensa o atraso do codificador
    from scipy.signal import correlate

    atraso = int(np.argmax(correlate(y, x, mode="full", method="fft"))) - (len(x) - 1)
    assert abs(atraso) <= 1


@pytest.mark.parametrize("nome", ["ulaw", "alaw", "gsm"])
def test_telefonia_corta_acima_de_4_khz(nome):
    x = _sinal()
    assert _alta(x) > 0.2
    assert _alta(codecs.codificar(x, SR, nome)) < 1e-3


@pytest.mark.parametrize("nome", ["mp3", "vorbis", "opus"])
def test_banda_larga_mantem_parte_da_banda_alta(nome):
    assert _alta(codecs.codificar(_sinal(), SR, nome, codecs.faixa_padrao(nome)[0])) > 0.05


def test_mais_compressao_mais_erro_no_opus():
    x = _sinal(seed=1)
    erro = [np.linalg.norm(codecs.codificar(x, SR, "opus", n) - x) for n in (0.9, 0.99)]
    assert erro[1] > erro[0]


def test_entrada_acima_de_1_nao_quebra():
    x = 1.5 * _sinal() / np.abs(_sinal()).max()
    for nome in codecs.CODECS:
        assert np.abs(codecs.codificar(x, SR, nome)).max() <= 1.5


def test_codec_desconhecido():
    with pytest.raises(ValueError, match="desconhecido"):
        codecs.codificar(_sinal(), SR, "amr")


def test_tipo_codec_no_gancho_sorteia_nome_e_nivel():
    d = Degradacao({"enabled": True, "etapas": [
        {"tipo": "codec", "prob": 1.0, "opcoes": ["mp3", "gsm"], "qualidade": {"mp3": [0.5, 0.6]}}]}, SR)
    etapa = d.etapas[0]
    assert etapa.faixas_codec == {"mp3": (0.5, 0.6), "gsm": None}
    assert {etapa.parametro(np.random.default_rng(s)) for s in range(20)} == {"mp3", "gsm"}
    x = _sinal()
    y = d(x, np.random.default_rng(0))
    assert y.shape == x.shape and not np.array_equal(x, y)
    np.testing.assert_array_equal(y, d(x, np.random.default_rng(0)))


@pytest.mark.parametrize("etapa, erro", [
    ({"tipo": "codec", "nivel": "amr"}, "desconhecido"),
    ({"tipo": "codec", "faixa": ["mp3", "gsm"]}, "não faixa"),
    ({"tipo": "codec", "nivel": "mp3", "qualidade": {"opus": [0.9, 0.95]}}, "não está nas opções"),
    ({"tipo": "codec", "nivel": "gsm", "qualidade": {"gsm": [0.1, 0.2]}}, "não tem nível"),
    ({"tipo": "codec", "nivel": "mp3", "qualidade": {"mp3": [0.5, 1.0]}}, r"\[0, 0.99\]"),
    ({"tipo": "ruido", "nivel": 20, "qualidade": {"mp3": [0.1, 0.2]}}, "só vale para codec"),
])
def test_config_de_codec_invalida(etapa, erro):
    with pytest.raises(ValueError, match=erro):
        Degradacao({"enabled": True, "etapas": [etapa]}, SR)
