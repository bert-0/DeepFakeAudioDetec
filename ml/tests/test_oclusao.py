"""Oclusão da banda alta (src/preprocess/channel.py, condições oclusao_* do robustness_eval).

O corte zera o espectro sem transição: acima dele não sobra nada, abaixo dele o
sinal fica igual.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess.channel import ChannelDegradation, ocluir_banda_alta  # noqa: E402

SR = 16000


def _sinal():
    t = np.arange(4 * SR) / SR
    return sum(np.sin(2 * np.pi * f * t) for f in (500, 3000, 7000, 7800)).astype(np.float32) * 0.1


@pytest.mark.parametrize("corte", [7600, 7000, 6000, 4000])
def test_zera_acima_do_corte_e_mantem_abaixo(corte):
    x = _sinal()
    y = ocluir_banda_alta(x, SR, corte)
    assert y.shape == x.shape and y.dtype == np.float32
    f = np.fft.rfftfreq(x.size, 1 / SR)
    X, Y = np.fft.rfft(x), np.fft.rfft(y)
    assert np.abs(Y[f >= corte]).max() < 1e-3 * np.abs(X).max()
    np.testing.assert_allclose(np.abs(Y[f < corte - 50]), np.abs(X[f < corte - 50]), atol=1e-2)


def test_7600_tira_so_o_tom_de_7800():
    x = _sinal()
    y = ChannelDegradation("oclusao", 7600, SR)(x)
    f = np.fft.rfftfreq(x.size, 1 / SR)
    Y = np.abs(np.fft.rfft(y))
    pico = lambda hz: Y[np.argmin(np.abs(f - hz))]  # noqa: E731
    assert pico(7000) > 100 * pico(7800)


def test_condicoes_no_robustness_eval():
    from scripts.robustness_eval import CHANNEL_CONDITIONS, construir_canal

    rotulos = [r for r, _ in CHANNEL_CONDITIONS]
    for corte in (7600, 7000, 6000, 4000):
        assert f"oclusao_{corte}" in rotulos
    # as condições de antes continuam lá, na mesma ordem
    assert rotulos[:7] == ["opus_30kbps", "opus_25kbps", "opus_15kbps", "banda_estreita",
                           "banda_estreita_opus", "captura_48k", "captura_48k_fir"]
    etapas = dict(CHANNEL_CONDITIONS)["oclusao_4000"]
    assert construir_canal(etapas, SR)(_sinal()).shape == (4 * SR,)
