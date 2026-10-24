"""Escala da tela com o limiar no meio (src/limiares.py, `centrado`).

Com a fusão nos log-odds o limiar fica em 0,0117 e a barra de 0 a 1 espremia a
decisão no primeiro 1% (RESUMO_TCC §10.2.12). A escala muda só a tela.
"""

import numpy as np
import pytest

from src.limiares import centrado


@pytest.mark.parametrize("limiar", [0.0117, 0.4351, 0.686, 0.9984])
def test_o_limiar_fica_no_meio_e_a_decisao_nao_muda(limiar):
    assert centrado(limiar, limiar) == pytest.approx(0.5)
    rng = np.random.default_rng(0)
    scores = np.r_[rng.uniform(0, 1, 500), 10 ** rng.uniform(-6, 0, 500)]
    na_tela = [centrado(s, limiar) for s in scores]
    assert all((t >= 0.5) == (s >= limiar) for s, t in zip(scores, na_tela, strict=True))
    ordem = np.argsort(scores)
    assert np.all(np.diff(np.array(na_tela)[ordem]) >= 0), "a ordem dos scores se mantém"


def test_fusao_nos_logodds_espalha_o_que_ficava_colado_no_zero():
    limiar = 0.0117
    humano, falso = 0.0003, 0.05
    assert humano < 0.01 and falso < 0.06, "na escala de 0 a 1, os dois parecem zero"
    assert centrado(humano, limiar) < 0.05 and centrado(falso, limiar) > 0.8


def test_extremos_e_ausencias():
    assert 0.0 <= centrado(0.0, 0.5) < 1e-6 and 1 - 1e-6 < centrado(1.0, 0.5) <= 1.0
    assert centrado(None, 0.5) is None
    assert centrado(0.3, None) == 0.3, "sem limiar, a tela mostra o score como está"


def test_barra_do_monitor_com_o_limiar_no_meio():
    from monitor import barra

    b = barra(centrado(0.0117, 0.0117), 0.5, largura=28)
    assert b.index("|") == 14
