"""Testes do detector de atalhos triviais na base."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_shortcut import eer_trivial  # noqa: E402


def _rotulos_e_valores(sep: float, n: int = 400, seed: int = 0):
    """Duas classes separadas por `sep` desvios-padrão."""
    rng = np.random.default_rng(seed)
    rot = np.array([i % 2 for i in range(n)])
    val = rng.standard_normal(n) + rot * sep
    return rot, val


def test_sem_separacao_fica_perto_de_50():
    rot, val = _rotulos_e_valores(sep=0.0)
    assert 0.40 < eer_trivial(rot, val) <= 0.50


def test_separacao_forte_e_detectada():
    rot, val = _rotulos_e_valores(sep=5.0)
    assert eer_trivial(rot, val) < 0.05


def test_detecta_atalho_invertido():
    """Um atalho pode correlacionar em qualquer sentido — spoof mais longo OU
    mais curto. Testar só um sentido deixaria metade dos casos passar."""
    rot, val = _rotulos_e_valores(sep=5.0)
    direto = eer_trivial(rot, val)
    invertido = eer_trivial(rot, -val)
    assert direto == pytest.approx(invertido), "os dois sentidos têm de dar o mesmo"
    assert invertido < 0.05


def test_nunca_passa_de_50_por_cento():
    """Por construção pega o melhor dos dois sentidos, então o teto é 50%."""
    for seed in range(8):
        rot, val = _rotulos_e_valores(sep=0.0, seed=seed)
        assert eer_trivial(rot, val) <= 0.50


# --------------------------------------------------------------------------- #
# Teste de permutação
#
# Regressão: a primeira versão comparava o EER trivial com um limiar fixo de
# 40%, escolhido a olho. Com classes desbalanceadas (~9 spoof por bonafide no
# LA) o acaso não produz 50%, e sim ~48,9% — então 43,8% passava como "OK"
# sendo, na verdade, sinal estatisticamente real.
# --------------------------------------------------------------------------- #
def _desbalanceado(n_bona=298, n_spoof=2702, sep=0.0, seed=7):
    rot = np.array([0] * n_bona + [1] * n_spoof)
    rng = np.random.default_rng(seed)
    val = np.where(rot == 0, rng.normal(0, 1, len(rot)), rng.normal(sep, 1, len(rot)))
    return rot, val


def test_o_acaso_nao_produz_50_por_cento_com_classes_desbalanceadas():
    from scripts.check_shortcut import teste_permutacao

    rot, val = _desbalanceado(sep=0.0)
    nulo, p5 = teste_permutacao(rot, val, n=200)
    assert nulo < 0.50, "o acaso fica abaixo de 50% quando as classes diferem em tamanho"
    assert p5 < nulo


def test_efeito_fraco_e_detectado_contra_o_nulo():
    """O caso real: bonafide 2,0s vs spoof 2,5s, desvio ~1s."""
    from scripts.check_shortcut import teste_permutacao

    rot, val = _desbalanceado(sep=0.45)
    _, p5 = teste_permutacao(rot, val, n=200)
    assert eer_trivial(rot, val) < p5, "um efeito real precisa ficar abaixo do p5"


def test_sem_efeito_nao_dispara():
    from scripts.check_shortcut import teste_permutacao

    rot, val = _desbalanceado(sep=0.0)
    _, p5 = teste_permutacao(rot, val, n=200)
    assert eer_trivial(rot, val) >= p5, "sem efeito não pode ser marcado como sinal"


def test_permutacao_e_reprodutivel():
    from scripts.check_shortcut import teste_permutacao

    rot, val = _desbalanceado(sep=0.3)
    assert teste_permutacao(rot, val, n=100, seed=1) == \
           teste_permutacao(rot, val, n=100, seed=1)
