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
