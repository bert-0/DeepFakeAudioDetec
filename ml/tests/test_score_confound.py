"""Testes do detector de dependência do score em grandezas triviais."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_score_confound import classificar, piso_de_ruido  # noqa: E402


def test_piso_cai_com_amostras_maiores():
    """O limiar certo depende do tamanho da amostra, não de um número fixo."""
    assert piso_de_ruido(100) > piso_de_ruido(1000) > piso_de_ruido(10000)


def test_piso_e_cerca_de_dois_erros_padrao():
    assert piso_de_ruido(401) == pytest.approx(0.10, abs=0.01)


def test_piso_nao_estoura_com_amostra_minima():
    assert 0 < piso_de_ruido(1) <= 2.0
    assert 0 < piso_de_ruido(2) <= 2.0


# --------------------------------------------------------------------------- #
# Regressão: a primeira versão usava um limiar fixo de 0,10. Com 300 pontos por
# classe o acaso já produz |rho| ~0,116, então um modelo que ignora o nível era
# marcado como dependente. O piso agora acompanha o tamanho da amostra.
# --------------------------------------------------------------------------- #
def test_correlacao_no_ruido_e_desprezivel_em_amostra_pequena():
    piso = piso_de_ruido(300)
    assert classificar(0.112, piso) == "desprezível"


def test_a_mesma_correlacao_vira_sinal_em_amostra_grande():
    piso = piso_de_ruido(5000)
    assert classificar(0.112, piso) == "acima do ruído"


def test_correlacao_forte_sempre_depende():
    for n in (100, 1000, 100000):
        assert classificar(0.95, piso_de_ruido(n)) == "DEPENDE"


def test_sinal_e_irrelevante_para_o_veredito():
    """Depender negativamente do nível também é depender."""
    piso = piso_de_ruido(1000)
    assert classificar(-0.85, piso) == classificar(0.85, piso) == "DEPENDE"
