"""Amostragem estratificada da avaliação de robustez (--amostra)."""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.robustness_eval import indices_estratificados  # noqa: E402

LABELS = [0] * 700 + [1] * 290 + [1] * 10
SYSTEMS = ["-"] * 700 + ["A07"] * 290 + ["A19"] * 10


def test_mantem_as_proporcoes_e_o_ataque_raro():
    idx = indices_estratificados(LABELS, SYSTEMS, 100, seed=42)
    grupos = Counter(SYSTEMS[i] for i in idx)
    assert grupos["-"] == 70
    assert grupos["A07"] == 29
    assert grupos["A19"] == 1, "o ataque raro não pode sumir da amostra"


def test_mesma_semente_mesma_amostra():
    """A amostra se repete entre condições — senão cada uma mediria outro conjunto."""
    a = indices_estratificados(LABELS, SYSTEMS, 100, seed=42)
    b = indices_estratificados(LABELS, SYSTEMS, 100, seed=42)
    assert a == b
    assert len(set(a)) == len(a)


def test_amostra_maior_que_o_conjunto_devolve_tudo():
    assert indices_estratificados(LABELS, SYSTEMS, 5000, seed=0) == list(range(len(LABELS)))
