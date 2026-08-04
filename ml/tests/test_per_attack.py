"""Testes da análise de EER por tipo de ataque."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.per_attack_eval import eer_per_attack  # noqa: E402


def test_isolates_hard_attack_hidden_by_global_average():
    """O EER global esconde um ataque ruim; a análise por ataque precisa expô-lo."""
    # 4 bonafide (score baixo) + A07 fácil (score alto) + A17 difícil (score baixo)
    labels = np.array([0, 0, 0, 0] + [1, 1, 1, 1] + [1, 1, 1, 1])
    systems = np.array(["-"] * 4 + ["A07"] * 4 + ["A17"] * 4)
    scores = np.array([0.1, 0.1, 0.2, 0.2] + [0.9, 0.9, 0.8, 0.8] + [0.1, 0.1, 0.1, 0.1])

    res = eer_per_attack(labels, scores, systems)

    assert set(res) == {"A07", "A17"}
    assert res["A07"]["eer"] == 0.0        # perfeitamente separável
    assert res["A17"]["eer"] > 0.4         # indistinguível dos bonafide
    assert res["A07"]["n_spoof"] == 4
    assert res["A17"]["n_spoof"] == 4


def test_each_attack_is_scored_against_all_bonafide():
    """Cada ataque usa TODOS os bonafide — protocolo padrão da ASVspoof."""
    labels = np.array([0, 0, 0, 0, 0, 0] + [1, 1])
    systems = np.array(["-"] * 6 + ["A07", "A08"])
    scores = np.array([0.1] * 6 + [0.9, 0.9])

    res = eer_per_attack(labels, scores, systems)

    assert res["A07"]["n_spoof"] == 1
    assert res["A08"]["n_spoof"] == 1
    for r in res.values():
        assert r["eer"] == 0.0


def test_ignores_bonafide_as_attack_category():
    labels = np.array([0, 0, 1, 1])
    systems = np.array(["-", "-", "A07", "A07"])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    assert list(eer_per_attack(labels, scores, systems)) == ["A07"]
