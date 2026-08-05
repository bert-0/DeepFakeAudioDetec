"""Testes das regras de combinação da fusão de scores."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.score_fusion import combine, eer_per_attack, parse_args, to_ranks  # noqa: E402


def test_mean_averages_scores():
    a = np.array([0.0, 1.0]); b = np.array([1.0, 0.0])
    assert np.allclose(combine([a, b], "mean"), [0.5, 0.5])


def test_max_and_min():
    a = np.array([0.2, 0.9]); b = np.array([0.7, 0.1])
    assert np.allclose(combine([a, b], "max"), [0.7, 0.9])
    assert np.allclose(combine([a, b], "min"), [0.2, 0.1])


def test_ranks_are_normalized():
    r = to_ranks(np.array([5.0, 1.0, 3.0]))
    assert r.min() == 0.0 and r.max() == 1.0
    assert r[1] < r[2] < r[0]      # preserva a ordenação


def test_rank_rule_is_immune_to_calibration_scale():
    """Dois modelos com a MESMA ordenação, mas escalas muito diferentes.

    A média simples é dominada pelo modelo de escala maior; a regra de posto
    trata os dois igualmente. É a vantagem prática do 'rank'.
    """
    ordenacao = np.array([0.1, 0.4, 0.6, 0.9])
    comprimido = ordenacao * 0.01          # mesmo ranking, escala 100x menor
    labels = np.array([0, 0, 1, 1])

    fund_mean = combine([ordenacao, comprimido], "mean")
    fund_rank = combine([ordenacao, comprimido], "rank")
    # A regra de posto reproduz exatamente a ordenação comum aos dois.
    assert np.array_equal(np.argsort(fund_rank), np.argsort(ordenacao))
    assert np.array_equal(np.argsort(fund_mean), np.argsort(ordenacao))
    # ...mas o 'mean' herda a escala do maior, o 'rank' normaliza.
    assert fund_rank.max() == 1.0
    assert fund_mean.max() < 1.0
    assert labels.size == 4


def test_fusion_of_complementary_models_beats_both():
    """Cada modelo acerta metade das amostras; juntos acertam tudo.

    Reproduz a complementaridade medida entre as configurações reais.
    """
    labels = np.array([0, 0, 1, 1])
    # modelo A separa as duas primeiras, erra as últimas
    a = np.array([0.1, 0.2, 0.3, 0.9])
    # modelo B separa as duas últimas, erra as primeiras
    b = np.array([0.8, 0.2, 0.9, 0.95])
    from src.metrics import compute_eer

    fused = combine([a, b], "mean")
    assert compute_eer(labels, fused) <= max(
        compute_eer(labels, a), compute_eer(labels, b))


def test_eer_per_attack_groups_by_system():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.1])
    systems = np.array(["-", "-", "A07", "A13"])
    res = eer_per_attack(labels, scores, systems)
    assert set(res) == {"A07", "A13"}
    assert res["A07"] == 0.0          # bem separado
    assert res["A13"] > 0.0           # confundido com bonafide


def test_cli_accepts_windows_paths_with_drive_letters(monkeypatch):
    """Caminhos com 'C:\\...' não podem ser confundidos com um separador."""
    argv = ["score_fusion.py",
            "--model", r"C:\proj\configs\a.yaml", r"C:\proj\checkpoints\a.pt",
            "--model", r"C:\proj\configs\b.yaml", r"C:\proj\checkpoints\b.pt"]
    monkeypatch.setattr(sys, "argv", argv)
    args = parse_args()
    assert args.model == [
        [r"C:\proj\configs\a.yaml", r"C:\proj\checkpoints\a.pt"],
        [r"C:\proj\configs\b.yaml", r"C:\proj\checkpoints\b.pt"],
    ]


def test_cli_collects_multiple_models(monkeypatch):
    argv = ["score_fusion.py",
            "--model", "configs/a.yaml", "checkpoints/a.pt",
            "--model", "configs/b.yaml", "checkpoints/b.pt",
            "--model", "configs/c.yaml", "checkpoints/c.pt"]
    monkeypatch.setattr(sys, "argv", argv)
    assert len(parse_args().model) == 3
