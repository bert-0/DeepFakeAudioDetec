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
    """Mesma ordenação em escalas diferentes: o 'rank' trata os dois modelos igualmente."""
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
    """Cada modelo acerta metade das amostras; juntos acertam tudo."""
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


# --------------------------------------------------------------------------- #
# Regressão: empates recebem o posto médio. O softmax satura em 1.0, e desempatar
# pela ordem do array inventaria uma ordenação que o modelo não produziu.
# --------------------------------------------------------------------------- #
def test_ranks_give_equal_scores_the_same_rank():
    r = to_ranks(np.array([0.5, 0.5, 0.5, 0.9]))
    assert r[0] == r[1] == r[2], "scores idênticos receberam postos diferentes"
    assert r[3] > r[0]


def test_ranks_preserve_eer_of_original_scores():
    """A regra rank não pode alterar o EER de um único modelo — só reescalar."""
    from src.metrics import compute_eer

    rng = np.random.default_rng(0)
    labels = np.concatenate([np.zeros(200, int), np.ones(200, int)])
    scores = np.concatenate([rng.uniform(0, 0.6, 200), rng.uniform(0.4, 1.0, 200)])
    scores[:150] = 1.0          # empates saturados, como no softmax real
    scores[200:350] = 1.0
    assert abs(compute_eer(labels, to_ranks(scores)) - compute_eer(labels, scores)) < 1e-9


def test_ranks_do_not_launder_nan_into_finite_values():
    """NaN não pode virar posto finito e escapar da proteção de compute_eer."""
    r = to_ranks(np.array([0.1, np.nan, 0.9]))
    assert not np.isfinite(r).all()


def test_rank_rule_uses_logodds_when_given():
    """Com o softmax saturado, só os log-odds preservam a separação do modelo."""
    from src.metrics import compute_eer

    labels = np.array([0] * 20 + [1] * 20)
    probs = [np.ones(40), np.ones(40)]
    logodds = [np.r_[np.full(20, 20.0), np.full(20, 30.0)],
               np.r_[np.full(20, 18.0), np.full(20, 25.0)]]
    assert compute_eer(labels, combine(probs, "rank")) >= 0.4
    assert compute_eer(labels, combine(probs, "rank", logodds)) == 0.0


def test_mean_rule_stays_on_probabilities():
    """`mean` é a conta do monitor ao vivo — continua sobre a probabilidade."""
    probs = [np.array([0.2, 0.4]), np.array([0.6, 0.8])]
    logodds = [np.array([-9.0, 9.0]), np.array([-9.0, 9.0])]
    np.testing.assert_allclose(combine(probs, "mean", logodds), [0.4, 0.6])
