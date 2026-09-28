"""Testes do diagnóstico de saturação dos scores (scripts/checar_saturacao.py)."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.checar_saturacao import contar, main  # noqa: E402


def test_softmax_float32_satura_em_exatamente_1():
    """O fato que motiva o script, medido: margem de logit 17 já dá 1.0 exato."""
    p16 = torch.softmax(torch.tensor([[0.0, 16.0]]), dim=1)[0, 1].item()
    p17 = torch.softmax(torch.tensor([[0.0, 17.0]]), dim=1)[0, 1].item()
    assert p16 < 1.0
    assert p17 == 1.0


def test_saturacao_destroi_a_ordem_e_o_eer():
    """Dois conjuntos perfeitamente separados pelo logit viram EER ~50% no softmax.

    Bonafide com margem 20 e spoof com margem 30: o modelo separa as classes
    sem erro nenhum, mas em float32 todos viram 1.0 e o EER passa a medir o
    arredondamento.
    """
    from src.metrics import compute_eer

    logits_b = torch.tensor([[0.0, 20.0]] * 50)
    logits_s = torch.tensor([[0.0, 30.0]] * 50)
    probs = torch.softmax(torch.cat([logits_b, logits_s]), dim=1)[:, 1].numpy()
    labels = np.array([0] * 50 + [1] * 50)

    assert (probs == 1.0).all(), "todos saturados"
    log_odds = np.array([20.0] * 50 + [30.0] * 50)
    assert compute_eer(labels, log_odds) == 0.0, "pelo logit, separação perfeita"
    assert compute_eer(labels, probs) >= 0.4, "pelo softmax, o EER mede o empate"


def test_contagem_por_classe():
    scores = np.array([1.0, 1.0, 0.5, 1.0, 0.2, 0.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    c = contar(scores, labels)
    assert c["bonafide"] == (2, 0, 3)
    assert c["spoof"] == (1, 1, 3)


def test_cli_le_os_arquivos(tmp_path, capsys):
    arq = tmp_path / "x_eval_scores.npz"
    np.savez(arq, scores=np.array([1.0, 0.3, 1.0, 1.0]), labels=np.array([0, 0, 1, 1]))
    assert main([str(arq)]) == 0
    saida = capsys.readouterr().out
    assert "1/2" in saida and "2/2" in saida
