"""Testes de utilidades do script de treino."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import archive_previous_checkpoints, class_weights_from  # noqa: E402


def test_archive_renames_existing_checkpoint(tmp_path):
    """Um treino novo não pode destruir o melhor modelo do treino anterior."""
    ckpt = tmp_path / "exp.pt"
    ckpt.write_text("modelo-bom")

    archive_previous_checkpoints(ckpt)

    assert not ckpt.exists()
    assert (tmp_path / "exp_prev.pt").read_text() == "modelo-bom"


def test_archive_is_noop_when_missing(tmp_path):
    archive_previous_checkpoints(tmp_path / "inexistente.pt")  # não deve levantar


def test_archive_overwrites_older_backup(tmp_path):
    """O backup guarda a execução imediatamente anterior, não acumula arquivos."""
    ckpt = tmp_path / "exp.pt"
    (tmp_path / "exp_prev.pt").write_text("antigo")
    ckpt.write_text("recente")

    archive_previous_checkpoints(ckpt)

    assert (tmp_path / "exp_prev.pt").read_text() == "recente"


def test_archive_handles_multiple_paths(tmp_path):
    best = tmp_path / "exp.pt"
    last = tmp_path / "exp_last.pt"
    best.write_text("b")
    last.write_text("l")

    archive_previous_checkpoints(best, last)

    assert (tmp_path / "exp_prev.pt").read_text() == "b"
    assert (tmp_path / "exp_last_prev.pt").read_text() == "l"


def test_class_weights_auto_is_inverse_frequency():
    import torch

    labels = [0] * 10 + [1] * 30  # 25% bonafide, 75% spoof
    w = class_weights_from(labels, 2, torch.device("cpu"), mode="auto")
    assert abs(float(w[0]) - 2.0) < 1e-6      # 40 / (2*10)
    assert abs(float(w[1]) - 0.6667) < 1e-3   # 40 / (2*30)


def test_class_weights_sqrt_is_milder():
    import math

    import torch

    labels = [0] * 10 + [1] * 30
    auto = class_weights_from(labels, 2, torch.device("cpu"), mode="auto")
    sqrt = class_weights_from(labels, 2, torch.device("cpu"), mode="sqrt")
    assert abs(float(sqrt[0]) - math.sqrt(float(auto[0]))) < 1e-5
    # A razão entre as classes fica menor (compensação mais suave).
    assert float(sqrt[0]) / float(sqrt[1]) < float(auto[0]) / float(auto[1])
