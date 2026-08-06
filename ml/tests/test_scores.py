"""Testes do reaproveitamento de scores entre evaluate / per_attack / fusão.

Reusar um score errado é pior que recalcular: as métricas do relatório
descreveriam um modelo que não é o avaliado, sem nenhum sinal de erro. Por isso
quase todo teste aqui é sobre a RECUSA do reuso.
"""

import numpy as np
import pytest
import torch

from src.scores import checkpoint_fingerprint, load_scores, save_scores, scores_path

IDS = ["LA_E_0001", "LA_E_0002", "LA_E_0003"]
LABELS = [0, 1, 1]
SCORES = [0.10, 0.90, 0.75]
SYSTEMS = ["-", "A07", "A10"]
FP = "abc123"


def gravar(path, **override):
    dados = {"ids": IDS, "labels": LABELS, "scores": SCORES, "systems": SYSTEMS,
             "fingerprint": FP, "partition": "eval"}
    dados.update(override)
    save_scores(path, **dados)
    return path


def test_roundtrip_preserves_values(tmp_path):
    caminho = gravar(tmp_path / "s.npz")
    (labels, scores, systems), motivo = load_scores(
        caminho, ids=IDS, fingerprint=FP, partition="eval")
    assert motivo == "reaproveitados"
    assert list(labels) == LABELS
    assert list(systems) == SYSTEMS
    np.testing.assert_array_equal(scores, SCORES)


def test_scores_keep_full_precision(tmp_path):
    """O .txt do ASVspoof arredonda em 6 casas; o .npz não pode arredondar.

    Scores muito próximos viram empates quando arredondados, e empates mudam o
    EER — foi por isso que a fusão por posto precisou tratar empates.
    """
    finos = [0.5, 0.5 + 1e-12, 1.0 - 1e-15]
    caminho = gravar(tmp_path / "s.npz", ids=IDS, labels=[0, 1, 1], scores=finos)
    (_, scores, _), _ = load_scores(caminho, ids=IDS, fingerprint=FP, partition="eval")
    assert len(set(scores.tolist())) == 3, "os scores foram arredondados"


def test_refuses_when_checkpoint_changed(tmp_path):
    caminho = gravar(tmp_path / "s.npz")
    reuso, motivo = load_scores(caminho, ids=IDS, fingerprint="outro", partition="eval")
    assert reuso is None
    assert "checkpoint" in motivo


def test_refuses_when_partition_differs(tmp_path):
    caminho = gravar(tmp_path / "s.npz")
    reuso, motivo = load_scores(caminho, ids=IDS, fingerprint=FP, partition="dev")
    assert reuso is None
    assert "partição" in motivo


def test_refuses_when_ids_change(tmp_path):
    caminho = gravar(tmp_path / "s.npz")
    reuso, _ = load_scores(caminho, ids=IDS[:2], fingerprint=FP, partition="eval")
    assert reuso is None
    reuso, _ = load_scores(caminho, ids=["x", "y", "z"], fingerprint=FP, partition="eval")
    assert reuso is None


def test_refuses_when_ids_are_reordered(tmp_path):
    """Mesma lista noutra ordem alinharia score com o áudio errado."""
    caminho = gravar(tmp_path / "s.npz")
    reuso, _ = load_scores(caminho, ids=list(reversed(IDS)), fingerprint=FP,
                           partition="eval")
    assert reuso is None


def test_refuses_when_file_is_missing(tmp_path):
    reuso, motivo = load_scores(tmp_path / "nao_existe.npz", ids=IDS,
                                fingerprint=FP, partition="eval")
    assert reuso is None
    assert "nenhum arquivo" in motivo


def test_refuses_when_file_is_corrupt(tmp_path):
    caminho = tmp_path / "s.npz"
    caminho.write_bytes(b"isso nao e um npz")
    reuso, _ = load_scores(caminho, ids=IDS, fingerprint=FP, partition="eval")
    assert reuso is None


def test_refuses_when_fields_are_missing(tmp_path):
    caminho = tmp_path / "s.npz"
    np.savez(caminho, version=1, ids=np.asarray(IDS))
    reuso, motivo = load_scores(caminho, ids=IDS, fingerprint=FP, partition="eval")
    assert reuso is None
    assert motivo == "formato antigo"


# --------------------------------------------------------------------------- #
# Impressão digital do checkpoint
# --------------------------------------------------------------------------- #
def test_fingerprint_changes_with_weights():
    a = {"w": torch.zeros(4), "b": torch.ones(2)}
    b = {"w": torch.zeros(4), "b": torch.ones(2)}
    assert checkpoint_fingerprint(a) == checkpoint_fingerprint(b)

    b["w"][0] = 1e-7  # uma diferença mínima já é outro modelo
    assert checkpoint_fingerprint(a) != checkpoint_fingerprint(b)


def test_fingerprint_reacts_to_renamed_layers():
    a = {"encoder.0.weight": torch.ones(3)}
    b = {"encoder.1.weight": torch.ones(3)}
    assert checkpoint_fingerprint(a) != checkpoint_fingerprint(b)


def test_fingerprint_ignores_key_order():
    a = {"w": torch.ones(3), "b": torch.zeros(3)}
    b = {"b": torch.zeros(3), "w": torch.ones(3)}
    assert checkpoint_fingerprint(a) == checkpoint_fingerprint(b)


def test_fingerprint_survives_a_save_load_cycle(tmp_path):
    """Salvar e recarregar o mesmo modelo não pode invalidar os scores."""
    original = {"w": torch.randn(8, 4), "b": torch.randn(8)}
    torch.save({"model_state": original}, tmp_path / "ckpt.pt")
    recarregado = torch.load(tmp_path / "ckpt.pt", weights_only=False)["model_state"]
    assert checkpoint_fingerprint(original) == checkpoint_fingerprint(recarregado)


def test_scores_path_follows_the_output_convention():
    caminho = scores_path("outputs", "baseline_lcnn_v4", "eval")
    assert caminho.name == "baseline_lcnn_v4_eval_scores.npz"


@pytest.mark.parametrize("particao", ["train", "dev", "eval"])
def test_each_partition_gets_its_own_file(particao):
    assert particao in scores_path("outputs", "exp", particao).name
