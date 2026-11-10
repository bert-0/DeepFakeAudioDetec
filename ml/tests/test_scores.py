"""Reaproveitamento de scores entre evaluate, per_attack e fusão.

Reusar um score errado descreveria outro modelo sem nenhum aviso; por isso quase
todo teste aqui é sobre a recusa do reuso.
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
    (labels, scores, systems, _), motivo = load_scores(
        caminho, ids=IDS, fingerprint=FP, partition="eval")
    assert motivo == "reaproveitados"
    assert list(labels) == LABELS
    assert list(systems) == SYSTEMS
    np.testing.assert_array_equal(scores, SCORES)


def test_scores_keep_full_precision(tmp_path):
    """O .npz não arredonda como o .txt do ASVspoof (6 casas): empates mudariam o EER."""
    finos = [0.5, 0.5 + 1e-12, 1.0 - 1e-15]
    caminho = gravar(tmp_path / "s.npz", ids=IDS, labels=[0, 1, 1], scores=finos)
    (_, scores, _, _), _ = load_scores(caminho, ids=IDS, fingerprint=FP, partition="eval")
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
# Log-odds: o EER não pode depender do arredondamento do softmax
# --------------------------------------------------------------------------- #

def test_logodds_are_saved_and_returned(tmp_path):
    lo = [-3.0, 25.0, 40.0]
    caminho = gravar(tmp_path / "s.npz", logodds=lo)
    (_, _, _, logodds), motivo = load_scores(caminho, ids=IDS, fingerprint=FP,
                                             partition="eval")
    assert motivo == "reaproveitados"
    np.testing.assert_array_equal(logodds, lo)


def test_old_file_without_saturated_bonafide_is_still_reused(tmp_path):
    """Arquivos de 2019, sem log-odds e sem bonafide em 1,0, continuam valendo.

    O log-odds derivado ordena como a probabilidade, então o EER não muda.
    """
    probs = [0.10, 1.0, 0.75]
    caminho = gravar(tmp_path / "s.npz", scores=probs)
    (_, _, _, logodds), motivo = load_scores(caminho, ids=IDS, fingerprint=FP,
                                             partition="eval")
    assert motivo == "reaproveitados"
    assert np.isfinite(logodds).all()
    assert list(np.argsort(logodds)) == list(np.argsort(probs))


def test_old_file_with_saturated_bonafide_is_refused(tmp_path):
    """Bonafide em 1,0 sem log-odds: o EER mediria o empate. Recalcula."""
    caminho = gravar(tmp_path / "s.npz", scores=[1.0, 1.0, 0.75])
    reuso, motivo = load_scores(caminho, ids=IDS, fingerprint=FP, partition="eval")
    assert reuso is None
    assert "log-odds" in motivo


def test_saturated_file_with_logodds_is_accepted(tmp_path):
    caminho = gravar(tmp_path / "s.npz", scores=[1.0, 1.0, 1.0],
                     logodds=[20.0, 30.0, 31.0])
    reuso, _ = load_scores(caminho, ids=IDS, fingerprint=FP, partition="eval")
    assert reuso is not None


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
