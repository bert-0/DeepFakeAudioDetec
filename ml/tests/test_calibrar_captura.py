"""Recalibração do limiar para a captura ao vivo (scripts/calibrar_captura.py)."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.calibrar_captura import calibrar, caminho_saida, main  # noqa: E402


def _deslocado(rng, n=2000):
    """O que a captura faz: a ordenação sobrevive, tudo sobe."""
    labels = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
    lo = np.r_[rng.normal(-1, 1, n // 2), rng.normal(1, 1, n // 2)] + 2.5
    return labels, 1 / (1 + np.exp(-lo)), lo


def test_limiar_novo_devolve_o_equilibrio():
    labels, probs, lo = _deslocado(np.random.default_rng(0))
    r = calibrar(labels, probs, lo, limiar_original=0.65)
    fa_antes, _ = r["antes"]
    fa, fr = r["depois"]
    assert fa_antes > 0.6, "no limiar antigo, a maioria dos humanos passa"
    assert abs(fa - fr) < 0.03, "no limiar novo, os dois erros se igualam (EER)"
    assert r["limiar"] > 0.65 and r["serve"]


def test_recusa_quando_a_probabilidade_satura():
    """O fusion_v4 na captura: todo bonafide em 1,0 — não há limiar."""
    labels = np.r_[np.zeros(100, int), np.ones(100, int)]
    probs = np.ones(200)
    lo = np.r_[np.full(100, 20.0), np.full(100, 30.0)]
    r = calibrar(labels, probs, lo, limiar_original=0.7)
    assert not r["serve"]
    assert r["eer_logodds"] == 0.0, "a ordenação existe; a probabilidade é que não serve"


def test_saida_nao_sobrescreve_o_original():
    assert caminho_saida("checkpoints/baseline_lfcc_cnn_v2.pt").name == \
        "baseline_lfcc_cnn_v2_captura.pt"


def test_fluxo_smoke_grava_copia_com_metadados(tmp_path):
    """Ponta a ponta com o dataset sintético: mesmos pesos, limiar novo."""
    from src.config import load_config
    from src.models import build_model

    config = load_config("configs/baseline.yaml")
    original = tmp_path / "m.pt"
    modelo = build_model(config["model"])
    torch.save({"model_state": modelo.state_dict(), "config": config,
                "threshold": 0.5}, original)
    saida = tmp_path / "m_captura.pt"
    codigo = main(["--config", "configs/baseline.yaml", "--checkpoint", str(original),
                   "--smoke", "--device", "cpu", "--saida", str(saida)])
    if codigo == 2:        # modelo aleatório pode saturar; a recusa também é válida
        assert not saida.exists()
        return
    assert codigo == 0
    novo = torch.load(saida, weights_only=False)
    orig = torch.load(original, weights_only=False)
    assert novo["calibracao"]["particao"] == "dev"
    assert novo["calibracao"]["threshold_original"] == 0.5
    for k in orig["model_state"]:
        assert torch.equal(novo["model_state"][k], orig["model_state"][k])
    assert orig["threshold"] == 0.5 and "calibracao" not in orig
