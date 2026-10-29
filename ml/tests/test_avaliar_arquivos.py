"""Tabela de scores por arquivo (scripts/avaliar_arquivos.py)."""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.avaliar_arquivos import frase_e_versao, main  # noqa: E402
from src.config import load_config  # noqa: E402
from src.models import build_model  # noqa: E402


def test_frase_e_versao():
    assert frase_e_versao("frase1_griffinlim") == ("frase1", "griffinlim")
    assert frase_e_versao("frase_2_original") == ("frase_2", "original")
    assert frase_e_versao("outro") == ("outro", None)


def test_compara_versoes_da_mesma_frase(tmp_path, capsys):
    config = load_config("configs/baseline.yaml")
    torch.manual_seed(0)
    ckpt = tmp_path / "m.pt"
    torch.save({"model_state": build_model(config["model"]).state_dict(),
                "config": config, "threshold": 0.5}, ckpt)
    pasta = tmp_path / "demo"
    pasta.mkdir()
    rng = np.random.default_rng(0)
    for v in ("original", "griffinlim", "world"):
        sf.write(pasta / f"frase1_{v}.wav", (rng.standard_normal(48000 * 3) * 0.1).astype(np.float32), 48000)
    sf.write(pasta / "pareado.wav", np.zeros(48000, np.float32), 48000)

    assert main([str(pasta), "--config", "configs/baseline.yaml", "--checkpoint", str(ckpt)]) == 0
    saida = capsys.readouterr().out
    assert "pareado" not in saida, "a playlist mistura tudo; fica de fora"
    assert "frase1_world" in saida and "de 2." in saida
