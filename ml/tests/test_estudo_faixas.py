"""Estudo por faixa de frequência (scripts/estudo_faixas.py), com sinais sintéticos."""

import json

import numpy as np
import soundfile as sf
import torch

from scripts.estudo_faixas import FAIXAS, main, resumir, sem_faixa
from src.config import load_config
from src.models import build_model
from src.scores import checkpoint_fingerprint

SR = 16000


def _energia(x, a, b):
    f = np.fft.rfftfreq(len(x), 1 / SR)
    p = np.abs(np.fft.rfft(x)) ** 2
    return p[(f >= a) & (f < b)].sum() / p.sum()


def test_tira_so_a_faixa_pedida():
    x = np.random.default_rng(0).standard_normal(SR * 2).astype(np.float32)
    y = sem_faixa(x, SR, (2000, 3000))
    assert _energia(y, 2200, 2800) < 1e-3 * _energia(x, 2200, 2800)
    assert abs(_energia(y, 500, 1500) / _energia(x, 500, 1500) - 1) < 0.2
    assert _energia(sem_faixa(x, SR, (0, 500)), 0, 400) < 1e-3
    assert _energia(sem_faixa(x, SR, (7000, 8000)), 7200, 8000) < 1e-3


def test_resumo_da_variacao_e_dos_que_passam():
    estudo = {"g": {"inteiro": [-5.0, -10.0], "sem 0-500": [-7.0, -12.0]}}
    t = resumir(estudo, -8.0)
    assert t["g"]["inteiro"] == {"media": -7.5, "acima": 1, "n": 2}
    assert t["g"]["sem 0-500"] == {"variacao": -2.0, "acima": 1, "n": 2}


def test_caminho_completo(tmp_path, capsys):
    config = load_config("configs/baseline.yaml")
    torch.manual_seed(0)
    estado = build_model(config["model"]).state_dict()
    ckpt = tmp_path / "m.pt"
    torch.save({"model_state": estado, "config": config, "threshold": 0.5}, ckpt)
    limiar = tmp_path / "l.json"
    limiar.write_text(json.dumps({"regra": "logodds", "limiar": 0.0, "modelos": [
        {"config": "configs/baseline.yaml", "checkpoint": str(ckpt),
         "impressao": checkpoint_fingerprint(estado)}]}), encoding="utf-8")
    rng = np.random.default_rng(1)
    for sub, taxa in (("real", 48000), ("clone", 24000)):
        (tmp_path / "v" / sub).mkdir(parents=True)
        sf.write(tmp_path / "v" / sub / "f01.wav", (rng.standard_normal(taxa * 5) * 0.1).astype(np.float32), taxa)
    saida = tmp_path / "e.json"
    assert main(["--limiar", str(limiar), "--conjunto", f"L09_x={tmp_path / 'v'}:clone",
                 "--json", str(saida)]) == 0
    r = json.loads(saida.read_text(encoding="utf-8"))
    assert set(r["tabela"]) == {"L09_real", "L09_x"}
    assert len(r["tabela"]["L09_x"]) == 1 + len(FAIXAS)
    assert "sem 7000-8000 Hz" in capsys.readouterr().out
