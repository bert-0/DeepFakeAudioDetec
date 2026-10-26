"""Pente na taxa de quadros (scripts/estudo_pente.py), com sinais sintéticos."""

import json

import numpy as np
import soundfile as sf

from scripts.estudo_linhas import excesso
from scripts.estudo_pente import auc, forca, main, varrido


def _som(sr, quadro=None, seed=0, segundos=4.0):
    """Ruído e, se pedido, um pulso fraco a cada `quadro` amostras (o relógio do vocoder)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(sr * segundos)) * 0.05
    if quadro:
        x[::quadro] += 0.3
    return x


def test_pulso_a_cada_480_amostras_vira_pente_de_50_hz():
    ex = excesso(_som(24000, 480), 24000)
    assert forca(ex, 50.0) > 3
    assert abs(forca(ex, 93.75)) < 1
    base, f = varrido(ex)
    assert base in (50.0, 100.0, 150.0, 200.0) and f > 3


def test_varredura_rapida_faz_a_mesma_conta():
    ex = excesso(_som(24000, 480, seed=5), 24000)
    base, f = varrido(ex)
    assert abs(f - forca(ex, base)) < 1e-9
    base_m, f_m = varrido(ex, (4000.0, 7600.0))
    assert abs(f_m - forca(ex, base_m, (4000.0, 7600.0))) < 1e-9


def test_sem_relogio_sem_pente():
    ex = excesso(_som(48000), 48000)
    assert abs(forca(ex, 50.0)) < 1 and abs(forca(ex, 93.75)) < 1


def test_auc():
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1], [1]) == 0.5
    assert auc([0], [1, 2]) == 0.0


def test_caminho_completo(tmp_path, capsys):
    for nome, sr, quadro in (("real", 48000, None), ("clone", 24000, 480)):
        (tmp_path / nome).mkdir()
        for s in range(2):
            sf.write(tmp_path / nome / f"f{s}.wav", _som(sr, quadro, seed=s).astype(np.float32), sr)
    saida = tmp_path / "p.json"
    assert main(["--grupo", f"real={tmp_path / 'real'}", "--grupo", f"clone={tmp_path / 'clone'}",
                 "--real", "real", "--base", "cb=50", "--comparar", "clone:real",
                 "--json", str(saida)]) == 0
    r = json.loads(saida.read_text(encoding="utf-8"))["resumo"]
    assert r["nativo"]["clone"]["cb"]["auc"] == 1.0
    assert r["modelo"]["clone"]["cb"]["auc"] == 1.0
    assert r["nativo"]["comparacoes"]["clone:real"]["cb"] == 1.0
    assert "Caminho modelo" in capsys.readouterr().out
