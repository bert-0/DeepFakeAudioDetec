"""Linhas fixas no espectro (scripts/estudo_linhas.py), com sinais sintéticos."""

import json

import numpy as np
import soundfile as sf

from scripts.estudo_linhas import (escolher_linhas, espectros_no_modelo, excesso, main,
                                   no_modelo, presenca)


def _som(sr, tom_hz=None, seed=0, segundos=4.0):
    """Ruído colorido (como fala no espectro médio) e, se pedido, um tom fixo fraco."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * segundos)) / sr
    x = np.convolve(rng.standard_normal(t.size), [1.0, 0.6, 0.2], "same") * 0.05
    if tom_hz:
        x += 0.002 * np.sin(2 * np.pi * tom_hz * t + seed)
    return x


def test_tom_fixo_vira_linha_e_ruido_nao():
    ex = excesso(_som(24000, 7850), 24000)
    assert ex[7850] > 10
    assert np.nanmax(excesso(_som(24000), 24000)[300:]) < 10


def test_acima_da_metade_da_taxa_fica_vazio():
    ex = excesso(_som(16000), 16000)
    assert np.isnan(ex[9000]) and not np.isnan(ex[7000])


def test_escolhe_e_conta_sem_confundir_os_grupos():
    clones = np.array([excesso(_som(24000, 6077, seed=s), 24000) for s in range(3)])
    reais = np.array([excesso(_som(48000, seed=10 + s), 48000) for s in range(3)])
    linhas = escolher_linhas(clones, n=1)
    assert abs(linhas[0][0] - 6077) <= 1
    assert presenca(clones, linhas[0][0]) == 3
    assert presenca(reais, linhas[0][0]) == 0


def test_janela_de_25_ms_apaga_a_linha(tmp_path):
    """A linha passa pelo caminho do modelo, mas a janela do detector a dilui."""
    for s in range(3):
        sf.write(tmp_path / f"f{s}.wav", _som(24000, 6078, seed=s).astype(np.float32), 24000)
    r = no_modelo(espectros_no_modelo(sorted(tmp_path.glob("*.wav"))), 6078)
    assert r["1hz"] > 6
    assert r["25ms"] < 1
    assert abs(r["controle_25ms"]) < 1 and r["controle_1hz"] < 4


def test_caminho_completo(tmp_path, capsys):
    for nome, sr, tom in (("real", 48000, None), ("clone", 24000, 9468)):
        (tmp_path / nome).mkdir()
        for s in range(2):
            sf.write(tmp_path / nome / f"f0{s}.wav", _som(sr, tom, seed=s).astype(np.float32), sr)
    saida = tmp_path / "l.json"
    assert main(["--grupo", f"real={tmp_path / 'real'}", "--grupo", f"clone={tmp_path / 'clone'}",
                 "--escolher", "clone", "--n", "2", "--json", str(saida)]) == 0
    r = json.loads(saida.read_text(encoding="utf-8"))
    primeira = r["clone"][0]
    assert abs(primeira["hz"] - 9468) <= 1
    assert primeira["presenca"] == {"real": [0, 2], "clone": [2, 2]}
    assert "modelo" not in primeira  # acima de 8 kHz não chega ao modelo
    assert "Linhas escolhidas em clone" in capsys.readouterr().out
