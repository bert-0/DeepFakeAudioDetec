"""Ensaios com o canal igualado (scripts/ensaio_canal_igualado.py)."""

import numpy as np
import soundfile as sf

from scripts.ensaio_canal_igualado import clones_com_canal, montar_origem


def test_clone_recebe_o_ruido_na_snr_pedida(tmp_path):
    t = np.arange(24000 * 2) / 24000
    clone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    (tmp_path / "c").mkdir()
    sf.write(tmp_path / "c" / "f01.wav", clone, 24000)
    ruido = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.01
    saida = clones_com_canal({"x": [tmp_path / "c" / "f01.wav"]}, ruido, 20.0, tmp_path / "out")
    y, sr = sf.read(saida["x"][0])
    assert sr == 16000 and abs(len(y) - 32000) <= 2 and np.abs(y).max() <= 1.0
    limpo = np.sin(2 * np.pi * 220 * np.arange(len(y)) / 16000)
    a = np.dot(y, limpo) / np.dot(limpo, limpo)          # o que sobra fora do tom é o ruído
    snr = 10 * np.log10(np.mean((a * limpo) ** 2) / np.mean((y - a * limpo) ** 2))
    assert 17 < snr < 23


def test_pasta_vazia_para_com_mensagem(tmp_path, capsys):
    import pytest

    from scripts.ensaio_canal_igualado import main

    (tmp_path / "vazia").mkdir()
    with pytest.raises(SystemExit):
        main(["--real", f"r={tmp_path / 'vazia'}", "--clone", f"c={tmp_path / 'vazia'}",
              "--aluno", "a=x.pt", "--sem-ruido"])
    assert "pasta sem .wav" in capsys.readouterr().err


def test_origem_com_rotulos():
    o = montar_origem({"L01_real": ["r1.wav", "r2.wav"]}, {"L01_xtts": ["c1.wav"]})
    assert list(o["rotulo"]) == [0, 0, 1] and list(o["classe"]) == ["L01_real", "L01_real", "L01_xtts"]
