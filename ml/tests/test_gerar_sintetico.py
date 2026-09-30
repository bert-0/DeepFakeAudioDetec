"""Versões sintéticas da própria voz (scripts/gerar_sintetico.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.gerar_sintetico import griffin_lim, main, world  # noqa: E402

SR = 16000


def _voz(segundos=1.5, f0=140.0):
    """Fonte harmônica com F0 variando e envelope silábico: o WORLD precisa de F0."""
    t = np.arange(int(segundos * SR)) / SR
    f = f0 * (1 + 0.1 * np.sin(2 * np.pi * 1.5 * t))
    fase = 2 * np.pi * np.cumsum(f) / SR
    x = sum(np.sin(k * fase) / k for k in range(1, 20))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2
    return (0.3 * x * env / np.abs(x).max()).astype(np.float32)


def test_griffin_lim_refaz_com_o_mesmo_nivel_e_outra_forma_de_onda():
    x = _voz()
    y = griffin_lim(x, SR, n_iter=10)
    assert abs(len(y) - len(x)) < 1024
    assert np.abs(y).max() == pytest.approx(np.abs(x).max(), rel=1e-4)
    n = min(len(x), len(y))
    assert abs(np.corrcoef(x[:n], y[:n])[0, 1]) < 0.9, "a fase é reconstruída, não copiada"


def test_world_refaz_a_voz():
    pytest.importorskip("pyworld")
    x = _voz()
    y = world(x, SR)
    assert abs(len(y) - len(x)) < 400
    assert np.abs(y).max() == pytest.approx(np.abs(x).max(), rel=1e-4)


def test_gera_pares_e_playlist_a_48k(tmp_path):
    entrada = tmp_path / "grav"
    entrada.mkdir()
    for i in range(2):
        sf.write(entrada / f"frase{i}.wav", _voz(f0=120 + 30 * i), SR)
    saida = tmp_path / "demo"
    assert main(["--entrada", str(entrada), "--saida", str(saida),
                 "--metodos", "griffinlim"]) == 0
    assert sf.info(saida / "pareado.wav").samplerate == 48000
    for i in range(2):
        for tipo in ("original", "griffinlim"):
            assert (saida / f"frase{i}_{tipo}.wav").is_file()
    cola = (saida / "pareado_cola.txt").read_text(encoding="utf-8")
    ordem = [ln.split()[1] for ln in cola.splitlines() if "frase" in ln]
    assert ordem == ["frase0_original", "frase0_griffinlim",
                     "frase1_original", "frase1_griffinlim"]
