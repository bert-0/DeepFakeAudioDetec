"""Marca Perth e linhas fixas (scripts/estudo_perth.py), com um marcador falso.

O `perth` real só existe no ambiente do Chatterbox; aqui o marcador falso soma
um tom fixo, para conferir que o estudo conta as linhas antes e depois da marca
e apaga os áudios marcados.
"""

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.estudo_perth import estudar


class MarcadorFalso:
    def apply_watermark(self, y, watermark=None, sample_rate=24000):
        t = np.arange(len(y)) / sample_rate
        return y + 0.01 * np.sin(2 * np.pi * 7850 * t)

    def get_watermark(self, y, sample_rate=24000):
        f = np.fft.rfftfreq(len(y), 1 / sample_rate)
        p = np.abs(np.fft.rfft(y)) ** 2
        return float(p[np.argmin(np.abs(f - 7850))] > 100 * np.median(p))


def test_conta_antes_e_depois_e_apaga_os_marcados(tmp_path):
    (tmp_path / "real").mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        # 12 s: em 1 Hz de resolução, ruído curto cria "linhas" por acaso
        sf.write(tmp_path / "real" / f"f{i}.wav", (rng.standard_normal(24000 * 12) * 0.05).astype(np.float32), 24000)
    antes = set(Path(tempfile.gettempdir()).glob("perth_*"))
    r = estudar(MarcadorFalso(), {"real": sorted((tmp_path / "real").glob("*.wav"))}, ["real"], [7850.0])
    assert r["real"]["linhas"]["7850"] == 0 and sum(r["real"]["marca"]) == 0
    assert r["real+perth"]["linhas"]["7850"] == 3 and sum(r["real+perth"]["marca"]) == 3
    assert set(Path(tempfile.gettempdir()).glob("perth_*")) == antes   # nada de voz sobra no disco
