"""Playlists de demonstração (scripts/montar_demo.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.montar_demo import GAP_S, INICIO_S, main, montar  # noqa: E402

SR = 16000


def test_silencio_entre_audios_cobre_a_janela_do_monitor():
    """Com 4 s de silêncio, nenhuma janela de 4 s pega dois áudios."""
    audios = [(("a",), np.ones(SR * 2, np.float32)), (("b",), np.ones(SR * 3, np.float32))]
    sinal, marcas = montar(audios, SR)
    assert GAP_S >= 4.0
    assert marcas[0][0] == INICIO_S
    assert marcas[1][0] == pytest.approx(INICIO_S + 2 + GAP_S)
    ini_b = int(marcas[1][0] * SR)
    assert not sinal[ini_b - int(GAP_S * SR):ini_b].any()


def _base(tmp_path):
    audio = tmp_path / "flac"
    audio.mkdir()
    linhas = []
    for i, (rot, sis) in enumerate([("bonafide", "-"), ("bonafide", "-"),
                                    ("spoof", "A09"), ("spoof", "A11")]):
        nome = f"LA_E_{i}"
        sf.write(audio / f"{nome}.flac", np.full(SR, 0.1 * (i + 1), np.float32), SR)
        linhas.append(f"LA_0099 {nome} - {sis} {rot}")
    proto = tmp_path / "eval.txt"
    proto.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    cfg = {"audio": {"sample_rate": SR},
           "data": {"protocols": {"eval": str(proto)}, "audio_dir": {"eval": str(audio)}}}
    caminho = tmp_path / "c.yaml"
    caminho.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return caminho


def test_gera_as_tres_playlists_com_cola(tmp_path):
    cfg = _base(tmp_path)
    saida = tmp_path / "demo"
    assert main(["--config", str(cfg), "--reais", "LA_E_0,LA_E_1",
                 "--falsos", "LA_E_2,LA_E_3", "--saida", str(saida)]) == 0
    for nome in ("reais", "falsos", "misto"):
        assert (saida / f"{nome}.wav").is_file()
    cola = (saida / "misto_cola.txt").read_text(encoding="utf-8")
    ordem = [ln.split()[1] for ln in cola.splitlines() if "LA_E_" in ln]
    assert ordem == ["LA_E_0", "LA_E_2", "LA_E_1", "LA_E_3"]
    assert "A09" in cola and "0.9829" in cola


def test_recusa_rotulo_trocado(tmp_path):
    cfg = _base(tmp_path)
    with pytest.raises(SystemExit, match="não é bonafide"):
        main(["--config", str(cfg), "--reais", "LA_E_2", "--falsos", "LA_E_3",
              "--saida", str(tmp_path / "d")])
