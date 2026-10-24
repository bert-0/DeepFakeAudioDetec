"""Clonagem com o XTTS v2 (scripts/clonar_xtts.py), com um TTS falso.

O XTTS de verdade fica num ambiente conda separado e baixa ~2 GB de pesos; aqui
um falso grava ruído e registra os pedidos.
"""

import json
import sys
import types

import numpy as np
import soundfile as sf

from scripts.clonar_xtts import MODELO, ler_frases, main


class _TTSFalso:
    pedidos: list = []
    modelos: list = []

    def __init__(self, modelo):
        _TTSFalso.modelos.append(modelo)

    def to(self, dispositivo):
        self.dispositivo = dispositivo
        return self

    def tts_to_file(self, text, speaker_wav, language, file_path):
        _TTSFalso.pedidos.append((text, speaker_wav, language))
        sf.write(file_path, np.random.default_rng(0).standard_normal(24000 * 2) * 0.1, 24000)


def _instalar_falso(monkeypatch):
    _TTSFalso.pedidos, _TTSFalso.modelos = [], []
    api = types.ModuleType("TTS.api")
    api.TTS = _TTSFalso
    monkeypatch.setitem(sys.modules, "TTS", types.ModuleType("TTS"))
    monkeypatch.setitem(sys.modules, "TTS.api", api)


def _entradas(tmp_path, n=3):
    ref = tmp_path / "ref.wav"
    sf.write(ref, np.zeros(24000, np.float32), 24000)
    frases = tmp_path / "frases.txt"
    frases.write_text("\n".join(f"Frase {i}." for i in range(1, n + 1)) + "\n", encoding="utf-8")
    return ref, frases


def test_uma_fala_por_frase_com_a_voz_da_referencia(monkeypatch, tmp_path, capsys):
    _instalar_falso(monkeypatch)
    ref, frases = _entradas(tmp_path)
    saida = tmp_path / "clone"
    assert main(["--referencia", str(ref), "--frases", str(frases), "--saida", str(saida),
                 "--dispositivo", "cpu"]) == 0
    assert _TTSFalso.modelos == [MODELO]
    assert [p[0] for p in _TTSFalso.pedidos] == ["Frase 1.", "Frase 2.", "Frase 3."]
    assert {p[1] for p in _TTSFalso.pedidos} == {str(ref)} and {p[2] for p in _TTSFalso.pedidos} == {"pt"}
    assert sorted(x.name for x in saida.glob("*.wav")) == ["f01.wav", "f02.wav", "f03.wav"]
    reg = json.loads((saida / "clonagem.json").read_text(encoding="utf-8"))
    assert reg["modelo"] == MODELO and reg["idioma"] == "pt"
    assert [f["semente"] for f in reg["frases"]] == [1, 2, 3]
    assert all(f["duracao_s"] == 2.0 for f in reg["frases"])
    assert "x o tempo real" in capsys.readouterr().out


def test_continua_de_onde_parou(monkeypatch, tmp_path):
    _instalar_falso(monkeypatch)
    ref, frases = _entradas(tmp_path)
    saida = tmp_path / "clone"
    saida.mkdir()
    sf.write(saida / "f01.wav", np.zeros(2400, np.float32), 24000)
    assert main(["--referencia", str(ref), "--frases", str(frases), "--saida", str(saida),
                 "--dispositivo", "cpu"]) == 0
    assert [p[0] for p in _TTSFalso.pedidos] == ["Frase 2.", "Frase 3."]


def test_referencia_inexistente(monkeypatch, tmp_path, capsys):
    _instalar_falso(monkeypatch)
    _, frases = _entradas(tmp_path)
    assert main(["--referencia", str(tmp_path / "nada.wav"), "--frases", str(frases),
                 "--saida", str(tmp_path / "c")]) == 1
    assert "referência não encontrada" in capsys.readouterr().out


def test_frases_do_teste_viram_dez_arquivos():
    assert len(ler_frases("docs/etica/frases_teste_voz_propria.txt")) == 10
