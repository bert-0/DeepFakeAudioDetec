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
    assert [p[0] for p in _TTSFalso.pedidos] == ["Frase 1", "Frase 2", "Frase 3"], "sem o ponto final"
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
    assert [p[0] for p in _TTSFalso.pedidos] == ["Frase 2", "Frase 3"]


def test_referencia_inexistente(monkeypatch, tmp_path, capsys):
    _instalar_falso(monkeypatch)
    _, frases = _entradas(tmp_path)
    assert main(["--referencia", str(tmp_path / "nada.wav"), "--frases", str(frases),
                 "--saida", str(tmp_path / "c")]) == 1
    assert "referência não encontrada" in capsys.readouterr().out


def test_frases_do_teste_viram_dez_arquivos():
    assert len(ler_frases("docs/etica/frases_teste_voz_propria.txt")) == 10


def test_frase_livre_vira_frase_nn_sem_sobrescrever(monkeypatch, tmp_path, capsys):
    import pytest

    _instalar_falso(monkeypatch)
    ref, _ = _entradas(tmp_path)
    saida = tmp_path / "demo"
    saida.mkdir()
    sf.write(saida / "frase_01.wav", np.zeros(2400, np.float32), 24000)
    assert main(["--referencia", str(ref), "--saida", str(saida), "--dispositivo", "cpu",
                 "--texto", "Bom dia a todos da banca, obrigado por estarem aqui hoje.",
                 "--texto", "Oi."]) == 0
    assert sorted(p.name for p in saida.glob("frase_*.wav")) == [
        "frase_01.wav", "frase_02.wav", "frase_03.wav"]
    reg = json.loads((saida / "clonagem.json").read_text(encoding="utf-8"))
    assert [f["semente"] for f in reg["frases"]] == [2, 3]
    assert "menos de 8 palavras" in capsys.readouterr().out, "frase curta não enche a janela"
    with pytest.raises(SystemExit):
        main(["--referencia", str(ref), "--saida", str(saida)])


def test_interativo_refaz_a_ultima_com_outra_semente(monkeypatch, tmp_path):
    _instalar_falso(monkeypatch)
    ref, _ = _entradas(tmp_path)
    frase = "A banca escolheu esta frase para o clone dizer agora mesmo."
    respostas = iter(["r", "", frase, "", "r", "sair"])
    monkeypatch.setattr("builtins.input", lambda *_: next(respostas))
    saida = tmp_path / "demo"
    assert main(["--referencia", str(ref), "--saida", str(saida), "--dispositivo", "cpu",
                 "--interativo"]) == 0
    assert [p[0] for p in _TTSFalso.pedidos] == [frase[:-1], frase[:-1]]
    reg = json.loads((saida / "clonagem.json").read_text(encoding="utf-8"))
    sementes = [f["semente"] for f in reg["frases"]]
    assert len(set(sementes)) == 2, "refazer com a mesma semente daria o mesmo áudio"
    assert [f["arquivo"] for f in reg["frases"]] == ["frase_01.wav", "frase_02.wav"]


def test_interativo_mostra_como_usar_e_ignora_enter_vazio(monkeypatch, tmp_path, capsys):
    """24/10/2026: `r` antes de qualquer frase e um Enter vazio encerravam sem gerar nada."""
    _instalar_falso(monkeypatch)
    ref, _ = _entradas(tmp_path)
    respostas = iter(["r", "", ""])
    def ler(*_):
        try:
            return next(respostas)
        except StopIteration:
            raise EOFError from None
    monkeypatch.setattr("builtins.input", ler)
    assert main(["--referencia", str(ref), "--saida", str(tmp_path / "d"), "--dispositivo", "cpu",
                 "--interativo"]) == 0
    saida = capsys.readouterr().out
    assert "Digite a frase que o clone vai dizer" in saida
    assert "digite uma frase primeiro" in saida and _TTSFalso.pedidos == []


def test_fora_do_ambiente_do_xtts_diz_o_que_fazer(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setitem(sys.modules, "TTS", None)
    monkeypatch.setitem(sys.modules, "TTS.api", None)
    ref, frases = _entradas(tmp_path)
    with pytest.raises(SystemExit, match="conda activate xtts"):
        main(["--referencia", str(ref), "--frases", str(frases), "--saida", str(tmp_path / "c"),
              "--dispositivo", "cpu"])


def test_sem_ponto_final_tira_so_o_ponto():
    """Clones do L02 (25/10/2026): o XTTS leu o ponto final como "ponto"."""
    from scripts.clonar_xtts import sem_ponto

    assert sem_ponto("Vamos ao parque amanhã cedo.") == "Vamos ao parque amanhã cedo"
    assert sem_ponto("Fim. ") == "Fim"
    assert sem_ponto("Você vem?") == "Você vem?" and sem_ponto("Que dia!") == "Que dia!"
    assert sem_ponto("E então...") == "E então..."


def test_padrao_sem_ponto_final_nos_tres_modos(monkeypatch, tmp_path):
    _instalar_falso(monkeypatch)
    ref, frases = _entradas(tmp_path)
    assert main(["--referencia", str(ref), "--frases", str(frases), "--saida", str(tmp_path / "a"),
                 "--dispositivo", "cpu"]) == 0
    assert [p[0] for p in _TTSFalso.pedidos] == ["Frase 1", "Frase 2", "Frase 3"]
    reg = json.loads((tmp_path / "a" / "clonagem.json").read_text(encoding="utf-8"))
    assert reg["sem_ponto_final"] is True
    assert reg["frases"][0]["frase"] == "Frase 1." and reg["frases"][0]["texto_enviado"] == "Frase 1"
    _TTSFalso.pedidos.clear()
    texto = "Uma frase escolhida pela banca para o clone dizer agora."
    assert main(["--referencia", str(ref), "--saida", str(tmp_path / "b"), "--dispositivo", "cpu",
                 "--texto", texto]) == 0
    respostas = iter([texto, "sair"])
    monkeypatch.setattr("builtins.input", lambda *_: next(respostas))
    assert main(["--referencia", str(ref), "--saida", str(tmp_path / "c"), "--dispositivo", "cpu",
                 "--interativo"]) == 0
    assert [p[0] for p in _TTSFalso.pedidos] == [texto[:-1]] * 2


def test_com_ponto_final_repete_os_ensaios_1_e_2(monkeypatch, tmp_path):
    _instalar_falso(monkeypatch)
    ref, frases = _entradas(tmp_path)
    assert main(["--referencia", str(ref), "--frases", str(frases), "--saida", str(tmp_path / "a"),
                 "--dispositivo", "cpu", "--com-ponto-final"]) == 0
    assert _TTSFalso.pedidos[0][0] == "Frase 1."
    reg = json.loads((tmp_path / "a" / "clonagem.json").read_text(encoding="utf-8"))
    assert reg["sem_ponto_final"] is False
