"""Aba Clonar (web/clonagem.py e as rotas em web/app.py).

O XTTS e o Chatterbox não rodam aqui: o `subprocess.run` falso escreve um WAV de
ruído onde o `clonar_xtts.py` escreveria o clone.
"""

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

from src.config import load_config  # noqa: E402
from src.models import build_model  # noqa: E402


def _vozes(tmp_path, autorizada=True):
    ref = tmp_path / "ref.wav"
    sf.write(ref, np.zeros(16000, np.float32), 16000)
    caminho = tmp_path / "vozes.json"
    caminho.write_text(json.dumps({"vozes": [
        {"codigo": "L01", "referencia": str(ref), "clonagem_autorizada": autorizada},
        {"codigo": "L09", "referencia": str(tmp_path / "sumiu.wav"), "clonagem_autorizada": True},
        {"codigo": "L03", "referencia": str(ref), "clonagem_autorizada": False}]}), encoding="utf-8")
    return caminho


def _run_falso(chamadas, codigo=0, ambientes=None):
    def run(cmd, cwd=None, stdout=None, stderr=None, env=None, timeout=None):
        chamadas.append(cmd)
        if ambientes is not None:
            ambientes.append(env)
        saida = Path(cmd[cmd.index("--saida") + 1])
        if codigo == 0:
            rng = np.random.default_rng(0)
            sf.write(saida / "frase_01.wav", (0.1 * rng.standard_normal(16000 * 5)).astype(np.float32), 16000)
        else:
            stdout.write("Traceback: CUDA out of memory\n")
        return types.SimpleNamespace(returncode=codigo)
    return run


def test_so_vozes_autorizadas_com_referencia(tmp_path):
    from web.clonagem import ler_vozes

    vozes = ler_vozes(_vozes(tmp_path))
    assert [v.codigo for v in vozes] == ["L01"], "sem termo ou sem referência, fica de fora"
    assert ler_vozes(tmp_path / "nao_existe.json") == []


def test_acha_o_python_de_cada_ambiente(tmp_path, monkeypatch):
    from web import clonagem

    base = tmp_path / "Miniconda"
    nome = "python.exe" if clonagem.os.name == "nt" else "bin/python"
    (base / "envs" / "xtts" / nome).parent.mkdir(parents=True)
    (base / "envs" / "xtts" / nome).write_text("")
    (base / nome).parent.mkdir(parents=True, exist_ok=True)
    (base / nome).write_text("")
    monkeypatch.setattr(sys, "executable", str(base / nome))
    monkeypatch.delenv("DETECTOR_PYTHON_XTTS", raising=False)
    monkeypatch.delenv("DETECTOR_PYTHON_CHATTERBOX", raising=False)
    assert clonagem.python_do_ambiente("xtts") == (base / "envs" / "xtts" / nome).resolve()
    assert clonagem.python_do_ambiente("chatterbox") is None
    assert clonagem.ferramentas_disponiveis() == {"xtts": "XTTS v2"}
    monkeypatch.setenv("DETECTOR_PYTHON_CHATTERBOX", str(base / nome))
    assert "chatterbox" in clonagem.ferramentas_disponiveis()


@pytest.mark.parametrize("windows", [True, False])
def test_candidatos_nas_duas_organizacoes_do_conda(windows):
    """Windows: Miniconda/python.exe e envs/x/python.exe; Linux: bin/python nos dois."""
    from pathlib import PurePosixPath

    from web.clonagem import candidatos

    py = "python.exe" if windows else "bin/python"
    base = PurePosixPath("/conda")
    do_base = candidatos(base / py, "xtts", windows)
    de_outro = candidatos(base / "envs" / "chatterbox" / py, "xtts", windows)
    assert base / "envs" / "xtts" / py in do_base
    assert base / "envs" / "xtts" / py in de_outro


def test_clonar_chama_o_script_e_devolve_o_wav(tmp_path, monkeypatch):
    from web import clonagem

    monkeypatch.setenv("DETECTOR_PYTHON_XTTS", sys.executable)
    chamadas = []
    monkeypatch.setattr(clonagem.subprocess, "run", _run_falso(chamadas))
    voz = clonagem.ler_vozes(_vozes(tmp_path))[0]
    wav = clonagem.clonar(voz, "uma frase de teste com oito palavras ao todo", "xtts", tmp_path / "c")
    assert wav.name == "frase_01.wav" and wav.is_file()
    cmd = chamadas[0]
    assert cmd[cmd.index("--ferramenta") + 1] == "xtts" and cmd[cmd.index("--referencia") + 1] == str(voz.referencia)
    assert "clonar_xtts.py" in cmd[2] and cmd[1] == "-u"

    monkeypatch.setattr(clonagem.subprocess, "run", _run_falso([], codigo=1))
    with pytest.raises(clonagem.ErroClonagem, match="out of memory"):
        clonagem.clonar(voz, "outra frase", "xtts", tmp_path / "d")
    with pytest.raises(ValueError):
        clonagem.validar_frase("   ")
    assert clonagem.validar_frase("  duas   palavras ") == "duas palavras"


@pytest.fixture()
def cliente(tmp_path, monkeypatch):
    config = load_config("configs/baseline.yaml")
    torch.manual_seed(0)
    ckpt = tmp_path / "m.pt"
    torch.save({"model_state": build_model(config["model"]).state_dict(), "config": config,
                "threshold": 0.5}, ckpt)
    monkeypatch.setenv("DETECTOR_CONFIG", "configs/baseline.yaml")
    monkeypatch.setenv("DETECTOR_CHECKPOINT", str(ckpt))
    monkeypatch.setenv("DETECTOR_BANCO", str(tmp_path / "h.db"))
    monkeypatch.setenv("DETECTOR_VOZES", str(_vozes(tmp_path)))
    monkeypatch.setenv("DETECTOR_PASTA_CLONES", str(tmp_path / "clones"))
    monkeypatch.setenv("DETECTOR_PYTHON_XTTS", sys.executable)
    monkeypatch.setenv("DETECTOR_PYTHON_CHATTERBOX", str(tmp_path / "nao_instalado.exe"))
    monkeypatch.delenv("DETECTOR_XLSR", raising=False)
    monkeypatch.delenv("DETECTOR_LIMIAR_FUSAO", raising=False)
    from web import clonagem

    monkeypatch.setattr(clonagem.subprocess, "run", _run_falso([]))
    import web.app as modulo

    importlib.reload(modulo)
    return TestClient(modulo.app), modulo, tmp_path


def test_aba_clonar_gera_analisa_toca_e_apaga(cliente):
    c, modulo, tmp_path = cliente
    pagina = c.get("/clonar").text
    assert 'value="L01"' in pagina and "L03" not in pagina and "XTTS v2" in pagina
    assert "Chatterbox" not in pagina, "sem o ambiente, a ferramenta não aparece"
    r = c.post("/api/clonar", json={"voz": "L01", "ferramenta": "xtts",
                                    "frase": "a biblioteca abre mais cedo para os alunos estudarem"})
    assert r.status_code == 200
    analise = r.json()["analise_id"]
    resultado = c.get(f"/analises/{analise}").text
    assert "Clone XTTS v2 de L01" in resultado and f'/analises/{analise}/audio' in resultado
    audio = c.get(f"/analises/{analise}/audio")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    pastas = list((tmp_path / "clones").glob("clone_*"))
    assert len(pastas) == 1
    c.post("/historico/excluir", data={"ids": [analise]})
    assert not pastas[0].exists(), "o clone sai com o registro"


def test_aba_clonar_recusa_o_que_nao_pode(cliente, monkeypatch):
    c, modulo, tmp_path = cliente
    frase = "a biblioteca abre mais cedo para os alunos estudarem"
    assert c.post("/api/clonar", json={"voz": "L03", "frase": frase}).status_code == 400
    assert c.post("/api/clonar", json={"voz": "L01", "ferramenta": "chatterbox", "frase": frase}).status_code == 400
    assert c.post("/api/clonar", json={"voz": "L01", "frase": " "}).status_code == 400
    monkeypatch.setitem(modulo._estado, "sessao", {"proc": types.SimpleNamespace(poll=lambda: None)})
    r = c.post("/api/clonar", json={"voz": "L01", "frase": frase})
    assert r.status_code == 409 and "ao vivo" in r.json()["detail"]


def test_audio_so_de_dentro_das_pastas(cliente):
    c, modulo, tmp_path = cliente
    fora = tmp_path / "fora.wav"
    sf.write(fora, np.zeros(16000, np.float32), 16000)
    assert modulo._audio_da_analise({"gravacao": str(fora)}) is None
    assert modulo._audio_da_analise({"gravacao": None}) is None


def test_clonagem_offline_primeiro_e_com_rede_se_faltar_cache(tmp_path, monkeypatch):
    from web import clonagem

    monkeypatch.setenv("DETECTOR_PYTHON_XTTS", sys.executable)
    voz = clonagem.ler_vozes(_vozes(tmp_path))[0]
    ambientes = []
    monkeypatch.setattr(clonagem.subprocess, "run", _run_falso([], ambientes=ambientes))
    clonagem.clonar(voz, "uma frase de teste com oito palavras ao todo", "xtts", tmp_path / "a")
    assert len(ambientes) == 1 and ambientes[0]["HF_HUB_OFFLINE"] == "1"

    tentativas = []

    def run(cmd, cwd=None, stdout=None, stderr=None, env=None, timeout=None):
        tentativas.append(env)
        if env.get("HF_HUB_OFFLINE") == "1":
            stdout.write("huggingface_hub.errors.IncompleteSnapshotError: ...")
            return types.SimpleNamespace(returncode=1)
        sf.write(Path(cmd[cmd.index("--saida") + 1]) / "frase_01.wav", np.zeros(16000, np.float32), 16000)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(clonagem.subprocess, "run", run)
    assert clonagem.clonar(voz, "outra frase", "xtts", tmp_path / "b").is_file()
    assert len(tentativas) == 2 and "HF_HUB_OFFLINE" not in tentativas[1]
