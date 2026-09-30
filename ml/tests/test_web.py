"""Interface web (web/app.py): envio, resultado, histórico e API."""

import importlib
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


@pytest.fixture()
def cliente(tmp_path, monkeypatch):
    config = load_config("configs/baseline.yaml")
    torch.manual_seed(0)
    ckpt = tmp_path / "m.pt"
    torch.save({"model_state": build_model(config["model"]).state_dict(),
                "config": config, "threshold": 0.5}, ckpt)
    monkeypatch.setenv("DETECTOR_CONFIG", "configs/baseline.yaml")
    monkeypatch.setenv("DETECTOR_CHECKPOINT", str(ckpt))
    monkeypatch.setenv("DETECTOR_BANCO", str(tmp_path / "h.db"))
    import web.app as modulo

    importlib.reload(modulo)
    return TestClient(modulo.app)


def _wav(tmp_path, taxa=16000, segundos=5.0):
    rng = np.random.default_rng(0)
    arq = tmp_path / "voz.wav"
    sf.write(arq, (rng.standard_normal(int(taxa * segundos)) * 0.1).astype(np.float32), taxa)
    return arq


def test_paginas_abrem(cliente):
    for rota in ("/", "/historico", "/sobre", "/ao-vivo", "/resultado"):
        assert cliente.get(rota).status_code == 200


def test_envio_gera_resultado_e_entra_no_historico(cliente, tmp_path):
    with open(_wav(tmp_path), "rb") as fh:
        r = cliente.post("/analisar", files={"arquivo": ("voz.wav", fh, "audio/wav")},
                         follow_redirects=False)
    assert r.status_code == 303
    pagina = cliente.get(r.headers["location"])
    assert pagina.status_code == 200
    assert "voz.wav" in pagina.text and 'class="barra' in pagina.text
    assert "arquivo nativo de 16 kHz" in pagina.text
    assert "protótipo" not in pagina.text and "confiança" not in pagina.text
    assert "voz.wav" in cliente.get("/historico").text

    rel = cliente.get(r.headers["location"] + "/relatorio.json")
    assert rel.status_code == 200 and "attachment" in rel.headers["content-disposition"]
    assert rel.json()["arquivo"] == "voz.wav" and rel.json()["janelas"]


def test_api_devolve_json_e_escolhe_limiar_pela_taxa(cliente, tmp_path):
    with open(_wav(tmp_path, taxa=48000), "rb") as fh:
        r = cliente.post("/api/analisar", files={"arquivo": ("voz.wav", fh, "audio/wav")})
    dados = r.json()
    assert r.status_code == 200
    assert dados["janelas"] and 0 <= dados["score_medio"] <= 1
    assert "48 kHz" in dados["origem_limiar"]


def test_formato_recusado(cliente):
    r = cliente.post("/analisar", files={"arquivo": ("x.txt", b"ola", "text/plain")})
    assert r.status_code == 400


def test_analise_inexistente(cliente):
    assert cliente.get("/analises/999").status_code == 404


def test_arquivo_curto_nao_e_inconclusivo(cliente, tmp_path):
    """Os áudios do ASVspoof (2-3 s) cabem numa janela só — a unidade em que o
    modelo foi medido. Inconclusivo é só quando não há fala nenhuma."""
    with open(_wav(tmp_path, segundos=2.5), "rb") as fh:
        r = cliente.post("/analisar", files={"arquivo": ("curto.wav", fh, "audio/wav")})
    assert r.status_code == 200
    assert "Inconclusivo" not in r.text
    import re

    assert re.search(r"(Nenhuma\s+ficou|1 de 1\s+ficou) acima do limiar", r.text)


def test_ao_vivo_sem_sessao(cliente):
    assert cliente.get("/api/ao-vivo").json() == {"existe": False, "rodando": False}


class _ProcFalso:
    """Faz o papel do monitor.py: roda até o arquivo de parada aparecer."""

    def __init__(self, parar: Path):
        self.parar = parar
        self.returncode = None

    def poll(self):
        if self.parar.exists():
            self.returncode = 0
        return self.returncode

    def wait(self, timeout=None):
        return self.poll()

    def terminate(self):
        self.returncode = -1


def test_sessao_ao_vivo_inicia_mostra_so_a_atual_e_vai_para_resultados(cliente, tmp_path,
                                                                     monkeypatch):
    import web.app as modulo
    from monitor import gravar_json
    from src.capture.analyzer import Agregador, Leitura

    monkeypatch.setenv("DETECTOR_PASTA_AO_VIVO", str(tmp_path / "ao_vivo"))
    monkeypatch.setattr(modulo, "_iniciar_monitor",
                        lambda j, w, parar, log: _ProcFalso(parar))

    assert cliente.post("/api/ao-vivo/iniciar").json() == {"ok": True}
    assert cliente.post("/api/ao-vivo/iniciar").status_code == 409, "uma sessão por vez"
    d = cliente.get("/api/ao-vivo").json()
    assert d["rodando"] and not d["existe"], "ainda sem janela: nada de sessão antiga"

    ag = Agregador(janela_s=4.0, passo_s=2.0)
    for i in range(50):
        ag.adicionar(Leitura(indice=i, instante=2.0 * i, score=0.1 + 0.01 * i, rms=0.1))
    gravar_json(ag, modulo._estado["sessao"]["json"], limiar=0.98,
                origem_limiar="recalibrado", ativo=True)
    d = cliente.get("/api/ao-vivo").json()
    assert d["existe"] and d["rodando"] and len(d["leituras"]) == 40
    assert "peso" in d["leituras"][0]

    parou = cliente.post("/api/ao-vivo/parar").json()
    assert parou["ok"] and parou["analise_id"]
    assert not cliente.get("/api/ao-vivo").json()["rodando"]
    assert "Sessão ao vivo" in cliente.get("/historico").text
    assert cliente.get(f"/analises/{parou['analise_id']}").status_code == 200


def test_monitor_para_quando_o_arquivo_de_parada_aparece(tmp_path):
    """O caminho real que a interface usa: o monitor vê o arquivo e encerra como
    num Ctrl+C, gravando o JSON final com ativo=false."""
    import json
    import subprocess
    import sys

    config = load_config("configs/baseline.yaml")
    ckpt = tmp_path / "m.pt"
    torch.save({"model_state": build_model(config["model"]).state_dict(),
                "config": config}, ckpt)
    parar = tmp_path / "parar"
    parar.touch()
    saida = tmp_path / "s.json"
    r = subprocess.run([sys.executable, "monitor.py", "--config", "configs/baseline.yaml",
                        "--checkpoint", str(ckpt), "--arquivo", str(_wav(tmp_path, segundos=30)),
                        "--json", str(saida), "--parar-com", str(parar), "--device", "cpu"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(saida.read_text(encoding="utf-8"))["ativo"] is False


def _enviar(cliente, tmp_path, nome):
    with open(_wav(tmp_path), "rb") as fh:
        r = cliente.post("/analisar", files={"arquivo": (nome, fh, "audio/wav")},
                         follow_redirects=False)
    return int(r.headers["location"].rsplit("/", 1)[1])


def test_aba_resultado_leva_a_analise_mais_recente(cliente, tmp_path):
    assert "Nenhuma análise ainda" in cliente.get("/resultado").text
    _enviar(cliente, tmp_path, "a.wav")
    ultima = _enviar(cliente, tmp_path, "b.wav")
    r = cliente.get("/resultado", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == f"/analises/{ultima}"


def test_excluir_alguns_e_limpar_tudo(cliente, tmp_path):
    ids = [_enviar(cliente, tmp_path, f"f{i}.wav") for i in range(3)]
    r = cliente.post("/historico/excluir", data={"ids": [ids[0], ids[2]]})
    assert r.status_code == 200
    assert "f1.wav" in r.text and "f0.wav" not in r.text and "f2.wav" not in r.text
    assert cliente.get(f"/analises/{ids[0]}").status_code == 404

    r = cliente.post("/historico/limpar")
    assert "O histórico está vazio" in r.text
    assert "Nenhuma análise ainda" in cliente.get("/resultado").text


def test_excluir_sem_selecao_nao_quebra(cliente):
    assert cliente.post("/historico/excluir", data={}).status_code == 200
