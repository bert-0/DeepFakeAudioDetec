"""Interface web (web/app.py): envio, resultado, histórico e API."""

import importlib

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
    for rota in ("/", "/historico", "/sobre"):
        assert cliente.get(rota).status_code == 200


def test_envio_gera_resultado_e_entra_no_historico(cliente, tmp_path):
    with open(_wav(tmp_path), "rb") as fh:
        r = cliente.post("/analisar", files={"arquivo": ("voz.wav", fh, "audio/wav")},
                         follow_redirects=False)
    assert r.status_code == 303
    pagina = cliente.get(r.headers["location"])
    assert pagina.status_code == 200
    assert "voz.wav" in pagina.text and "<svg" in pagina.text
    assert "arquivo nativo de 16 kHz" in pagina.text
    assert "voz.wav" in cliente.get("/historico").text


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
