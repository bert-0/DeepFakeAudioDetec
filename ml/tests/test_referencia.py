"""Gravar a referência de uma voz pela aba Clonar (web/referencia.py e as rotas).

O microfone é falso: uma fonte que entrega um sinal pronto, em blocos, na taxa de
48 kHz do Realtek. A "fala" é ruído forte entre trechos de ruído fraco, como nos
testes do gravador de frases.
"""

import importlib
import json
import sys
import types

import numpy as np
import pytest
import soundfile as sf

SR = 48000


def _leitura(fala_s=25.0, antes_s=2.0, depois_s=2.0, seed=0):
    """O que o microfone entrega: o segundo descartado, silêncio, fala e silêncio."""
    rng = np.random.default_rng(seed)
    n = int((1.0 + antes_s + fala_s + depois_s) * SR)
    wav = rng.standard_normal(n) * 0.002
    i = int((1.0 + antes_s) * SR)
    wav[i:i + int(fala_s * SR)] += rng.standard_normal(int(fala_s * SR)) * 0.1
    return wav.astype(np.float32)


class _MicrofoneFalso:
    def __init__(self, sinal, bloco=4800, ao_entregar=None):
        self.sinal, self.bloco, self.ao_entregar = sinal, bloco, ao_entregar
        self.taxa_dispositivo, self.aviso, self.fechado = SR, None, False

    def blocos(self):
        for k, i in enumerate(range(0, len(self.sinal), self.bloco)):
            if self.ao_entregar:
                self.ao_entregar(k)
            yield self.sinal[i:i + self.bloco]

    def fechar(self):
        self.fechado = True


def _lista(tmp_path):
    ref = tmp_path / "antiga.wav"
    sf.write(ref, np.zeros(SR, np.float32), SR)
    caminho = tmp_path / "vozes_autorizadas.json"
    caminho.write_text(json.dumps({"apagar_clones_ate": "2026-11-19", "vozes": [
        {"codigo": "L05", "referencia": str(ref), "clonagem_autorizada": True, "observacao": "x"},
        {"codigo": "L07", "referencia": "data/vozes/referencias/L07.wav", "clonagem_autorizada": True},
        {"codigo": "L08", "referencia": str(ref), "clonagem_autorizada": False}]}), encoding="utf-8")
    return caminho


def test_grava_so_vozes_autorizadas_mesmo_sem_referencia(tmp_path):
    from web.clonagem import ler_vozes
    from web.referencia import vozes_para_gravar

    _lista(tmp_path)
    assert vozes_para_gravar() == ["L05", "L07"], "a L08 não autorizou; a L07 ainda não tem referência"
    assert [v.codigo for v in ler_vozes()] == ["L05"]


def test_leitura_aprovada_vira_a_referencia_da_voz(tmp_path):
    from web.clonagem import ler_vozes
    from web.referencia import Gravador

    lista = _lista(tmp_path)
    g = Gravador()
    mic = _MicrofoneFalso(_leitura())
    g.iniciar("L07", abrir=lambda: mic)
    g.esperar(60)
    r = g.estado()["resultado"]
    assert r["ok"], r
    assert r["fala_s"] >= 20 and mic.fechado
    assert all(c["ok"] for c in r["checagens"])
    gravacoes = list((tmp_path / "vozes" / "L07" / "referencia").glob("gravacao_*.wav"))
    refs = list((tmp_path / "vozes" / "referencias").glob("L07_*.wav"))
    assert len(gravacoes) == len(refs) == 1
    assert sf.info(refs[0]).subtype == "PCM_16" and sf.info(gravacoes[0]).subtype == "PCM_24"
    dados = json.loads(lista.read_text(encoding="utf-8"))
    l07 = next(v for v in dados["vozes"] if v["codigo"] == "L07")
    assert l07["referencia"] == str(refs[0].resolve()) and l07["clonagem_autorizada"]
    assert dados["vozes"][0] == {"codigo": "L05", "referencia": str(tmp_path / "antiga.wav"),
                                 "clonagem_autorizada": True, "observacao": "x"}, "as outras vozes não mudam"
    assert dados["apagar_clones_ate"] == "2026-11-19"
    assert [v.codigo for v in ler_vozes()] == ["L05", "L07"]


def test_leitura_curta_pede_regravar_e_nao_guarda_nada(tmp_path):
    from web.referencia import Gravador

    lista = _lista(tmp_path)
    antes = lista.read_text(encoding="utf-8")
    g = Gravador()
    g.iniciar("L07", abrir=lambda: _MicrofoneFalso(_leitura(fala_s=8.0)))
    g.esperar(60)
    r = g.estado()["resultado"]
    assert not r["ok"] and "fala" in r["motivo"] and "20 s" in r["motivo"]
    assert not (tmp_path / "vozes").exists(), "voz de tentativa reprovada não fica guardada"
    assert lista.read_text(encoding="utf-8") == antes


def test_parar_segue_mais_um_segundo_para_o_silencio_do_fim(tmp_path):
    from web.referencia import FOLGA_FINAL_S, Gravador

    _lista(tmp_path)
    g = Gravador()
    bloco = 4800

    def clicar(k):
        if k == 100:  # 10 s depois de abrir o microfone
            g.parar()

    g.iniciar("L07", abrir=lambda: _MicrofoneFalso(_leitura(fala_s=60.0), bloco, clicar))
    g.esperar(60)
    estado = g.estado()
    assert not estado["gravando"]
    # Até o clique: 10 s, menos o 1 s descartado, mais a folga do fim.
    assert estado["decorrido_s"] == pytest.approx(10.0 - 1.0 + FOLGA_FINAL_S, abs=0.11)


def test_microfone_que_falha_avisa_sem_derrubar(tmp_path):
    from web.referencia import Gravador

    _lista(tmp_path)

    def quebrado():
        raise RuntimeError("driver travou")

    g = Gravador()
    g.iniciar("L07", abrir=quebrado)
    g.esperar(10)
    assert g.estado()["resultado"]["motivo"] == "Não foi possível gravar pelo microfone."
    g._estado["gravando"] = True
    with pytest.raises(RuntimeError):
        g.iniciar("L05")


@pytest.fixture()
def cliente(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("multipart")
    torch = pytest.importorskip("torch")
    from fastapi.testclient import TestClient

    from src.config import load_config
    from src.models import build_model

    config = load_config("configs/baseline.yaml")
    torch.manual_seed(0)
    ckpt = tmp_path / "m.pt"
    torch.save({"model_state": build_model(config["model"]).state_dict(), "config": config,
                "threshold": 0.5}, ckpt)
    monkeypatch.setenv("DETECTOR_CONFIG", "configs/baseline.yaml")
    monkeypatch.setenv("DETECTOR_CHECKPOINT", str(ckpt))
    _lista(tmp_path)
    # Uma ferramenta "instalada" em qualquer máquina: no CI não há o ambiente do XTTS,
    # e a página não mostrava o formulário do clone.
    monkeypatch.setenv("DETECTOR_PYTHON_XTTS", sys.executable)
    monkeypatch.setenv("DETECTOR_PYTHON_CHATTERBOX", str(tmp_path / "nao_instalado.exe"))
    monkeypatch.setattr("src.capture.sources.MicrofoneWasapiSource",
                        lambda *a, **k: _MicrofoneFalso(_leitura()))
    import web.app as modulo

    importlib.reload(modulo)
    return TestClient(modulo.app), modulo


def test_aba_clonar_grava_a_referencia(cliente):
    c, modulo = cliente
    pagina = c.get("/clonar").text
    assert "Gravar a referência de" in pagina and 'value="L07"' in pagina and "L08" not in pagina
    assert "família inteira se reúne" in pagina, "o texto de referência aparece para ler"
    assert c.post("/api/referencia/gravar", json={"voz": "L08"}).status_code == 400
    assert c.post("/api/referencia/gravar", json={"voz": "L07"}).status_code == 200
    modulo.gravador_referencia().esperar(60)
    estado = c.get("/api/referencia").json()
    assert not estado["gravando"] and estado["resultado"]["ok"]
    formulario_do_clone = c.get("/clonar").text.split('id="referencia"')[0]
    assert 'value="L07"' in formulario_do_clone, "com a referência gravada, a L07 já pode ser clonada"


def test_gravar_nao_disputa_o_microfone_com_o_ao_vivo_nem_com_o_clone(cliente, monkeypatch):
    c, modulo = cliente
    monkeypatch.setitem(modulo._estado, "sessao", {"proc": types.SimpleNamespace(poll=lambda: None)})
    assert c.post("/api/referencia/gravar", json={"voz": "L07"}).status_code == 409
    modulo._estado.pop("sessao")
    monkeypatch.setitem(modulo._estado, "clonando", True)
    assert c.post("/api/referencia/gravar", json={"voz": "L07"}).status_code == 409
    modulo._estado.pop("clonando")
    monkeypatch.setitem(modulo._estado, "gravador", types.SimpleNamespace(gravando=lambda: True))
    assert c.post("/api/ao-vivo/iniciar", json={"fonte": "microfone"}).status_code == 409
    frase = "uma frase de teste com oito palavras ao todo"
    assert c.post("/api/clonar", json={"voz": "L05", "frase": frase}).status_code == 409
