"""Conferência antes da apresentação (web/conferencia.py, a página e o scripts/conferir.py).

Nada de hardware: microfone, GPU e saídas de som são falsos.
"""

import importlib
import time
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from web import conferencia as C  # noqa: E402


def _limiar(tmp_path, locutores=None, frase=-1.0, pasta=None, por_frase=None):
    import json

    from src.xlsr_final import criar_cabeca, gravar_limiar

    c = criar_cabeca(25, 16, 120)
    ck = tmp_path / "c.pt"
    torch.save({"estado": c.state_dict(), "config": c.config}, ck)
    caminho = gravar_limiar(tmp_path / "l.json", str(ck), -0.5, origem="x", regra="r",
                            locutores=locutores or {}, limiar_frase=frase,
                            arquivo_por_frase=frase is not None if por_frase is None else por_frase)
    if pasta is not None:
        d = json.loads(caminho.read_text(encoding="utf-8"))
        d["modelo_local"] = str(pasta)
        caminho.write_text(json.dumps(d), encoding="utf-8")
    return str(caminho)


def _estados(itens):
    return {i.nome: i.estado for i in itens}


def test_modelo_sem_xlsr_e_com_limiar_de_uma_ou_duas_vozes(tmp_path):
    assert _estados(C.modelo_final(None)) == {"Modelo": "atencao"}
    um = C.modelo_final(_limiar(tmp_path, {"L02": {}}, pasta=tmp_path))
    assert _estados(um) == {"Limiar": "atencao", "XLS-R no disco": "ok"}
    assert "1 voz" in um[0].detalhe and "−" not in um[0].acao
    dois = C.modelo_final(_limiar(tmp_path, {"L02": {}, "L03": {}}, pasta=tmp_path / "sumiu"))
    assert _estados(dois) == {"Limiar": "ok", "XLS-R no disco": "atencao"}
    assert _estados(C.modelo_final(_limiar(tmp_path, frase=None)))["Limiar"] == "falha"


def test_enviar_arquivo_pelo_arquivo_inteiro_pede_atencao(tmp_path):
    itens = C.modelo_final(_limiar(tmp_path, {"L02": {}, "L03": {}}, pasta=tmp_path, por_frase=False))
    assert _estados(itens)["Enviar arquivo"] == "atencao"


def test_carga_do_modelo_ok_ou_falha(tmp_path):
    caminho = _limiar(tmp_path, {"L02": {}, "L03": {}})
    assert _estados(C.modelo_final(caminho, carregar=lambda: None))["Carga do modelo"] == "ok"

    def quebra():
        raise OSError("sem cache")

    assert _estados(C.modelo_final(caminho, carregar=quebra))["Carga do modelo"] == "falha"
    (tmp_path / "l.json").write_text("{}", encoding="utf-8")
    assert _estados(C.modelo_final(str(tmp_path / "l.json"))) == {"Modelo": "falha"}


class _Fonte:
    def __init__(self, nivel, modo="compartilhado"):
        self.nivel, self.modo, self.nome = nivel, modo, "Grupo de microfones"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def blocos(self):
        rng = np.random.default_rng(0)
        while True:
            yield (self.nivel * rng.standard_normal(1600)).astype(np.float32)


@pytest.mark.parametrize("nivel,modo,esperado", [
    (0.003, "compartilhado", "ok"),          # ~ −50 dBFS: o ruído normal da sala
    (0.0, "compartilhado", "falha"),         # mudo
    (0.05, "compartilhado", "atencao"),      # ~ −26 dBFS: alguém falando
    (0.003, "exclusivo", "atencao"),
])
def test_microfone(nivel, modo, esperado):
    item = C.microfone(lambda sr: _Fonte(nivel, modo))
    assert item.estado == esperado and "dBFS" in item.detalhe


def test_microfone_que_nao_abre():
    def nao_abre(sr):
        raise RuntimeError("sem permissão")

    assert C.microfone(nao_abre).estado == "falha"


def test_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert C.gpu().estado == "atencao"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "GTX 1650")
    for livre, esperado in ((3.8e9, "ok"), (2.5e9, "atencao"), (1.0e9, "falha")):
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda livre=livre: (livre, 4.29e9))
        assert C.gpu().estado == esperado


def test_saida_de_som_e_disco(monkeypatch, tmp_path):
    assert C.saida_de_som(lambda: {"padrao": {"sistema": "Alto-falantes"}}).estado == "ok"
    assert C.saida_de_som(lambda: {"padrao": {}}).estado == "falha"

    def quebra():
        raise RuntimeError("sem soundcard")

    assert C.saida_de_som(quebra).estado == "atencao"
    for livre, esperado in ((50e9, "ok"), (3e9, "atencao"), (1e9, "falha")):
        monkeypatch.setattr(C.shutil, "disk_usage", lambda p, livre=livre: SimpleNamespace(free=livre))
        assert C.disco(tmp_path).estado == esperado


def test_resumo_e_situacao():
    itens = [C.Item("a", "ok", ""), C.Item("b", "lembrete", "")]
    assert C.situacao(itens) == "ok" and C.resumo(itens) == "pronto para apresentar"
    itens.append(C.Item("c", "atencao", ""))
    assert C.situacao(itens) == "atencao" and "1 ponto" in C.resumo(itens)
    itens.append(C.Item("d", "falha", ""))
    assert C.situacao(itens) == "falha" and "1 falha" in C.resumo(itens)


def test_com_o_ao_vivo_rodando_nao_abre_o_microfone(monkeypatch):
    monkeypatch.setattr(C, "microfone", lambda *a, **k: pytest.fail("o microfone está ocupado pelo ao vivo"))
    monkeypatch.setattr(C, "gpu", lambda: C.Item("GPU", "ok", ""))
    itens = C.conferir_tudo(None, rodando=True, listar=lambda: {"padrao": {"sistema": "x"}})
    assert _estados(itens)["Ao vivo"] == "atencao"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("multipart")
    from fastapi.testclient import TestClient

    from src.config import load_config
    from src.models import build_model

    config = load_config("configs/baseline.yaml")
    ckpt = tmp_path / "m.pt"
    torch.save({"model_state": build_model(config["model"]).state_dict(), "config": config, "threshold": 0.5}, ckpt)
    monkeypatch.setenv("DETECTOR_CONFIG", "configs/baseline.yaml")
    monkeypatch.setenv("DETECTOR_CHECKPOINT", str(ckpt))
    monkeypatch.setenv("DETECTOR_BANCO", str(tmp_path / "h.db"))
    monkeypatch.delenv("DETECTOR_XLSR", raising=False)
    monkeypatch.setattr(C, "conferir_tudo", lambda *a, **k: [C.Item("GPU", "ok", "GTX"),
                                                              C.Item("Limiar", "atencao", "1 voz", "calibre")])
    import web.app as modulo

    importlib.reload(modulo)
    return TestClient(modulo.app), modulo


def _esperar(cliente):
    for _ in range(100):
        d = cliente.get("/api/conferir").json()
        if not d.get("rodando"):
            return d
        time.sleep(0.05)
    pytest.fail("a conferência não terminou")


def test_conferencia_automatica_e_de_novo(app, monkeypatch):
    cliente, modulo = app
    monkeypatch.setenv("DETECTOR_CONFERIR", "0")
    modulo._conferir_ao_iniciar()
    assert modulo._estado.get("conferencia") is None, "desligada por DETECTOR_CONFERIR=0"
    monkeypatch.setenv("DETECTOR_CONFERIR", "1")
    modulo._conferir_ao_iniciar()
    d = _esperar(cliente)
    assert d["situacao"] == "atencao" and len(d["itens"]) == 2 and d["quando"]
    pagina = cliente.get("/").text
    assert 'id="situacao-conferencia"' in pagina and "Atenção" in pagina
    assert cliente.get("/conferir").status_code == 200
    assert cliente.post("/api/conferir").status_code == 200
    assert _esperar(cliente)["resumo"].startswith("pronto, com 1 ponto")


def test_comando_do_terminal(monkeypatch, capsys):
    from scripts import conferir

    monkeypatch.setattr(C, "conferir_tudo", lambda *a, **k: [C.Item("Microfone", "falha", "mudo", "confira")])
    assert conferir.main(["--sem-modelo"]) == 1
    saida = capsys.readouterr().out
    assert "FALHA" in saida and "Microfone: mudo" in saida


def test_microfone_travado_vira_falha_sem_travar_a_conferencia():
    """O driver de áudio travado: abrir o microfone não volta (29/10/2026)."""
    import threading

    preso = threading.Event()

    def abre_e_trava(sr):
        preso.wait(5)
        return _Fonte(0.003)

    inicio = time.time()
    item = C.microfone(abre_e_trava, tempo_maximo_s=0.3)
    assert item.estado == "falha" and "não respondeu" in item.detalhe and time.time() - inicio < 2
    preso.set()
