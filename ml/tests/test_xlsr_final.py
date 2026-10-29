"""Modelo final na interface e no monitor (src/xlsr_final.py, src/capture/frases.py).

O XLS-R de verdade não roda nos testes: uma rede pequena devolve as mesmas saídas
(25 camadas), e a conferência é contra o caminho do treino (`estudo_ssl.py`).
"""

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")

DIM = 16
SR = 16000


class _RedeFalsa(torch.nn.Module):
    """Quadros de 20 ms, como o XLS-R; cada camada é uma função fixa dos quadros."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.w = torch.randn(320, DIM, generator=g) / 10

    def forward(self, x, output_hidden_states=True):
        t = x.shape[1] // 320
        q = x[0, :t * 320].reshape(t, 320) @ self.w
        return SimpleNamespace(extract_features=q[None, :, :8],
                               hidden_states=tuple(torch.tanh(q * (i + 1) / 25)[None] for i in range(25)))


def _cabeca(caminho, semente=0):
    from src.xlsr_final import criar_cabeca

    torch.manual_seed(semente)
    c = criar_cabeca(25, DIM, 120)
    with torch.no_grad():
        c.pesos.normal_()
        c.media_lfcc.normal_()
    torch.save({"estado": c.state_dict(), "config": c.config, "epoca": 1}, caminho)
    return caminho


def _modelo(caminho):
    from src.xlsr_final import ModeloFinal

    return ModeloFinal(caminho, "cpu", rede=_RedeFalsa().eval(), dev=torch.device("cpu"))


def _voz(segundos, semente=0, tom=220.0):
    """Algo com cara de fala: harmônicos acima do passa-altas, amplitude variando."""
    rng = np.random.default_rng(semente)
    t = np.arange(int(segundos * SR)) / SR
    x = sum(np.sin(2 * np.pi * tom * k * t) / k for k in range(1, 6))
    x *= 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 3 * t))
    return (0.2 * x / np.abs(x).max() + 0.002 * rng.standard_normal(t.size)).astype(np.float32)


def _ruido(segundos, semente=1):
    return (0.0005 * np.random.default_rng(semente).standard_normal(int(segundos * SR))).astype(np.float32)


def _fala_continua(segundos):
    """Sílabas de 0,5 s com respiros de 0,15 s, curtos demais para fechar a frase."""
    partes = []
    for i in range(int(segundos / 0.65)):
        partes += [_voz(0.5, semente=i), _ruido(0.15, semente=100 + i)]
    return np.concatenate(partes)


def test_pontuar_wav_igual_ao_caminho_do_treino(tmp_path):
    from scripts.estudo_ssl import lfcc_medio, preparar, representar_com
    from scripts.xlsr_congelado import carregar, pontuar

    arq = tmp_path / "a.wav"
    sf.write(arq, np.concatenate([_ruido(0.5), _voz(9.0), _ruido(0.5)]), SR)
    ck = _cabeca(tmp_path / "c.pt")

    wav, _ = preparar(arq)
    _, camadas = representar_com(_RedeFalsa().eval(), torch.device("cpu"))(wav)
    d = {"camadas": camadas.astype(np.float16)[None], "lfcc": lfcc_medio(wav)[None], "rotulo": np.zeros(1)}
    cabeca, _, dev = carregar(ck, "cpu")
    esperado = float(pontuar(cabeca, d, dev)[0])

    p = _modelo(ck).pontuar_arquivo(arq)
    assert p.logodds == pytest.approx(esperado, abs=1e-6)
    assert [round(t, 3) for t, _ in p.janelas] == [0.0, 2.0, 4.0]
    assert p.duracao_s == pytest.approx(10.0) and p.silencio_s > 0


def test_fala_curta_demais(tmp_path):
    from src.xlsr_final import FalaCurta

    with pytest.raises(FalaCurta):
        _modelo(_cabeca(tmp_path / "c.pt")).pontuar_wav(_voz(0.1))


def test_limiar_confere_os_pesos_da_cabeca(tmp_path):
    from src.xlsr_final import gravar_limiar, ler_limiar, sigmoide

    ck = _cabeca(tmp_path / "c.pt")
    destino = gravar_limiar(tmp_path / "l.json", str(ck), -0.5, origem="calibrado com L02", regra="maior")
    lim = ler_limiar(destino)
    assert lim.limiar == -0.5 and lim.probabilidade == pytest.approx(sigmoide(-0.5))
    assert lim.descricao() == "calibrado com L02"
    assert not lim.calibrado_por_frase and lim.probabilidade_frase == lim.probabilidade
    assert "sem calibração por frase" in lim.descricao_frase()
    lim = ler_limiar(gravar_limiar(tmp_path / "f.json", str(ck), -0.5, origem="x", regra="maior",
                                   limiar_frase=-1.0))
    assert lim.calibrado_por_frase and lim.probabilidade_frase == pytest.approx(sigmoide(-1.0))
    _cabeca(ck, semente=1)  # treinada de novo: o limiar antigo não vale mais
    with pytest.raises(ValueError, match="mudou"):
        ler_limiar(destino)


def test_limiar_pela_maior_voz_real(tmp_path):
    from scripts.limiar_xlsr import limiar_por_reais, pontuar_pastas

    pasta = tmp_path / "L02"
    pasta.mkdir()
    for i in range(3):
        # Como as gravações reais: ruído da sala antes e depois da fala.
        sf.write(pasta / f"f{i}.wav", np.concatenate([_ruido(0.5, i), _voz(3.0 + i, semente=i, tom=150 + 40 * i),
                                                      _ruido(1.0, 10 + i)]), SR)
    sf.write(pasta / "curto.wav", _voz(0.1), SR)
    modelo = _modelo(_cabeca(tmp_path / "c.pt"))
    scores = pontuar_pastas(modelo, {"L02": pasta})
    assert set(scores["L02"]) == {"f0.wav", "f1.wav", "f2.wav"}, "o curto fica de fora"
    assert all(len(x["frases"]) == 1 for x in scores["L02"].values())
    limiar, limiar_frase, locutores = limiar_por_reais(scores)
    assert limiar == max(x["arquivo"] for x in scores["L02"].values())
    assert limiar_frase == max(x["frases"][0] for x in scores["L02"].values())
    assert locutores["L02"]["n"] == 3 and locutores["L02"]["n_frases"] == 3


def test_limiar_do_dev_grava_o_json(tmp_path):
    from scripts.limiar_xlsr import main
    from src.xlsr_final import ler_limiar

    rng = np.random.default_rng(0)
    n = 40
    y = (np.arange(n) % 2).astype(np.int8)
    npz = tmp_path / "dev.npz"
    np.savez(npz, arquivo=np.array([f"a{i}" for i in range(n)]), classe=np.array(["x"] * n), rotulo=y,
             camadas=rng.standard_normal((n, 25, DIM)).astype(np.float16),
             lfcc=rng.standard_normal((n, 120)).astype(np.float32))
    saida = tmp_path / "l.json"
    ck = _cabeca(tmp_path / "c.pt")
    assert main(["--cabeca", str(ck), "--dev", str(npz), "--saida", str(saida), "--dispositivo", "cpu"]) == 0
    assert "dev" in ler_limiar(saida).origem


# --------------------------------------------------------------- frase por frase

class _Gravador:
    """Modelo falso: guarda o que recebeu; o log-odds é a duração."""

    def __init__(self):
        self.recebidos = []

    def pontuar_wav(self, wav, sample_rate=SR):
        from src.xlsr_final import Pontuacao

        self.recebidos.append(wav)
        return Pontuacao(logodds=len(wav) / sample_rate, duracao_s=len(wav) / sample_rate, silencio_s=0.0)


def _rodar(analisador, sinal, bloco=1600):
    leituras = []
    for i in range(0, len(sinal), bloco):
        leituras += list(analisador.processar(sinal[i:i + bloco]))
    return leituras + list(analisador.finalizar())


def test_frases_separadas_por_pausa_e_tosse_ignorada():
    from src.capture.frases import AnalisadorFrases

    sinal = np.concatenate([_ruido(1.0), _voz(2.0), _ruido(1.5, 2), _voz(0.3, 3), _ruido(1.5, 4),
                            _voz(2.5, 5, tom=180)])
    modelo = _Gravador()
    leituras = _rodar(AnalisadorFrases(modelo, limiar=0.5), sinal)
    assert len(leituras) == 2, "a tosse de 0,3 s não vira frase"
    assert leituras[0].instante == pytest.approx(0.7, abs=0.05), "começa 0,3 s antes da fala"
    assert leituras[1].instante == pytest.approx(6.3 - 0.3, abs=0.05)
    assert 2.0 < len(modelo.recebidos[0]) / SR < 3.3, "a fala, a margem e a pausa que fechou"
    assert leituras[1].logodds == pytest.approx(len(modelo.recebidos[1]) / SR)
    assert all(x.confiavel and x.peso == 1.0 for x in leituras)


def test_fala_sem_pausa_e_cortada():
    from src.capture.frases import MAX_FRASE_S, AnalisadorFrases

    modelo = _Gravador()
    leituras = _rodar(AnalisadorFrases(modelo), np.concatenate([_ruido(0.5), _fala_continua(20.0)]))
    assert len(leituras) == 2
    assert len(modelo.recebidos[0]) / SR == pytest.approx(MAX_FRASE_S, abs=0.05)


def test_silencio_digital_nao_vira_frase():
    from src.capture.frases import AnalisadorFrases

    assert _rodar(AnalisadorFrases(_Gravador()), np.zeros(5 * SR, np.float32)) == []


# --------------------------------------------------------------- interface e monitor

@pytest.fixture()
def limiar_xlsr(tmp_path, monkeypatch):
    from src.xlsr_final import gravar_limiar

    monkeypatch.setattr("src.features.xlsr.carregar_rede",
                        lambda *a, **k: (_RedeFalsa().eval(), torch.device("cpu")))
    ck = _cabeca(tmp_path / "xlsr_congelado.pt")
    return gravar_limiar(tmp_path / "limiar.json", str(ck), -0.5, origem="calibrado com as vozes de L02",
                         regra="maior voz real")


def test_enviar_arquivo_com_o_modelo_final(limiar_xlsr, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("multipart")
    from fastapi.testclient import TestClient

    from src.xlsr_final import sigmoide

    monkeypatch.setenv("DETECTOR_XLSR", str(limiar_xlsr))
    monkeypatch.setenv("DETECTOR_BANCO", str(tmp_path / "h.db"))
    import web.app as modulo

    importlib.reload(modulo)
    cliente = TestClient(modulo.app)
    arq = tmp_path / "voz.wav"
    sf.write(arq, np.concatenate([_ruido(0.3), _voz(7.0), _ruido(0.3)]), SR)
    with open(arq, "rb") as fh:
        dados = cliente.post("/api/analisar", files={"arquivo": ("voz.wav", fh, "audio/wav")}).json()
    esperado = _modelo(json.loads(limiar_xlsr.read_text(encoding="utf-8"))["cabeca"]).pontuar_arquivo(arq)
    assert dados["limiar"] == pytest.approx(sigmoide(-0.5))
    assert len(dados["janelas"]) == len(esperado.janelas) == 2
    assert dados["indicio_de_sintese"] == (esperado.logodds >= -0.5)
    pagina = cliente.get(f"/analises/{dados['id']}").text
    assert "xlsr_congelado" in pagina and "Score do arquivo" in pagina
    sobre = cliente.get("/sobre").text
    assert "XLS-R 300M" in sobre and "0,921" in sobre and "18,99%" not in sobre
    assert "frase por frase" in cliente.get("/ao-vivo").text

    with open(tmp_path / "curto.wav", "wb") as fh:
        sf.write(fh, _voz(0.1), SR)
    with open(tmp_path / "curto.wav", "rb") as fh:
        r = cliente.post("/api/analisar", files={"arquivo": ("curto.wav", fh, "audio/wav")})
    assert r.status_code == 422 and "pouca fala" in r.json()["detail"]


def test_ao_vivo_roda_o_modelo_final(limiar_xlsr, tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setenv("DETECTOR_XLSR", str(limiar_xlsr))
    import web.app as modulo

    importlib.reload(modulo)
    capturado = {}
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: capturado.setdefault("cmd", cmd))
    modulo._iniciar_monitor(tmp_path / "s.json", tmp_path / "s.wav", tmp_path / "s.parar",
                            tmp_path / "s.log", fonte="microfone")
    cmd = capturado["cmd"]
    assert cmd[cmd.index("--xlsr") + 1] == str(limiar_xlsr) and "--checkpoint" not in cmd
    assert modulo._modelo_ao_vivo() == "xlsr_congelado"


def test_monitor_frase_por_frase_num_arquivo(limiar_xlsr, tmp_path):
    import monitor

    from src.xlsr_final import gravar_limiar, sigmoide

    cabeca = json.loads(limiar_xlsr.read_text(encoding="utf-8"))["cabeca"]
    limiar_xlsr = gravar_limiar(tmp_path / "frase.json", cabeca, -0.5, origem="calibrado com L02",
                                regra="maior", limiar_frase=-1.0)
    arq = tmp_path / "sessao.wav"
    sf.write(arq, np.concatenate([_ruido(0.5), _voz(3.0), _ruido(1.5, 2), _voz(4.0, 3, tom=170), _ruido(0.5)]), SR)
    saida = tmp_path / "s.json"
    assert monitor.main(["--xlsr", str(limiar_xlsr), "--arquivo", str(arq), "--json", str(saida),
                         "--device", "cpu"]) == 0
    dados = json.loads(saida.read_text(encoding="utf-8"))
    assert dados["unidade"] == "frase" and len(dados["leituras"]) == 2
    assert dados["limiar"] == pytest.approx(sigmoide(-1.0)), "o ao vivo usa o limiar da frase"
    assert dados["duracao_s"] == pytest.approx(9.5, abs=0.05)
    assert monitor.main(["--xlsr", str(limiar_xlsr), "--checkpoint", "x.pt"]) == 1


def test_arquivo_que_comeca_falando_nao_perde_o_comeco():
    """Sem aquecer o piso, a própria fala vira piso e a frase não é achada."""
    from src.capture.frases import frases_de

    sinal = np.concatenate([_fala_continua(6.0), np.zeros(SR, np.float32)])
    modelo = _Gravador()
    leituras = frases_de(modelo, sinal, SR)
    assert len(leituras) == 1 and leituras[0].instante == 0.0
    assert frases_de(_Gravador(), sinal, SR, aquecer=False)[0].instante > 0.0


def test_pontuar_por_frases_e_a_media_das_frases(tmp_path):
    from src.xlsr_final import FalaCurta

    modelo = _modelo(_cabeca(tmp_path / "c.pt"))
    sinal = np.concatenate([_ruido(1.0), _voz(3.0), _ruido(1.5, 2), _voz(4.0, 3, tom=170), _ruido(1.0, 4)])
    p = modelo.pontuar_por_frases(sinal)
    assert p.unidade == "frase" and len(p.janelas) == 2
    assert p.logodds == pytest.approx(np.mean([z for _, z in p.janelas]))
    with pytest.raises(FalaCurta):
        modelo.pontuar_por_frases(_ruido(3.0))


def test_json_do_arquivo_por_frase_e_da_pasta_local(tmp_path):
    from src.xlsr_final import MODELO_HF, gravar_limiar, ler_limiar, modelo_local, sigmoide

    ck = _cabeca(tmp_path / "c.pt")
    with pytest.raises(ValueError):
        gravar_limiar(tmp_path / "x.json", str(ck), -0.5, origem="x", regra="r", arquivo_por_frase=True)
    lim = ler_limiar(gravar_limiar(tmp_path / "l.json", str(ck), -0.5, origem="x", regra="r",
                                   limiar_frase=-1.0, arquivo_por_frase=True))
    assert lim.arquivo_por_frase and lim.probabilidade_arquivo == pytest.approx(sigmoide(-1.0))
    assert modelo_local("modelo/que-nao-existe") is None
    d = json.loads((tmp_path / "l.json").read_text(encoding="utf-8"))
    d["modelo_local"] = str(tmp_path)
    (tmp_path / "l.json").write_text(json.dumps(d), encoding="utf-8")
    assert ler_limiar(tmp_path / "l.json").fonte_modelo == str(tmp_path)
    d["modelo_local"] = str(tmp_path / "apagada")
    (tmp_path / "l.json").write_text(json.dumps(d), encoding="utf-8")
    assert ler_limiar(tmp_path / "l.json").fonte_modelo == MODELO_HF, "pasta sumiu: volta ao nome"


def test_interface_carrega_o_xlsr_so_na_primeira_analise(limiar_xlsr, tmp_path, monkeypatch):
    import web.analise as analise

    cargas = []
    monkeypatch.setattr("src.features.xlsr.carregar_rede",
                        lambda *a, **k: cargas.append(1) or (_RedeFalsa().eval(), torch.device("cpu")))
    d = analise.Detector(xlsr=str(limiar_xlsr))
    assert d.xlsr and d.modelo == "xlsr_congelado" and cargas == []
    arq = tmp_path / "v.wav"
    sf.write(arq, np.concatenate([_ruido(0.5), _voz(5.0), _ruido(1.0)]), SR)
    d.analisar(arq)
    d.analisar(arq)
    assert cargas == [1]


def test_enviar_arquivo_por_frase_usa_o_limiar_da_frase(tmp_path, monkeypatch):
    import web.analise as analise
    from src.xlsr_final import gravar_limiar, sigmoide

    monkeypatch.setattr("src.features.xlsr.carregar_rede",
                        lambda *a, **k: (_RedeFalsa().eval(), torch.device("cpu")))
    ck = _cabeca(tmp_path / "c.pt")
    lim = gravar_limiar(tmp_path / "l.json", str(ck), -0.5, origem="calibrado com L02", regra="r",
                        limiar_frase=-1.0, arquivo_por_frase=True)
    arq = tmp_path / "v.wav"
    sf.write(arq, np.concatenate([_ruido(1.0), _voz(3.0), _ruido(1.5, 2), _voz(4.0, 3, tom=170), _ruido(1.0, 4)]), SR)
    r = analise.Detector(xlsr=str(lim)).analisar(arq)
    assert r.limiar == pytest.approx(sigmoide(-1.0)) and "frase por frase" in r.origem_limiar
    assert len(r.janelas) == 2


def test_modelo_local_acha_a_pasta_no_cache_sem_readme(tmp_path):
    from src.xlsr_final import modelo_local

    raiz = tmp_path / "models--org--xlsr"
    pasta = raiz / "snapshots" / "abc123"
    pasta.mkdir(parents=True)
    (raiz / "refs").mkdir()
    (raiz / "refs" / "main").write_text("abc123")
    (pasta / "config.json").write_text("{}")
    assert modelo_local("org/xlsr", cache=str(tmp_path)) is None, "sem os pesos não serve"
    (pasta / "model.safetensors").write_bytes(b"0")
    assert modelo_local("org/xlsr", cache=str(tmp_path)) == str(pasta).replace("\\", "/")


def test_estudo_corte_fala_compara_os_dois_caminhos(tmp_path, monkeypatch):
    from scripts.estudo_corte_fala import main, resumir

    r = resumir(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.5, np.nan, 2.5, 3.5]),
                np.array([0, 0, 1, 1]), np.array([1, 0, 2, 1]))
    assert r["sem_frase"] == 1 and r["mais_de_uma_frase"] == 1
    assert r["reais"]["mediana"] == pytest.approx(0.5) and r["auc_b"] == 1.0

    monkeypatch.setattr("src.features.xlsr.carregar_rede",
                        lambda *a, **k: (_RedeFalsa().eval(), torch.device("cpu")))
    arquivos = []
    for i in range(4):
        arq = tmp_path / f"a{i}.wav"
        sf.write(arq, np.concatenate([_ruido(0.5, i), _voz(3.0, i, tom=150 + 30 * i), _ruido(1.0, 9 + i)]), SR)
        arquivos.append(str(arq))
    npz = tmp_path / "dev.npz"
    rng = np.random.default_rng(0)
    np.savez(npz, arquivo=np.array(arquivos), classe=np.array(["bonafide", "bonafide", "A01", "A01"]),
             rotulo=np.array([0, 0, 1, 1], np.int8), camadas=rng.standard_normal((4, 25, DIM)).astype(np.float16),
             lfcc=rng.standard_normal((4, 120)).astype(np.float32))
    saida = tmp_path / "r.json"
    assert main(["--cabeca", str(_cabeca(tmp_path / "c.pt")), "--dev", f"x={npz}", "--dispositivo", "cpu",
                 "--json", str(saida)]) == 0
    x = json.loads(saida.read_text(encoding="utf-8"))["x"]
    assert x["n"] == 4 and x["sem_frase"] == 0 and "auc_b" in x


def test_placar_do_ensaio_com_o_modelo_final(tmp_path, monkeypatch):
    from scripts.teste_voz_propria import main
    from src.xlsr_final import gravar_limiar, sigmoide

    monkeypatch.setattr("src.features.xlsr.carregar_rede",
                        lambda *a, **k: (_RedeFalsa().eval(), torch.device("cpu")))
    pasta = tmp_path / "ensaio"
    for sub, tom in (("real", 150), ("clone", 260), ("clone_ar", 200)):
        (pasta / sub).mkdir(parents=True)
        for i in range(2):
            sf.write(pasta / sub / f"f{i + 1:02d}.wav",
                     np.concatenate([_ruido(1.0, i), _voz(3.0, i, tom=tom + 10 * i), _ruido(1.0, 7 + i)]), SR)
    lim = gravar_limiar(tmp_path / "l.json", str(_cabeca(tmp_path / "c.pt")), -0.5, origem="calibrado com L02",
                        regra="r", limiar_frase=-1.0)
    with pytest.raises(SystemExit):
        main(["--pasta", str(pasta), "--xlsr", str(lim)])
    saida = tmp_path / "r.json"
    assert main(["--pasta", str(pasta), "--xlsr", str(lim), "--microfone", "frase", "--json", str(saida)]) == 0
    r = json.loads(saida.read_text(encoding="utf-8"))
    assert r["placar"]["arquivo"]["limiar"] == pytest.approx(sigmoide(-0.5))
    assert r["placar"]["microfone"]["limiar"] == pytest.approx(sigmoide(-1.0))
    assert r["placar"]["microfone"]["unidade"] == "frase" and r["placar"]["arquivo"]["unidade"] == "arquivo"
    assert r["placar"]["arquivo"]["real"]["n"] == 2 and r["placar"]["microfone"]["clone_ar"]["n"] == 2
    assert len(r["arquivos"]["microfone"]) == 4


def test_histerese_mantem_a_fala_fraca_depois_do_comeco():
    """Com eco e ruído, a fala cai para perto do piso entre as sílabas: depois de entrar
    a 10 dB, ela fica enquanto passar de 5 dB."""
    from src.capture.frases import frases_de

    rng = np.random.default_rng(0)
    t = np.arange(int(1.7 * SR)) / SR
    tom = np.sin(2 * np.pi * 1000 * t) * np.sqrt(2)
    forte, fraca = int(0.2 * SR), int(1.5 * SR)
    fala = np.concatenate([0.04 * tom[:forte], 0.0224 * tom[forte:forte + fraca]])   # 12 e 7 dB acima
    sinal = 0.01 * rng.standard_normal(int(4.7 * SR))
    sinal[SR:SR + len(fala)] += fala
    leituras = frases_de(_Gravador(), sinal.astype(np.float32), SR)
    assert len(leituras) == 1 and leituras[0].instante == pytest.approx(0.7, abs=0.05)


def test_monitor_grava_o_json_e_para_mesmo_sem_fala(limiar_xlsr, tmp_path, monkeypatch):
    """Com o som do computador em silêncio, a fonte só devolve blocos vazios: o JSON sai
    no primeiro (a aba sai de "Iniciando a captura") e o pedido de parada funciona."""
    import monitor

    class _FonteMuda:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def blocos(self):
            for _ in range(200):
                yield np.zeros(0, np.float32)
            raise AssertionError("o monitor não parou")

    vistos = []
    original = monitor.gravar_json

    def espiar(agregador, destino, *a, **k):
        vistos.append(k.get("ativo"))
        return original(agregador, destino, *a, **k)

    monkeypatch.setattr(monitor, "abrir", lambda args, sr: _FonteMuda())
    monkeypatch.setattr(monitor, "gravar_json", espiar)
    parar = tmp_path / "parar"
    parar.touch()
    saida = tmp_path / "s.json"
    assert monitor.main(["--xlsr", str(limiar_xlsr), "--json", str(saida), "--parar-com", str(parar),
                         "--device", "cpu"]) == 0
    assert vistos[0] is True, "o JSON sai antes da primeira frase"
    dados = json.loads(saida.read_text(encoding="utf-8"))
    assert dados["leituras"] == [] and dados["ativo"] is False


def test_limite_de_segundos_vale_sem_fala(limiar_xlsr, tmp_path, monkeypatch):
    import monitor

    class _FonteRuido:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def blocos(self):
            for i in range(100):
                yield _ruido(0.1, i)
            raise AssertionError("o limite não parou o monitor")

    monkeypatch.setattr(monitor, "abrir", lambda args, sr: _FonteRuido())
    assert monitor.main(["--xlsr", str(limiar_xlsr), "--segundos", "2", "--device", "cpu"]) == 0


def test_monitor_da_interface_sem_buffer_e_em_utf8(limiar_xlsr, tmp_path, monkeypatch):
    import subprocess
    import time

    monkeypatch.setenv("DETECTOR_XLSR", str(limiar_xlsr))
    import web.app as modulo

    importlib.reload(modulo)
    capturado = {}
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: capturado.update(cmd=cmd, kw=kw))
    modulo._iniciar_monitor(tmp_path / "s.json", tmp_path / "s.wav", tmp_path / "s.parar",
                            tmp_path / "s.log", fonte="sistema")
    assert capturado["cmd"][1] == "-u" and capturado["kw"]["env"]["PYTHONIOENCODING"] == "utf-8"

    # Registro antigo, no código de página do Windows: a aba mostra o erro em vez de cair.
    log = tmp_path / "velho.log"
    log.write_bytes("Nenhuma janela com áudio".encode("cp1252"))
    morto = SimpleNamespace(poll=lambda: 1)
    monkeypatch.setitem(modulo._estado, "sessao", {"json": tmp_path / "nao.json", "log": log, "proc": morto,
                                                   "inicio": time.time(), "analise_id": None, "fonte": "sistema"})
    r = modulo.api_ao_vivo()
    assert r["rodando"] is False and "Nenhuma janela" in r["erro"]


def _tom(segundos, rms, hz=1000.0):
    t = np.arange(int(segundos * SR)) / SR
    return (rms * np.sqrt(2) * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _alimentar(analisador, sinal, bloco=1600):
    avisos = []
    for i in range(0, len(sinal), bloco):
        list(analisador.processar(sinal[i:i + bloco]))
        avisos.append(analisador.aviso)
    return avisos


def test_aviso_quando_ha_som_mas_nenhuma_fala():
    """Som só de graves (vibração na mesa, abaixo do passa-altas da fala), 10 dB acima
    do ruído: não vira frase, e a tela avisa em vez de calar (28/10/2026)."""
    from src.capture.frases import AnalisadorFrases

    rng = np.random.default_rng(0)
    ruido = lambda s: (0.003 * rng.standard_normal(int(s * SR))).astype(np.float32)  # noqa: E731
    fraco = lambda s: ruido(s) + _tom(s, 0.009, hz=80.0)  # noqa: E731
    a = AnalisadorFrases(_Gravador())
    avisos = _alimentar(a, np.concatenate([ruido(3.0), fraco(4.0)]))
    assert avisos[25] is None and avisos[-1] == "sem_fala"
    assert _alimentar(a, ruido(4.0))[-1] is None, "o aviso some quando o som some"


def test_aviso_de_saturacao_dura_uns_segundos():
    from src.capture.frases import AnalisadorFrases

    rng = np.random.default_rng(1)
    estouro = np.clip(1.5 * np.sin(2 * np.pi * 200 * np.arange(SR // 2) / SR), -1, 1).astype(np.float32)
    a = AnalisadorFrases(_Gravador())
    avisos = _alimentar(a, np.concatenate([(0.003 * rng.standard_normal(SR)).astype(np.float32), estouro]))
    assert avisos[-1] == "saturado"
    assert _alimentar(a, (0.003 * rng.standard_normal(4 * SR)).astype(np.float32))[-1] is None


def test_monitor_e_interface_mostram_o_aviso(limiar_xlsr, tmp_path, monkeypatch):
    import time

    import monitor

    rng = np.random.default_rng(2)

    class _Fonte:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def blocos(self):
            sinal = np.concatenate([(0.003 * rng.standard_normal(3 * SR)).astype(np.float32),
                                    (0.003 * rng.standard_normal(6 * SR)).astype(np.float32) + _tom(6.0, 0.009, hz=80.0)])
            for i in range(0, len(sinal), 1600):
                yield sinal[i:i + 1600]

    vistos = []
    original = monitor.gravar_json
    monkeypatch.setattr(monitor, "gravar_json", lambda *a, **k: vistos.append(k.get("aviso")) or original(*a, **k))
    monkeypatch.setattr(monitor, "abrir", lambda args, sr: _Fonte())
    saida = tmp_path / "s.json"
    assert monitor.main(["--xlsr", str(limiar_xlsr), "--json", str(saida), "--device", "cpu"]) == 0
    assert "sem_fala" in vistos

    # A aba lê o código no JSON e mostra o texto, só enquanto a sessão roda.
    monkeypatch.setenv("DETECTOR_XLSR", str(limiar_xlsr))
    import web.app as modulo

    importlib.reload(modulo)
    dados = json.loads(saida.read_text(encoding="utf-8"))
    dados["aviso"] = "sem_fala"
    saida.write_text(json.dumps(dados), encoding="utf-8")
    vivo = SimpleNamespace(poll=lambda: None)
    monkeypatch.setitem(modulo._estado, "sessao", {"json": saida, "log": tmp_path / "s.log", "proc": vivo,
                                                   "inicio": time.time(), "analise_id": None, "fonte": "microfone"})
    assert "nenhuma fala" in modulo.api_ao_vivo()["aviso"]


def test_xlsr_so_carrega_do_cache(monkeypatch):
    """Nada de download escondido: sem o modelo no cache, erro claro."""
    transformers = pytest.importorskip("transformers")
    from src.features import xlsr

    pedidos = []

    def from_pretrained(modelo, **kw):
        pedidos.append(kw)
        raise OSError("não achou")

    monkeypatch.setattr(transformers.Wav2Vec2Model, "from_pretrained", staticmethod(from_pretrained))
    with pytest.raises(OSError, match="HF_HOME"):
        xlsr.carregar_rede("cpu", "org/modelo")
    assert pedidos == [{"local_files_only": True}]
