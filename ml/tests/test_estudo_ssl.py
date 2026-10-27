"""Parte 6, o XLS-R como instrumento (scripts/estudo_ssl.py).

Sem download e sem GPU: as sondas usam vetores sintéticos, e o caminho do
modelo usa uma rede wav2vec 2.0 minúscula com pesos aleatórios.
"""

import numpy as np
import pytest
import soundfile as sf

from scripts.estudo_ssl import (SR, eer, explicar, extrair, faixas, janelas, medidas_simples, preparar,
                                representar_com, selecionar_grupos, sondar)


def _conjunto(n, informativa=2, seed=0, deslocamento=1.5, classes=("bonafide", "A01")):
    """Vetores falsos: só a camada `informativa` separa humano de falso."""
    rng = np.random.default_rng(seed)
    rotulo = np.array([0] * n + [1] * n, dtype=np.int8)
    camadas = rng.standard_normal((2 * n, 3, 8)).astype(np.float16)
    camadas[n:, informativa, :3] += deslocamento
    medidas = rng.standard_normal((2 * n, 2)).astype(np.float32)
    medidas[:, 0] = camadas[:, informativa, 0]  # a 1ª medida acompanha o que a camada usa
    return {"arquivo": np.array([f"f{i}.wav" for i in range(2 * n)]),
            "classe": np.array([classes[0]] * n + [classes[1]] * n), "rotulo": rotulo,
            "cnn": rng.standard_normal((2 * n, 4)).astype(np.float16), "camadas": camadas,
            "lfcc": rng.standard_normal((2 * n, 6)).astype(np.float32),
            "medidas": medidas, "nomes_medidas": np.array(["acompanha", "ruido"])}


def test_janelas_de_4_s_com_passo_de_2_s():
    assert len(janelas(np.zeros(3 * SR))) == 1
    js = janelas(np.zeros(10 * SR))
    assert len(js) == 4 and all(len(j) == 4 * SR for j in js)


def test_eer():
    assert eer([0, 1, 2], [5, 6, 7])[0] == 0.0
    e, _ = eer(np.arange(10), np.arange(10))
    assert 0.4 <= e <= 0.6


def test_sondar_acha_a_camada_informativa():
    r = sondar(_conjunto(60, seed=1), _conjunto(40, seed=2, classes=("bonafide", "A09")),
               {"ensaios": _conjunto(10, seed=3, classes=("L01_real", "L01_xtts"))})
    assert r["melhor"] == "camada_02"
    x = r["representacoes"]["camada_02"]
    assert x["auc_media_dev"] > 0.9 and x["eer_dev"] < 0.2
    assert x["avaliar"]["ensaios"]["classes"]["L01_xtts"]["auc"] > 0.8
    assert r["representacoes"]["lfcc"]["auc_media_dev"] < 0.75


def test_explicar_reconhece_a_medida_que_acompanha():
    treino, dev = _conjunto(60, seed=1), _conjunto(40, seed=2, classes=("bonafide", "A09"))
    r = sondar(treino, dev, {})
    ex = explicar(r, treino, {"dev": dev})
    c = ex["conjuntos"]["dev"]
    assert c["correlacao"]["acompanha"]["falsos"] > 0.2
    assert abs(c["correlacao"]["ruido"]["falsos"]) < 0.35
    assert c["so_medidas"]["auc"] > 0.7
    assert set(c["efeitos"]) == {"A09"}


def test_auc_por_codec(tmp_path):
    from scripts.estudo_ssl import ler_codecs, por_codec

    (tmp_path / "a.tsv").write_text(
        "E_1 E_0001 M - 0 x AC1 - bonafide -\nE_1 E_0002 M C04 7 x AC1 A17 spoof -\n"
        "E_1 E_0003 F C04 7 x AC1 - bonafide -\nE_1 E_0004 F - 0 x AC1 A18 spoof -\n", encoding="utf-8")
    codecs = ler_codecs(tmp_path / "a.tsv")
    assert codecs == {"E_0001": "-", "E_0002": "C04", "E_0003": "C04", "E_0004": "-"}
    arquivos = [f"data/x/flac/E_000{i}.flac" for i in (1, 2, 3, 4)]
    r = por_codec(np.array([0.0, 1.0, 2.0, 3.0]), arquivos, np.array([0, 1, 0, 1]), codecs)
    assert r["-"]["auc"] == 1.0 and r["C04"]["auc"] == 0.0     # no EnCodec o humano ficou acima
    assert r["C04"]["nome"] == "EnCodec (neural)" and r["C04"]["humanos"] == 1


def test_tamanho_do_efeito():
    from scripts.estudo_ssl import efeitos

    e = efeitos(_conjunto(400, seed=5, classes=("bonafide", "A12")))["A12"]
    assert 1.2 < e["acompanha"] < 1.8 and abs(e["ruido"]) < 0.2


def test_medidas_simples():
    t = np.arange(4 * SR) / SR
    vogal = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 20)).astype(np.float32)
    ruido = np.random.default_rng(0).standard_normal(4 * SR).astype(np.float32)
    mv, mr = medidas_simples(vogal / 3, 0.2), medidas_simples(ruido / 4, 0.0)
    assert mv["hnr_db"] > mr["hnr_db"] + 10
    assert mv["silencio_s"] == 0.2 and mv["duracao_s"] == 4.0
    assert mr["planura_alta"] > mv["planura_alta"]
    assert np.isnan(medidas_simples(ruido[:SR], 0.0)["linhas_fracao"])


def test_grupo_com_rotulo_errado():
    with pytest.raises(ValueError):
        selecionar_grupos(["x=pasta:clone"])


def test_detector_nos_mesmos_arquivos(tmp_path):
    torch = pytest.importorskip("torch")
    from scripts.estudo_ssl import comparar_detector, pontuar_detector
    from src.config import load_config
    from src.models import build_model

    config = load_config("configs/baseline.yaml")
    torch.manual_seed(0)
    ckpt = tmp_path / "m.pt"
    torch.save({"model_state": build_model(config["model"]).state_dict(), "config": config}, ckpt)
    lista = _wavs(tmp_path)
    s = pontuar_detector([str(a) for a, _, _ in lista], "configs/baseline.yaml", str(ckpt), "cpu")
    assert s.shape == (2,) and np.isfinite(s).all()
    conj = {"arquivo": np.array(["a", "b"]), "classe": np.array(["real", "clone"]),
            "rotulo": np.array([0, 1], dtype=np.int8)}
    r = comparar_detector({"dev": np.array([-1.0, 2.0])}, {"dev": conj})
    assert r["eer_dev"] == 0.0 and r["conjuntos"]["dev"]["classes"]["clone"]["certos"] == 1


@pytest.fixture
def rede_crua():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Wav2Vec2Config(
        hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
        conv_dim=(8, 8), conv_kernel=(10, 3), conv_stride=(5, 2), num_conv_pos_embeddings=8,
        num_conv_pos_embedding_groups=2, do_stable_layer_norm=True, feat_extract_norm="layer")
    return transformers.Wav2Vec2Model(cfg).eval(), torch.device("cpu")


@pytest.fixture
def rede_pequena(rede_crua):
    return representar_com(*rede_crua)


def test_ordem_dos_quadros():
    from scripts.estudo_ssl import BLOCO_QUADROS, ordem

    rng = np.random.default_rng(0)
    assert ordem("original", 12, rng) is None
    assert list(ordem("invertido", 4, rng)) == [3, 2, 1, 0]
    p = ordem("embaralhado", 50, rng)
    assert sorted(p) == list(range(50)) and list(p) != list(range(50))
    b = ordem("embaralhado_100ms", 23, rng)
    for i in range(0, 23, BLOCO_QUADROS):  # cada quadro fica no próprio bloco de 100 ms
        assert sorted(b[i:i + BLOCO_QUADROS]) == list(range(i, min(i + BLOCO_QUADROS, 23)))


def test_resumir_tempo():
    from scripts.estudo_ssl import VARIANTES, resumir_tempo

    def reg(rotulo, s):
        return {"conjunto": "dev", "rotulo": rotulo, "scores": {v: s for v in VARIANTES},
                "quadros": {"pausa": s, "meio": s, "fala": None, "transicao": s}}
    r = resumir_tempo([reg(0, -1.0), reg(0, -2.0), reg(1, 3.0)])["dev"]
    assert r["original"] == 1.0 and r["quadros_pausa"] == 1.0 and r["quadros_fala"] is None


def test_tempo_com_rede_pequena(tmp_path, rede_crua):
    from scripts.estudo_ssl import aplicar, tempo

    rede, dev = rede_crua
    lista = [(a, c, r, "ensaios") for a, c, r in _wavs(tmp_path)]
    rng = np.random.default_rng(1)
    sonda = {"media": [0.0] * 16, "escala": [1.0] * 16, "coef": rng.standard_normal(16).tolist(),
             "intercepto": 0.1, "C": 1.0}
    r = {"melhor": "camada_01", "representacoes": {"camada_01": {"sonda": sonda}}}
    regs = tempo(r, lista, rede, dev)
    # Sonda linear: o score original é o da média das camadas, como na extração.
    _, cam = representar_com(rede, dev)(preparar(lista[0][0])[0])
    esperado = aplicar({k: np.asarray(v) if isinstance(v, list) else v for k, v in sonda.items()},
                       cam[1][None])[0]
    assert abs(regs[0]["scores"]["original"] - esperado) < 1e-4
    assert regs[0]["scores"]["invertido"] != regs[0]["scores"]["original"]
    assert set(regs[0]["quadros"]) == {"pausa", "meio", "fala", "transicao"}
    assert not rede.feature_extractor._forward_hooks  # o gancho sai no fim


def _wavs(tmp_path):
    rng = np.random.default_rng(4)
    for nome, sr in (("a", 48000), ("b", 24000)):
        x = np.concatenate([np.zeros(sr // 2), rng.standard_normal(int(4.5 * sr)) * 0.1, np.zeros(sr // 2)])
        sf.write(tmp_path / f"{nome}.wav", x.astype(np.float32), sr)
    return [(tmp_path / "a.wav", "real", 0), (tmp_path / "b.wav", "clone", 1)]


def test_extrair_com_rede_pequena(tmp_path, rede_pequena):
    destino = tmp_path / "e.npz"
    extrair(_wavs(tmp_path), rede_pequena, destino)
    with np.load(destino) as d:
        assert d["camadas"].shape == (2, 3, 16) and d["cnn"].shape == (2, 8)
        assert d["lfcc"].shape == (2, 120)
        assert list(d["rotulo"]) == [0, 1]
        i = list(d["nomes_medidas"]).index("silencio_s")
        assert 0.8 < d["medidas"][0, i] < 1.2  # as duas pontas de 0,5 s saem no trim


def test_faixas_com_rede_pequena(tmp_path, rede_pequena):
    lista = _wavs(tmp_path)
    r = {"melhor": "camada_01", "representacoes": {"camada_01": {"sonda": {
        "media": [0.0] * 16, "escala": [1.0] * 16, "coef": [0.1] * 16, "intercepto": 0.0, "C": 1.0}}}}
    saida = faixas(r, lista, rede_pequena)
    assert set(saida) == {"real", "clone"}
    assert saida["clone"]["falso"] and len(saida["clone"]["sem 3000-4000"]) == 1


def test_grupos_aceitam_flac_e_mp3(tmp_path):
    """O BRSpeech-DF vem em FLAC: a extração pegava só .wav e não achava nada."""
    import soundfile as sf

    for nome in ("a.wav", "b.flac"):
        sf.write(tmp_path / nome, np.zeros(1600, np.float32), 16000)
    (tmp_path / "notas.txt").write_text("x", encoding="utf-8")
    lista = selecionar_grupos([f"g={tmp_path}:spoof"])
    assert [a.name for a, _, _ in lista] == ["a.wav", "b.flac"] and all(r == 1 for _, _, r in lista)
