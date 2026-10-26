"""Piloto de destilação (scripts/destilacao.py): sem download e sem GPU.

O professor é uma rede wav2vec 2.0 minúscula com pesos aleatórios; o aluno, um
transformer de uma camada. O que se confere é o encanamento: formatos,
alinhamento dos quadros, máscara da perda, checkpoint e o critério.
"""

import json

import numpy as np
import pytest
import soundfile as sf

from scripts.destilacao import MAX_ALVO, MAX_ENTRADA, entradas

torch = pytest.importorskip("torch")


def test_entradas_normalizadas():
    x = np.random.default_rng(0).standard_normal(16000 * 4).astype(np.float32) * 0.1
    e = entradas(x)
    assert e.shape == (MAX_ENTRADA, 140)
    assert np.allclose(e.mean(axis=0), 0, atol=1e-3) and np.allclose(e.std(axis=0), 1, atol=1e-2)


def test_aluno_reduz_para_o_passo_do_professor():
    from scripts.destilacao import criar_aluno

    aluno = criar_aluno(dim=32, camadas=1, cabecas=2, dim_alvo=8).eval()
    x = torch.randn(2, MAX_ENTRADA, 140)
    pred, n = aluno(x, torch.tensor([MAX_ENTRADA, 300]))
    assert pred.shape == (2, 201, 8) and n.tolist() == [201, 150]
    assert MAX_ALVO <= 201  # o professor tem 199 quadros em 4 s


def test_quadros_do_professor_sem_roda_lo():
    from scripts.destilacao import quadros_do_professor

    assert quadros_do_professor(64000) == MAX_ALVO   # 4 s
    assert quadros_do_professor(48000) == 149         # 3 s, como no teste com a rede pequena


def test_preparar_sem_professor(tmp_path):
    from scripts.destilacao import carregar_dados, preparar

    origem = _base(tmp_path, "e", n=2)
    r = preparar(origem, tmp_path / "sem", None, None)
    d = carregar_dados(tmp_path / "sem")
    assert r["janelas"] == 2 and d["alvos"] is None
    assert (d["n_alvo"] == 149).all() and d["entradas"].shape == (2, 1, MAX_ENTRADA, 140)  # 1 versão: a limpa


def test_pca_guarda_a_variancia_de_dados_de_posto_baixo():
    from scripts.destilacao import ajustar_pca

    rng = np.random.default_rng(0)
    base = rng.standard_normal((3, 12))
    alvos = np.zeros((20, MAX_ALVO, 12), np.float32)
    for i in range(20):
        alvos[i, :50] = rng.standard_normal((50, 3)) @ base + 5.0
    pca, fracao = ajustar_pca({"alvos": alvos, "n_alvo": np.full(20, 50)}, k=3, amostra=20)
    assert pca["componentes"].shape == (3, 12) and fracao > 0.999
    q = alvos[0, :50]
    volta = (q - pca["media"]) @ pca["componentes"].T @ pca["componentes"] + pca["media"]
    assert np.allclose(volta, q, atol=1e-3)


def test_perda_ignora_quadros_vazios():
    from scripts.destilacao import perda

    alvo = torch.randn(1, 6, 4)
    pred = alvo.clone()
    pred[:, 3:] = 100.0            # lixo só depois dos 3 quadros válidos
    p = perda(pred, alvo, torch.tensor([3]))
    assert abs(float(p) - float(-torch.nn.functional.logsigmoid(torch.tensor(1.0)))) < 1e-5


@pytest.fixture
def professor():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Wav2Vec2Config(
        hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
        conv_dim=(8, 8), conv_kernel=(10, 3), conv_stride=(5, 2), num_conv_pos_embeddings=8,
        num_conv_pos_embedding_groups=2, do_stable_layer_norm=True, feat_extract_norm="layer")
    return transformers.Wav2Vec2Model(cfg).eval(), torch.device("cpu")


def _base(tmp_path, nome, n=4, seed=0):
    """Metade humanos (ruído), metade falsos (ruído com um tom), como um .npz da parte 6."""
    rng = np.random.default_rng(seed)
    arquivos, classes = [], []
    for i in range(n):
        t = np.arange(16000 * 3) / 16000
        x = rng.standard_normal(t.size) * 0.1
        falso = i % 2 == 1
        if falso:
            x += 0.3 * np.sin(2 * np.pi * 440 * t)
        caminho = tmp_path / f"{nome}_{i}.wav"
        sf.write(caminho, x.astype(np.float32), 16000)
        arquivos.append(str(caminho))
        classes.append("A01" if falso else "bonafide")
    origem = tmp_path / f"{nome}.npz"
    np.savez(origem, arquivo=np.array(arquivos), classe=np.array(classes),
             rotulo=np.array([int(c != "bonafide") for c in classes], dtype=np.int8))
    return origem


def test_preparar_treinar_avaliar(tmp_path, professor, capsys):
    from scripts.destilacao import carregar_dados, main, preparar

    rede, dev = professor
    pastas = {}
    # 10 no treino: a sonda escolhe o C por validação cruzada em 5 partes
    for nome, seed, n in (("treino", 0, 10), ("dev", 1, 4), ("asv19_train", 2, 4)):
        pastas[nome] = tmp_path / nome          # o nome da pasta é o da régua (asv19_train)
        r = preparar(_base(tmp_path, nome, n=n, seed=seed), pastas[nome], rede, dev, camada=1)
        assert r == {"arquivos": n, "janelas": n}
    d = carregar_dados(pastas["treino"])
    assert d["entradas"].shape == (10, 1, MAX_ENTRADA, 140) and d["alvos"].shape == (10, MAX_ALVO, 16)
    assert (d["n_alvo"] > 0).all() and (d["n_entrada"] == 301).all()

    ck = tmp_path / "aluno.pt"
    assert main(["treinar", "--treino", str(pastas["treino"]), "--dev", str(pastas["dev"]), "--saida", str(ck),
                 "--epocas", "2", "--lote", "2", "--dispositivo", "cpu", "--dim", "32", "--camadas", "1"]) == 0
    hist = json.loads(ck.with_suffix(".json").read_text())["historico"]
    assert ck.is_file() and hist
    assert all(np.isfinite([h["perda_treino"], h["perda_dev"]]).all() for h in hist)

    sondas = {"representacoes": {
        "camada_01": {"sonda": {"media": [0.0] * 16, "escala": [1.0] * 16, "coef": [0.1] * 16,
                                "intercepto": 0.0, "C": 1.0},
                      "auc_media_dev": 0.99, "avaliar": {"asv19_train": {"auc": 0.98}}},
        "lfcc": {"auc_media_dev": 0.8, "avaliar": {"asv19_train": {"auc": 0.6}}}}}
    (tmp_path / "s.json").write_text(json.dumps(sondas))
    saida = tmp_path / "av.json"
    assert main(["avaliar", "--aluno", str(ck), "--sondas", str(tmp_path / "s.json"), "--treino",
                 str(pastas["treino"]), "--dev", str(pastas["dev"]), "--avaliar", str(pastas["asv19_train"]),
                 "--dispositivo", "cpu", "--json", str(saida)]) == 0
    r = json.loads(saida.read_text(encoding="utf-8"))
    assert set(r["sondas"]) == {"professor", "nova"} and "passa" in r["criterio"]
    assert "Critério" in capsys.readouterr().out

    # Piloto 2: versões degradadas e alvos por PCA; o dev continua com 16 dimensões
    from scripts.destilacao import ajustar_pca

    pca, _ = ajustar_pca(d, k=4, amostra=10)
    np.savez(tmp_path / "pca.npz", **pca)

    def degradar(x, rng):
        return x + rng.standard_normal(len(x)).astype(np.float32) * 0.05

    p2 = tmp_path / "treino2"
    preparar(_base(tmp_path, "t2", n=10, seed=5), p2, rede, dev, camada=1, degradar=degradar, versoes=2, pca=pca)
    d2 = carregar_dados(p2)
    assert d2["entradas"].shape == (10, 3, MAX_ENTRADA, 140) and d2["alvos"].shape[2] == 4
    assert not np.allclose(np.asarray(d2["entradas"][0, 0]), np.asarray(d2["entradas"][0, 1]))
    ck2 = tmp_path / "aluno2.pt"
    assert main(["treinar", "--treino", str(p2), "--dev", str(pastas["dev"]), "--saida", str(ck2),
                 "--epocas", "1", "--lote", "2", "--dispositivo", "cpu", "--dim", "32", "--camadas", "1"]) == 0
    assert main(["avaliar", "--aluno", str(ck2), "--sondas", str(tmp_path / "s.json"), "--treino", str(p2),
                 "--dev", str(pastas["dev"]), "--avaliar", str(pastas["asv19_train"]), "--dispositivo", "cpu",
                 "--json", str(tmp_path / "av2.json")]) == 0
    r2 = json.loads((tmp_path / "av2.json").read_text(encoding="utf-8"))
    assert set(r2["sondas"]) == {"professor", "nova"}
