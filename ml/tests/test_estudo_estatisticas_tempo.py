"""Parte 9: estatísticas temporais dos quadros (scripts/estudo_estatisticas_tempo.py)."""

import json

import numpy as np
import pytest
import soundfile as sf

from scripts.estudo_estatisticas_tempo import criterio, estatisticas, extrair, main, sondar


def test_autocorrelacao_separa_lento_de_tremido():
    rng = np.random.default_rng(0)
    lento = np.cumsum(rng.standard_normal((400, 2)), axis=0)   # muda devagar
    tremido = rng.standard_normal((400, 2))                     # muda a cada quadro
    el, et = estatisticas(lento, (1, 8)), estatisticas(tremido, (1, 8))
    assert (el["ac1"] > 0.9).all() and (np.abs(et["ac1"]) < 0.2).all()
    assert (el["delta"] < et["delta"]).all()
    assert set(el) == {"media", "desvio", "delta", "ac1", "ac8"}


def test_ordem_muda_so_as_estatisticas_temporais():
    x = np.cumsum(np.random.default_rng(1).standard_normal((300, 3)), axis=0)
    embaralhado = x[np.random.default_rng(2).permutation(300)]
    a, b = estatisticas(x, (1,)), estatisticas(embaralhado, (1,))
    assert np.allclose(a["media"], b["media"]) and np.allclose(a["desvio"], b["desvio"])
    assert (a["ac1"] - b["ac1"] > 0.5).all()


def _conjunto(n, seed, classes=("bonafide", "A01"), dim=6):
    """Só a estatística temporal separa: a média e o desvio são iguais nas duas classes."""
    rng = np.random.default_rng(seed)
    rotulo = np.array([0] * n + [1] * n, dtype=np.int8)
    d = {"arquivo": np.array([f"f{i}" for i in range(2 * n)]), "rotulo": rotulo,
         "classe": np.array([classes[0]] * n + [classes[1]] * n),
         "lfcc": rng.standard_normal((2 * n, 4)).astype(np.float32)}
    for base, tempo in (("cnn_media_desvio", "cnn_tempo"), ("lfcc_media_desvio", "lfcc_tempo")):
        md = rng.standard_normal((2 * n, dim))
        ac = rng.standard_normal((2 * n, dim)) * 0.3
        ac[n:, :2] += 1.0
        d[base] = md.astype(np.float32)
        d[tempo] = np.column_stack([md, ac]).astype(np.float32)
    d["cnn_media"] = d["cnn_media_desvio"][:, :3]
    return d


def test_sondar_e_criterio():
    r = sondar(_conjunto(60, 1), _conjunto(40, 2, ("bonafide", "A12")),
               {"asv19_train": _conjunto(40, 3)})
    assert r["cnn_tempo"]["auc_media_dev"] > 0.9 and r["cnn_media_desvio"]["auc_media_dev"] < 0.75
    c = criterio(r, "asv19_train")
    assert c["cnn_tempo"]["passa"] and c["lfcc_tempo"]["passa"]


def test_extrair_com_rede_pequena(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Wav2Vec2Config(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        conv_dim=(8, 8), conv_kernel=(10, 3), conv_stride=(5, 2), num_conv_pos_embeddings=8,
        num_conv_pos_embedding_groups=2, do_stable_layer_norm=True, feat_extract_norm="layer")
    rede = transformers.Wav2Vec2Model(cfg).eval()
    arquivos = []
    for i in range(2):
        caminho = tmp_path / f"a{i}.wav"
        sf.write(caminho, (np.random.default_rng(i).standard_normal(16000 * 3) * 0.1).astype(np.float32), 16000)
        arquivos.append(str(caminho))
    origem = tmp_path / "o.npz"
    np.savez(origem, arquivo=np.array(arquivos), classe=np.array(["bonafide", "A01"]),
             rotulo=np.array([0, 1], dtype=np.int8), lfcc=np.zeros((2, 120), np.float32))
    destino = tmp_path / "d.npz"
    extrair(origem, destino, rede, torch.device("cpu"))
    with np.load(destino) as d:
        assert d["cnn_media"].shape == (2, 8) and d["cnn_tempo"].shape == (2, 8 * 7)
        assert d["lfcc_tempo"].shape == (2, 20 * 8) and d["lfcc_media_desvio"].shape == (2, 40)


def test_linha_de_comando(tmp_path, capsys):
    for nome, (n, seed, cl) in {"t": (60, 1, ("bonafide", "A01")), "dv": (40, 2, ("bonafide", "A12")),
                                "asv19_train": (40, 3, ("bonafide", "A01"))}.items():
        np.savez(tmp_path / f"{nome}.npz", **_conjunto(n, seed, cl))
    saida = tmp_path / "s.json"
    assert main(["sondar", "--treino", str(tmp_path / "t.npz"), "--dev", str(tmp_path / "dv.npz"),
                 "--avaliar", str(tmp_path / "asv19_train.npz"), "--json", str(saida)]) == 0
    assert json.loads(saida.read_text(encoding="utf-8"))["criterio"]["cnn_tempo"]["passa"]
    assert "Critério" in capsys.readouterr().out
