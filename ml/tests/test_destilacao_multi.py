"""Piloto 4 (scripts/destilacao_multi.py): professor minúsculo, detector falso, sem GPU."""

import json

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture
def professor():
    torch.manual_seed(0)
    cfg = transformers.Wav2Vec2Config(
        hidden_size=16, num_hidden_layers=4, num_attention_heads=2, intermediate_size=32,
        conv_dim=(8, 8), conv_kernel=(10, 3), conv_stride=(5, 2), num_conv_pos_embeddings=8,
        num_conv_pos_embedding_groups=2, do_stable_layer_norm=True, feat_extract_norm="layer")
    return transformers.Wav2Vec2Model(cfg).eval(), torch.device("cpu")


@pytest.fixture
def detector_falso(tmp_path):
    from src.config import load_config
    from src.models import build_model

    config = load_config("configs/baseline.yaml")
    torch.manual_seed(1)
    caminho = tmp_path / "detector.pt"
    torch.save({"model_state": build_model(config["model"]).state_dict(), "config": config}, caminho)
    return "configs/baseline.yaml", str(caminho)


def _origem(tmp_path, nome, n=4, seed=0, segundos=3):
    rng = np.random.default_rng(seed)
    arquivos, classes = [], []
    for i in range(n):
        t = np.arange(16000 * segundos) / 16000
        x = rng.standard_normal(t.size) * 0.1 + (0.3 * np.sin(2 * np.pi * 440 * t) if i % 2 else 0)
        caminho = tmp_path / f"{nome}_{i}.wav"
        sf.write(caminho, x.astype(np.float32), 16000)
        arquivos.append(str(caminho))
        classes.append("A01" if i % 2 else "bonafide")
    return {"arquivo": np.array(arquivos), "classe": np.array(classes),
            "rotulo": np.array([int(c != "bonafide") for c in classes], dtype=np.int8)}


def _preparado(tmp_path, professor, nome, n=4, seed=0, segundos=3):
    from scripts.destilacao import preparar

    rede, dev = professor
    preparar(_origem(tmp_path, nome, n, seed, segundos), tmp_path / nome, rede, dev, camada=1)
    return tmp_path / nome


def test_acrescentar_grava_a_camada_extra_e_o_detector_nas_mesmas_janelas(tmp_path, professor, detector_falso):
    from scripts.destilacao_multi import Detector, acrescentar, carregar, pca_camada

    rede, dev = professor
    pasta = _preparado(tmp_path, professor, "t", segundos=7)   # 7 s: 2 janelas por arquivo
    pca, fracao = pca_camada(pasta, rede, dev, camada=4, n_janelas=6, k=8)
    assert pca["componentes"].shape == (8, 16) and 0 < fracao <= 1
    acrescentar(pasta, rede, dev, pca, Detector(*detector_falso, dev), camada=4)
    d = carregar(pasta)
    assert d["alvos_extra"].shape == (len(d["n_alvo"]), 199, 8)
    assert d["detector"].shape == (len(d["n_alvo"]),) and np.isfinite(d["detector"]).all()
    assert len(d["n_alvo"]) == 8


def test_acrescentar_recusa_pasta_de_outra_origem(tmp_path, professor, detector_falso):
    from scripts.destilacao_multi import Detector, acrescentar

    pasta = _preparado(tmp_path, professor, "t")
    meta = dict(np.load(pasta / "meta.npz"))
    meta["arquivo"] = meta["arquivo"][:2]
    np.savez(pasta / "meta.npz", **meta)
    with pytest.raises(ValueError, match="não é desta origem"):
        acrescentar(pasta, None, None, None, Detector(*detector_falso, torch.device("cpu")))


def test_mascara_quando_uma_pasta_nao_tem_os_alvos_extras(tmp_path, professor, detector_falso):
    from scripts.destilacao_multi import (Detector, acrescentar, carregar, estatisticas, juntar_multi, lote_multi,
                                          perda_total, criar_aluno4)

    rede, dev = professor
    com = _preparado(tmp_path, professor, "com")
    sem = _preparado(tmp_path, professor, "sem", seed=1)
    acrescentar(com, rede, dev, None, Detector(*detector_falso, dev), camada=4)
    j = juntar_multi([carregar(com), carregar(sem)])
    assert list(j["tem_extra"]) == [True] * 4 + [False] * 4 and list(j["tem_detector"]) == [True] * 4 + [False] * 4
    est = estatisticas(j, amostra=8)
    a = lote_multi(j, np.arange(8), est, "cpu")
    assert a[4].shape == (8, 199, 16)
    assert a[5][:4].all() and not a[5][4:].any(), "a máscara tira da perda as janelas sem o alvo"
    assert a[7][:4].all() and not a[7][4:].any()
    aluno = criar_aluno4(dim=32, camadas=1, dim_alvo=16, dim_extra=16)
    total, partes = perda_total(aluno(a[0], a[1]), a)
    assert torch.isfinite(total) and set(partes) == {"c7", "c4", "detector", "classe"}


def test_treinar_a_partir_de_um_aluno_e_avaliar(tmp_path, professor, detector_falso):
    from scripts.destilacao import main as main_destilacao
    from scripts.destilacao_multi import Detector, acrescentar, main

    rede, dev = professor
    treino = _preparado(tmp_path, professor, "treino", n=6)
    dev_p = _preparado(tmp_path, professor, "dev", n=4, seed=2)
    det = Detector(*detector_falso, dev)
    for pasta in (treino, dev_p):
        acrescentar(pasta, rede, dev, None, det, camada=4)
    inicial = tmp_path / "aluno3b.pt"
    assert main_destilacao(["treinar", "--treino", str(treino), "--dev", str(dev_p), "--saida", str(inicial),
                            "--epocas", "1", "--lote", "2", "--dispositivo", "cpu", "--dim", "32",
                            "--camadas", "1"]) == 0
    ck = tmp_path / "aluno4.pt"
    assert main(["treinar", "--treino", str(treino), "--dev", str(dev_p), "--inicial", str(inicial),
                 "--saida", str(ck), "--epocas", "1", "--lote", "2", "--dispositivo", "cpu"]) == 0
    h = json.loads(ck.with_suffix(".json").read_text())["historico"][0]
    assert set(h["partes_treino"]) == {"c7", "c4", "detector", "classe"} and np.isfinite(h["perda_dev"])
    saida = tmp_path / "av.json"
    assert main(["avaliar", "--aluno", str(ck), "--dev", str(dev_p), "--regua", str(treino), "--dispositivo", "cpu",
                 "--json", str(saida)]) == 0
    r = json.loads(saida.read_text(encoding="utf-8"))
    assert "treino" in r["reguas"] and len(r["scores"]["treino"]) == 6
