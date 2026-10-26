"""Parte 10: surpresa do modelo de fala natural (scripts/estudo_surpresa.py).

Sem download: uma rede de pré-treino wav2vec 2.0 minúscula com pesos aleatórios.
"""

import numpy as np
import pytest

from scripts.estudo_surpresa import criterio, resumir, surpresa_janela


@pytest.fixture
def rede_pequena():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Wav2Vec2Config(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        conv_dim=(8, 8), conv_kernel=(10, 3), conv_stride=(5, 2), num_conv_pos_embeddings=8,
        num_conv_pos_embedding_groups=2, do_stable_layer_norm=True, feat_extract_norm="layer",
        codevector_dim=8, proj_codevector_dim=8, num_codevector_groups=2, num_codevectors_per_group=4,
        num_negatives=5, mask_time_prob=0.2, mask_time_length=3)
    return transformers.Wav2Vec2ForPreTraining(cfg).eval(), torch.device("cpu")


def test_minha_perda_bate_com_a_do_modelo(rede_pequena):
    rede, dev = rede_pequena
    x = np.random.default_rng(0).standard_normal(4000).astype(np.float32)
    r = surpresa_janela(rede, dev, x, np.random.default_rng(1), repeticoes=3)
    assert r["quadros"] > 0 and 0 <= r["acerto"] <= 1 and -1 <= r["similaridade"] <= 1
    assert abs(r["perda"] - r["perda_modelo"]) < 1e-3


def test_mesma_semente_mesmo_resultado(rede_pequena):
    rede, dev = rede_pequena
    x = np.random.default_rng(2).standard_normal(4000).astype(np.float32)
    a = surpresa_janela(rede, dev, x, np.random.default_rng(7), repeticoes=2)
    b = surpresa_janela(rede, dev, x, np.random.default_rng(7), repeticoes=2)
    assert a == b


def _reg(conj, classe, rotulo, perda):
    return {"conjunto": conj, "classe": classe, "rotulo": rotulo, "perda": perda,
            "acerto": 1 - perda / 10, "similaridade": 0.5}


def test_resumo_e_criterio():
    regs = ([_reg("asv5_dev", "bonafide", 0, p) for p in (1.0, 1.2, 1.1)]
            + [_reg("asv5_dev", "A12", 1, p) for p in (2.0, 2.5)]
            + [_reg("asv19_train", "bonafide", 0, p) for p in (1.0, 1.3)]
            + [_reg("asv19_train", "A01", 1, p) for p in (1.9, 2.2)])
    r = resumir(regs)
    assert r["asv5_dev"]["perda"]["auc"] == 1.0 and r["asv5_dev"]["acerto"]["auc"] == 0.0
    assert r["asv5_dev"]["perda"]["classes"]["A12"]["auc"] == 1.0
    c = criterio(r)
    assert c["passa"] and c["sentido"] == "falsos surpreendem mais"
    regs[-1]["perda"] = regs[-2]["perda"] = 1.15  # 2019 no meio: não passa
    assert not criterio(resumir(regs))["passa"]


def test_banco_de_ruido_so_pega_as_pausas(tmp_path):
    import soundfile as sf

    from scripts.estudo_surpresa import banco_de_ruido

    rng = np.random.default_rng(0)
    t = np.arange(16000 * 4) / 16000
    fala = (np.sin(2 * np.pi * 0.5 * t) > 0)          # 1 s de "fala", 1 s de pausa
    x = rng.standard_normal(t.size) * np.where(fala, 0.3, 0.003)
    sf.write(tmp_path / "r.wav", x.astype(np.float32), 16000)
    ruido = banco_de_ruido([tmp_path / "r.wav"])
    assert len(ruido) > 0.3 * 16000
    assert np.sqrt(np.mean(ruido ** 2)) < 0.01         # nada da "fala" entrou


def test_canais():
    from scripts.estudo_surpresa import aplicar_canal

    rng = np.random.default_rng(0)
    t = np.arange(16000) / 16000
    wav = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    assert aplicar_canal(wav, "limpo", rng) is wav
    y = aplicar_canal(wav, "ruido_branco_20db", np.random.default_rng(1))
    assert len(y) == len(wav) and np.abs(y).max() <= 1.0
    assert 0.9 < np.corrcoef(y, wav)[0, 1] < 1.0   # ruído 20 dB abaixo: ainda é o mesmo som
    assert len(aplicar_canal(wav, "opus", rng)) == len(wav)
    ruido = rng.standard_normal(500).astype(np.float32)
    assert len(aplicar_canal(wav, "microfone_20db", rng, ruido)) == len(wav)
    with pytest.raises(ValueError):
        aplicar_canal(wav, "microfone_20db", rng)


def test_criterio_do_canal():
    from scripts.estudo_surpresa import criterio_canal

    def linha(limpo, *outras):
        return {"asv5_dev": {"limpo": {"auc": limpo}, **{c: {"auc": a} for c, a in
                zip(("ruido_branco_20db", "opus", "microfone_20db"), outras)}}}
    assert criterio_canal(linha(0.9, 0.85, 0.82, 0.5))["passa"]
    assert not criterio_canal(linha(0.9, 0.7, 0.85, 0.5))["passa"]


def test_combinar_com_o_detector():
    from scripts.estudo_surpresa import combinar

    regs = [dict(_reg("asv5_dev", c, rot, p), arquivo=f"d{i}")
            for i, (c, rot, p) in enumerate([("bonafide", 0, 1.0), ("bonafide", 0, 1.4), ("A12", 1, 2.2),
                                              ("A09", 1, 1.2)])]
    regs.append(dict(_reg("ensaios", "L01_real", 0, 3.0), arquivo="e0"))
    regs.append(dict(_reg("ensaios", "L01_xtts", 1, 1.5), arquivo="e1"))
    # o detector erra o A12 e acerta o A09; a surpresa, o contrário
    detector = {"d0": -10.0, "d1": -8.0, "d2": -12.0, "d3": 5.0, "e0": -9.0, "e1": 4.0}
    r = combinar(regs, detector)
    dev = r["conjuntos"]["asv5_dev"]
    assert dev["z_detector"]["classes"]["A12"] == 0.0 and dev["z_perda"]["classes"]["A12"] == 1.0
    assert r["norma"]["perda"][0] == 1.2  # média dos humanos do dev
    assert set(r["conjuntos"]) == {"asv5_dev", "ensaios"}
