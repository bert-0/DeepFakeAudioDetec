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
