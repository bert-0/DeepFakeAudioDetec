"""Testes dos modelos dos três incrementos (formato da saída)."""

import pytest
import torch

from src.models import build_model


def _features(batch=2, freq_lfcc=60, freq_spec=40, frames=64):
    return {
        "lfcc": torch.randn(batch, 1, freq_lfcc, frames),
        "spectrogram": torch.randn(batch, 1, freq_spec, frames),
    }


@pytest.mark.parametrize("name", ["baseline_cnn", "fusion", "attention"])
@pytest.mark.parametrize("pooling", ["avg", "stats"])
def test_model_forward_output_shape(name, pooling):
    model = build_model({"name": name, "n_classes": 2, "dropout": 0.3, "pooling": pooling})
    model.eval()
    with torch.no_grad():
        logits = model(_features())
    assert logits.shape == (2, 2)


@pytest.mark.parametrize("name", ["baseline_cnn", "fusion", "attention"])
def test_stats_pooling_doubles_feature_dim(name):
    """O pooling estatístico deve gerar um classificador com o dobro da entrada."""
    avg = build_model({"name": name, "pooling": "avg"})
    stats = build_model({"name": name, "pooling": "stats"})
    in_avg = [m for m in avg.classifier if isinstance(m, torch.nn.Linear)][0].in_features
    in_stats = [m for m in stats.classifier if isinstance(m, torch.nn.Linear)][0].in_features
    assert in_stats == 2 * in_avg


@pytest.mark.parametrize("name", ["baseline_cnn", "fusion", "attention"])
def test_model_backward_no_nan(name):
    """Gradientes finitos — protege o sqrt do pooling estatístico contra NaN."""
    model = build_model({"name": name, "pooling": "stats"})
    logits = model(_features())
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize(
    "name,expected",
    [
        # Chaves gravadas pelos checkpoints treinados antes do pooling
        # configurável. Um checkpoint antigo precisa continuar carregando com
        # pooling="avg" — por isso a numeração das camadas não pode mudar.
        ("baseline_cnn", {"classifier.2.weight", "classifier.2.bias"}),
        ("fusion", {"classifier.1.weight", "classifier.4.weight"}),
        ("attention", {"lfcc_attn.score.weight", "spec_attn.score.weight"}),
    ],
)
def test_avg_pooling_keeps_legacy_state_dict_keys(name, expected):
    keys = set(build_model({"name": name, "pooling": "avg"}).state_dict())
    assert expected <= keys, f"chaves ausentes: {sorted(expected - keys)}"


@pytest.mark.parametrize("name", ["baseline_cnn", "fusion", "attention"])
def test_checkpoint_roundtrip_with_avg_pooling(name):
    """Salvar e recarregar com pooling='avg' não pode exigir strict=False."""
    src = build_model({"name": name, "pooling": "avg"})
    dst = build_model({"name": name, "pooling": "avg"})
    dst.load_state_dict(src.state_dict())  # levanta RuntimeError se divergir


@pytest.mark.parametrize("name", ["baseline_cnn", "fusion", "attention"])
def test_deeper_encoder_forward(name):
    """Encoder de 4 blocos (config v3) precisa rodar nos três modelos."""
    model = build_model({"name": name, "pooling": "stats",
                         "channels": [32, 64, 128, 128]})
    model.eval()
    with torch.no_grad():
        # 4 blocos = 4 maxpools: 60->3 e 80->5 na frequência, 400->25 no tempo
        logits = model(_features(freq_lfcc=60, freq_spec=80, frames=400))
    assert logits.shape == (2, 2)


def test_channels_default_matches_legacy():
    """Sem a chave `channels`, o modelo deve manter a arquitetura antiga."""
    legacy = build_model({"name": "baseline_cnn", "pooling": "avg"})
    explicit = build_model({"name": "baseline_cnn", "pooling": "avg",
                            "channels": [16, 32, 64]})
    assert set(legacy.state_dict()) == set(explicit.state_dict())
    legacy.load_state_dict(explicit.state_dict())


@pytest.mark.parametrize("pool_cls", ["StatsPool", "AttentiveStatsPool"])
def test_stats_pool_survives_large_values(pool_cls):
    """Valores grandes fazem a soma de quadrados estourar o fp16.

    Reproduz a divergência que ocorria sob AMP: com x~1e3, x^2~1e6 passa do
    máximo do float16 (~65504). O pooling calcula em float32, então a saída
    precisa continuar finita.
    """
    from src.models import blocks

    pool = getattr(blocks, pool_cls)(8)
    x = torch.randn(2, 8, 10, 50) * 1000.0
    out = pool(x)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("pool_cls", ["StatsPool", "AttentiveStatsPool"])
def test_stats_pool_zero_variance_has_finite_grad(pool_cls):
    """Variância exatamente zero: o sqrt teria gradiente infinito sem o clamp."""
    from src.models import blocks

    pool = getattr(blocks, pool_cls)(4)
    x = torch.ones(2, 4, 6, 20, requires_grad=True)  # variância temporal = 0
    pool(x).sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("name", ["baseline_cnn", "fusion", "attention"])
def test_stats_models_finite_with_large_inputs(name):
    """Modelo inteiro com entradas de magnitude alta não pode gerar NaN."""
    model = build_model({"name": name, "pooling": "stats"})
    model.eval()
    with torch.no_grad():
        big = {k: v * 500.0 for k, v in _features().items()}
        logits = model(big)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("name", ["baseline_cnn", "fusion", "attention"])
@pytest.mark.parametrize("pooling", ["avg", "stats", "freq_stats"])
def test_lcnn_encoder_forward(name, pooling):
    """A LCNN precisa servir aos três incrementos, em todos os poolings."""
    model = build_model({"name": name, "encoder": "lcnn", "pooling": pooling})
    model.eval()
    with torch.no_grad():
        logits = model(_features(freq_lfcc=60, freq_spec=80, frames=400))
    assert logits.shape == (2, 2)


def test_mfm_halves_channels_taking_maximum():
    from src.models.blocks import MFM

    x = torch.tensor([[[[1.0]], [[5.0]], [[9.0]], [[2.0]]]])   # (1, 4, 1, 1)
    out = MFM()(x)
    assert out.shape == (1, 2, 1, 1)
    # max(canal 0, canal 2) = 9 ; max(canal 1, canal 3) = 5
    assert out[0, 0, 0, 0] == 9.0
    assert out[0, 1, 0, 0] == 5.0


def test_freq_stats_pool_preserves_frequency_information():
    """Dois sinais com a MESMA média espectral, mas energia em faixas diferentes.

    O StatsPool (que faz média na frequência) não consegue distingui-los;
    o FreqStatsPool precisa distinguir.
    """
    from src.models.blocks import FreqStatsPool, StatsPool

    baixa = torch.zeros(1, 1, 8, 20)
    baixa[:, :, :4, :] = 1.0          # energia nas frequências baixas
    alta = torch.zeros(1, 1, 8, 20)
    alta[:, :, 4:, :] = 1.0           # energia nas frequências altas

    plano = StatsPool(1)
    assert torch.allclose(plano(baixa), plano(alta))       # indistinguíveis

    preserva = FreqStatsPool(1, freq_bins=4)
    assert not torch.allclose(preserva(baixa), preserva(alta))


def test_freq_stats_pool_output_dim_independent_of_input_freq():
    """freq_bins fixo: a saída não depende do tamanho do eixo de frequência."""
    from src.models.blocks import FreqStatsPool

    pool = FreqStatsPool(8, freq_bins=4)
    for n_freq in (7, 16, 60):
        out = pool(torch.randn(2, 8, n_freq, 30))
        assert out.shape == (2, pool.out_dim)


@pytest.mark.parametrize("name", ["baseline_cnn", "fusion", "attention"])
def test_default_encoder_keeps_legacy_state_dict(name):
    """Sem a chave `encoder`, o modelo deve continuar sendo a CNN de antes."""
    legacy = build_model({"name": name, "pooling": "avg"})
    explicit = build_model({"name": name, "pooling": "avg", "encoder": "cnn"})
    assert set(legacy.state_dict()) == set(explicit.state_dict())
    legacy.load_state_dict(explicit.state_dict())


def test_unknown_encoder_raises():
    with pytest.raises(ValueError):
        build_model({"name": "baseline_cnn", "encoder": "inexistente"})


@pytest.mark.parametrize("name", ["fusion", "attention"])
def test_branches_select_which_features_feed_each_branch(name):
    """Com branches=('lfcc','lfcc_hi') o modelo consome as duas resoluções."""
    model = build_model({"name": name, "encoder": "lcnn", "pooling": "freq_stats",
                         "branches": ["lfcc", "lfcc_hi"]})
    model.eval()
    feats = {"lfcc": torch.randn(2, 1, 60, 400),
             "lfcc_hi": torch.randn(2, 1, 60, 400)}
    with torch.no_grad():
        assert model(feats).shape == (2, 2)


@pytest.mark.parametrize("name", ["fusion", "attention"])
def test_branches_default_to_tc1_pair(name):
    """Sem a chave `branches`, mantém LFCC + espectrograma (o par do TC1)."""
    model = build_model({"name": name})
    assert model.branches == ("lfcc", "spectrogram")


def test_each_branch_receives_a_different_feature():
    """Os dois ramos não podem estar lendo a mesma entrada por engano."""
    model = build_model({"name": "fusion", "encoder": "lcnn",
                         "pooling": "freq_stats", "branches": ["lfcc", "lfcc_hi"]})
    model.eval()
    a = torch.randn(1, 1, 60, 200)
    b = torch.randn(1, 1, 60, 200)
    with torch.no_grad():
        trocado = model({"lfcc": b, "lfcc_hi": a})
        normal = model({"lfcc": a, "lfcc_hi": b})
    # Se ambos os ramos lessem a mesma chave, trocar as entradas não mudaria nada.
    assert not torch.allclose(normal, trocado)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model({"name": "inexistente"})


def test_unknown_pooling_raises():
    with pytest.raises(ValueError):
        build_model({"name": "baseline_cnn", "pooling": "inexistente"})


# --------------------------------------------------------------------------- #
# Regressão: pooling 'freq_stats' precisa preservar a frequência TAMBÉM no
# modelo de atenção. Antes, 'stats' e 'freq_stats' caíam no mesmo ramo e o
# Incremento 3 descartava o eixo espectral que o Incremento 2 preservava —
# a comparação entre eles deixava de isolar o efeito da atenção.
# --------------------------------------------------------------------------- #
def test_attention_freq_stats_matches_fusion_dimension():
    fus = build_model({"name": "fusion", "encoder": "lcnn", "pooling": "freq_stats"})
    att = build_model({"name": "attention", "encoder": "lcnn", "pooling": "freq_stats"})
    dim_fus = [m for m in fus.classifier if isinstance(m, torch.nn.Linear)][0].in_features
    dim_att = [m for m in att.classifier if isinstance(m, torch.nn.Linear)][0].in_features
    assert dim_att == dim_fus, "atenção descartou a frequência que a fusão preserva"


@pytest.mark.parametrize("freq_bins", [1, 4, 8])
def test_attention_freq_stats_honours_freq_bins(freq_bins):
    m = build_model({"name": "attention", "encoder": "lcnn",
                     "pooling": "freq_stats", "freq_bins": freq_bins})
    dim = [x for x in m.classifier if isinstance(x, torch.nn.Linear)][0].in_features
    assert dim == 2 * 2 * 32 * freq_bins   # 2 ramos x (media+desvio) x canais x faixas


def test_attention_freq_stats_distinguishes_frequency_bands():
    from src.models.blocks import AttentiveFreqStatsPool, AttentiveStatsPool

    baixa = torch.zeros(1, 1, 8, 20); baixa[:, :, :4, :] = 1.0
    alta = torch.zeros(1, 1, 8, 20); alta[:, :, 4:, :] = 1.0
    plano = AttentiveStatsPool(1)
    assert torch.allclose(plano(baixa), plano(alta))          # não distingue
    preserva = AttentiveFreqStatsPool(1, freq_bins=4)
    assert not torch.allclose(preserva(baixa), preserva(alta))
