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


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model({"name": "inexistente"})


def test_unknown_pooling_raises():
    with pytest.raises(ValueError):
        build_model({"name": "baseline_cnn", "pooling": "inexistente"})
