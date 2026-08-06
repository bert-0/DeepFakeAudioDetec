"""Testes do espectro de potência compartilhado entre ramos de features."""

import numpy as np
import pytest
import yaml
from pathlib import Path

from src.features.extractor import FeatureExtractor
from src.features.lfcc import compute_lfcc
from src.features.spectrogram import compute_log_mel
from src.features.stft import power_spectrum, stft_params

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def wav():
    return np.random.default_rng(7).standard_normal(16000 * 4).astype(np.float32)


def test_stft_params_identifica_janelas_iguais():
    a = {"n_fft": 512, "win_length": 400, "hop_length": 160, "n_mels": 80}
    b = {"n_fft": 512, "win_length": 400, "hop_length": 160, "n_filter": 70}
    c = {"n_fft": 1024, "win_length": 400, "hop_length": 160}
    assert stft_params(a) == stft_params(b), "só a janela define o espectro"
    assert stft_params(a) != stft_params(c)


# --------------------------------------------------------------------------- #
# O ponto central: compartilhar o espectro NÃO pode mudar o resultado.
# Se mudasse, invalidaria os caches em disco e os checkpoints já treinados.
# --------------------------------------------------------------------------- #
def test_lfcc_identico_com_e_sem_espectro_pronto(wav):
    cfg = {"n_fft": 512, "win_length": 400, "hop_length": 160,
           "n_filter": 70, "n_lfcc": 20, "deltas": True}
    spec = power_spectrum(wav, 512, 400, 160)
    assert np.array_equal(compute_lfcc(wav, 16000, cfg),
                          compute_lfcc(wav, 16000, cfg, spec=spec))


def test_log_mel_identico_com_e_sem_espectro_pronto(wav):
    cfg = {"n_fft": 512, "win_length": 400, "hop_length": 160, "n_mels": 80}
    spec = power_spectrum(wav, 512, 400, 160)
    assert np.array_equal(compute_log_mel(wav, 16000, cfg),
                          compute_log_mel(wav, 16000, cfg, spec=spec))


def test_extractor_do_fusion_v4_bate_bit_a_bit_com_o_caminho_antigo(wav):
    cfg = yaml.safe_load((CONFIGS / "fusion_v4.yaml").read_text(encoding="utf-8"))
    ex = FeatureExtractor(cfg["audio"], cfg["features"])
    novo = ex(wav)

    def norma(f, eps=1e-8):
        return (f - f.mean()) / (f.std() + eps)

    antigo = {
        "lfcc": norma(compute_lfcc(wav, 16000, cfg["features"]["lfcc"])),
        "spectrogram": norma(compute_log_mel(wav, 16000, cfg["features"]["spectrogram"])),
    }
    for chave, esperado in antigo.items():
        assert np.array_equal(novo[chave].squeeze(0).numpy(), esperado), chave


def test_ramos_com_janelas_diferentes_nao_compartilham(wav):
    """Compartilhar aqui seria um bug silencioso: espectros diferentes."""
    feat_cfg = {
        "types": ["lfcc", "lfcc_hi"],
        "lfcc": {"n_fft": 512, "win_length": 400, "hop_length": 160,
                 "n_filter": 20, "n_lfcc": 20, "deltas": False},
        "lfcc_hi": {"n_fft": 1024, "win_length": 800, "hop_length": 160,
                    "n_filter": 70, "n_lfcc": 20, "deltas": False},
    }
    out = FeatureExtractor({"sample_rate": 16000}, feat_cfg)(wav)
    esperado = compute_lfcc(wav, 16000, feat_cfg["lfcc_hi"])
    assert np.array_equal(out["lfcc_hi"].squeeze(0).numpy(),
                          (esperado - esperado.mean()) / (esperado.std() + 1e-8))


def test_o_espectro_e_calculado_uma_vez_so(wav, monkeypatch):
    """Regressão direta: era chamado 2x com parâmetros idênticos."""
    chamadas = []
    import src.features.extractor as mod
    original = mod.power_spectrum

    def espiao(w, *args):
        chamadas.append(args)
        return original(w, *args)

    monkeypatch.setattr(mod, "power_spectrum", espiao)
    cfg = yaml.safe_load((CONFIGS / "fusion_v4.yaml").read_text(encoding="utf-8"))
    FeatureExtractor(cfg["audio"], cfg["features"])(wav)

    assert len(chamadas) == 1, f"esperava 1 STFT para os 2 ramos, houve {len(chamadas)}"
