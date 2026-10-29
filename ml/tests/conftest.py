"""Configuração compartilhada dos testes (pytest)."""

import os
import sys

import numpy as np
import pytest

#: Raiz do projeto Python (`ml/`), onde ficam `configs/`, `src/` e `scripts/`.
RAIZ_ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Permite `import src...` de qualquer diretório.
sys.path.insert(0, RAIZ_ML)


@pytest.fixture(autouse=True)
def _roda_de_dentro_de_ml(monkeypatch):
    """Roda todo teste com o diretório atual em `ml/`, de onde quer que o pytest
    tenha sido chamado: vários testes usam caminhos relativos a `ml/`."""
    monkeypatch.chdir(RAIZ_ML)


@pytest.fixture(autouse=True)
def _interface_fora_das_pastas_reais(tmp_path, monkeypatch):
    """A limpeza do histórico varre a pasta do ao vivo e a dos clones. Sem isto, um
    teste apagou as gravações reais de outputs/ao_vivo (29/10/2026). Também não deixa
    o modelo do terminal (DETECTOR_XLSR de quem subiu a interface) vazar para os testes."""
    monkeypatch.setenv("DETECTOR_PASTA_AO_VIVO", str(tmp_path / "ao_vivo"))
    monkeypatch.setenv("DETECTOR_PASTA_CLONES", str(tmp_path / "clones"))
    monkeypatch.setenv("DETECTOR_BANCO", str(tmp_path / "historico.db"))
    monkeypatch.setenv("DETECTOR_CONFERIR", "0")
    for nome in ("DETECTOR_XLSR", "DETECTOR_LIMIAR_FUSAO"):
        monkeypatch.delenv(nome, raising=False)


@pytest.fixture
def audio_cfg():
    """Config de áudio curta (1 s) para testes rápidos."""
    return {
        "sample_rate": 16000,
        "duration": 1.0,
        "trim_silence": False,
        "top_db": 30,
        "peak_normalize": True,
    }


@pytest.fixture
def feat_cfg():
    return {
        "types": ["lfcc", "spectrogram"],
        "lfcc": {"n_lfcc": 20, "n_fft": 512, "win_length": 400,
                 "hop_length": 160, "n_filter": 20, "deltas": True},
        "spectrogram": {"n_fft": 512, "win_length": 400, "hop_length": 160, "n_mels": 40},
    }


@pytest.fixture
def sine_wave():
    """1 s de um tom de 180 Hz a 16 kHz."""
    sr = 16000
    t = np.arange(sr) / sr
    return (0.5 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
