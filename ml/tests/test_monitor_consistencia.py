"""O monitor dá o mesmo score que o caminho offline para o mesmo áudio.

O offline (`infer.py`, `evaluate.py`) já foi validado contra o eval; se os dois
batem, um resultado estranho ao vivo vem do áudio ou do modelo, não do programa.
"""

import numpy as np
import pytest
import soundfile as sf
import torch

from src.capture import FileSource
from src.capture.analyzer import AnalisadorContinuo
from src.config import load_config
from src.features import FeatureExtractor
from src.models import build_model
from src.preprocess import load_audio, preprocess_waveform


@pytest.mark.parametrize("duracao", [2.7, 4.0])
def test_score_do_monitor_igual_ao_offline(tmp_path, duracao):
    config = load_config("configs/baseline.yaml")
    torch.manual_seed(0)
    modelo = build_model(config["model"]).eval()
    ckpt = tmp_path / "m.pt"
    torch.save({"model_state": modelo.state_dict(), "config": config}, ckpt)

    rng = np.random.default_rng(0)
    n = int(16000 * duracao)
    x = (rng.standard_normal(n) * np.linspace(0.05, 0.3, n)).astype(np.float32)
    arquivo = tmp_path / "a.wav"
    sf.write(arquivo, x, 16000)

    extractor = FeatureExtractor(config["audio"], config["features"])
    w = preprocess_waveform(load_audio(arquivo, 16000), config["audio"])
    with torch.no_grad():
        feats = {k: v.unsqueeze(0) for k, v in extractor(w).items()}
        offline = float(torch.softmax(modelo(feats), dim=1)[0, 1])

    analisador = AnalisadorContinuo(config, str(ckpt), torch.device("cpu"))
    leituras = []
    with FileSource(arquivo, 16000) as fonte:
        for bloco in fonte.blocos():
            leituras += list(analisador.processar(bloco))
    leituras += list(analisador.finalizar())

    assert len(leituras) == 1
    assert leituras[0].score == pytest.approx(offline, abs=1e-6)
