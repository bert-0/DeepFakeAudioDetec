"""Representações do XLS-R 300M congelado e o LFCC médio, por arquivo.

O preparo do `scripts/estudo_ssl.py extrair`, que deu os .npz de treino e das
réguas do modelo final (RESUMO_TCC §10.2.22 e §10.2.43). Fica aqui para a
interface e o monitor usarem o mesmo código; qualquer mudança muda os números.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MODELO_HF = "facebook/wav2vec2-xls-r-300m"
SR = 16000
#: Mesma janela do detector (4 s, passo de 2 s); áudio curto entra inteiro, sem repetir.
JANELA_S, PASSO_S = 4.0, 2.0
TOP_DB = 30
#: Mesmo LFCC dos configs/fusion_v4*.yaml.
LFCC_CFG = {"n_lfcc": 20, "n_fft": 512, "win_length": 400, "hop_length": 160,
            "n_filter": 70, "deltas": True}
#: Abaixo disto, a primeira convolução do XLS-R falha (um clone de 17 ms, §10.2.40).
MIN_AMOSTRAS = int(0.25 * SR)


def preparar(caminho: Path) -> tuple[np.ndarray, float]:
    """Como no treino: 16 kHz pelo FIR, sem silêncio nas pontas, pico em 1.
    Devolve (áudio, segundos de silêncio retirados)."""
    from src.preprocess.audio import load_audio

    return preparar_wav(load_audio(caminho, SR))


def preparar_wav(bruto: np.ndarray) -> tuple[np.ndarray, float]:
    """O `preparar` sobre um áudio já lido a 16 kHz."""
    from src.preprocess.audio import peak_normalize, trim_silence

    wav = trim_silence(bruto, TOP_DB)
    return peak_normalize(wav).astype(np.float32), (len(bruto) - len(wav)) / SR


def inicios(n_amostras: int) -> list[int]:
    """Onde começa cada janela, em amostras."""
    n, passo = int(JANELA_S * SR), int(PASSO_S * SR)
    if n_amostras <= n:
        return [0]
    return list(range(0, n_amostras - n + 1, passo))


def janelas(wav: np.ndarray) -> list[np.ndarray]:
    n = int(JANELA_S * SR)
    return [wav[i:i + n] for i in inicios(len(wav))]


def carregar_rede(dispositivo: str = "cuda", modelo: str = MODELO_HF):
    import torch
    from transformers import Wav2Vec2Model

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    dev = torch.device(dispositivo if torch.cuda.is_available() or dispositivo == "cpu" else "cpu")
    return Wav2Vec2Model.from_pretrained(modelo).eval().to(dev), dev


def carregar_xlsr(dispositivo: str = "cuda", modelo: str = MODELO_HF):
    """Função wav -> (cnn (512,), camadas (25, 1024)), médias no tempo, em fp32."""
    return representar_com(*carregar_rede(dispositivo, modelo))


def por_janela(rede, dev, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cnn (512,), camadas (25, 1024)) de uma janela: médias no tempo, em fp32."""
    import torch

    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)  # do_normalize do XLS-R
    with torch.inference_mode():
        s = rede(torch.from_numpy(x).float()[None].to(dev), output_hidden_states=True)
    return (s.extract_features[0].float().mean(0).cpu().numpy(),
            torch.stack([h[0].float().mean(0) for h in s.hidden_states]).cpu().numpy())


def representar_com(rede, dev):
    def rodar(wav: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cnns, camadas = zip(*(por_janela(rede, dev, x) for x in janelas(wav)), strict=True)
        return np.mean(cnns, axis=0), np.mean(camadas, axis=0)
    return rodar


def lfcc_medio(wav: np.ndarray) -> np.ndarray:
    """Média e desvio no tempo do LFCC com deltas (120 valores)."""
    from src.features.lfcc import compute_lfcc

    c = compute_lfcc(wav, SR, LFCC_CFG)
    return np.concatenate([c.mean(axis=1), c.std(axis=1)]).astype(np.float32)
