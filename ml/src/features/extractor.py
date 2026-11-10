"""Extração das features configuradas (ExtratorCaracteristicas da APS).

Recebe um waveform pré-processado e devolve um dicionário de tensores, um por
tipo de feature — a mesma interface para baseline e fusão.
"""

from __future__ import annotations

import numpy as np
import torch

from .lfcc import compute_lfcc
from .spectrogram import compute_log_mel
from .stft import power_spectrum, stft_params


class FeatureExtractor:
    """Extrai as features listadas em `features.types`.

    O prefixo do nome (`lfcc`, `spectrogram`) escolhe o cálculo e o nome
    completo, o bloco de config; assim cabem variantes da mesma feature::

        features:
          types: [lfcc, lfcc_hi]
          lfcc:    {n_filter: 20, ...}
          lfcc_hi: {n_filter: 70, ...}
    """

    def __init__(self, audio_cfg: dict, feat_cfg: dict):
        self.sample_rate = audio_cfg["sample_rate"]
        self.types = list(feat_cfg["types"])
        self.cfgs = {t: feat_cfg.get(t, {}) for t in self.types}
        for t in self.types:
            if _kind_of(t) is None:
                raise ValueError(
                    f"tipo de feature desconhecido: {t!r} — o nome deve começar "
                    "com 'lfcc' ou 'spectrogram'")
        # Compatibilidade com código que acessava estes atributos diretamente.
        self.lfcc_cfg = feat_cfg.get("lfcc", {})
        self.spec_cfg = feat_cfg.get("spectrogram", {})

    def fingerprint(self) -> dict:
        """Parâmetros das features ativas, usados na chave do cache em disco."""
        return {"types": sorted(self.types),
                **{t: self.cfgs[t] for t in sorted(self.types)}}

    def __call__(self, wav: np.ndarray) -> dict[str, torch.Tensor]:
        """Extrai as features pedidas. Cada tensor tem shape (1, freq, frames).

        Ramos com a mesma janela de STFT reusam o mesmo espectro de potência.
        """
        espectros: dict[tuple[int, int, int], np.ndarray] = {}
        out: dict[str, torch.Tensor] = {}
        for name in self.types:
            cfg = self.cfgs[name]
            chave = stft_params(cfg)
            if chave not in espectros:
                espectros[chave] = power_spectrum(wav, *chave)
            feat = _kind_of(name)(wav, self.sample_rate, cfg, spec=espectros[chave])
            out[name] = _to_chw(_normalize(feat))
        return out


def _kind_of(name: str):
    """Mapeia o nome do tipo para a função de cálculo, pelo prefixo."""
    if name.startswith("lfcc"):
        return compute_lfcc
    if name.startswith("spectrogram"):
        return compute_log_mel
    return None


def _normalize(feat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalização por instância (média 0, desvio 1)."""
    return (feat - feat.mean()) / (feat.std() + eps)


def _to_chw(feat: np.ndarray) -> torch.Tensor:
    """(freq, frames) -> tensor (1, freq, frames) com canal explícito."""
    return torch.from_numpy(feat).unsqueeze(0).float()
