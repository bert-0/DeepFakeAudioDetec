"""ExtratorCaracteristicas — orquestra a extração das features configuradas.

Recebe um waveform já pré-processado e devolve um dicionário de tensores
(uma entrada por tipo de feature), pronto para os modelos. Manter um dicionário
uniforme permite que baseline (só LFCC) e fusão/atenção (LFCC + espectrograma)
compartilhem a mesma interface.
"""

from __future__ import annotations

import numpy as np
import torch

from .lfcc import compute_lfcc
from .spectrogram import compute_log_mel


class FeatureExtractor:
    def __init__(self, audio_cfg: dict, feat_cfg: dict):
        self.sample_rate = audio_cfg["sample_rate"]
        self.types = list(feat_cfg["types"])
        self.lfcc_cfg = feat_cfg.get("lfcc", {})
        self.spec_cfg = feat_cfg.get("spectrogram", {})

    def fingerprint(self) -> dict:
        """Parâmetros que afetam as features extraídas.

        Usado para invalidar o cache em disco: mudar `n_filter`, `n_lfcc` etc.
        precisa gerar uma chave diferente, senão features antigas seriam reusadas
        silenciosamente. Só os tipos ativos entram, para não invalidar o cache à
        toa quando se altera uma feature que nem está em uso.
        """
        active = {"types": sorted(self.types)}
        if "lfcc" in self.types:
            active["lfcc"] = self.lfcc_cfg
        if "spectrogram" in self.types:
            active["spectrogram"] = self.spec_cfg
        return active

    def __call__(self, wav: np.ndarray) -> dict[str, torch.Tensor]:
        """Extrai as features pedidas. Cada tensor tem shape (1, freq, frames)."""
        out: dict[str, torch.Tensor] = {}
        if "lfcc" in self.types:
            feat = compute_lfcc(wav, self.sample_rate, self.lfcc_cfg)
            out["lfcc"] = _to_chw(_normalize(feat))
        if "spectrogram" in self.types:
            feat = compute_log_mel(wav, self.sample_rate, self.spec_cfg)
            out["spectrogram"] = _to_chw(_normalize(feat))
        return out


def _normalize(feat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalização por instância (média 0, desvio 1)."""
    return (feat - feat.mean()) / (feat.std() + eps)


def _to_chw(feat: np.ndarray) -> torch.Tensor:
    """(freq, frames) -> tensor (1, freq, frames) com canal explícito."""
    return torch.from_numpy(feat).unsqueeze(0).float()
