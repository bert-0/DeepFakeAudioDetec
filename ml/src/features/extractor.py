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
    """Extrai as features listadas em `features.types`.

    O nome de cada tipo determina o *cálculo* pelo seu prefixo e o *bloco de
    configuração* pelo nome completo. Isso permite mais de uma variante da mesma
    feature no mesmo modelo — por exemplo, dois LFCC com resoluções diferentes::

        features:
          types: [lfcc, lfcc_hi]
          lfcc:    {n_filter: 20, ...}
          lfcc_hi: {n_filter: 70, ...}

    Qualquer nome iniciado por `lfcc` usa o cálculo de LFCC; qualquer nome
    iniciado por `spectrogram`, o de espectrograma log-mel.
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
        """Parâmetros que afetam as features extraídas.

        Usado para invalidar o cache em disco: mudar `n_filter`, `n_lfcc` etc.
        precisa gerar uma chave diferente, senão features antigas seriam reusadas
        silenciosamente. Só os tipos ativos entram, para não invalidar o cache à
        toa quando se altera uma feature que nem está em uso.
        """
        return {"types": sorted(self.types),
                **{t: self.cfgs[t] for t in sorted(self.types)}}

    def __call__(self, wav: np.ndarray) -> dict[str, torch.Tensor]:
        """Extrai as features pedidas. Cada tensor tem shape (1, freq, frames)."""
        out: dict[str, torch.Tensor] = {}
        for name in self.types:
            compute = _kind_of(name)
            feat = compute(wav, self.sample_rate, self.cfgs[name])
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
