"""Coeficientes dinâmicos: delta (1ª ordem) e delta-delta (2ª ordem).

Capturam a variação temporal das características — falas sintéticas frequentemente
falham em reproduzir essas dinâmicas (TC1 §3.3).
"""

from __future__ import annotations

import librosa
import numpy as np


def add_deltas(feat: np.ndarray, width: int = 9) -> np.ndarray:
    """Concatena [feat; delta; delta-delta] ao longo do eixo de frequência.

    `feat` tem shape (n_coef, n_frames); a saída tem shape (3*n_coef, n_frames).
    """
    # `width` deve ser ímpar e menor que o número de frames.
    width = min(width, _largest_odd_leq(feat.shape[1]))
    if width < 3:
        # Poucos frames para calcular deltas — devolve zeros do mesmo shape.
        d1 = np.zeros_like(feat)
        d2 = np.zeros_like(feat)
    else:
        d1 = librosa.feature.delta(feat, width=width, order=1)
        d2 = librosa.feature.delta(feat, width=width, order=2)
    return np.concatenate([feat, d1, d2], axis=0)


def _largest_odd_leq(n: int) -> int:
    """Maior número ímpar <= n."""
    return n if n % 2 == 1 else n - 1
