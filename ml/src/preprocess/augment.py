"""Aumentação e perturbação de áudio no domínio do tempo.

`Augmenter` faz a aumentação aleatória do treino (TC1 §5.5); `make_perturbation`
aplica um nível fixo para o `scripts/robustness_eval.py`. Com features
normalizadas por instância, o ganho quase não muda nada; o ruído é o que pesa.
"""

from __future__ import annotations

import numpy as np


def add_noise(wav: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Adiciona ruído gaussiano branco a uma relação sinal-ruído (SNR) alvo, em dB."""
    sig_power = float(np.mean(wav ** 2)) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = rng.standard_normal(wav.shape).astype(np.float32) * np.sqrt(noise_power)
    return (wav + noise).astype(np.float32)


def apply_gain(wav: np.ndarray, gain_db: float) -> np.ndarray:
    """Aplica um ganho de amplitude, em dB (positivo amplifica, negativo atenua)."""
    return (wav * (10 ** (gain_db / 20.0))).astype(np.float32)


def time_shift(wav: np.ndarray, shift: int) -> np.ndarray:
    """Desloca o sinal circularmente por `shift` amostras."""
    return np.roll(wav, int(shift)).astype(np.float32)


class Augmenter:
    """Aumentação aleatória configurável, só para o treino.

    Exemplo (em `audio.augment` no YAML)::

        augment:
          enabled: true
          noise: {prob: 0.5, snr_db: [10, 30]}
          gain:  {prob: 0.5, gain_db: [-6, 6]}
          shift: {prob: 0.5, max_fraction: 0.1}

    O `rng` vem de fora, por amostra: um gerador guardado no objeto se repetiria
    em cada worker do DataLoader e a cada época.
    """

    def __init__(self, cfg: dict | None, seed: int | None = None):
        self.cfg = cfg or {}
        self.seed = 0 if seed is None else int(seed)

    def __call__(self, wav: np.ndarray,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        if not self.cfg.get("enabled", False):
            return wav
        # Uso avulso: gerador da semente, determinístico e sem variar por época.
        if rng is None:
            rng = np.random.default_rng(self.seed)

        noise = self.cfg.get("noise")
        if noise and rng.random() < noise.get("prob", 0.0):
            lo, hi = noise["snr_db"]
            wav = add_noise(wav, rng.uniform(lo, hi), rng)

        gain = self.cfg.get("gain")
        if gain and rng.random() < gain.get("prob", 0.0):
            lo, hi = gain["gain_db"]
            wav = apply_gain(wav, rng.uniform(lo, hi))

        shift = self.cfg.get("shift")
        if shift and rng.random() < shift.get("prob", 0.0):
            max_shift = int(len(wav) * shift.get("max_fraction", 0.1))
            if max_shift > 0:
                wav = time_shift(wav, rng.integers(-max_shift, max_shift + 1))

        return wav


class Perturbation:
    """Perturbação determinística de nível fixo, com a assinatura do `Augmenter`.

    `kind`: 'clean', 'noise' (level = SNR em dB), 'gain' (dB) ou 'shift'
    (amostras). É classe, e não lambda, para o pickle levar o dataset aos
    workers; o ruído sai do `rng` da amostra e não depende do nº de workers.
    """

    def __init__(self, kind: str, level: float | None, seed: int = 0):
        if kind not in ("clean", "noise", "gain", "shift"):
            raise ValueError(f"perturbação desconhecida: {kind!r}")
        self.kind = kind
        self.level = level
        self.seed = int(seed)

    def __call__(self, wav: np.ndarray,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        if self.kind == "clean":
            return wav
        if self.kind == "gain":
            return apply_gain(wav, self.level)
        if self.kind == "shift":
            return time_shift(wav, int(self.level))
        # Uso avulso, sem `rng`: usa a semente própria.
        return add_noise(wav, self.level, rng or np.random.default_rng(self.seed))


def make_perturbation(kind: str, level: float, seed: int = 0) -> Perturbation:
    """Atalho para `Perturbation`, com o nome que os scripts usam."""
    return Perturbation(kind, level, seed)
