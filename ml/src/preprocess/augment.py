"""Aumentação e perturbação de áudio no domínio do tempo.

Usado em dois contextos:

- **Treino** — aumentação ALEATÓRIA (`Augmenter`), que melhora a generalização e
  a robustez a ruído (TC1 §5.5). Aplicada após o pré-processamento, antes da
  extração de características.
- **Avaliação de robustez** — perturbação DETERMINÍSTICA com nível fixo
  (`make_perturbation`), usada por `scripts/robustness_eval.py` para medir a
  degradação do modelo sob ruído/ganho controlados.

Observação: como o extrator normaliza cada feature por instância (média 0,
desvio 1), a aumentação de **ganho** tem efeito pequeno nas features — o ganho
real está em treinar para ser invariante a ela. A aumentação de **ruído** é a
mais impactante para a robustez.
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
    """Aumentação aleatória configurável (aplicar apenas no conjunto de treino).

    Exemplo de configuração (em `audio.augment` no YAML)::

        augment:
          enabled: true
          noise: {prob: 0.5, snr_db: [10, 30]}
          gain:  {prob: 0.5, gain_db: [-6, 6]}
          shift: {prob: 0.5, max_fraction: 0.1}

    Cada transformação é aplicada com a sua própria probabilidade.

    **O gerador aleatório vem de fora, por amostra.** Guardar um `rng` como
    estado do objeto não funciona com `DataLoader(num_workers>0)`: cada worker
    recebe uma *cópia* do dataset com o mesmo estado inicial, então todos
    produziriam a mesma sequência de aumentação — e, como os workers são
    recriados a cada época, a sequência ainda se repetiria época após época.
    Recebendo o `rng` derivado de (semente, época, índice), cada amostra tem
    aumentação própria, reprodutível e diferente a cada época.
    """

    def __init__(self, cfg: dict | None, seed: int | None = None):
        self.cfg = cfg or {}
        self.seed = 0 if seed is None else int(seed)

    def __call__(self, wav: np.ndarray,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        if not self.cfg.get("enabled", False):
            return wav
        # Sem rng externo (uso avulso, fora do DataLoader), cai num gerador
        # derivado da semente — determinístico, porém sem variação por época.
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


def make_perturbation(kind: str, level: float, seed: int = 0):
    """Devolve um callable determinístico para a avaliação de robustez.

    Assinatura `(wav, rng=None) -> wav`, compatível com a do `Augmenter` para
    que o dataset possa usar os dois de forma intercambiável. O `rng` recebido
    é ignorado: a perturbação usa nível fixo e semente própria, de modo que a
    mesma condição de teste seja idêntica entre modelos comparados.

    kind: 'clean' | 'noise' (level = SNR em dB) | 'gain' (level = dB) |
          'shift' (level = nº de amostras).
    """
    própria = np.random.default_rng(seed)
    if kind == "clean":
        return lambda w, rng=None: w
    if kind == "noise":
        return lambda w, rng=None: add_noise(w, level, própria)
    if kind == "gain":
        return lambda w, rng=None: apply_gain(w, level)
    if kind == "shift":
        return lambda w, rng=None: time_shift(w, int(level))
    raise ValueError(f"perturbação desconhecida: {kind!r}")
