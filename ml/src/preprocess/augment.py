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


class Perturbation:
    """Perturbação determinística de nível fixo, para a avaliação de robustez.

    Assinatura `(wav, rng=None) -> wav`, igual à do `Augmenter`, para que o
    dataset use os dois de forma intercambiável.

    kind: 'clean' | 'noise' (level = SNR em dB) | 'gain' (level = dB) |
          'shift' (level = nº de amostras).

    **É uma classe, e não uma closure.** A versão anterior devolvia `lambda`, que
    o pickle não serializa — e o dataset inteiro é serializado para os workers do
    DataLoader (no Windows, `spawn` faz isso sempre). Por isso a robustez rodava
    presa a `num_workers=0`: um único processo extraindo features de 71.237
    áudios, seis vezes (uma por condição), sem poder usar cache — a aumentação
    muda as features, então elas têm mesmo de ser recalculadas.

    **O ruído passa a ser derivado do `rng` da amostra.** Antes, um único gerador
    era compartilhado por todas as chamadas, então a perturbação de cada áudio
    dependia da *ordem* em que ele era processado. Com vários workers essa ordem
    deixa de existir, e o resultado passaria a variar com o nº de workers. Usando
    o gerador que o dataset deriva de (semente, época, índice), cada áudio recebe
    sempre o mesmo ruído — reprodutível com 0, 2 ou 8 workers.
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
        # Sem `rng` (uso avulso, fora do DataLoader) cai na semente própria.
        return add_noise(wav, self.level, rng or np.random.default_rng(self.seed))


def make_perturbation(kind: str, level: float, seed: int = 0) -> Perturbation:
    """Atalho para `Perturbation` (mantido pelo nome já usado nos scripts)."""
    return Perturbation(kind, level, seed)
