"""Degradação de canal no treino, no áudio bruto, antes do pré-processamento.

O `Augmenter` entra depois do corte de silêncio e da normalização pelo pico:
atinge um trecho que não existe numa gravação real. Aqui a degradação atinge o
áudio como foi lido, com as pausas, que é onde o ruído desloca o score
(RESUMO_TCC §10.2.1). Plano do Incremento 4, "gancho de degradação".

Config (`audio.degradacao`)::

    degradacao:
      enabled: true
      etapas:                     # em ordem, cada uma com a sua chance
        - {tipo: captura_fir, prob: 0.5, nivel: 48000}
        - {tipo: opus, prob: 0.3, faixa: [0.3, 0.9]}
        - {tipo: band, prob: 0.1, opcoes: [8000, 11025]}
        - {tipo: ruido, prob: 0.3, faixa: [10, 30]}

`nivel` fixa o parâmetro, `faixa` sorteia um valor entre dois por amostra e
`opcoes` sorteia um da lista. O parâmetro é o do `ChannelDegradation` (opus:
compression_level; band: taxa intermediária; captura e captura_fir: taxa do
dispositivo) ou, no ruído, a SNR em dB.
"""

from __future__ import annotations

import numpy as np

from .augment import add_noise
from .channel import ChannelDegradation

TIPOS = ("opus", "band", "captura", "captura_fir", "ruido")
_FORMAS = ("nivel", "faixa", "opcoes")


class _Etapa:
    def __init__(self, cfg: dict, sample_rate: int):
        self.tipo = cfg.get("tipo")
        if self.tipo not in TIPOS:
            raise ValueError(f"degradação desconhecida: {self.tipo!r} (use uma de {TIPOS})")
        self.prob = float(cfg.get("prob", 1.0))
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError(f"{self.tipo}: prob {self.prob} fora de [0, 1]")
        formas = [f for f in _FORMAS if f in cfg]
        if len(formas) != 1:
            raise ValueError(f"{self.tipo}: dê exatamente um de {_FORMAS}, não {formas}")
        self.forma, self.valor = formas[0], cfg[formas[0]]
        if self.forma == "faixa" and len(self.valor) != 2:
            raise ValueError(f"{self.tipo}: faixa precisa de dois valores, não {self.valor}")
        if self.forma == "opcoes" and not self.valor:
            raise ValueError(f"{self.tipo}: opcoes vazia")
        self.sample_rate = int(sample_rate)

    def parametro(self, rng: np.random.Generator) -> float:
        if self.forma == "nivel":
            return float(self.valor)
        if self.forma == "faixa":
            lo, hi = self.valor
            return float(rng.uniform(lo, hi))
        return float(self.valor[int(rng.integers(len(self.valor)))])

    def __call__(self, wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        p = self.parametro(rng)
        if self.tipo == "ruido":
            return add_noise(wav, p, rng)
        return ChannelDegradation(self.tipo, p, self.sample_rate)(wav)

    def __str__(self) -> str:
        return f"{self.tipo} {self.prob:.0%}"


class Degradacao:
    """Cadeia de etapas com a assinatura `(wav, rng) -> wav`.

    Classe, e não lambda, para o pickle levá-la aos workers do DataLoader. O
    `rng` vem da amostra; cada etapa sorteia se entra e, se entrar, o parâmetro.
    """

    def __init__(self, cfg: dict, sample_rate: int):
        etapas = cfg.get("etapas") or []
        if not etapas:
            raise ValueError("audio.degradacao ligada sem nenhuma etapa")
        self.etapas = [_Etapa(e, sample_rate) for e in etapas]

    def __call__(self, wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        for etapa in self.etapas:
            if rng.random() < etapa.prob:
                wav = etapa(wav, rng)
        return wav.astype(np.float32)

    def descricao(self) -> str:
        return ", ".join(str(e) for e in self.etapas)


def construir(cfg: dict | None, sample_rate: int) -> Degradacao | None:
    """A degradação do config, ou None quando ausente ou desligada."""
    if not cfg or not cfg.get("enabled", False):
        return None
    return Degradacao(cfg, sample_rate)
