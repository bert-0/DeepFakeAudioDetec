"""Portão de qualidade: quando o canal não permite julgar.

O detector foi medido sob degradação de canal (`robustness_eval.py`, eval
completo). O resultado decide o que este módulo faz:

    condição               fusion_v4    custo
    limpo                    20,18%        —
    opus 25 kbps             22,08%   +1,90 pp
    banda estreita (8 kHz)   25,53%   +5,35 pp

O codec custa pouco. **Perder a banda alta custa cinco vezes mais**, e é uma
condição detectável no próprio áudio: um canal de 8 kHz não transmite nada acima
de 4 kHz, e o LFCC deste projeto tem metade do banco de filtros justamente ali.

Medido com ruído rosa, energia acima de 4 kHz:

    banda larga (limpo)      11,21%
    opus 25 kbps              6,39%
    opus 15 kbps              8,32%
    banda estreita            0,00%
    banda estreita + opus     0,00%

A separação é de ordem de grandeza, não de margem apertada. Por isso o sistema
consegue reconhecer que está recebendo áudio fora do domínio em que foi medido —
e nesse caso a resposta correta é **não emitir veredito**, em vez de emitir um
que sabemos degradado.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Frequência acima da qual um canal de banda estreita (8 kHz) não transmite.
CORTE_HZ = 4000.0

#: Fração mínima de energia acima de `CORTE_HZ` para o áudio ser considerado de
#: banda larga. Fica entre os 0,00% da banda estreita e os 6,39% do pior caso de
#: banda larga medido — longe dos dois, porque a separação é ampla.
FRACAO_MINIMA = 0.02


@dataclass(frozen=True)
class Qualidade:
    """Diagnóstico do canal para uma janela de áudio."""

    fracao_alta: float          # energia acima de CORTE_HZ, em [0, 1]
    banda_larga: bool

    @property
    def avaliavel(self) -> bool:
        """Se falso, o score desta janela não deve virar veredito."""
        return self.banda_larga

    def descricao(self) -> str:
        if self.banda_larga:
            return f"banda larga ({100 * self.fracao_alta:.1f}% acima de 4 kHz)"
        return (f"BANDA ESTREITA ({100 * self.fracao_alta:.1f}% acima de 4 kHz) "
                "— fora do domínio medido")


def fracao_energia_alta(wav: np.ndarray, sample_rate: int,
                        corte_hz: float = CORTE_HZ) -> float:
    """Fração da energia espectral acima de `corte_hz`.

    Usa Welch em vez de um único FFT: a média sobre segmentos reduz a variância
    da estimativa, o que importa numa janela curta de fala real, onde um único
    FFT oscila muito entre trechos sonoros e surdos.
    """
    from scipy.signal import welch

    if wav.size == 0:
        return 0.0
    nperseg = min(512, wav.size)
    f, p = welch(wav.astype(np.float64), sample_rate, nperseg=nperseg)
    total = p.sum()
    if total <= 0:
        return 0.0
    return float(p[f >= corte_hz].sum() / total)


def avaliar(wav: np.ndarray, sample_rate: int,
            fracao_minima: float = FRACAO_MINIMA) -> Qualidade:
    """Diagnostica se a janela está no domínio em que o detector foi medido."""
    fracao = fracao_energia_alta(wav, sample_rate)
    return Qualidade(fracao_alta=fracao, banda_larga=fracao >= fracao_minima)
