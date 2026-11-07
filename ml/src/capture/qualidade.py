"""Verifica se o canal transmite alta frequência (acima de 4 kHz).

Presença de alta frequência prova banda larga; ausência não prova nada, pois uma
vogal sustentada também não tem. Por isso o veredito é da sessão, não da janela.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Frequência acima da qual um canal de banda estreita (8 kHz) não transmite.
CORTE_HZ = 4000.0

#: Alto de propósito: basta uma minoria de quadros com alta frequência.
PERCENTIL = 90

#: Medido: fala real dá 81%, ruído rosa 15%, banda estreita 0%.
FRACAO_MINIMA = 0.02

#: Janelas com áudio e sem alta frequência antes de declarar banda estreita.
JANELAS_PARA_CONCLUIR = 5


@dataclass(frozen=True)
class Qualidade:
    """Medição de uma janela. Quem decide é `EstadoDoCanal`."""

    fracao_alta: float          # p90 da fração de energia acima de CORTE_HZ
    tem_alta_frequencia: bool   # evidência POSITIVA de banda larga nesta janela


def fracao_energia_alta(wav: np.ndarray, sample_rate: int,
                        corte_hz: float = CORTE_HZ,
                        percentil: int = PERCENTIL) -> float:
    """Percentil `percentil`, entre quadros, da fração de energia acima de `corte_hz`.

    Quadros em silêncio ficam de fora para não puxar o percentil para baixo.
    """
    from scipy.signal import stft

    if wav.size < 2:
        return 0.0
    nperseg = min(256, wav.size)
    f, _, Z = stft(wav.astype(np.float64), sample_rate, nperseg=nperseg,
                   noverlap=nperseg // 2)
    potencia = np.abs(Z) ** 2
    total = potencia.sum(axis=0)
    com_energia = total > 0
    if not com_energia.any():
        return 0.0
    alta = potencia[f >= corte_hz][:, com_energia].sum(axis=0)
    return float(np.percentile(alta / total[com_energia], percentil))


def avaliar(wav: np.ndarray, sample_rate: int,
            fracao_minima: float = FRACAO_MINIMA) -> Qualidade:
    """Mede uma janela. O veredito do canal é de `EstadoDoCanal`."""
    fracao = fracao_energia_alta(wav, sample_rate)
    return Qualidade(fracao_alta=fracao, tem_alta_frequencia=fracao >= fracao_minima)


class EstadoDoCanal:
    """Veredito sobre o canal, acumulando evidência ao longo da sessão.

    Uma vez vista alta frequência, o veredito fica em banda larga para sempre.
    """

    LARGA = "banda larga"
    ESTREITA = "banda estreita"
    INDETERMINADO = "indeterminado"

    def __init__(self, janelas_para_concluir: int = JANELAS_PARA_CONCLUIR):
        self.janelas_para_concluir = int(janelas_para_concluir)
        self.observadas = 0
        self.sem_evidencia = 0
        self.maior_fracao = 0.0
        self._provada = False

    def observar(self, qualidade: Qualidade) -> None:
        """Registra uma janela com áudio (silêncio não conta)."""
        self.observadas += 1
        self.maior_fracao = max(self.maior_fracao, qualidade.fracao_alta)
        if qualidade.tem_alta_frequencia:
            self._provada = True
            self.sem_evidencia = 0
        else:
            self.sem_evidencia += 1

    @property
    def veredito(self) -> str:
        if self._provada:
            return self.LARGA
        if self.sem_evidencia >= self.janelas_para_concluir:
            return self.ESTREITA
        return self.INDETERMINADO

    @property
    def avaliavel(self) -> bool:
        """Falso só em banda estreita; `indeterminado` continua mostrando score."""
        return self.veredito != self.ESTREITA

    def descricao(self) -> str:
        pct = 100 * self.maior_fracao
        if self.veredito == self.LARGA:
            return f"banda larga confirmada (pico de {pct:.0f}% acima de 4 kHz)"
        if self.veredito == self.ESTREITA:
            return (f"BANDA ESTREITA provável — {self.sem_evidencia} janelas de "
                    "áudio sem nenhuma energia acima de 4 kHz")
        return (f"indeterminado ({self.observadas} janela(s), ainda sem "
                "evidência de alta frequência)")
