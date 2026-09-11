"""Portão de qualidade: o canal transmite alta frequência?

O detector foi medido sob degradação de canal (`robustness_eval.py`, eval
completo). O resultado motiva este módulo:

    condição               fusion_v4    custo
    limpo                    20,18%        —
    opus 25 kbps             22,08%   +1,90 pp
    banda estreita (8 kHz)   25,53%   +5,35 pp

O codec custa pouco; perder a banda alta custa cinco vezes mais. E metade do
banco de filtros do LFCC deste projeto olha justamente acima de 4 kHz.

**Por que não basta medir a energia alta de uma janela.** Medido aqui:

    sinal                    média da janela   p90 por quadro
    ruído rosa                       11,21%           14,84%
    fala (vogais + fricativas)        5,40%           81,39%
    vogal sustentada                  0,00%            0,00%
    fala em banda estreita            0,00%            0,00%

Uma **vogal sustentada não tem energia acima de 4 kHz** — exatamente como um
canal de banda estreita. De uma janela isolada, os dois são indistinguíveis, e
um portão que reprovasse por ausência rejeitaria fala perfeitamente normal.

A assimetria é o que salva: **presença de alta frequência prova que o canal a
transmite; ausência não prova nada.** Por isso o veredito é de sessão, não de
janela, e tem três estados — `banda larga` (confirmada e definitiva),
`indeterminado` (ainda sem evidência) e `banda estreita` (provável, após várias
janelas de fala sem nenhuma alta frequência).

A medição por janela usa o **percentil 90 dos quadros**, e não a média: a fala
alterna vogais (sem alta frequência) e fricativas (com muita), e a média dilui
as fricativas até sumirem. O p90 pergunta "algum quadro mostrou alta
frequência?", que é a pergunta sobre o canal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Frequência acima da qual um canal de banda estreita (8 kHz) não transmite.
CORTE_HZ = 4000.0

#: Percentil dos quadros usado como evidência. Alto de propósito: basta uma
#: minoria de quadros com alta frequência para provar que o canal a transmite.
PERCENTIL = 90

#: Fração mínima, no percentil acima, para considerar a banda larga provada.
#: Medido: fala real dá 81%, ruído rosa 15%, banda estreita 0,00%. O corte fica
#: muito abaixo dos casos positivos e muito acima do negativo.
FRACAO_MINIMA = 0.02

#: Quantas janelas com áudio e sem nenhuma evidência de alta frequência antes de
#: declarar banda estreita provável. Em fala corrida, fricativas aparecem a todo
#: momento; várias janelas seguidas sem elas indicam o canal, não o locutor.
JANELAS_PARA_CONCLUIR = 5


@dataclass(frozen=True)
class Qualidade:
    """Medição de uma janela. Não decide sozinha — ver `EstadoDoCanal`."""

    fracao_alta: float          # p90 da fração de energia acima de CORTE_HZ
    tem_alta_frequencia: bool   # evidência POSITIVA de banda larga nesta janela


def fracao_energia_alta(wav: np.ndarray, sample_rate: int,
                        corte_hz: float = CORTE_HZ,
                        percentil: int = PERCENTIL) -> float:
    """Percentil `percentil` da fração de energia acima de `corte_hz`, por quadro.

    Quadros sem energia nenhuma (silêncio entre palavras) ficam de fora: a
    fração seria 0/0, e incluí-los puxaria o percentil para baixo por um motivo
    que nada tem a ver com o canal.
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

    Uma vez observada alta frequência, o canal está **provado** de banda larga e
    o veredito não volta atrás: o canal de uma chamada não muda a cada dois
    segundos, e uma sequência de vogais não é motivo para desconfiar dele.
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
        """Registra uma janela COM ÁUDIO (silêncio não é evidência de nada)."""
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
        """Se falso, os scores não devem virar indício.

        `indeterminado` conta como avaliável: na dúvida o sistema continua
        medindo e mostrando o score, em vez de se calar por uma sequência de
        vogais. O que ele não faz é afirmar banda estreita sem evidência.
        """
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
