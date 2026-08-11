"""Janela deslizante: do fluxo contínuo para as janelas fixas do modelo.

O modelo classifica trechos de `audio.duration` segundos (4 s nos configs). Uma
chamada é um fluxo sem fim, e os blocos que chegam da placa de som não têm
relação com esse tamanho — são ~100 ms cada.

Esta classe acumula os blocos e emite janelas de tamanho fixo com sobreposição.
O passo (`hop`) menor que a janela existe para que um trecho sintético curto não
caia exatamente na fronteira entre duas janelas e seja diluído nas duas.
"""

from __future__ import annotations

import numpy as np


class JanelaDeslizante:
    """Acumula amostras e emite janelas de `tamanho` a cada `passo` amostras.

    >>> j = JanelaDeslizante(tamanho=4, passo=2)
    >>> list(j.alimentar(np.arange(6, dtype=np.float32)))
    [array([0., 1., 2., 3.], dtype=float32), array([2., 3., 4., 5.], dtype=float32)]
    """

    def __init__(self, tamanho: int, passo: int):
        if tamanho <= 0:
            raise ValueError("tamanho da janela precisa ser positivo")
        if not 0 < passo <= tamanho:
            raise ValueError("passo precisa estar entre 1 e o tamanho da janela")
        self.tamanho = int(tamanho)
        self.passo = int(passo)
        self._buffer = np.zeros(0, dtype=np.float32)
        #: nº de amostras já descartadas — dá a posição absoluta de cada janela
        self._consumidas = 0

    def alimentar(self, bloco: np.ndarray):
        """Adiciona um bloco e devolve as janelas completas que ele fechou."""
        if bloco.size:
            self._buffer = np.concatenate([self._buffer,
                                           np.asarray(bloco, dtype=np.float32)])
        while self._buffer.size >= self.tamanho:
            yield self._buffer[:self.tamanho].copy()
            self._buffer = self._buffer[self.passo:]
            self._consumidas += self.passo

    def resto(self) -> np.ndarray:
        """O que sobrou sem completar uma janela (fim do arquivo/da chamada)."""
        return self._buffer.copy()

    @property
    def inicio_da_proxima(self) -> int:
        """Índice absoluto da primeira amostra ainda no buffer."""
        return self._consumidas

    def instante_da_janela(self, indice: int, sample_rate: int) -> float:
        """Segundos, desde o início da captura, em que a janela `indice` começa."""
        return indice * self.passo / sample_rate
