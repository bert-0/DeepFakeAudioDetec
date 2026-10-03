"""Janela deslizante: do fluxo contínuo para as janelas fixas do modelo.

Acumula os blocos da placa de som (~100 ms) e emite janelas de
`audio.duration` segundos com sobreposição.
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
        #: amostras já descartadas; dá a posição absoluta de cada janela
        self._consumidas = 0
        #: amostra em que termina a última janela emitida
        self._cobertura = 0
        #: início da última janela emitida; a final não cai na grade do passo
        self.inicio_da_ultima = 0

    def alimentar(self, bloco: np.ndarray):
        """Adiciona um bloco e devolve as janelas completas que ele fechou."""
        if bloco.size:
            self._buffer = np.concatenate([self._buffer,
                                           np.asarray(bloco, dtype=np.float32)])
        while self._buffer.size >= self.tamanho:
            self.inicio_da_ultima = self._consumidas
            self._cobertura = self._consumidas + self.tamanho
            yield self._buffer[:self.tamanho].copy()
            self._buffer = self._buffer[self.passo:]
            self._consumidas += self.passo

    def resto(self) -> np.ndarray:
        """O que sobrou sem completar uma janela (fim do arquivo/da chamada)."""
        return self._buffer.copy()

    def finalizar(self):
        """Emite o trecho final se sobrou áudio que nenhuma janela cobriu.

        Sai mais curto que a janela; o `preprocess_waveform` completa por repetição.
        """
        total = self._consumidas + self._buffer.size
        if self._buffer.size == 0 or total <= self._cobertura:
            return
        self.inicio_da_ultima = self._consumidas
        self._cobertura = total
        yield self._buffer.copy()

    @property
    def inicio_da_proxima(self) -> int:
        """Índice absoluto da primeira amostra ainda no buffer."""
        return self._consumidas

    def instante_da_janela(self, indice: int, sample_rate: int) -> float:
        """Segundos, desde o início da captura, em que a janela `indice` começa."""
        return indice * self.passo / sample_rate
