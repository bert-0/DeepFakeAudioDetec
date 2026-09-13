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
        #: amostra em que termina a última janela emitida. Serve para saber se
        #: sobrou áudio que nenhuma janela chegou a cobrir.
        self._cobertura = 0
        #: amostra inicial da última janela emitida (por `alimentar` ou
        #: `finalizar`). É o que dá o instante correto da janela final, que não
        #: cai na grade regular do passo.
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
        """Emite a janela final quando sobrou áudio que nenhuma janela cobriu.

        Sem isto, **um arquivo mais curto que a janela não produz leitura
        nenhuma** — o `while` de `alimentar` nunca dispara e o resto é
        descartado em silêncio. É o caso da maioria dos áudios do ASVspoof:
        a janela tem 4 s e o enunciado típico é mais curto que isso.

        Só emite se houver áudio além do que a última janela já cobriu. Com
        passo igual a metade da janela, o fim de um arquivo longo normalmente
        já está dentro da última janela emitida, e repetir aquele trecho
        inflaria a contagem sem acrescentar informação.

        O trecho sai **mais curto que a janela**; quem completa é o
        `preprocess_waveform`, por repetição, exatamente como no treino. O peso
        da leitura cai na proporção — ver `Leitura.peso`.
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
