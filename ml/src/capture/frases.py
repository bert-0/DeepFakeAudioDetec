"""Ao vivo frase por frase, para o modelo final (src/xlsr_final.py).

O modelo final dá o score de um arquivo inteiro, e o limiar dele é calibrado com
frases gravadas uma a uma (RESUMO_TCC §10.2.39). A média de janelas soltas não dá
o mesmo número: no ensaio 5 ela se afasta do score do arquivo em até 0,67 nos
log-odds. Por isso o ao vivo junta a fala até uma pausa e analisa a frase pelo
mesmo caminho do arquivo.

A fala é achada pela regra da checagem depois de gravar (`checagens.trechos_de_fala`):
sem graves (pancadas de teclado) e 10 dB acima do piso de ruído, aqui o percentil
10 dos últimos 30 s. Com 10 s, uma fala longa com poucos respiros subia o piso até
a própria fala.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .analyzer import SILENCIO_RMS, Leitura
from .checagens import FALA_ACIMA_DO_PISO_DB, FALA_PASSA_ALTAS_HZ
from .qualidade import EstadoDoCanal, avaliar

QUADRO_S = 0.02
#: Silêncio que fecha a frase; as pausas entre palavras ficam abaixo disso.
PAUSA_S = 0.8
#: Menos fala que isso não vira frase (tosse, tecla).
MIN_FALA_S = 1.0
#: Fala sem pausa é cortada aqui, para o resultado não demorar.
MAX_FRASE_S = 15.0
#: Áudio guardado antes da fala, como a folga das frases gravadas.
MARGEM_S = 0.3
PISO_S = 30.0


class DetectorDeFala:
    """Quadro a quadro: o passa-altas guarda o estado entre um bloco e outro."""

    def __init__(self, sample_rate: int):
        from scipy.signal import butter

        self.sos = butter(4, FALA_PASSA_ALTAS_HZ, btype="highpass", fs=sample_rate, output="sos")
        self.zi = np.zeros((len(self.sos), 2))
        self.historico: deque[float] = deque(maxlen=int(PISO_S / QUADRO_S))

    def __call__(self, quadro: np.ndarray) -> bool:
        from scipy.signal import sosfilt

        voz, self.zi = sosfilt(self.sos, quadro.astype(np.float64), zi=self.zi)
        rms = float(np.sqrt(np.mean(voz ** 2)))
        self.historico.append(rms)
        piso = float(np.percentile(self.historico, 10))
        return rms >= max(SILENCIO_RMS, piso * 10 ** (FALA_ACIMA_DO_PISO_DB / 20))

    def aquecer(self, wav: np.ndarray, quadro: int) -> None:
        """Num áudio já gravado, o piso sai do arquivo inteiro antes do primeiro quadro.
        Sem isso, um arquivo que começa falando (um clone sem silêncio na frente) toma
        a própria fala por piso e perde o começo."""
        from scipy.signal import sosfilt

        voz = sosfilt(self.sos, np.asarray(wav, np.float64))
        n = len(voz) // quadro * quadro
        if n:
            self.historico.extend(np.sqrt((voz[:n].reshape(-1, quadro) ** 2).mean(axis=1)).tolist())


class AnalisadorFrases:
    """Mesma interface do `AnalisadorContinuo` no monitor: `processar(bloco)` e
    `finalizar()` devolvem leituras, aqui uma por frase."""

    regra = "logodds"
    n_modelos = 1

    def __init__(self, modelo, limiar: float | None = None, sample_rate: int = 16000):
        self.modelo = modelo
        self.threshold = limiar
        self.sample_rate = sample_rate
        self.quadro = int(QUADRO_S * sample_rate)
        self.fala = DetectorDeFala(sample_rate)
        self.canal = EstadoDoCanal()
        self._resto = np.zeros(0, np.float32)
        self._antes: deque[np.ndarray] = deque(maxlen=int(MARGEM_S / QUADRO_S))
        self._frase: list[np.ndarray] | None = None
        self._inicio = self._t = self._n = 0
        self._fala = self._pausa = 0

    @property
    def duracao_s(self) -> float:
        return self._t / self.sample_rate

    def processar(self, bloco: np.ndarray):
        x = np.concatenate([self._resto, np.asarray(bloco, np.float32).reshape(-1)])
        n = len(x) // self.quadro * self.quadro
        self._resto = x[n:]
        for i in range(0, n, self.quadro):
            leitura = self._quadro(x[i:i + self.quadro])
            if leitura is not None:
                yield leitura

    def finalizar(self):
        if self._frase is not None:
            leitura = self._fechar()
            if leitura is not None:
                yield leitura

    def _quadro(self, q: np.ndarray) -> Leitura | None:
        fala = self.fala(q)
        t, self._t = self._t, self._t + len(q)
        if self._frase is None:
            if not fala:
                self._antes.append(q)
                return None
            self._frase = list(self._antes)
            self._antes.clear()
            self._inicio = t - sum(len(a) for a in self._frase)
            self._fala = self._pausa = 0
        self._frase.append(q)
        if fala:
            self._fala, self._pausa = self._fala + 1, 0
        else:
            self._pausa += 1
        if self._pausa * QUADRO_S >= PAUSA_S or self._t - self._inicio >= MAX_FRASE_S * self.sample_rate:
            return self._fechar()
        return None

    def _fechar(self) -> Leitura | None:
        from src.xlsr_final import FalaCurta

        frase, self._frase = self._frase, None
        if self._fala * QUADRO_S < MIN_FALA_S:
            return None
        wav = np.concatenate(frase)
        try:
            p = self.modelo.pontuar_wav(wav, self.sample_rate)
        except FalaCurta:
            return None
        qualidade = avaliar(wav, self.sample_rate)
        self.canal.observar(qualidade)
        leitura = Leitura(indice=self._n, instante=self._inicio / self.sample_rate, score=p.score,
                          rms=float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))), qualidade=qualidade,
                          por_modelo=(p.score,), logodds=p.logodds, logodds_por_modelo=(p.logodds,),
                          canal=self.canal.veredito, fracao_fala=1.0)
        self._n += 1
        return leitura


def frases_de(modelo, wav: np.ndarray, sample_rate: int = 16000, bloco: int = 1600,
              aquecer: bool = True) -> list[Leitura]:
    """As leituras do ao vivo para um áudio já gravado: a calibração passa pelo mesmo
    caminho da demonstração. Ao vivo, o piso se forma nos segundos de sala antes da
    fala; aqui, com `aquecer`, sai do arquivo inteiro."""
    a = AnalisadorFrases(modelo, sample_rate=sample_rate)
    if aquecer:
        a.fala.aquecer(wav, a.quadro)
    leituras = [x for i in range(0, len(wav), bloco) for x in a.processar(wav[i:i + bloco])]
    return leituras + list(a.finalizar())
