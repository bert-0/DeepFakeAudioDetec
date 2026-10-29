"""Sala simulada: a resposta de uma sala real e o silêncio gravado nela (BUT ReverbDB).

O aluno da destilação imita mal a camada 7 do XLS-R no áudio que passou por
microfone e sala (RESUMO_TCC §10.2.37). Sem tocar nada em casa, a passagem pela
sala é simulada com as medidas do BUT ReverbDB (9 salas, 31 microfones, CC BY
4.0): convolução com a resposta ao impulso de uma posição e soma do silêncio que
o mesmo microfone gravou nela. É também o item "resposta ao impulso" do plano.

Estrutura esperada (a do pacote `RIR-Only`): cada posição é uma pasta com
`RIR/*.wav` (1 s a 16 kHz) e `silence/*.wav` (60 s).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Posicao:
    rirs: tuple[Path, ...]
    silencios: tuple[Path, ...] = field(default_factory=tuple)


def indexar(raiz: str | Path) -> list[Posicao]:
    """Posições com ao menos uma resposta ao impulso, em ordem fixa."""
    posicoes = []
    for pasta_rir in sorted(Path(raiz).rglob("RIR")):
        rirs = tuple(sorted(pasta_rir.glob("*.wav")))
        if not pasta_rir.is_dir() or not rirs:
            continue
        silencios = tuple(sorted((pasta_rir.parent / "silence").glob("*.wav")))
        posicoes.append(Posicao(rirs, silencios))
    return posicoes


def preparar_rir(h: np.ndarray, sample_rate: int, max_s: float = 1.0) -> np.ndarray:
    """Começa 1 ms antes do som direto (o pico), para a fala não atrasar; energia 1."""
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    inicio = max(0, int(np.argmax(np.abs(h))) - int(0.001 * sample_rate))
    h = h[inicio:inicio + int(max_s * sample_rate)]
    return h / (np.sqrt(np.sum(h ** 2)) + 1e-12)


def _ler(caminho: Path, sample_rate: int, inicio: int | None = None, n: int | None = None) -> np.ndarray:
    import soundfile as sf

    info = sf.info(str(caminho))
    if info.samplerate == sample_rate and inicio is not None:
        x, _ = sf.read(str(caminho), start=inicio, frames=n, dtype="float64", always_2d=True)
        return x[:, 0]
    x, taxa = sf.read(str(caminho), dtype="float64", always_2d=True)
    x = x[:, 0]
    if taxa != sample_rate:
        from .reamostragem import para_taxa

        x = np.asarray(para_taxa(x.astype(np.float32), taxa, sample_rate), dtype=np.float64)
    return x if inicio is None else x[inicio:inicio + n]


class Sala:
    """`sala(wav, rng)`: o mesmo comprimento, passado por uma sala sorteada, pico em 1.

    Serve também de condição do `robustness_eval.py`, cujos workers copiam o
    objeto entre processos: por isso o cache é um dicionário, não `lru_cache`."""

    def __init__(self, raiz: str | Path, sample_rate: int = 16000, faixa_snr: tuple[float, float] = (10.0, 30.0),
                 prob_ruido: float = 1.0, max_rir_s: float = 1.0):
        self.posicoes = indexar(raiz)
        if not self.posicoes:
            raise ValueError(f"nenhuma pasta RIR/ com .wav em {raiz}")
        self.sample_rate, self.faixa_snr = sample_rate, faixa_snr
        self.prob_ruido, self.max_rir_s = prob_ruido, max_rir_s
        self._cache: dict[Path, np.ndarray] = {}

    def _rir(self, caminho: Path) -> np.ndarray:
        if caminho not in self._cache:
            self._cache[caminho] = preparar_rir(_ler(caminho, self.sample_rate), self.sample_rate, self.max_rir_s)
        return self._cache[caminho]

    def descricao(self) -> str:
        n_rir = sum(len(p.rirs) for p in self.posicoes)
        return (f"{len(self.posicoes)} posições, {n_rir} respostas ao impulso; ruído da sala a "
                f"{self.faixa_snr[0]:g}-{self.faixa_snr[1]:g} dB com chance {self.prob_ruido:g}")

    def _silencio(self, caminho: Path, n: int, rng: np.random.Generator) -> np.ndarray:
        import soundfile as sf

        total = sf.info(str(caminho)).frames
        if total >= n:
            return _ler(caminho, self.sample_rate, int(rng.integers(0, total - n + 1)), n)
        x = _ler(caminho, self.sample_rate)
        return np.tile(x, int(np.ceil(n / max(len(x), 1))))[:n]

    def __call__(self, wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        from scipy.signal import fftconvolve

        x = np.asarray(wav, dtype=np.float64)
        pos = self.posicoes[int(rng.integers(len(self.posicoes)))]
        h = self._rir(pos.rirs[int(rng.integers(len(pos.rirs)))])
        y = fftconvolve(x, h)[:len(x)]
        if pos.silencios and rng.random() < self.prob_ruido:
            ruido = self._silencio(pos.silencios[int(rng.integers(len(pos.silencios)))], len(y), rng)
            snr = rng.uniform(*self.faixa_snr)
            nivel = np.sqrt(np.mean(y ** 2)) / (np.sqrt(np.mean(ruido ** 2)) + 1e-12)
            y = y + ruido * nivel * 10 ** (-snr / 20)
        pico = np.max(np.abs(y))
        return (y / pico if pico > 0 else y).astype(np.float32)
