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
        - {tipo: rawboost, prob: 0.7, opcoes: [lnl+isd, ssi]}
        - {tipo: codec, prob: 0.3, opcoes: [mp3, vorbis, opus, ulaw, gsm]}
        - {tipo: oclusao, prob: 0.5, faixa: [4000, 7900]}

`nivel` fixa o parâmetro, `faixa` sorteia um valor entre dois por amostra e
`opcoes` sorteia um da lista. O parâmetro é o do `ChannelDegradation` (opus:
compression_level; band: taxa intermediária; captura e captura_fir: taxa do
dispositivo; oclusao: frequência a partir da qual o espectro é zerado, o
descarte aleatório da banda alta), no ruído a SNR em dB, no RawBoost o nome da variante
(`src/preprocess/rawboost.py`), cujos padrões `parametros:` pode trocar, e no
codec o nome dele (`src/preprocess/codecs.py`), com o nível de compressão
sorteado na faixa padrão de cada um ou na de `qualidade: {mp3: [0.3, 0.9]}`.
"""

from __future__ import annotations

import numpy as np

from .augment import add_noise
from .channel import ChannelDegradation

TIPOS = ("opus", "band", "captura", "captura_fir", "oclusao", "ruido", "rawboost", "codec")
#: Tipos cujo parâmetro é um nome, e não um número.
_NOMINAIS = ("rawboost", "codec")
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
        nomes = [self.valor] if self.forma == "nivel" else self.valor
        if self.tipo in _NOMINAIS and self.forma == "faixa":
            raise ValueError(f"{self.tipo}: use nivel ou opcoes com o nome, não faixa")
        if self.tipo == "rawboost":
            from . import rawboost

            for nome in nomes:
                rawboost.variante(nome)
            self.parametros_rawboost = rawboost.parametros(cfg.get("parametros"))
        elif "parametros" in cfg:
            raise ValueError(f"{self.tipo}: 'parametros' só vale para rawboost")
        if self.tipo == "codec":
            from . import codecs

            qualidade = dict(cfg.get("qualidade") or {})
            self.faixas_codec = {nome: codecs.faixa_padrao(nome) for nome in nomes}
            for nome, faixa in qualidade.items():
                if nome not in self.faixas_codec:
                    raise ValueError(f"codec: qualidade de {nome!r}, que não está nas opções")
                if self.faixas_codec[nome] is None:
                    raise ValueError(f"codec: {nome} não tem nível de compressão")
                if len(faixa) != 2 or not 0 <= min(faixa) <= max(faixa) <= 0.99:
                    raise ValueError(f"codec: qualidade de {nome} precisa de dois valores em [0, 0.99]")
                self.faixas_codec[nome] = (float(min(faixa)), float(max(faixa)))
        elif "qualidade" in cfg:
            raise ValueError(f"{self.tipo}: 'qualidade' só vale para codec")
        self.sample_rate = int(sample_rate)

    def parametro(self, rng: np.random.Generator):
        if self.forma == "faixa":
            lo, hi = self.valor
            return float(rng.uniform(lo, hi))
        valor = (self.valor if self.forma == "nivel"
                 else self.valor[int(rng.integers(len(self.valor)))])
        return valor if self.tipo in _NOMINAIS else float(valor)

    def __call__(self, wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        p = self.parametro(rng)
        if self.tipo == "ruido":
            return add_noise(wav, p, rng)
        if self.tipo == "rawboost":
            from . import rawboost

            return rawboost.aplicar(wav, p, rng, self.parametros_rawboost, self.sample_rate)
        if self.tipo == "codec":
            from . import codecs

            faixa = self.faixas_codec[p]
            nivel = float(rng.uniform(*faixa)) if faixa else None
            return codecs.codificar(wav, self.sample_rate, p, nivel)
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
