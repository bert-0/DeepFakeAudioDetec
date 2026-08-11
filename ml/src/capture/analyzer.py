"""Analisador contínuo: liga a captura ao modelo já treinado.

Reaproveita integralmente o caminho de inferência do projeto — o mesmo
`FeatureExtractor`, o mesmo `build_model`, o mesmo threshold gravado no
checkpoint. A única coisa que muda em relação ao `infer.py` é a origem do
waveform: em vez de `load_audio`, vem da janela deslizante.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..features import FeatureExtractor
from ..models import build_model
from ..preprocess import preprocess_waveform
from .stream import JanelaDeslizante


@dataclass
class Leitura:
    """Resultado de uma janela."""
    indice: int
    instante: float           # segundos desde o início da captura
    score: float              # probabilidade de spoof, em [0, 1]
    rms: float                # energia do trecho; separa silêncio de fala

    @property
    def silencio(self) -> bool:
        return self.rms < SILENCIO_RMS


#: Abaixo disto o trecho é silêncio ou ruído de fundo, e o score não significa
#: nada — o modelo nunca viu silêncio rotulado. Analisar mesmo assim encheria o
#: histórico de valores arbitrários e faria a média perder o sentido.
SILENCIO_RMS = 1e-3


@dataclass
class Agregador:
    """Acumula as leituras e resume o que foi ouvido até agora.

    A média móvel existe porque uma janela isolada é frágil: 4 s de áudio
    passado por codec, com o modelo operando fora do domínio de treino. A
    decisão útil vem da tendência, não de um ponto.
    """
    janela_media: int = 5
    leituras: list[Leitura] = field(default_factory=list)

    def adicionar(self, leitura: Leitura) -> None:
        self.leituras.append(leitura)

    @property
    def uteis(self) -> list[Leitura]:
        """Só as janelas com áudio de verdade."""
        return [x for x in self.leituras if not x.silencio]

    def media_movel(self) -> float | None:
        """Média dos scores das últimas janelas úteis. `None` se não houver."""
        recentes = self.uteis[-self.janela_media:]
        if not recentes:
            return None
        return float(np.mean([x.score for x in recentes]))

    def resumo(self) -> dict:
        uteis = self.uteis
        scores = [x.score for x in uteis]
        return {
            "janelas_total": len(self.leituras),
            "janelas_uteis": len(uteis),
            "janelas_silencio": len(self.leituras) - len(uteis),
            "score_medio": float(np.mean(scores)) if scores else None,
            "score_mediano": float(np.median(scores)) if scores else None,
            "score_maximo": float(np.max(scores)) if scores else None,
            "duracao_s": uteis[-1].instante if uteis else 0.0,
        }


class AnalisadorContinuo:
    """Carrega o checkpoint uma vez e classifica janela a janela."""

    def __init__(self, config: dict, checkpoint: str, device: torch.device,
                 hop_s: float | None = None):
        dados = torch.load(checkpoint, map_location=device, weights_only=False)
        # O config do checkpoint é a fonte da verdade: garante que as features
        # extraídas aqui sejam as mesmas com que o modelo foi treinado.
        self.config = dados.get("config", config)
        self.threshold = dados.get("threshold")
        self.device = device
        self.extractor = FeatureExtractor(self.config["audio"], self.config["features"])
        self.model = build_model(self.config["model"]).to(device)
        self.model.load_state_dict(dados["model_state"])
        self.model.eval()

        audio_cfg = self.config["audio"]
        self.sample_rate = int(audio_cfg["sample_rate"])
        tamanho = int(self.sample_rate * float(audio_cfg["duration"]))
        passo = int(self.sample_rate * (hop_s if hop_s else float(audio_cfg["duration"]) / 2))
        self.janela = JanelaDeslizante(tamanho=tamanho, passo=max(1, passo))
        self._n = 0

    @torch.no_grad()
    def _classificar(self, wav: np.ndarray) -> float:
        # Sem `rng`: o recorte aleatório é de treino. Aqui a janela já tem o
        # comprimento exato, então `preprocess_waveform` só normaliza.
        pronto = preprocess_waveform(wav, self.config["audio"])
        feats = {k: v.unsqueeze(0).to(self.device)
                 for k, v in self.extractor(pronto).items()}
        probs = torch.softmax(self.model(feats), dim=1)
        return float(probs[0, 1])

    def processar(self, bloco: np.ndarray):
        """Consome um bloco da fonte e devolve as leituras que ele completou."""
        for wav in self.janela.alimentar(bloco):
            rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
            score = 0.0 if rms < SILENCIO_RMS else self._classificar(wav)
            leitura = Leitura(
                indice=self._n,
                instante=self.janela.instante_da_janela(self._n, self.sample_rate),
                score=score,
                rms=rms,
            )
            self._n += 1
            yield leitura
