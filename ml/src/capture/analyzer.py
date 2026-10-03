"""Analisador contínuo: aplica os modelos treinados às janelas da captura.

Mesmo caminho de inferência do `infer.py`, mas o waveform vem da janela
deslizante. Com vários checkpoints, os scores são fundidos pela média.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..features import FeatureExtractor
from ..models import build_model
from ..preprocess import preprocess_waveform
from ..preprocess.audio import trim_silence
from .qualidade import EstadoDoCanal, Qualidade, avaliar
from .stream import JanelaDeslizante

#: Abaixo disto é silêncio; o modelo nunca viu silêncio rotulado, o score não vale.
SILENCIO_RMS = 1e-3


@dataclass
class Leitura:
    """Resultado de uma janela."""
    indice: int
    instante: float           # segundos desde o início da captura
    score: float              # probabilidade de spoof fundida, em [0, 1]
    rms: float                # energia do trecho; separa silêncio de fala
    qualidade: Qualidade | None = None   # medição de canal desta janela
    por_modelo: tuple[float, ...] = ()   # score de cada modelo, antes da fusão
    canal: str = EstadoDoCanal.INDETERMINADO   # veredito da SESSÃO neste instante
    fracao_fala: float = 1.0   # quanto da janela sobrou após remover o silêncio

    @property
    def silencio(self) -> bool:
        return self.rms < SILENCIO_RMS

    @property
    def peso(self) -> float:
        """Peso desta janela nas médias, em [0, 1]: a fração de fala.

        Pouca fala vira muita repetição no preprocess, fora do domínio de treino.
        O canal não entra no peso; janela de banda estreita já é excluída.
        """
        if not self.confiavel:
            return 0.0
        return max(0.0, min(1.0, self.fracao_fala))

    @property
    def confiavel(self) -> bool:
        """Janela que pode entrar nas médias: nem silêncio, nem banda estreita.

        Canal `indeterminado` entra, para não calar o sistema sem evidência.
        """
        if self.silencio:
            return False
        return self.canal != EstadoDoCanal.ESTREITA


@dataclass
class Agregador:
    """Acumula as leituras e resume o que foi ouvido até agora.

    Como as janelas se sobrepõem, o resumo informa também quantas equivalem a
    janelas independentes.
    """
    janela_media: int = 5
    janela_s: float = 4.0
    passo_s: float = 2.0
    leituras: list[Leitura] = field(default_factory=list)

    def adicionar(self, leitura: Leitura) -> None:
        self.leituras.append(leitura)

    @property
    def uteis(self) -> list[Leitura]:
        """Só as janelas que podem entrar numa média."""
        return [x for x in self.leituras if x.confiavel]

    def efetivas(self, n_janelas: int) -> float:
        """Quantas janelas independentes equivalem a `n_janelas` sobrepostas."""
        if n_janelas <= 0:
            return 0.0
        coberto = (n_janelas - 1) * self.passo_s + self.janela_s
        return coberto / self.janela_s

    def media_movel(self) -> float | None:
        """Média das últimas janelas confiáveis, ponderada por `Leitura.peso`."""
        recentes = self.uteis[-self.janela_media:]
        if not recentes:
            return None
        pesos = np.array([x.peso for x in recentes], dtype=float)
        scores = np.array([x.score for x in recentes], dtype=float)
        if pesos.sum() <= 0:
            return None
        return float(np.average(scores, weights=pesos))

    def resumo(self) -> dict:
        uteis = self.uteis
        scores = [x.score for x in uteis]
        pesos = [x.peso for x in uteis]
        silencio = sum(1 for x in self.leituras if x.silencio)
        descartadas = len(self.leituras) - len(uteis) - silencio
        return {
            "janelas_total": len(self.leituras),
            "janelas_uteis": len(uteis),
            "janelas_silencio": silencio,
            "janelas_canal_ruim": descartadas,
            "janelas_independentes": round(self.efetivas(len(uteis)), 1),
            "score_medio": (float(np.average(scores, weights=pesos))
                            if scores and sum(pesos) > 0 else None),
            "score_medio_simples": float(np.mean(scores)) if scores else None,
            "score_mediano": float(np.median(scores)) if scores else None,
            "score_maximo": float(np.max(scores)) if scores else None,
            "peso_medio": float(np.mean(pesos)) if pesos else None,
            "duracao_s": uteis[-1].instante if uteis else 0.0,
        }


class _Modelo:
    """Um checkpoint carregado, com o seu próprio `FeatureExtractor`.

    Cada modelo usa features diferentes sobre o mesmo waveform.
    """

    def __init__(self, checkpoint: str, device: torch.device, config: dict):
        dados = torch.load(checkpoint, map_location=device, weights_only=False)
        # O config do checkpoint prevalece: garante as mesmas features do treino.
        self.config = dados.get("config", config)
        self.threshold = dados.get("threshold")
        # Só existe se o limiar foi recalibrado por scripts/calibrar_captura.py.
        self.calibracao = dados.get("calibracao")
        self.device = device
        self.extractor = FeatureExtractor(self.config["audio"], self.config["features"])
        self.rede = build_model(self.config["model"]).to(device)
        self.rede.load_state_dict(dados["model_state"])
        self.rede.eval()

    @torch.no_grad()
    def score(self, pronto: np.ndarray) -> float:
        feats = {k: v.unsqueeze(0).to(self.device)
                 for k, v in self.extractor(pronto).items()}
        return float(torch.softmax(self.rede(feats), dim=1)[0, 1])


class AnalisadorContinuo:
    """Carrega os checkpoints uma vez e classifica janela a janela.

    `checkpoint` aceita um caminho ou uma lista; com mais de um, faz a média dos scores.
    """

    def __init__(self, config: dict, checkpoint: str | list[str],
                 device: torch.device, hop_s: float | None = None):
        caminhos = [checkpoint] if isinstance(checkpoint, (str, bytes)) else list(checkpoint)
        if not caminhos:
            raise ValueError("é preciso ao menos um checkpoint")
        self.modelos = [_Modelo(c, device, config) for c in caminhos]

        # A janela vem do primeiro modelo; os demais precisam usar a mesma.
        self.config = self.modelos[0].config
        audio_cfg = self.config["audio"]
        for m in self.modelos[1:]:
            for chave in ("sample_rate", "duration"):
                if m.config["audio"][chave] != audio_cfg[chave]:
                    raise ValueError(
                        f"os checkpoints discordam em audio.{chave}: "
                        f"{audio_cfg[chave]} vs {m.config['audio'][chave]}. "
                        "A fusão exige a mesma janela nos dois modelos.")

        self.threshold = self.modelos[0].threshold
        self.calibracao = self.modelos[0].calibracao
        self.device = device
        self.sample_rate = int(audio_cfg["sample_rate"])
        tamanho = int(self.sample_rate * float(audio_cfg["duration"]))
        passo = int(self.sample_rate * (hop_s if hop_s else float(audio_cfg["duration"]) / 2))
        self.janela = JanelaDeslizante(tamanho=tamanho, passo=max(1, passo))
        # Canal é julgado por sessão: uma janela sem agudos pode ser só uma vogal.
        self.canal = EstadoDoCanal()
        self._n = 0

    @property
    def n_modelos(self) -> int:
        return len(self.modelos)

    def _classificar(self, wav: np.ndarray) -> tuple[float, tuple[float, ...]]:
        """Devolve (score fundido, score de cada modelo)."""
        # Sem `rng`: recorte aleatório é só de treino.
        pronto = preprocess_waveform(wav, self.config["audio"])
        individuais = tuple(m.score(pronto) for m in self.modelos)
        return float(np.mean(individuais)), individuais

    def _ler(self, wav: np.ndarray) -> Leitura:
        """Transforma uma janela (completa ou final) numa leitura."""
        rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
        if rms < SILENCIO_RMS:
            score, individuais, qualidade, fracao_fala = 0.0, (), None, 0.0
        else:
            qualidade = avaliar(wav, self.sample_rate)
            self.canal.observar(qualidade)
            # Divide pelo tamanho da janela, não do array: a janela final chega
            # mais curta e o restante é preenchido por repetição.
            limpo = trim_silence(wav, self.config["audio"].get("top_db", 30))
            fracao_fala = len(limpo) / max(self.janela.tamanho, 1)
            score, individuais = self._classificar(wav)
        leitura = Leitura(
            indice=self._n,
            instante=self.janela.inicio_da_ultima / self.sample_rate,
            score=score,
            rms=rms,
            qualidade=qualidade,
            por_modelo=individuais,
            canal=self.canal.veredito,
            fracao_fala=fracao_fala,
        )
        self._n += 1
        return leitura

    def processar(self, bloco: np.ndarray):
        """Consome um bloco da fonte e devolve as leituras que ele completou."""
        for wav in self.janela.alimentar(bloco):
            yield self._ler(wav)

    def finalizar(self):
        """Leitura do trecho final que não completou uma janela.

        Chamar ao fim da fonte; sem isso, áudio mais curto que a janela não gera leitura.
        """
        for wav in self.janela.finalizar():
            yield self._ler(wav)
