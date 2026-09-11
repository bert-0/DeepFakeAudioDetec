"""Analisador contínuo: liga a captura aos modelos já treinados.

Reaproveita integralmente o caminho de inferência do projeto — o mesmo
`FeatureExtractor`, o mesmo `build_model`, o mesmo threshold gravado no
checkpoint. A única coisa que muda em relação ao `infer.py` é a origem do
waveform: em vez de `load_audio`, vem da janela deslizante.

**Fusão de scores.** Aceita mais de um checkpoint e combina as saídas pela
média. Medido no eval completo: `baseline_v2` sozinho 18,99%, `fusion_v4`
sozinho 20,18%, os dois fundidos pela média **14,03%**. A regra `rank` chega a
13,13%, mas exige o conjunto inteiro de scores para atribuir postos — não existe
num fluxo ao vivo, onde há uma janela por vez. Os 0,90 pp de diferença são o
preço da viabilidade em tempo real.

**Resolução temporal.** A janela é de `audio.duration` segundos porque foi assim
que os modelos foram treinados (`fix_length` força esse comprimento em toda
amostra). Não é parâmetro livre: mudá-la exigiria retreinar, e as métricas
medidas deixariam de valer. A consequência prática é um limite duro — um trecho
sintético **mais curto que a janela** nunca ocupa uma janela inteira, e o modelo
sempre o vê misturado com áudio real. Hop menor não resolve isso.
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

#: Abaixo disto o trecho é silêncio ou ruído de fundo, e o score não significa
#: nada — o modelo nunca viu silêncio rotulado. Analisar mesmo assim encheria o
#: histórico de valores arbitrários e faria a média perder o sentido.
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
        """Peso desta janela nas médias, em [0, 1].

        **Só a parcela de fala é ponderada; o canal é binário.** A degradação de
        canal foi medida (+5,35 pp em banda estreita), e a resposta medida é
        excluir — não atenuar. Inventar um peso contínuo para o canal seria
        supor uma curva EER×qualidade que ninguém mediu.

        A parcela de fala tem outra justificativa, e ela é estrutural: o
        `preprocess_waveform` remove o silêncio e completa por **repetição**.
        Uma janela com 10% de fala vira um trecho de 0,48 s repetido 8 vezes —
        entrada que não existe no conjunto de treino. Quanto menos fala
        original, mais a janela se afasta do domínio, e menos ela deve pesar.

        O peso é a própria fração de fala: é a grandeza que causa a repetição,
        sem constante de ajuste no meio.
        """
        if not self.confiavel:
            return 0.0
        return max(0.0, min(1.0, self.fracao_fala))

    @property
    def confiavel(self) -> bool:
        """Janela que pode entrar nas médias e virar indício.

        Exclui silêncio (o modelo nunca viu silêncio rotulado) e canal julgado
        de banda estreita (medido: +5,35 pp de EER no `fusion_v4`, fora do
        domínio em que o sistema foi avaliado). `indeterminado` entra: na dúvida
        o sistema continua medindo, em vez de se calar sem evidência.
        """
        if self.silencio:
            return False
        return self.canal != EstadoDoCanal.ESTREITA


@dataclass
class Agregador:
    """Acumula as leituras e resume o que foi ouvido até agora.

    A média móvel existe porque uma janela isolada é frágil: 4 s de áudio
    passado por codec, com o modelo operando fora do domínio de treino. A
    decisão útil vem da tendência, não de um ponto.

    **As janelas se sobrepõem, então elas não são observações independentes.**
    Com janela de 4 s e passo de 2 s, vizinhas compartilham metade do áudio: uma
    média de 5 janelas cobre 12 s, o equivalente a **3** janelas independentes.
    Reportar "média de 5" sugeriria mais solidez do que existe, então o resumo
    devolve também o número efetivo.
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
        """Quantas janelas INDEPENDENTES cabem em `n_janelas` sobrepostas.

        `n` janelas com passo `h` cobrem `(n-1)*h + w` segundos de áudio, que
        equivalem a `coberto / w` janelas sem sobreposição.
        """
        if n_janelas <= 0:
            return 0.0
        coberto = (n_janelas - 1) * self.passo_s + self.janela_s
        return coberto / self.janela_s

    def media_movel(self) -> float | None:
        """Média das últimas janelas confiáveis, **ponderada** pelo peso de cada uma.

        Ver `Leitura.peso`. Se todos os pesos forem iguais, recai na média
        simples — o comportamento anterior.
        """
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
    """Um checkpoint carregado, com o extractor que combina com ele.

    Cada modelo tem o seu próprio `FeatureExtractor`: o `baseline_v2` usa
    `n_filter: 20` e só LFCC, o `fusion_v4` usa 70 e também espectrograma. O
    waveform é o mesmo; as features, não.
    """

    def __init__(self, checkpoint: str, device: torch.device, config: dict):
        dados = torch.load(checkpoint, map_location=device, weights_only=False)
        # O config do checkpoint é a fonte da verdade: garante que as features
        # extraídas aqui sejam as mesmas com que o modelo foi treinado.
        self.config = dados.get("config", config)
        self.threshold = dados.get("threshold")
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

    `checkpoint` aceita um caminho ou uma lista. Com mais de um, os scores são
    fundidos pela média — ver o cabeçalho do módulo para o porquê da média e não
    do posto.
    """

    def __init__(self, config: dict, checkpoint: str | list[str],
                 device: torch.device, hop_s: float | None = None):
        caminhos = [checkpoint] if isinstance(checkpoint, (str, bytes)) else list(checkpoint)
        if not caminhos:
            raise ValueError("é preciso ao menos um checkpoint")
        self.modelos = [_Modelo(c, device, config) for c in caminhos]

        # A janela vem do primeiro modelo; os demais precisam concordar, senão
        # estariam vendo trechos de comprimentos diferentes do mesmo áudio.
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
        self.device = device
        self.sample_rate = int(audio_cfg["sample_rate"])
        tamanho = int(self.sample_rate * float(audio_cfg["duration"]))
        passo = int(self.sample_rate * (hop_s if hop_s else float(audio_cfg["duration"]) / 2))
        self.janela = JanelaDeslizante(tamanho=tamanho, passo=max(1, passo))
        # O canal é propriedade da SESSÃO, não da janela: ausência de alta
        # frequência numa janela pode ser o locutor (uma vogal), e só a
        # acumulação ao longo do tempo distingue isso do canal.
        self.canal = EstadoDoCanal()
        self._n = 0

    @property
    def n_modelos(self) -> int:
        return len(self.modelos)

    def _classificar(self, wav: np.ndarray) -> tuple[float, tuple[float, ...]]:
        """Devolve (score fundido, score de cada modelo)."""
        # Sem `rng`: o recorte aleatório é de treino. Aqui a janela já tem o
        # comprimento exato, então `preprocess_waveform` só normaliza.
        pronto = preprocess_waveform(wav, self.config["audio"])
        individuais = tuple(m.score(pronto) for m in self.modelos)
        return float(np.mean(individuais)), individuais

    def processar(self, bloco: np.ndarray):
        """Consome um bloco da fonte e devolve as leituras que ele completou."""
        for wav in self.janela.alimentar(bloco):
            rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
            if rms < SILENCIO_RMS:
                score, individuais, qualidade, fracao_fala = 0.0, (), None, 0.0
            else:
                qualidade = avaliar(wav, self.sample_rate)
                self.canal.observar(qualidade)
                # Quanto da janela é fala, antes de o fix_length completar por
                # repetição. É o que determina o peso — ver `Leitura.peso`.
                limpo = trim_silence(wav, self.config["audio"].get("top_db", 30))
                fracao_fala = len(limpo) / max(wav.size, 1)
                score, individuais = self._classificar(wav)
            leitura = Leitura(
                indice=self._n,
                instante=self.janela.instante_da_janela(self._n, self.sample_rate),
                score=score,
                rms=rms,
                qualidade=qualidade,
                por_modelo=individuais,
                canal=self.canal.veredito,
                fracao_fala=fracao_fala,
            )
            self._n += 1
            yield leitura
