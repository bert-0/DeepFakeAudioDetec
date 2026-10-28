"""Modelo final: XLS-R congelado com pesos por camada e ramo LFCC (RESUMO_TCC §10.2.43).

Pelo mesmo caminho das réguas: `src/features/xlsr.py` dá a média de cada camada
em cada janela e o LFCC médio do arquivo, e a cabeça soma as camadas com os pesos
que aprendeu. O score é o log-odds da cabeça; acima do limiar, indício de síntese.

O limiar vem de um JSON do `scripts/limiar_xlsr.py`, que guarda a impressão dos
pesos da cabeça: se ela for treinada de novo, o limiar antigo deixa de valer.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.features.xlsr import MIN_AMOSTRAS, MODELO_HF, SR, inicios, janelas, lfcc_medio, por_janela, preparar_wav

CABECA_PADRAO = "checkpoints/xlsr_congelado.pt"
NOME = "xlsr_congelado"


class FalaCurta(RuntimeError):
    """Sobra menos de 0,25 s depois do corte de silêncio."""


def sigmoide(x: float) -> float:
    # Pela tangente: sem estouro de exp() com log-odds grandes.
    return float(0.5 * (1.0 + math.tanh(0.5 * x)))


def criar_cabeca(n_camadas: int = 25, dim: int = 1024, dim_lfcc: int = 120, dropout: float = 0.2):
    import torch
    from torch import nn

    class Cabeca(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {"n_camadas": n_camadas, "dim": dim, "dim_lfcc": dim_lfcc, "dropout": dropout}
            self.pesos = nn.Parameter(torch.zeros(n_camadas))
            self.register_buffer("media", torch.zeros(n_camadas, dim))
            self.register_buffer("escala", torch.ones(n_camadas, dim))
            self.register_buffer("media_lfcc", torch.zeros(dim_lfcc))
            self.register_buffer("escala_lfcc", torch.ones(dim_lfcc))
            self.queda = nn.Dropout(dropout)
            self.xlsr = nn.Linear(dim, 1)
            self.ramo_lfcc = nn.Linear(dim_lfcc, 1)

        def forward(self, camadas, lfcc):
            x = (camadas - self.media) / self.escala
            h = (torch.softmax(self.pesos, dim=0)[None, :, None] * x).sum(dim=1)
            z = (lfcc - self.media_lfcc) / self.escala_lfcc
            return (self.xlsr(self.queda(h)) + self.ramo_lfcc(self.queda(z))).squeeze(-1)

    return Cabeca()


def carregar_cabeca(caminho: str | Path, dispositivo: str = "cpu"):
    """(cabeça em modo de avaliação, o dicionário do checkpoint, o dispositivo)."""
    import torch

    dev = torch.device(dispositivo if torch.cuda.is_available() or dispositivo == "cpu" else "cpu")
    ck = torch.load(caminho, map_location=dev, weights_only=False)
    cabeca = criar_cabeca(**ck["config"]).to(dev)
    cabeca.load_state_dict(ck["estado"])
    return cabeca.eval(), ck, dev


def impressao(caminho: str | Path) -> str:
    import torch

    from src.scores import checkpoint_fingerprint

    return checkpoint_fingerprint(torch.load(caminho, map_location="cpu", weights_only=False)["estado"])


# --------------------------------------------------------------- limiar

@dataclass(frozen=True)
class LimiarXlsr:
    """Dois limiares, nos log-odds da cabeça: o do arquivo inteiro (Enviar arquivo) e o
    da frase cortada pelo ao vivo. Nas gravações do microfone, o arquivo leva segundos
    de ruído da sala que o corte de silêncio não tira, e a frase não; o score muda nos
    dois sentidos (até 1,7 nos log-odds nas vozes do L02), então cada caminho é
    calibrado com as mesmas vozes, pela mesma regra."""
    cabeca: str
    limiar: float
    origem: str
    regra: str = "maior voz real"
    modelo_hf: str = MODELO_HF
    locutores: dict = field(default_factory=dict)
    limiar_frase: float | None = None
    #: Opção (b) da decisão pendente: o Enviar arquivo também frase por frase.
    arquivo_por_frase: bool = False
    #: A pasta do XLS-R no cache, gravada na calibração: a interface não depende do HF_HOME.
    modelo_local: str | None = None

    @property
    def fonte_modelo(self) -> str:
        if self.modelo_local and Path(self.modelo_local).is_dir():
            return self.modelo_local
        return self.modelo_hf

    @property
    def probabilidade_arquivo(self) -> float:
        """O limiar do Enviar arquivo: o da frase, se o arquivo também vai por frase."""
        return self.probabilidade_frase if self.arquivo_por_frase else self.probabilidade

    @property
    def probabilidade(self) -> float:
        """O limiar na escala de 0 a 1, a que a tela e o histórico usam."""
        return sigmoide(self.limiar)

    @property
    def calibrado_por_frase(self) -> bool:
        return self.limiar_frase is not None

    @property
    def probabilidade_frase(self) -> float:
        """O do ao vivo; sem calibração por frase, o do arquivo."""
        return sigmoide(self.limiar if self.limiar_frase is None else self.limiar_frase)

    def descricao(self) -> str:
        return self.origem

    def descricao_frase(self) -> str:
        if self.limiar_frase is None:
            return self.origem + "; sem calibração por frase, vale o do arquivo"
        return self.origem + ", frase por frase"


def modelo_local(nome: str = MODELO_HF, cache: str | None = None) -> str | None:
    """A pasta do modelo no cache do Hugging Face, sem rede; None se não estiver lá."""
    # O snapshot_download recusa a cópia do cache, que não tem README nem .gitattributes.
    try:
        from huggingface_hub import try_to_load_from_cache

        config = try_to_load_from_cache(nome, "config.json", cache_dir=cache)
    except Exception:
        return None
    if not isinstance(config, str):
        return None
    pasta = Path(config).parent
    if not any((pasta / p).is_file() for p in ("pytorch_model.bin", "model.safetensors")):
        return None
    return str(pasta).replace("\\", "/")


def gravar_limiar(destino: str | Path, cabeca: str, limiar: float, origem: str, regra: str,
                  locutores: dict | None = None, data: str | None = None,
                  limiar_frase: float | None = None, arquivo_por_frase: bool = False) -> Path:
    import time

    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    if arquivo_por_frase and limiar_frase is None:
        raise ValueError("o arquivo por frase precisa do limiar da frase")
    dados = {"modelo": NOME, "cabeca": str(cabeca).replace("\\", "/"), "impressao": impressao(cabeca),
             "modelo_hf": MODELO_HF, "modelo_local": modelo_local(), "limiar": float(limiar),
             "limiar_frase": None if limiar_frase is None else float(limiar_frase),
             "arquivo_por_frase": bool(arquivo_por_frase),
             "regra": regra, "origem": origem,
             "locutores": locutores or {}, "data": data or time.strftime("%Y-%m-%d")}
    destino.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    return destino


def ler_limiar(caminho: str | Path) -> LimiarXlsr:
    """Lê o JSON e confere se a cabeça ainda tem os pesos com que o limiar foi escolhido."""
    r = json.loads(Path(caminho).read_text(encoding="utf-8"))
    if r.get("modelo") != NOME:
        raise ValueError(f"{caminho} não é um limiar do {NOME}")
    if impressao(r["cabeca"]) != r["impressao"]:
        raise ValueError(f"{r['cabeca']} mudou desde a escolha do limiar: rode o limiar_xlsr.py de novo")
    frase = r.get("limiar_frase")
    return LimiarXlsr(cabeca=r["cabeca"], limiar=float(r["limiar"]), origem=r["origem"],
                      regra=r.get("regra", ""), modelo_hf=r.get("modelo_hf", MODELO_HF),
                      locutores=r.get("locutores", {}), limiar_frase=None if frase is None else float(frase),
                      arquivo_por_frase=bool(r.get("arquivo_por_frase", False)),
                      modelo_local=r.get("modelo_local"))


# --------------------------------------------------------------- pontuar

@dataclass
class Pontuacao:
    logodds: float
    duracao_s: float              # o áudio lido, antes do corte de silêncio
    silencio_s: float
    #: (início em segundos, log-odds) de cada janela, só para ver; com `unidade` "frase",
    #: de cada frase, e o score é a média delas.
    janelas: list[tuple[float, float]] = field(default_factory=list)
    unidade: str = "janela"

    @property
    def score(self) -> float:
        return sigmoide(self.logodds)


class ModeloFinal:
    """O XLS-R e a cabeça, carregados uma vez. `rede` e `dev` substituem o XLS-R
    (nos testes, uma rede pequena)."""

    def __init__(self, cabeca: str | Path = CABECA_PADRAO, dispositivo: str = "cpu",
                 rede=None, dev=None, modelo_hf: str = MODELO_HF):
        self.cabeca, self.checkpoint, self.dev = carregar_cabeca(cabeca, dispositivo)
        if rede is None:
            from src.features.xlsr import carregar_rede

            rede, dev = carregar_rede(str(self.dev), modelo_hf)
        self.rede, self.dev_rede = rede, dev

    def _aplicar(self, camadas: np.ndarray, lfcc: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            c = torch.from_numpy(np.asarray(camadas, np.float32)).to(self.dev)
            z = self.cabeca(c, torch.from_numpy(np.asarray(lfcc, np.float32)).to(self.dev))
        return z.cpu().numpy()

    def pontuar_wav(self, bruto: np.ndarray, sample_rate: int = SR) -> Pontuacao:
        """Áudio já lido a 16 kHz, mono."""
        if sample_rate != SR:
            raise ValueError(f"o modelo final lê {SR} Hz, veio {sample_rate}")
        wav, silencio = preparar_wav(np.asarray(bruto, np.float32))
        if len(wav) < MIN_AMOSTRAS:
            raise FalaCurta(f"{len(wav) / SR:.3f} s de fala depois do corte de silêncio")
        camadas = np.stack([por_janela(self.rede, self.dev_rede, x)[1] for x in janelas(wav)])
        # Como no .npz do treino: média das janelas em float16.
        media = camadas.mean(axis=0).astype(np.float16).astype(np.float32)
        z = float(self._aplicar(media[None], lfcc_medio(wav)[None])[0])
        z_janelas = self._aplicar(camadas, np.stack([lfcc_medio(x) for x in janelas(wav)]))
        return Pontuacao(logodds=z, duracao_s=len(bruto) / SR, silencio_s=silencio,
                         janelas=[(i / SR, float(v)) for i, v in zip(inicios(len(wav)), z_janelas, strict=True)])

    def pontuar_por_frases(self, bruto: np.ndarray, sample_rate: int = SR) -> Pontuacao:
        """A opção (b) da decisão pendente (§10.2.44): o arquivo pelo caminho do ao vivo,
        frase por frase, e o score é a média dos log-odds das frases."""
        from src.capture.frases import frases_de

        leituras = [x for x in frases_de(self, np.asarray(bruto, np.float32), sample_rate) if x.confiavel]
        if not leituras:
            raise FalaCurta("nenhuma frase achada no áudio")
        return Pontuacao(logodds=float(np.mean([x.logodds for x in leituras])), duracao_s=len(bruto) / SR,
                         silencio_s=0.0, janelas=[(x.instante, x.logodds) for x in leituras], unidade="frase")

    def pontuar_arquivo(self, caminho: str | Path, por_frases: bool = False) -> Pontuacao:
        from src.preprocess.audio import load_audio

        bruto = load_audio(caminho, SR)
        return self.pontuar_por_frases(bruto) if por_frases else self.pontuar_wav(bruto)
