"""Análise de um arquivo de áudio para a interface web.

Usa exatamente o caminho do monitor (`AnalisadorContinuo` sobre `FileSource`),
que dá o mesmo score do `infer.py` para o mesmo áudio
(`tests/test_monitor_consistencia.py`). O limiar segue o tipo de áudio
(`src/limiares.py`): arquivo nativo de 16 kHz usa o original; gravado acima
disso, o recalibrado para conversão de taxa.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.capture import FileSource
from src.capture.analyzer import Agregador, AnalisadorContinuo
from src.config import load_config, resolve_device
from src.limiares import escolher_limiar, limiares_do_checkpoint, taxa_nativa

CONFIG_PADRAO = "configs/baseline_v2.yaml"
CHECKPOINT_PADRAO = "checkpoints/baseline_lfcc_cnn_v2.pt"


@dataclass
class Resultado:
    arquivo: str
    duracao_s: float
    taxa_nativa: int | None
    score_medio: float | None
    score_maximo: float | None
    limiar: float | None
    origem_limiar: str
    aviso_limiar: str | None
    janelas_uteis: int
    janelas_independentes: float
    janelas: list[dict] = field(default_factory=list)

    @property
    def indicio_de_sintese(self) -> bool | None:
        if self.score_medio is None or self.limiar is None:
            return None
        return self.score_medio >= self.limiar


class Detector:
    """Carrega config e checkpoint uma vez; cada análise cria o seu analisador
    (o estado do canal é da sessão, não pode vazar de um arquivo para outro)."""

    def __init__(self, config: str | None = None, checkpoint: str | None = None,
                 device: str | None = None):
        self.config_path = config or os.environ.get("DETECTOR_CONFIG", CONFIG_PADRAO)
        self.checkpoint = checkpoint or os.environ.get("DETECTOR_CHECKPOINT", CHECKPOINT_PADRAO)
        self.config = load_config(self.config_path)
        self.device = resolve_device(device or os.environ.get("DETECTOR_DEVICE", "cpu"))
        dados = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.limiares = limiares_do_checkpoint(self.checkpoint, dados)
        self.modelo = self.config["model"]["name"]

    def analisar(self, caminho: str | Path, nome_original: str | None = None) -> Resultado:
        analisador = AnalisadorContinuo(self.config, self.checkpoint, self.device)
        agregador = Agregador(
            janela_s=analisador.janela.tamanho / analisador.sample_rate,
            passo_s=analisador.janela.passo / analisador.sample_rate)
        with FileSource(caminho, analisador.sample_rate) as fonte:
            for bloco in fonte.blocos():
                for leitura in analisador.processar(bloco):
                    agregador.adicionar(leitura)
        for leitura in analisador.finalizar():
            agregador.adicionar(leitura)

        taxa = taxa_nativa(caminho)
        escolha = escolher_limiar(self.limiares, analisador.sample_rate, taxa)
        resumo = agregador.resumo()
        uteis = set(id(x) for x in agregador.uteis)
        return Resultado(
            arquivo=nome_original or Path(caminho).name,
            duracao_s=len(fonte._wav) / analisador.sample_rate,
            taxa_nativa=taxa,
            score_medio=resumo["score_medio"],
            score_maximo=resumo["score_maximo"],
            limiar=escolha.limiar,
            origem_limiar=escolha.origem,
            aviso_limiar=escolha.aviso,
            janelas_uteis=resumo["janelas_uteis"],
            janelas_independentes=resumo["janelas_independentes"],
            janelas=[{"t": round(x.instante, 2), "score": round(x.score, 4),
                      "util": id(x) in uteis, "peso": round(x.peso, 3)}
                     for x in agregador.leituras],
        )
