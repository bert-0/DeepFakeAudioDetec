"""Análise de um arquivo de áudio para a interface web.

Usa o mesmo caminho do monitor (`AnalisadorContinuo` sobre `FileSource`), que dá
o score do `infer.py` (tests/test_monitor_consistencia.py). O limiar depende da
taxa nativa do arquivo (`src/limiares.py`).

Com `DETECTOR_LIMIAR_FUSAO` (o JSON do `scripts/limiar_fusao.py`), a análise usa
os modelos, a regra e o limiar desse arquivo, com a taxa comum de 48 kHz: o mesmo
caminho do ensaio 2 (RESUMO_TCC §10.2.13). A aba Ao vivo lê a mesma variável.

Com `DETECTOR_XLSR` (o JSON do `scripts/limiar_xlsr.py`), a análise usa o modelo
final, o XLS-R congelado (§10.2.43), pelo caminho das réguas: o score é o do
arquivo inteiro; as barras por janela são só para ver onde ele sobe.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.capture import FileSource
from src.capture.analyzer import Agregador, AnalisadorContinuo
from src.config import load_config, resolve_device
from src.limiares import (
    Escolha,
    escolher_limiar,
    ler_limiar_fusao,
    limiares_do_checkpoint,
    taxa_nativa,
)

CONFIG_PADRAO = "configs/baseline_v2.yaml"
CHECKPOINT_PADRAO = "checkpoints/baseline_lfcc_cnn_v2.pt"
#: A do ensaio 2: clone de 24 kHz e gravação de 48 kHz pela mesma conversão final.
TAXA_COMUM_FUSAO = 48000


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


def nome_dos_modelos(checkpoints: list[str] | tuple[str, ...]) -> str:
    """Um modelo: o nome dele; dois: "fusão: a + b" (o mesmo da aba Ao vivo)."""
    nomes = [Path(c).stem for c in checkpoints]
    return nomes[0] if len(nomes) == 1 else "fusão: " + " + ".join(nomes)


class Detector:
    """Carrega config e checkpoint uma vez; cada análise cria seu analisador,
    para o estado do canal não vazar de um arquivo para outro."""

    def __init__(self, config: str | None = None, checkpoint: str | None = None,
                 device: str | None = None, limiar_fusao: str | None = None,
                 xlsr: str | None = None):
        self.device = resolve_device(device or os.environ.get("DETECTOR_DEVICE", "cpu"))
        self.limiar_xlsr = None
        self._modelo_final = None
        caminho_xlsr = xlsr or os.environ.get("DETECTOR_XLSR")
        if caminho_xlsr:
            self._iniciar_xlsr(caminho_xlsr)
            return
        caminho_fusao = limiar_fusao or os.environ.get("DETECTOR_LIMIAR_FUSAO")
        self.fusao = ler_limiar_fusao(caminho_fusao) if caminho_fusao else None
        if self.fusao:
            self.config_path = self.fusao.config
            self.checkpoints = list(self.fusao.checkpoints)
            self.checkpoint = self.checkpoints[0]
            self.limiares = None
            self.modelo = nome_dos_modelos(self.checkpoints)
        else:
            self.config_path = config or os.environ.get("DETECTOR_CONFIG", CONFIG_PADRAO)
            self.checkpoint = checkpoint or os.environ.get("DETECTOR_CHECKPOINT",
                                                           CHECKPOINT_PADRAO)
            self.checkpoints = [self.checkpoint]
            dados = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            self.limiares = limiares_do_checkpoint(self.checkpoint, dados)
        self.config = load_config(self.config_path)
        if not self.fusao:
            self.modelo = self.config["model"]["name"]

    def _iniciar_xlsr(self, caminho: str) -> None:
        from src.xlsr_final import NOME, ler_limiar

        self.limiar_xlsr = ler_limiar(caminho)
        self.fusao, self.limiares, self.config = None, None, None
        self.config_path, self.checkpoint = None, self.limiar_xlsr.cabeca
        self.checkpoints = [self.checkpoint]
        self.modelo = NOME

    @property
    def xlsr(self) -> bool:
        return self.limiar_xlsr is not None

    def modelo_final(self):
        """O XLS-R só carrega na primeira análise (~1,2 GB e uns 13 s na GPU): abrir a
        página Sobre não precisa dele."""
        if self._modelo_final is None:
            from src.xlsr_final import ModeloFinal

            self._modelo_final = ModeloFinal(self.limiar_xlsr.cabeca, str(self.device),
                                             modelo_hf=self.limiar_xlsr.fonte_modelo)
        return self._modelo_final

    @property
    def regra(self) -> str:
        if self.xlsr:
            return "logodds"
        return self.fusao.regra if self.fusao else "probabilidade"

    def _analisar_xlsr(self, caminho: str | Path, nome_original: str | None) -> Resultado:
        from src.xlsr_final import sigmoide

        lim = self.limiar_xlsr
        p = self.modelo_final().pontuar_arquivo(caminho, por_frases=lim.arquivo_por_frase)
        scores = [sigmoide(z) for _, z in p.janelas]
        n = len(scores)
        return Resultado(
            arquivo=nome_original or Path(caminho).name, duracao_s=p.duracao_s,
            taxa_nativa=taxa_nativa(caminho), score_medio=p.score, score_maximo=max(scores),
            limiar=lim.probabilidade_arquivo,
            origem_limiar=lim.descricao_frase() if lim.arquivo_por_frase else lim.descricao(),
            aviso_limiar=None, janelas_uteis=n,
            janelas_independentes=Agregador().efetivas(n),
            janelas=[{"t": round(t, 2), "score": significativos(s), "util": True, "peso": 1.0}
                     for (t, _), s in zip(p.janelas, scores, strict=True)])

    def analisar(self, caminho: str | Path, nome_original: str | None = None) -> Resultado:
        if self.xlsr:
            return self._analisar_xlsr(caminho, nome_original)
        analisador = AnalisadorContinuo(self.config, self.checkpoints, self.device,
                                        regra=self.regra)
        agregador = Agregador(
            janela_s=analisador.janela.tamanho / analisador.sample_rate,
            passo_s=analisador.janela.passo / analisador.sample_rate, regra=self.regra)
        taxa_comum = TAXA_COMUM_FUSAO if self.fusao else None
        with FileSource(caminho, analisador.sample_rate, taxa_comum=taxa_comum) as fonte:
            for bloco in fonte.blocos():
                for leitura in analisador.processar(bloco):
                    agregador.adicionar(leitura)
        for leitura in analisador.finalizar():
            agregador.adicionar(leitura)

        taxa = taxa_nativa(caminho)
        if self.fusao:
            escolha = Escolha(self.fusao.probabilidade, self.fusao.descricao())
        else:
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
            janelas=[{"t": round(x.instante, 2), "score": significativos(x.score),
                      "util": id(x) in uteis, "peso": round(x.peso, 3)}
                     for x in agregador.leituras],
        )


def significativos(x: float, n: int = 6) -> float:
    """Arredonda em algarismos significativos: com limiar de 0,00017 (modelo do
    recorte), 4 casas decimais trocariam o lado do limiar de uma janela."""
    return float(f"{x:.{n}g}")
