"""Lê os protocolos da ASVspoof 5 (trilha 1) e converte para o formato de 2019.

Campos, separados por espaço apesar do `.tsv` (README do Zenodo, registro
14498691): locutor arquivo gênero codec qualidade_do_codec original_do_codec
condição_de_adaptação ataque chave reservado. O README fala em "cinco colunas"
e lista dez; o último é reservado ("-"), então 9 ou 10 campos são aceitos e
cada linha é conferida pelo conteúdo, não só pela posição.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Formato de 2019 e amostra estratificada por (chave, ataque): os do 2021
# servem, porque só leem locutor, arquivo, ataque e chave.
from .asvspoof2021 import escrever_protocolo, linha_de_protocolo, subamostrar

__all__ = ["Enunciado", "ProtocoloInvalido", "conferir_tamanho", "escrever_protocolo",
           "ler_protocolo", "linha_de_protocolo", "recorte", "resumo", "subamostrar"]

CHAVES = ("bonafide", "spoof")
GENEROS = ("F", "M")
CAMPOS_MINIMOS = 9

#: Linhas de cada protocolo segundo o README; um download truncado não bate.
LINHAS_OFICIAIS = {
    "ASVspoof5.train.tsv": 182_357,
    "ASVspoof5.dev.track_1.tsv": 140_950,
    "ASVspoof5.eval.track_1.tsv": 680_774,
}

_ATAQUE = re.compile(r"A\d{2}$")
_ARQUIVO = re.compile(r"[TDE]_\d+$")


@dataclass(frozen=True)
class Enunciado:
    """Uma linha do protocolo. Os nomes dos campos que o resto do projeto lê
    (`locutor`, `arquivo`, `ataque`, `chave`) são os mesmos do Trial de 2021."""
    locutor: str
    arquivo: str
    genero: str          # F | M
    codec: str           # C01…C11, ou "-" (sem codec)
    codec_q: str         # configuração de qualidade do codec, ou "-"
    codec_origem: str    # enunciado original de onde a versão codificada veio, ou "-"
    adaptacao: str       # condição de adaptação do atacante (AC*), ou "-"
    ataque: str          # A01…A32, ou "-" para bonafide, como no protocolo de 2019
    chave: str           # bonafide | spoof

    @property
    def particao(self) -> str:
        """train, dev ou eval, pelo prefixo do arquivo (T_, D_, E_)."""
        return {"T": "train", "D": "dev", "E": "eval"}[self.arquivo[0]]


class ProtocoloInvalido(ValueError):
    """O arquivo não se parece com um protocolo da trilha 1 da ASVspoof 5."""


def _ler_linha(partes: list[str]) -> Enunciado | None:
    if len(partes) < CAMPOS_MINIMOS:
        return None
    locutor, arquivo, genero, codec, codec_q, origem, adaptacao, rotulo, chave = partes[:9]
    if chave not in CHAVES or genero not in GENEROS or not _ARQUIVO.match(arquivo):
        return None
    # O rótulo do ataque tem de bater com a chave; senão as colunas estão trocadas.
    if (chave == "bonafide") != (rotulo == "bonafide"):
        return None
    if chave == "spoof" and not _ATAQUE.match(rotulo):
        return None
    return Enunciado(locutor=locutor, arquivo=arquivo, genero=genero, codec=codec,
                     codec_q=codec_q, codec_origem=origem, adaptacao=adaptacao,
                     ataque="-" if chave == "bonafide" else rotulo, chave=chave)


def ler_protocolo(caminho: str | Path, relatorio: dict | None = None) -> list[Enunciado]:
    """Lê o protocolo inteiro.

    `relatorio`, se passado, recebe `ignoradas` e até 3 `exemplos`. Levanta
    `ProtocoloInvalido` se nenhuma linha for reconhecida.
    """
    enunciados: list[Enunciado] = []
    ignoradas: list[str] = []
    with open(caminho, "r", encoding="utf-8") as fh:
        for linha in fh:
            partes = linha.split()
            if not partes or partes[0] == "SPEAKER_ID":
                continue
            e = _ler_linha(partes)
            if e is None:
                ignoradas.append(linha.rstrip())
            else:
                enunciados.append(e)
    if relatorio is not None:
        relatorio["ignoradas"] = len(ignoradas)
        relatorio["exemplos"] = ignoradas[:3]
    if not enunciados:
        raise ProtocoloInvalido(
            f"{caminho}: nenhuma linha reconhecida ({len(ignoradas)} ignoradas). Confira "
            "se é o protocolo da trilha 1 (ASVspoof5.train.tsv, .dev.track_1.tsv ou "
            ".eval.track_1.tsv, 10 campos), e não o de 2019 (5 campos) nem os trials "
            "da trilha 2 (5 campos).")
    return enunciados


def conferir_tamanho(caminho: str | Path, enunciados: list[Enunciado]) -> str | None:
    """Aviso se o número de linhas não for o do README (download truncado ou
    arquivo editado). None quando bate ou quando o nome não é um dos oficiais."""
    esperado = LINHAS_OFICIAIS.get(Path(caminho).name)
    if esperado is None or esperado == len(enunciados):
        return None
    return (f"{Path(caminho).name}: {len(enunciados)} enunciados lidos, o README diz "
            f"{esperado}. Baixe o protocolo de novo.")


def resumo(enunciados: list[Enunciado]) -> dict:
    """Contagens para conferir a base antes de usar: chave, ataque, codec, gênero
    e locutores por chave."""
    locutores = {c: len({e.locutor for e in enunciados if e.chave == c}) for c in CHAVES}
    return {
        "chave": dict(Counter(e.chave for e in enunciados)),
        "ataque": dict(sorted(Counter(e.ataque for e in enunciados).items())),
        "codec": dict(sorted(Counter(e.codec for e in enunciados).items())),
        "genero": dict(Counter(e.genero for e in enunciados)),
        "locutores": locutores,
    }


def recorte(enunciados: list[Enunciado], falsos_por_ataque: int,
            seed: int = 42) -> list[Enunciado]:
    """Todos os bonafide e até `falsos_por_ataque` spoof de cada ataque.

    É a seleção do recorte de treino do plano (19 mil humanos, cerca de 5 mil
    falsos por ataque). A mesma semente dá o mesmo recorte; a ordem segue a
    do protocolo.
    """
    import random

    rng = random.Random(seed)
    por_ataque: dict[str, list[int]] = {}
    for i, e in enumerate(enunciados):
        if e.chave == "spoof":
            por_ataque.setdefault(e.ataque, []).append(i)
    escolhidos = {i for i, e in enumerate(enunciados) if e.chave == "bonafide"}
    for ataque in sorted(por_ataque):
        indices = por_ataque[ataque]
        escolhidos.update(rng.sample(indices, min(falsos_por_ataque, len(indices))))
    return [enunciados[i] for i in sorted(escolhidos)]
