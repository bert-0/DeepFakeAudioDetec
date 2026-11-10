"""Lê os protocolos da ASVspoof 5 (trilha 1) e converte para o formato de 2019.

Campos, separados por espaço apesar do `.tsv` (README do Zenodo, registro
14498691): locutor arquivo gênero codec qualidade_do_codec original_do_codec
condição_de_adaptação ataque chave reservado. O README fala em "cinco colunas"
e lista dez; o último é reservado ("-"), então 9 ou 10 campos são aceitos e
cada linha é conferida pelo conteúdo, não só pela posição.

Os FLAC são lidos direto dos `.tar` do Zenodo, sem extrair: os de treino e dev
somam 57,5 GB, e extraídos dobrariam o espaço.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Formato de 2019 e amostra estratificada por (chave, ataque): os do 2021
# servem, porque só leem locutor, arquivo, ataque e chave.
from .asvspoof2021 import escrever_protocolo, linha_de_protocolo, subamostrar

__all__ = ["Enunciado", "ProtocoloInvalido", "amostra_estratificada", "conferir_arquivo",
           "conferir_tamanho", "cortar_trecho", "escrever_original", "escrever_protocolo",
           "ler_protocolo", "ler_tar", "linha_de_protocolo", "linha_original", "recorte",
           "resumo", "subamostrar"]

CHAVES = ("bonafide", "spoof")
GENEROS = ("F", "M")
CAMPOS_MINIMOS = 9

#: Linhas de cada protocolo segundo o README; um download truncado não bate.
LINHAS_OFICIAIS = {
    "ASVspoof5.train.tsv": 182_357,
    "ASVspoof5.dev.track_1.tsv": 140_950,
    "ASVspoof5.eval.track_1.tsv": 680_774,
}

#: Tamanho em bytes e MD5 de cada arquivo, da API do Zenodo (registro 14498691),
#: conferidos contra a página do registro em 15/10/2026.
ARQUIVOS_OFICIAIS = {
    "ASVspoof5_protocols.tar.gz": (20_669_392, "865d0e894ea9f686f0f37e5ae3ae3616"),
    "flac_T_aa.tar": (7_522_498_560, "b0cc86b14826a7701b52aad4f53daf9c"),
    "flac_T_ab.tar": (7_489_587_200, "d05be3f4be7a343fdbdd0ed29fdff2e1"),
    "flac_T_ac.tar": (7_497_256_960, "70d1ba4ad75a20aef3dae541fbd321e3"),
    "flac_T_ad.tar": (7_509_688_320, "c9bb56af2cc410d98338d74babcb95c9"),
    "flac_T_ae.tar": (7_482_521_600, "a3969816982f3e52300d6147ad796df7"),
    "flac_D_aa.tar": (6_647_265_280, "df0be44957623991028cce59792beb17"),
    "flac_D_ab.tar": (6_641_459_200, "1e8cd685d89b64502692f1bcf1a13db3"),
    "flac_D_ac.tar": (6_771_660_800, "5e0031f08c30e4bdbf0c59f91b2d662b"),
    "flac_E_aa.tar": (8_449_781_760, "a8c800766f3d4ef87971e2b4f29663e2"),
    "flac_E_ab.tar": (8_459_868_160, "c35064188f54f07c87aba58de22534a0"),
    "flac_E_ac.tar": (8_461_475_840, "ed7cbcd1b2847998b72472ddf6b445e3"),
    "flac_E_ad.tar": (8_467_200_000, "626a080e994b05df49c577d7b3dede8d"),
    "flac_E_ae.tar": (8_456_785_920, "05f19a5e64fa556714ae1b38eb2ea70b"),
    "flac_E_af.tar": (8_457_410_560, "2ca2f52a3bbf827f7ec155ecb47e85d6"),
    "flac_E_ag.tar": (8_461_946_880, "671200b5de2fc1e74a563b07b057af8f"),
    "flac_E_ah.tar": (8_455_321_600, "eaf885b1299a61eb07f71e82e96dba37"),
    "flac_E_ai.tar": (8_464_230_400, "b2625683bf440abae2af901a45542ad9"),
    "flac_E_aj.tar": (8_623_206_400, "c50281c8233af6f3b0899604d21ade44"),
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


def amostra_estratificada(enunciados: list[Enunciado], n: int, seed: int = 42,
                          por: tuple[str, ...] = ("chave", "ataque")) -> list[Enunciado]:
    """Amostra de `n` com a proporção de cada estrato, sem perder estrato raro.

    Com o padrão (chave, ataque) dá exatamente a `subamostrar` do 2021, que montou
    o dev; no eval, ("chave", "ataque", "codec") garante áudio de cada condição
    de codec, para separar os codecs que o treino viu dos que não viu.
    """
    import random

    if n <= 0 or n >= len(enunciados):
        return list(enunciados)
    estratos: dict[tuple, list[Enunciado]] = {}
    for e in enunciados:
        estratos.setdefault(tuple(getattr(e, campo) for campo in por), []).append(e)
    rng = random.Random(seed)
    fracao = n / len(enunciados)
    escolhidos: list[Enunciado] = []
    for chave in sorted(estratos):
        grupo = estratos[chave]
        k = max(1, round(len(grupo) * fracao))
        escolhidos.extend(rng.sample(grupo, min(k, len(grupo))))
    rng.shuffle(escolhidos)
    return escolhidos


def linha_original(e: Enunciado) -> str:
    """De volta ao formato da ASVspoof 5 (10 campos), que guarda o codec; o de
    2019 o perde."""
    rotulo = "bonafide" if e.chave == "bonafide" else e.ataque
    return " ".join([e.locutor, e.arquivo, e.genero, e.codec, e.codec_q, e.codec_origem,
                     e.adaptacao, rotulo, e.chave, "-"])


def escrever_original(enunciados: list[Enunciado], destino: str | Path) -> int:
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("".join(linha_original(e) + "\n" for e in enunciados), encoding="utf-8")
    return len(enunciados)


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


def conferir_arquivo(caminho: str | Path, md5: bool = False) -> str | None:
    """Problema com um arquivo baixado, ou None. O tamanho é conferido sempre (é
    instantâneo e pega download incompleto); o MD5 só com `md5=True` (lê o
    arquivo inteiro). Arquivos fora da tabela oficial passam sem conferência."""
    caminho = Path(caminho)
    oficial = ARQUIVOS_OFICIAIS.get(caminho.name)
    if oficial is None:
        return None
    tamanho, esperado = oficial
    if caminho.stat().st_size != tamanho:
        return (f"{caminho.name}: {caminho.stat().st_size} bytes, o Zenodo diz {tamanho}. "
                "O download não terminou; retome com curl -C -.")
    if md5:
        import hashlib

        h = hashlib.md5()
        with open(caminho, "rb") as fh:
            for bloco in iter(lambda: fh.read(1 << 24), b""):
                h.update(bloco)
        if h.hexdigest() != esperado:
            return f"{caminho.name}: MD5 {h.hexdigest()}, o Zenodo diz {esperado}. Baixe de novo."
    return None


def ler_tar(caminho, ids: set[str] | None = None):
    """Gera (id, áudio float32, taxa) dos FLAC de um `.tar`, na ordem em que estão
    gravados, sem extrair nada. `ids` restringe aos arquivos pedidos.

    Lê em fluxo ("r|"): serve também para um `.tar` chegando pela rede.
    """
    import io
    import tarfile

    import soundfile as sf

    abrir = (tarfile.open(fileobj=caminho, mode="r|*") if hasattr(caminho, "read")
             else tarfile.open(caminho, mode="r|*"))
    with abrir as tar:
        for membro in tar:
            if not membro.isfile() or not membro.name.endswith(".flac"):
                continue
            nome = Path(membro.name).stem
            if ids is not None and nome not in ids:
                continue
            dados = tar.extractfile(membro).read()
            wav, taxa = sf.read(io.BytesIO(dados), dtype="float32")
            yield nome, wav, int(taxa)


def cortar_trecho(wav, taxa: int, segundos: float = 6.0, margem: float = 0.5,
                  top_db: float = 30.0):
    """No máximo `segundos` de áudio, começando `margem` s antes da primeira fala.

    A margem de silêncio fica de propósito: a degradação de canal do treino
    também age nas pausas, e ruído nas pausas desloca o score (RESUMO_TCC
    §10.2.1). Fala que começa tarde puxa a janela para trás, até o fim do áudio.
    """
    import librosa

    n = int(round(segundos * taxa))
    if len(wav) <= n:
        return wav
    trechos = librosa.effects.split(wav, top_db=top_db)
    inicio = int(trechos[0][0]) - int(round(margem * taxa)) if len(trechos) else 0
    inicio = min(max(inicio, 0), len(wav) - n)
    return wav[inicio:inicio + n]
