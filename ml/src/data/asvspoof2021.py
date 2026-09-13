"""Leitura do metadado do ASVspoof 2021 e conversão para o formato do projeto.

O ASVspoof 2021 distribui os rótulos num `trial_metadata.txt` com mais campos
que o protocolo de 2019, e **em outra ordem**:

    2019:  LA_0079 LA_E_1234567 -     A07   spoof
    2021:  LA_0009 LA_E_9332881 alaw  ita_tx  A07  spoof  notrim  eval
           locutor arquivo      codec canal   ataque chave  trim   fase

O parser de 2019 (`parse_protocol_with_systems`) lê o ataque em `parts[3]` e a
chave em `parts[4]`. Aplicado ao arquivo de 2021 ele leria `ita_tx` e `A07` —
e como `A07` não é uma chave válida, **descartaria todas as linhas em
silêncio**, devolvendo uma lista vazia. Falha silenciosa, que é a pior forma.

Por isso este módulo não usa índice fixo: ele **localiza** o token da chave
(`bonafide`/`spoof`) e deduz os demais a partir dele. Assim funciona para LA e
sobrevive à ordem diferente dos campos de outras trilhas.

**Por que isso importa cientificamente.** O metadado traz codec e canal por
áudio, e os ataques são os mesmos A07–A19 do eval de 2019. Isso permite comparar
*o mesmo ataque* em condição limpa e transmitida — inclusive contra a simulação
de canal deste projeto (`scripts/robustness_eval.py`), que usa Opus.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CHAVES = ("bonafide", "spoof")

#: O metadado de 2021 traz no mínimo locutor, arquivo, codec, canal, ataque e
#: chave. O protocolo de 2019 tem exatamente 5 campos, então a contagem separa
#: os dois com segurança — e é o que impede o arquivo errado de passar.
CAMPOS_MINIMOS_2021 = 6


@dataclass(frozen=True)
class Trial:
    """Uma linha do metadado do ASVspoof 2021."""
    locutor: str
    arquivo: str
    codec: str        # alaw, ulaw, opus, gsm, g722, … ou o valor de "sem codec"
    canal: str        # ita_tx, pstn, … (meio de transmissão)
    ataque: str       # A07…A19, ou "-" para bonafide
    chave: str        # bonafide | spoof

    @property
    def condicao(self) -> str:
        """Identificador da condição: o par (codec, canal)."""
        return f"{self.codec}/{self.canal}"


class MetadadoInvalido(ValueError):
    """O arquivo não se parece com um trial_metadata do ASVspoof 2021."""


def ler_metadata(caminho: str | Path) -> list[Trial]:
    """Lê o `trial_metadata.txt` inteiro.

    Levanta em vez de devolver lista vazia: um protocolo vazio propagado adiante
    vira um EER calculado sobre nada, e ninguém percebe até o número sair errado.
    """
    trials: list[Trial] = []
    ignoradas = 0
    with open(caminho, "r", encoding="utf-8") as fh:
        for linha in fh:
            partes = linha.split()
            if not partes:
                continue
            # Discriminador contra o protocolo de 2019, que tem exatamente 5
            # campos e TAMBÉM casaria com a busca pela chave abaixo — lá
            # `parts[3]` é o ataque, aqui é o canal. Aceitá-lo produziria
            # condições inventadas ("-/A07") que pareceriam canais reais.
            if len(partes) < CAMPOS_MINIMOS_2021:
                ignoradas += 1
                continue
            # A chave é o âncora: tudo se posiciona em relação a ela.
            indices = [i for i, p in enumerate(partes) if p in CHAVES]
            if len(indices) != 1 or indices[0] < 4:
                ignoradas += 1
                continue
            i = indices[0]
            trials.append(Trial(locutor=partes[0], arquivo=partes[1],
                                codec=partes[2], canal=partes[3],
                                ataque=partes[i - 1], chave=partes[i]))
    if not trials:
        raise MetadadoInvalido(
            f"{caminho}: nenhuma linha com chave 'bonafide'/'spoof' reconhecida "
            f"({ignoradas} linhas ignoradas). Confira se é o trial_metadata.txt "
            f"do eval-package do ASVspoof 2021 (8 campos), e não o protocolo "
            f"de 2019 (5 campos).")
    return trials


def condicoes(trials: list[Trial]) -> list[tuple[str, int, int]]:
    """Lista as condições presentes: (condição, nº bonafide, nº spoof)."""
    bona: Counter = Counter()
    spoof: Counter = Counter()
    for t in trials:
        (bona if t.chave == "bonafide" else spoof)[t.condicao] += 1
    nomes = sorted(set(bona) | set(spoof))
    return [(n, bona[n], spoof[n]) for n in nomes]


def filtrar(trials: list[Trial], condicao: str | None = None,
            codec: str | None = None) -> list[Trial]:
    """Seleciona por condição completa (`codec/canal`) ou só por codec."""
    saida = trials
    if condicao:
        saida = [t for t in saida if t.condicao == condicao]
    if codec:
        saida = [t for t in saida if t.codec == codec]
    return saida


def linha_de_protocolo(t: Trial) -> str:
    """Converte para o formato de 2019, que o `evaluate.py` já lê."""
    return f"{t.locutor} {t.arquivo} - {t.ataque} {t.chave}"


def escrever_protocolo(trials: list[Trial], destino: str | Path) -> int:
    """Grava o protocolo convertido. Devolve quantas linhas foram escritas."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(linha_de_protocolo(t) for t in trials) + "\n",
                       encoding="utf-8")
    return len(trials)
