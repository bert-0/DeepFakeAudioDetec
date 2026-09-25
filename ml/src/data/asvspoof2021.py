"""Leitura do metadado do ASVspoof 2021 e conversão para o formato do projeto.

O ASVspoof 2021 distribui os rótulos num `trial_metadata.txt` com mais campos
que o protocolo de 2019, e **em outra ordem**:

    2019:  LA_0079 LA_E_1234567 -     A07   spoof
    2021:  LA_0009 LA_E_9332881 alaw  ita_tx  A07  spoof  notrim  eval
           locutor arquivo      codec canal   ataque chave  trim   fase

O parser de 2019 (`parse_protocol_with_systems`) lê o ataque em `parts[3]` e a
chave em `parts[4]`. No arquivo real do 2021 o bonafide traz `bonafide` também
na coluna de ataque (`... alaw ita_tx bonafide bonafide notrim eval`), então o
parser de 2019 **aceita as linhas bonafide com o canal no lugar do ataque e
descarta todos os spoof** — um protocolo de uma classe só, que parece válido.

Por isso este módulo não usa índice fixo: ele **localiza** o último token
`bonafide`/`spoof` da linha (a chave) e deduz os demais a partir dele.

A primeira versão deste módulo exigia uma única ocorrência desse token, e por
isso descartava todo bonafide do arquivo real — o `--listar` saiu com 163.114
trials e zero bonafide. O formato tinha sido suposto, não conferido; os testes
usavam o formato suposto e passavam. Agora o descarte é sempre relatado.

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
    fase: str = ""    # progress | eval | … (última coluna, quando houver)

    @property
    def condicao(self) -> str:
        """Identificador da condição: o par (codec, canal)."""
        return f"{self.codec}/{self.canal}"


class MetadadoInvalido(ValueError):
    """O arquivo não se parece com um trial_metadata do ASVspoof 2021."""


def ler_metadata(caminho: str | Path,
                 relatorio: dict | None = None) -> list[Trial]:
    """Lê o `trial_metadata.txt` inteiro.

    **A chave é o ÚLTIMO token `bonafide`/`spoof` da linha.** No arquivo real,
    as linhas bonafide trazem `bonafide` também na coluna de ataque:

        LA_0007 LA_E_5932896 alaw ita_tx bonafide bonafide notrim eval

    A versão anterior exigia uma ocorrência só e descartava essas linhas em
    silêncio — o `--listar` saiu com 163.114 trials e **zero bonafide**. Os
    testes não pegaram porque o arquivo de exemplo usava `-` no lugar, que era
    o formato suposto, não o real.

    `relatorio`, se passado, recebe `ignoradas` e até 3 `exemplos` de linhas
    descartadas — para que um descarte nunca mais seja silencioso.

    Levanta em vez de devolver lista vazia: um protocolo vazio propagado adiante
    vira um EER calculado sobre nada, e ninguém percebe até o número sair errado.
    """
    trials: list[Trial] = []
    ignoradas: list[str] = []
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
                ignoradas.append(linha.rstrip())
                continue
            indices = [i for i, p in enumerate(partes) if p in CHAVES]
            if not indices or indices[-1] < 4:
                ignoradas.append(linha.rstrip())
                continue
            i = indices[-1]
            ataque = partes[i - 1]
            if ataque in CHAVES:        # coluna de ataque do bonafide real
                ataque = "-"
            trials.append(Trial(locutor=partes[0], arquivo=partes[1],
                                codec=partes[2], canal=partes[3],
                                ataque=ataque, chave=partes[i],
                                fase=partes[-1] if len(partes) > i + 2 else ""))
    if relatorio is not None:
        relatorio["ignoradas"] = len(ignoradas)
        relatorio["exemplos"] = ignoradas[:3]
    if not trials:
        raise MetadadoInvalido(
            f"{caminho}: nenhuma linha com chave 'bonafide'/'spoof' reconhecida "
            f"({len(ignoradas)} linhas ignoradas). Confira se é o trial_metadata.txt "
            f"do eval-package do ASVspoof 2021 (8 campos), e não o protocolo "
            f"de 2019 (5 campos).")
    return trials


def fases(trials: list[Trial]) -> dict[str, tuple[int, int]]:
    """Contagem (bonafide, spoof) por fase do desafio."""
    saida: dict[str, list[int]] = {}
    for t in trials:
        par = saida.setdefault(t.fase or "-", [0, 0])
        par[0 if t.chave == "bonafide" else 1] += 1
    return {k: (v[0], v[1]) for k, v in sorted(saida.items())}


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


def subamostrar(trials: list[Trial], n: int, seed: int = 42) -> list[Trial]:
    """Subamostra estratificada por (chave, ataque), preservando as proporções.

    O eval do 2021 tem 181.566 trials e não é preciso rodar todos: no regime
    deste projeto (EER ~20%, ~10% de bonafide), 10.000 por condição dão IC 95%
    de ±1,28 pp — suficiente para os efeitos em jogo (ver RESUMO_TCC, 12.2).

    Estratificar importa porque uma amostra aleatória simples pode sub-representar
    justamente A10 e A12, que dominam o erro. Cada estrato recebe a sua fração
    proporcional, arredondada; a mesma semente dá a mesma amostra, então duas
    condições sorteadas com a mesma semente são comparáveis entre si.
    """
    import random

    if n <= 0 or n >= len(trials):
        return list(trials)
    estratos: dict[tuple[str, str], list[Trial]] = {}
    for t in trials:
        estratos.setdefault((t.chave, t.ataque), []).append(t)

    rng = random.Random(seed)
    fracao = n / len(trials)
    escolhidos: list[Trial] = []
    for chave in sorted(estratos):
        grupo = estratos[chave]
        k = max(1, round(len(grupo) * fracao))
        escolhidos.extend(rng.sample(grupo, min(k, len(grupo))))
    rng.shuffle(escolhidos)
    return escolhidos


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
