"""Lê o `trial_metadata.txt` do ASVspoof 2021 e converte para o protocolo de 2019.

Colunas: locutor arquivo codec canal ataque chave trim fase. A ordem difere da
de 2019, então a chave é localizada (último token `bonafide`/`spoof`), não lida
por índice fixo.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CHAVES = ("bonafide", "spoof")

#: O protocolo de 2019 tem 5 campos; a contagem basta para recusá-lo.
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

    A chave é o último `bonafide`/`spoof` da linha: o bonafide real repete
    `bonafide` na coluna de ataque. `relatorio`, se passado, recebe `ignoradas`
    e até 3 `exemplos`. Levanta `MetadadoInvalido` se nada for reconhecido.
    """
    trials: list[Trial] = []
    ignoradas: list[str] = []
    with open(caminho, "r", encoding="utf-8") as fh:
        for linha in fh:
            partes = linha.split()
            if not partes:
                continue
            # Recusa o protocolo de 2019, que geraria condições falsas ("-/A07").
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
            codec: str | None = None, fase: str | None = None) -> list[Trial]:
    """Seleciona por condição completa (`codec/canal`), por codec e/ou por fase.

    Só a fase `eval` é o conjunto oficial (14.816 bonafide, 133.360 spoof).
    """
    saida = trials
    if condicao:
        saida = [t for t in saida if t.condicao == condicao]
    if codec:
        saida = [t for t in saida if t.codec == codec]
    if fase:
        saida = [t for t in saida if t.fase == fase]
    return saida


def subamostrar(trials: list[Trial], n: int, seed: int = 42) -> list[Trial]:
    """Subamostra estratificada por (chave, ataque), preservando as proporções.

    A mesma semente dá a mesma amostra, então condições diferentes ficam
    comparáveis. 10.000 por condição dão IC 95% de ±1,28 pp no EER.
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
