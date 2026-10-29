"""Aba Clonar: gera o clone de uma voz autorizada e o analisa como arquivo.

É o roteiro da demonstração (plano, decisões pendentes): o clone gerado na hora e
analisado pelo caminho do Enviar arquivo, sem comando no terminal.

- Só as vozes de `data/vozes/vozes_autorizadas.json` (fora do Git, com as vozes):
  código do locutor e gravação de referência, e só quem assinou o termo com a
  clonagem autorizada. A página manda o código; o caminho do áudio nunca vem dela.
- Cada ferramenta roda no seu ambiente conda (`xtts`, `chatterbox`), pelo
  `scripts/clonar_xtts.py`, uma vez por clone: a GPU (4 GB) fica livre para o
  XLS-R entre um clone e outro. Medido em 28/10/2026, com a interface na CPU
  (`DETECTOR_DEVICE=cpu`): XTTS 55 s (pico de 2,5 GB na GPU), Chatterbox 90 s
  (3,8 GB). Com o XLS-R da interface na GPU, o Chatterbox levou 128 s.
- O clone fica em `outputs/clones/` e é apagado junto com o registro no Histórico
  (os clones são apagados depois da banca, pelo termo).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

RAIZ_ML = Path(__file__).resolve().parent.parent
VOZES_PADRAO = "data/vozes/vozes_autorizadas.json"
#: nome na tela e ambiente conda de cada ferramenta.
FERRAMENTAS = {"xtts": ("XTTS v2", "xtts"), "chatterbox": ("Chatterbox", "chatterbox")}
MAX_CARACTERES = 300
#: O Chatterbox levou até 36 s por frase, mais a carga do modelo.
TEMPO_MAXIMO_S = 600


class ErroClonagem(RuntimeError):
    """Falha ao gerar o clone; o detalhe vai para o terminal, não para a tela."""


@dataclass(frozen=True)
class Voz:
    codigo: str
    referencia: Path


def ler_vozes(caminho: str | Path | None = None) -> list[Voz]:
    """As vozes autorizadas; arquivo ausente ou referência sumida ficam de fora."""
    caminho = Path(caminho or os.environ.get("DETECTOR_VOZES", VOZES_PADRAO))
    if not caminho.is_file():
        return []
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    vozes = []
    for v in dados.get("vozes", []):
        if not v.get("clonagem_autorizada"):
            continue
        ref = Path(v["referencia"])
        if not ref.is_absolute():
            ref = RAIZ_ML / ref
        if ref.is_file():
            vozes.append(Voz(codigo=str(v["codigo"]), referencia=ref))
    return vozes


def python_do_ambiente(ambiente: str) -> Path | None:
    """O Python do ambiente conda; `DETECTOR_PYTHON_<AMBIENTE>` tem prioridade."""
    escolhido = os.environ.get(f"DETECTOR_PYTHON_{ambiente.upper()}")
    if escolhido:
        return Path(escolhido) if Path(escolhido).is_file() else None
    for candidato in candidatos(Path(sys.executable).resolve(), ambiente, os.name == "nt"):
        if candidato.is_file():
            return candidato
    return None


def candidatos(atual: Path, ambiente: str, windows: bool) -> list[Path]:
    """Onde o conda põe o Python do ambiente, a partir do Python atual. No Windows ele
    fica na raiz do ambiente; no Linux, em bin/ (o CI quebrou sem isso, 29/10/2026)."""
    raiz = atual.parent if windows else atual.parent.parent
    nome = "python.exe" if windows else "bin/python"
    # No ambiente base, os outros ficam em envs/; dentro de um deles, ao lado.
    return [raiz / "envs" / ambiente / nome, raiz.parent / ambiente / nome]


def ferramentas_disponiveis() -> dict[str, str]:
    """{ferramenta: nome na tela}, só as que têm o ambiente instalado."""
    return {f: nome for f, (nome, amb) in FERRAMENTAS.items() if python_do_ambiente(amb)}


def pasta_clones() -> Path:
    return Path(os.environ.get("DETECTOR_PASTA_CLONES", "outputs/clones"))


def validar_frase(frase: str) -> str:
    frase = " ".join((frase or "").split())
    if not frase:
        raise ValueError("Escreva a frase que o clone vai falar.")
    if len(frase) > MAX_CARACTERES:
        raise ValueError(f"Frase longa demais (máximo de {MAX_CARACTERES} caracteres).")
    return frase


def clonar(voz: Voz, frase: str, ferramenta: str, destino: Path | None = None,
           tempo_maximo_s: int = TEMPO_MAXIMO_S) -> Path:
    """Gera o clone numa pasta nova e devolve o WAV."""
    if ferramenta not in FERRAMENTAS:
        raise ValueError(f"ferramenta desconhecida: {ferramenta}")
    python = python_do_ambiente(FERRAMENTAS[ferramenta][1])
    if python is None:
        raise ErroClonagem(f"o ambiente conda '{FERRAMENTAS[ferramenta][1]}' não foi encontrado")
    destino = destino or pasta_clones() / f"clone_{time.strftime('%Y%m%d_%H%M%S')}_{ferramenta}_{voz.codigo}"
    destino.mkdir(parents=True, exist_ok=True)
    cmd = [str(python), "-u", str(RAIZ_ML / "scripts" / "clonar_xtts.py"), "--referencia", str(voz.referencia),
           "--texto", frase, "--saida", str(destino), "--ferramenta", ferramenta]
    log = destino / "clonagem.log"
    # Primeiro sem rede: a banca pode não ter internet, e com o cache completo as duas
    # ferramentas rodam offline (conferido em 28/10/2026). Se faltar arquivo no cache,
    # uma segunda tentativa com rede o completa.
    ambiente = {**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"}
    r = _rodar(cmd, log, ambiente, tempo_maximo_s)
    if r.returncode != 0 and _faltou_cache(log):
        ambiente.pop("HF_HUB_OFFLINE")
        r = _rodar(cmd, log, ambiente, tempo_maximo_s)
    wavs = sorted(destino.glob("frase_*.wav"))
    if r.returncode != 0 or not wavs:
        fim = log.read_text(encoding="utf-8", errors="replace")[-800:]
        raise ErroClonagem(f"a clonagem falhou (código {r.returncode}); registro {log}:\n{fim}")
    return wavs[-1]


def _rodar(cmd: list[str], log: Path, ambiente: dict, tempo_maximo_s: int):
    with open(log, "a", encoding="utf-8") as saida:
        try:
            return subprocess.run(cmd, cwd=RAIZ_ML, stdout=saida, stderr=subprocess.STDOUT,
                                  env=ambiente, timeout=tempo_maximo_s)
        except subprocess.TimeoutExpired as erro:
            raise ErroClonagem(f"a clonagem passou de {tempo_maximo_s} s (registro: {log})") from erro


def _faltou_cache(log: Path) -> bool:
    texto = log.read_text(encoding="utf-8", errors="replace")
    return any(m in texto for m in ("IncompleteSnapshotError", "OfflineModeIsEnabled", "LocalEntryNotFoundError"))
