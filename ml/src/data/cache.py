"""Cache de features num único arquivo mapeado em memória (memory-map).

Antes, cada áudio virava um arquivo `.pt` próprio. No ASVspoof LA isso significa
cerca de **96 mil arquivos pequenos** (24.844 no dev + 71.237 no eval), reabertos
a cada época de treino e a cada avaliação.

Medido neste projeto (Linux, cache de página quente, features do `baseline_v4`):

| leitura de uma amostra do cache | ms   |
|---------------------------------|------|
| `torch.load` de um `.pt`        | 0,32 |
| linha de um memmap              | 0,01 |

São 30x. No Windows a diferença tende a ser maior, porque cada abertura passa
pelo NTFS e pelo antivírus em tempo real — e 96 mil arquivos numa única pasta
também penalizam backup, cópia e o próprio Explorer.

Layout de `<cache>/<fingerprint>/<partição>/`:

    meta.json    versão, shapes de cada feature, nº de amostras, hash dos ids
    data.npy     matriz (n_amostras, floats_por_amostra) float32
    filled.npy   vetor (n_amostras,) uint8 — 1 quando a linha já foi gravada

As linhas são indexadas pela **posição da amostra no protocolo**, não pelo nome
do arquivo. Por isso `meta.json` guarda o hash da lista de ids: se o protocolo
mudar (outra partição, outra ordem, linhas a mais), o cache é reconstruído em
vez de devolver as features do áudio errado.

**Concorrência.** Os workers do DataLoader são processos separados e gravam ao
mesmo tempo. Cada um escreve apenas a linha do índice que está processando, e
`np.memmap` usa um mapeamento *compartilhado* do mesmo arquivo — dois processos
que escrevem bytes diferentes da mesma página escrevem na mesma página física,
sem o risco de sobrescrita que haveria com I/O em buffer. `filled` só é marcado
depois que a linha inteira foi gravada.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

# Muda quando o layout em disco muda de forma incompatível: o `meta.json` antigo
# deixa de bater e o cache é reconstruído sozinho.
VERSION = 1

META = "meta.json"
DATA = "data.npy"
FILLED = "filled.npy"


def ids_fingerprint(ids) -> str:
    """Hash da lista de ids, na ordem — identifica o protocolo e a ordenação."""
    h = hashlib.md5()
    for item in ids:
        h.update(str(item).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


class FeatureCache:
    """Cache de dicionários de features com shape fixo, indexado por posição.

    `shapes` mapeia o nome de cada feature ao seu shape (ex.: `{"lfcc": (1, 60,
    401)}`). Os shapes são constantes porque `fix_length` padroniza a duração do
    áudio antes da extração — é isso que permite guardar tudo numa matriz única.
    """

    def __init__(self, directory: str | Path, ids, shapes: dict[str, tuple[int, ...]]):
        self.dir = Path(directory)
        self.n = len(ids)
        self.shapes = {k: tuple(int(x) for x in v) for k, v in shapes.items()}
        self.keys = sorted(self.shapes)

        self.slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for key in self.keys:
            size = int(np.prod(self.shapes[key]))
            self.slices[key] = (offset, offset + size)
            offset += size
        self.row_floats = offset

        self.meta = {
            "version": VERSION,
            "n_items": self.n,
            "row_floats": self.row_floats,
            "shapes": {k: list(self.shapes[k]) for k in self.keys},
            "ids_fingerprint": ids_fingerprint(ids),
        }

        # Os memmaps são abertos por processo (ver `_open`): um objeto memmap
        # herdado por pickle para um worker apontaria para um mapeamento que não
        # existe naquele processo.
        self._pid: int | None = None
        self._data: np.ndarray | None = None
        self._filled: np.ndarray | None = None
        self._prepare()

    # ------------------------------------------------------------------ #
    # criação / abertura
    # ------------------------------------------------------------------ #
    def _prepare(self) -> None:
        """Cria os arquivos se não existirem, ou os recria se `meta` não bater."""
        self.dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.dir / META

        atual = None
        if meta_path.exists():
            try:
                atual = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                atual = None  # meta corrompido -> reconstrói

        completo = (atual == self.meta
                    and (self.dir / DATA).exists()
                    and (self.dir / FILLED).exists())
        if not completo:
            self._create(meta_path)

    def _create(self, meta_path: Path) -> None:
        # `meta.json` é escrito por ÚLTIMO: se a criação for interrompida no meio
        # (falta de espaço, Ctrl+C), a próxima execução não encontra o meta e
        # reconstrói, em vez de ler uma matriz pela metade como se estivesse boa.
        meta_path.unlink(missing_ok=True)
        for nome in (DATA, FILLED):
            (self.dir / nome).unlink(missing_ok=True)

        data = np.lib.format.open_memmap(
            self.dir / DATA, mode="w+", dtype=np.float32,
            shape=(self.n, self.row_floats))
        data.flush()
        del data
        filled = np.lib.format.open_memmap(
            self.dir / FILLED, mode="w+", dtype=np.uint8, shape=(self.n,))
        filled.flush()
        del filled

        meta_path.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    def _open(self) -> None:
        """Abre (ou reabre) os memmaps no processo atual."""
        if self._pid == os.getpid():
            return
        self._data = np.load(self.dir / DATA, mmap_mode="r+")
        self._filled = np.load(self.dir / FILLED, mmap_mode="r+")
        self._pid = os.getpid()

    def __getstate__(self) -> dict:
        """Não leva os memmaps ao serializar o dataset para os workers."""
        estado = self.__dict__.copy()
        estado["_pid"] = None
        estado["_data"] = None
        estado["_filled"] = None
        return estado

    # ------------------------------------------------------------------ #
    # leitura / escrita
    # ------------------------------------------------------------------ #
    def get(self, idx: int) -> dict[str, torch.Tensor] | None:
        """Devolve as features da posição `idx`, ou None se ainda não estiverem lá."""
        self._open()
        if not self._filled[idx]:
            return None
        # Uma cópia da linha inteira (não do memmap): os tensores precisam de
        # memória própria e gravável, e uma cópia só é mais barata que várias.
        row = np.array(self._data[idx], dtype=np.float32)
        return {key: torch.from_numpy(row[a:b].reshape(self.shapes[key]))
                for key, (a, b) in self.slices.items()}

    def put(self, idx: int, features: dict[str, torch.Tensor]) -> None:
        """Grava as features da posição `idx` (no-op se já estiverem gravadas)."""
        self._open()
        if self._filled[idx]:
            return
        row = np.empty(self.row_floats, dtype=np.float32)
        for key, (a, b) in self.slices.items():
            tensor = features[key]
            if tuple(tensor.shape) != self.shapes[key]:
                raise ValueError(
                    f"o cache espera {key} com shape {self.shapes[key]}, mas a "
                    f"amostra {idx} veio com {tuple(tensor.shape)}. O cache exige "
                    "shape constante — verifique se `audio.duration` está fixo.")
            row[a:b] = tensor.detach().numpy().reshape(-1)
        self._data[idx] = row
        # Só depois da linha inteira gravada: marcar antes deixaria uma janela em
        # que um outro worker leria uma linha pela metade como se fosse válida.
        self._filled[idx] = 1

    # ------------------------------------------------------------------ #
    # informação
    # ------------------------------------------------------------------ #
    def n_filled(self) -> int:
        self._open()
        return int(self._filled.sum())

    def nbytes(self) -> int:
        return self.n * self.row_floats * 4

    def describe(self) -> str:
        gb = self.nbytes() / 1e9
        return (f"{self.dir} — {self.n} amostras, "
                f"{self.row_floats * 4 / 1024:.0f} KiB cada (~{gb:.1f} GB)")


def legacy_pt_files(directory: str | Path) -> tuple[int, int]:
    """Conta os arquivos `.pt` do formato antigo e o total de bytes.

    Usa `os.scandir` em vez de `Path.glob` + `stat()`: no Windows a enumeração do
    diretório já traz o tamanho, e são dezenas de milhares de entradas.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return 0, 0
    quantos = tamanho = 0
    try:
        with os.scandir(directory) as entradas:
            for entrada in entradas:
                if entrada.name.endswith(".pt") and entrada.is_file():
                    quantos += 1
                    tamanho += entrada.stat().st_size
    except OSError:
        return 0, 0
    return quantos, tamanho
