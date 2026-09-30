"""Histórico das análises (SQLite, da biblioteca padrão).

Guarda só o resultado — nunca o áudio. O arquivo enviado é apagado assim que a
análise termina: voz é dado pessoal (LGPD), e o histórico não precisa dela.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

ESQUEMA = """
CREATE TABLE IF NOT EXISTS analises (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    criado_em TEXT NOT NULL,
    arquivo TEXT NOT NULL,
    duracao_s REAL,
    taxa_nativa INTEGER,
    score_medio REAL,
    score_maximo REAL,
    limiar REAL,
    origem_limiar TEXT,
    aviso_limiar TEXT,
    janelas_uteis INTEGER,
    janelas_independentes REAL,
    janelas TEXT,
    modelo TEXT
)
"""


class Banco:
    def __init__(self, caminho: str | Path):
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        with self._conectar() as con:
            con.execute(ESQUEMA)

    def _conectar(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.caminho)
        con.row_factory = sqlite3.Row
        return con

    def salvar(self, r, modelo: str) -> int:
        with self._conectar() as con:
            cur = con.execute(
                "INSERT INTO analises (criado_em, arquivo, duracao_s, taxa_nativa, "
                "score_medio, score_maximo, limiar, origem_limiar, aviso_limiar, "
                "janelas_uteis, janelas_independentes, janelas, modelo) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), r.arquivo, r.duracao_s,
                 r.taxa_nativa, r.score_medio, r.score_maximo, r.limiar, r.origem_limiar,
                 r.aviso_limiar, r.janelas_uteis, r.janelas_independentes,
                 json.dumps(r.janelas), modelo))
            return int(cur.lastrowid)

    def buscar(self, analise_id: int) -> dict | None:
        with self._conectar() as con:
            linha = con.execute("SELECT * FROM analises WHERE id = ?", (analise_id,)).fetchone()
        if linha is None:
            return None
        d = dict(linha)
        d["janelas"] = json.loads(d["janelas"] or "[]")
        return d

    def listar(self, limite: int = 100) -> list[dict]:
        with self._conectar() as con:
            linhas = con.execute("SELECT id, criado_em, arquivo, duracao_s, score_medio, "
                                 "limiar FROM analises ORDER BY id DESC LIMIT ?",
                                 (limite,)).fetchall()
        return [dict(x) for x in linhas]
