"""Histórico das análises (SQLite, da biblioteca padrão).

Guarda só o resultado — nunca o áudio. O arquivo enviado é apagado assim que a
análise termina: voz é dado pessoal (LGPD), e o histórico não precisa dela. Da
sessão ao vivo fica só o caminho da gravação, para apagá-la junto com o registro.
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
    modelo TEXT,
    gravacao TEXT
)
"""


class Banco:
    def __init__(self, caminho: str | Path):
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        with self._conectar() as con:
            con.execute(ESQUEMA)
            colunas = {x["name"] for x in con.execute("PRAGMA table_info(analises)")}
            if "gravacao" not in colunas:
                con.execute("ALTER TABLE analises ADD COLUMN gravacao TEXT")

    def _conectar(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.caminho)
        con.row_factory = sqlite3.Row
        return con

    def salvar(self, r, modelo: str, gravacao: str | None = None) -> int:
        with self._conectar() as con:
            cur = con.execute(
                "INSERT INTO analises (criado_em, arquivo, duracao_s, taxa_nativa, "
                "score_medio, score_maximo, limiar, origem_limiar, aviso_limiar, "
                "janelas_uteis, janelas_independentes, janelas, modelo, gravacao) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(timespec="seconds"), r.arquivo, r.duracao_s,
                 r.taxa_nativa, r.score_medio, r.score_maximo, r.limiar, r.origem_limiar,
                 r.aviso_limiar, r.janelas_uteis, r.janelas_independentes,
                 json.dumps(r.janelas), modelo, gravacao))
            return int(cur.lastrowid)

    def buscar(self, analise_id: int) -> dict | None:
        with self._conectar() as con:
            linha = con.execute("SELECT * FROM analises WHERE id = ?", (analise_id,)).fetchone()
        if linha is None:
            return None
        d = dict(linha)
        d["janelas"] = json.loads(d["janelas"] or "[]")
        return d

    def ultima(self) -> int | None:
        with self._conectar() as con:
            linha = con.execute("SELECT MAX(id) FROM analises").fetchone()
        return linha[0]

    def gravacoes(self, ids: list[int] | None = None) -> list[str]:
        """Caminhos das gravações ao vivo dos registros `ids` (todos, se None)."""
        sql = "SELECT gravacao FROM analises WHERE gravacao IS NOT NULL"
        args: list[int] = []
        if ids is not None:
            if not ids:
                return []
            sql += f" AND id IN ({','.join('?' * len(ids))})"
            args = list(ids)
        with self._conectar() as con:
            return [x[0] for x in con.execute(sql, args)]

    def excluir(self, ids: list[int]) -> int:
        if not ids:
            return 0
        with self._conectar() as con:
            marcas = ",".join("?" * len(ids))
            return con.execute(f"DELETE FROM analises WHERE id IN ({marcas})", ids).rowcount

    def limpar(self) -> int:
        with self._conectar() as con:
            return con.execute("DELETE FROM analises").rowcount

    def listar(self, limite: int = 100) -> list[dict]:
        with self._conectar() as con:
            linhas = con.execute("SELECT id, criado_em, arquivo, duracao_s, score_medio, "
                                 "limiar FROM analises ORDER BY id DESC LIMIT ?",
                                 (limite,)).fetchall()
        return [dict(x) for x in linhas]
