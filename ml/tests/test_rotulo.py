"""Testes da consulta de rótulo (scripts/rotulo.py).

Motivação: procurar o ID com `findstr` falha em silêncio — saída vazia é
indistinguível de protocolo errado, pasta errada ou ID de outra base. Estes
testes garantem que a falha seja explicada, não muda.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.rotulo import identificador, main  # noqa: E402


@pytest.fixture
def projeto(tmp_path):
    protos = {}
    for particao, linhas in (
        ("train", ["LA_0079 LA_T_1000001 - - bonafide",
                   "LA_0079 LA_T_1000002 - A01 spoof"]),
        ("dev", ["LA_0080 LA_D_2000001 - - bonafide"]),
        ("eval", ["LA_0039 LA_E_1000147 - - bonafide",
                  "LA_0040 LA_E_1000148 - A10 spoof"]),
    ):
        p = tmp_path / f"{particao}.txt"
        p.write_text("\n".join(linhas) + "\n", encoding="utf-8")
        protos[particao] = str(p)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"data": {"protocols": protos}}), encoding="utf-8")
    return cfg


def _rodar(cfg, alvo):
    sys.argv = ["x", alvo, "--config", str(cfg)]
    return main()


def test_encontra_bonafide(projeto, capsys):
    assert _rodar(projeto, "LA_E_1000147") == 0
    saida = capsys.readouterr().out
    assert "BONAFIDE" in saida and "eval" in saida


def test_encontra_spoof_com_o_ataque(projeto, capsys):
    assert _rodar(projeto, "LA_E_1000148") == 0
    saida = capsys.readouterr().out
    assert "SPOOF" in saida and "A10" in saida


def test_encontra_em_outra_particao(projeto, capsys):
    assert _rodar(projeto, "LA_T_1000002") == 0
    assert "train" in capsys.readouterr().out


def test_lembra_a_direcao_do_score(projeto, capsys):
    """O score já foi lido ao contrário uma vez; a dica anda junto do rótulo."""
    _rodar(projeto, "LA_E_1000147")
    saida = capsys.readouterr().out
    assert "BAIXO = voz humana" in saida


# --------------------------------------------------------------------------- #
# A falha precisa explicar, não sumir
# --------------------------------------------------------------------------- #
def test_id_inexistente_mostra_onde_procurou(projeto, capsys):
    assert _rodar(projeto, "LA_E_A9898607") == 1
    saida = capsys.readouterr().out
    assert "NÃO ENCONTRADO" in saida
    assert "train" in saida and "dev" in saida and "eval" in saida


def test_id_inexistente_mostra_exemplos_do_formato_real(projeto, capsys):
    """É o que revela um ID com formato diferente num relance."""
    _rodar(projeto, "LA_E_A9898607")
    saida = capsys.readouterr().out
    assert "LA_E_1000147" in saida, "precisa mostrar como são os IDs de verdade"


def test_protocolo_faltando_e_denunciado(tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump(
        {"data": {"protocols": {"eval": str(tmp_path / "sumiu.txt")}}}),
        encoding="utf-8")
    assert _rodar(cfg, "qualquer") == 1
    assert "ARQUIVO NÃO EXISTE" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Aceitar caminho ou ID
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("entrada", [
    "LA_E_1000147",
    "data/LA/ASVspoof2019_LA_eval/flac/LA_E_1000147.flac",
    r"data\LA\ASVspoof2019_LA_eval\flac\LA_E_1000147.flac",
])
def test_aceita_id_ou_caminho(entrada):
    assert identificador(entrada) == "LA_E_1000147"


def test_caminho_windows_completo_funciona(projeto, capsys):
    assert _rodar(projeto, r"E:\TCC\ml\data\LA\flac\LA_E_1000148.flac") == 0
    assert "SPOOF" in capsys.readouterr().out
