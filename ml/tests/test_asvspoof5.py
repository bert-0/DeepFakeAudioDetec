"""Leitor dos protocolos da ASVspoof 5 (src/data/asvspoof5.py), com linhas sintéticas.

O README fala em cinco colunas e lista dez; o leitor aceita 9 ou 10 campos e
confere cada linha pelo conteúdo. Protocolos de outro formato são recusados,
não lidos errado em silêncio.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.asvspoof5 import (  # noqa: E402
    ProtocoloInvalido,
    conferir_tamanho,
    escrever_protocolo,
    ler_protocolo,
    recorte,
    resumo,
    subamostrar,
)
from src.data.dataset import parse_protocol_with_systems  # noqa: E402

# locutor arquivo gênero codec codec_q original adaptação ataque chave reservado
LINHAS = """\
SPEAKER_ID FLAC_FILE_NAME SPEAKER_GENDER CODEC CODEC_Q CODEC_SEED ATTACK_TAG ATTACK_LABEL KEY TMP
T_0001 T_0000000000 F - - - - bonafide bonafide -
T_0001 T_0000000001 F - - - AC1 A01 spoof -
T_0002 T_0000000002 M - - - AC2 A02 spoof -
T_0002 T_0000000003 M - - - - bonafide bonafide -
E_0003 E_0000000004 M C05 2 E_0000000009 AC3 A20 spoof -
"""


def _protocolo(tmp_path, texto=LINHAS, nome="ASVspoof5.train.tsv"):
    p = tmp_path / nome
    p.write_text(texto, encoding="utf-8")
    return p


def test_le_as_linhas_e_pula_o_cabecalho(tmp_path):
    rel = {}
    e = ler_protocolo(_protocolo(tmp_path), rel)
    assert len(e) == 5 and rel["ignoradas"] == 0
    assert (e[0].chave, e[0].ataque, e[0].genero) == ("bonafide", "-", "F")
    assert (e[1].ataque, e[1].adaptacao, e[1].particao) == ("A01", "AC1", "train")
    assert (e[4].codec, e[4].codec_q, e[4].codec_origem, e[4].particao) == (
        "C05", "2", "E_0000000009", "eval")


def test_nove_campos_sem_o_reservado_dao_o_mesmo(tmp_path):
    sem_tmp = "\n".join(linha.rsplit(" ", 1)[0] for linha in LINHAS.splitlines()[1:])
    a = ler_protocolo(_protocolo(tmp_path))
    b = ler_protocolo(_protocolo(tmp_path, sem_tmp, "nove.tsv"))
    assert a == b


def test_recusa_o_protocolo_de_2019(tmp_path):
    p = _protocolo(tmp_path, "LA_0079 LA_E_1234567 - A07 spoof\nLA_0079 LA_E_7654321 - - bonafide\n")
    with pytest.raises(ProtocoloInvalido, match="2019"):
        ler_protocolo(p)


def test_recusa_os_trials_da_trilha_2(tmp_path):
    p = _protocolo(tmp_path, "E_0001 E_0000000004 M A20 spoof\nE_0001 E_0000000005 M bonafide target\n")
    with pytest.raises(ProtocoloInvalido, match="trilha 2"):
        ler_protocolo(p)


def test_linha_com_rotulo_e_chave_trocados_e_ignorada(tmp_path):
    texto = LINHAS + "T_0004 T_0000000010 F - - - AC1 A01 bonafide -\n" \
                     "T_0004 T_0000000011 F - - - - bonafide spoof -\n"
    rel = {}
    assert len(ler_protocolo(_protocolo(tmp_path, texto), rel)) == 5
    assert rel["ignoradas"] == 2 and "T_0000000010" in rel["exemplos"][0]


def test_tamanho_contra_o_readme(tmp_path):
    p = _protocolo(tmp_path)
    assert "182357" in conferir_tamanho(p, ler_protocolo(p))
    outro = _protocolo(tmp_path, LINHAS, "meu_recorte.tsv")
    assert conferir_tamanho(outro, ler_protocolo(outro)) is None


def test_resumo():
    from src.data.asvspoof5 import Enunciado

    e = [Enunciado("T_1", "T_0", "F", "-", "-", "-", "-", "-", "bonafide"),
         Enunciado("T_1", "T_1", "F", "-", "-", "-", "AC1", "A01", "spoof"),
         Enunciado("T_2", "T_2", "M", "-", "-", "-", "AC1", "A01", "spoof")]
    r = resumo(e)
    assert r["chave"] == {"bonafide": 1, "spoof": 2}
    assert r["ataque"] == {"-": 1, "A01": 2}
    assert r["locutores"] == {"bonafide": 1, "spoof": 2}


def test_convertido_e_lido_pelo_resto_do_projeto(tmp_path):
    """O protocolo no formato de 2019 é o que dataset, evaluate e robustez já leem."""
    destino = tmp_path / "train_2019.txt"
    assert escrever_protocolo(ler_protocolo(_protocolo(tmp_path)), destino) == 5
    lido = parse_protocol_with_systems(destino)
    assert lido[0] == ("T_0000000000", 0, "-")
    assert lido[1] == ("T_0000000001", 1, "A01")
    assert lido[4] == ("E_0000000004", 1, "A20")


def _base(n_bona=30, n_por_ataque=(50, 3)):
    from src.data.asvspoof5 import Enunciado

    e, k = [], 0
    for _ in range(n_bona):
        e.append(Enunciado("T_1", f"T_{k:010d}", "F", "-", "-", "-", "-", "-", "bonafide"))
        k += 1
    for a, n in enumerate(n_por_ataque, start=1):
        for _ in range(n):
            e.append(Enunciado("T_2", f"T_{k:010d}", "M", "-", "-", "-", "AC1", f"A{a:02d}", "spoof"))
            k += 1
    return e


def test_recorte_tem_todos_os_humanos_e_ate_n_por_ataque():
    base = _base()
    r = recorte(base, falsos_por_ataque=10)
    assert sum(e.chave == "bonafide" for e in r) == 30
    assert sum(e.ataque == "A01" for e in r) == 10
    assert sum(e.ataque == "A02" for e in r) == 3      # ataque menor fica inteiro
    assert r == recorte(base, falsos_por_ataque=10)   # mesma semente, mesmo recorte
    assert r != recorte(base, falsos_por_ataque=10, seed=7)
    assert [base.index(e) for e in r] == sorted(base.index(e) for e in r)


def test_subamostra_estratificada_mantem_as_proporcoes():
    base = _base(n_bona=100, n_por_ataque=(300, 100))
    s = subamostrar(base, 100)
    assert len(s) == 100
    assert sum(e.chave == "bonafide" for e in s) == 20
    assert sum(e.ataque == "A01" for e in s) == 60
