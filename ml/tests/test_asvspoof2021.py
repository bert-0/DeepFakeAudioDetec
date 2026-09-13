"""Testes da importação do ASVspoof 2021.

O metadado de 2021 tem oito campos, contra cinco do protocolo de 2019, e em
ordem diferente. Aplicar o parser de 2019 nele não levanta erro: ele descarta
todas as linhas em silêncio e devolve lista vazia. O teste
`test_parser_de_2019_falharia_em_silencio` registra exatamente esse cenário —
é a razão de este módulo existir.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.asvspoof2021 import (  # noqa: E402
    MetadadoInvalido,
    condicoes,
    escrever_protocolo,
    filtrar,
    ler_metadata,
    linha_de_protocolo,
)
from src.data.dataset import parse_protocol_with_systems  # noqa: E402

# Linhas no formato real do eval-package:
#   locutor arquivo codec canal ataque chave trim fase
LINHAS = """\
LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval
LA_0009 LA_E_1000001 alaw ita_tx - bonafide notrim eval
LA_0010 LA_E_1000002 opus ita_tx A10 spoof notrim eval
LA_0010 LA_E_1000003 opus ita_tx - bonafide notrim eval
LA_0011 LA_E_1000004 nocodec nocodec A12 spoof notrim eval
LA_0011 LA_E_1000005 nocodec nocodec - bonafide notrim eval
LA_0012 LA_E_1000006 gsm pstn A19 spoof notrim eval
"""


@pytest.fixture
def metadata(tmp_path):
    p = tmp_path / "trial_metadata.txt"
    p.write_text(LINHAS, encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# A regressão que motiva o módulo
# --------------------------------------------------------------------------- #
def test_parser_de_2019_falharia_em_silencio(metadata):
    """Sem conversão, o protocolo sai VAZIO — e nada avisa."""
    assert parse_protocol_with_systems(metadata) == [], \
        "se isto passar a devolver linhas, o parser de 2019 mudou"


def test_parser_de_2021_le_todas_as_linhas(metadata):
    assert len(ler_metadata(metadata)) == 7


# --------------------------------------------------------------------------- #
# Leitura dos campos
# --------------------------------------------------------------------------- #
def test_campos_sao_atribuidos_na_ordem_certa(metadata):
    t = ler_metadata(metadata)[0]
    assert (t.locutor, t.arquivo) == ("LA_0009", "LA_E_9332881")
    assert (t.codec, t.canal) == ("alaw", "ita_tx")
    assert (t.ataque, t.chave) == ("A07", "spoof")


def test_bonafide_nao_ganha_ataque(metadata):
    bona = [t for t in ler_metadata(metadata) if t.chave == "bonafide"]
    assert bona and all(t.ataque == "-" for t in bona)


def test_ataques_sao_os_mesmos_de_2019(metadata):
    """O valor científico do 2021 LA depende disto: mesmos A07–A19."""
    ataques = {t.ataque for t in ler_metadata(metadata) if t.ataque != "-"}
    assert ataques <= {f"A{i:02d}" for i in range(7, 20)}


def test_ordem_diferente_dos_campos_ainda_e_lida(tmp_path):
    """A chave é o âncora, não o índice — outra trilha pode ter campos a mais."""
    p = tmp_path / "m.txt"
    p.write_text("LA_0009 LA_E_1 opus ita_tx extra A07 spoof notrim eval\n",
                 encoding="utf-8")
    t = ler_metadata(p)[0]
    assert (t.ataque, t.chave) == ("A07", "spoof")


# --------------------------------------------------------------------------- #
# Falha ruidosa
# --------------------------------------------------------------------------- #
def test_protocolo_de_2019_e_recusado(tmp_path):
    """Apontar para o arquivo errado precisa doer na hora, não no EER."""
    p = tmp_path / "2019.txt"
    p.write_text("LA_0079 LA_E_1234567 - A07 spoof\n", encoding="utf-8")
    with pytest.raises(MetadadoInvalido, match="trial_metadata"):
        ler_metadata(p)


def test_arquivo_sem_chave_reconhecida_levanta(tmp_path):
    p = tmp_path / "lixo.txt"
    p.write_text("uma linha qualquer sem nada\noutra linha\n", encoding="utf-8")
    with pytest.raises(MetadadoInvalido):
        ler_metadata(p)


# --------------------------------------------------------------------------- #
# Condições — é o que permite a comparação pareada
# --------------------------------------------------------------------------- #
def test_condicoes_sao_contadas_por_classe(metadata):
    achadas = dict((n, (b, s)) for n, b, s in condicoes(ler_metadata(metadata)))
    assert achadas["alaw/ita_tx"] == (1, 1)
    assert achadas["opus/ita_tx"] == (1, 1)
    assert achadas["nocodec/nocodec"] == (1, 1)
    assert achadas["gsm/pstn"] == (0, 1)


def test_filtro_por_condicao_completa(metadata):
    sel = filtrar(ler_metadata(metadata), condicao="opus/ita_tx")
    assert len(sel) == 2 and all(t.codec == "opus" for t in sel)


def test_filtro_por_codec_ignora_o_canal(metadata):
    assert len(filtrar(ler_metadata(metadata), codec="nocodec")) == 2


def test_filtro_sem_criterio_devolve_tudo(metadata):
    assert len(filtrar(ler_metadata(metadata))) == 7


# --------------------------------------------------------------------------- #
# A saída precisa entrar no evaluate.py sem conversão manual
# --------------------------------------------------------------------------- #
def test_protocolo_convertido_e_lido_pelo_parser_de_2019(metadata, tmp_path):
    trials = ler_metadata(metadata)
    destino = tmp_path / "convertido.txt"
    assert escrever_protocolo(trials, destino) == 7

    lidos = parse_protocol_with_systems(destino)
    assert len(lidos) == 7
    assert [n for n, _, _ in lidos] == [t.arquivo for t in trials]
    assert [s for _, s, _ in lidos] == [0 if t.chave == "bonafide" else 1
                                        for t in trials]
    assert [a for _, _, a in lidos] == [t.ataque for t in trials]


def test_linha_tem_os_cinco_campos_de_2019(metadata):
    assert len(linha_de_protocolo(ler_metadata(metadata)[0]).split()) == 5


def test_escrever_cria_o_diretorio(metadata, tmp_path):
    destino = tmp_path / "novo" / "sub" / "p.txt"
    escrever_protocolo(ler_metadata(metadata), destino)
    assert destino.is_file()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_recusa_selecao_de_uma_classe_so(metadata, tmp_path, capsys):
    """gsm/pstn só tem spoof no exemplo: EER não existe com uma classe."""
    import argparse

    from scripts.importar_asvspoof2021 import main

    sys.argv = ["x", "--metadata", str(metadata), "--condicao", "gsm/pstn",
                "--saida", str(tmp_path / "p.txt")]
    assert main() == 1
    assert "uma classe só" in capsys.readouterr().out
    del argparse


def test_cli_lista_condicoes(metadata, capsys):
    from scripts.importar_asvspoof2021 import main

    sys.argv = ["x", "--metadata", str(metadata), "--listar"]
    assert main() == 0
    saida = capsys.readouterr().out
    assert "nocodec/nocodec" in saida and "opus/ita_tx" in saida
    assert "A07" in saida and "A19" in saida


def test_cli_gera_protocolo_utilizavel(metadata, tmp_path, capsys):
    from scripts.importar_asvspoof2021 import main

    destino = tmp_path / "opus.txt"
    sys.argv = ["x", "--metadata", str(metadata), "--codec", "opus",
                "--saida", str(destino)]
    assert main() == 0
    assert len(parse_protocol_with_systems(destino)) == 2
