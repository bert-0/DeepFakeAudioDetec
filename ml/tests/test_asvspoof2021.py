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

# Linhas no formato do eval-package:
#   locutor arquivo codec canal ataque chave trim fase
#
# No arquivo REAL, o bonafide traz `bonafide` também na coluna de ataque. A
# primeira versão deste fixture usava `-` ali — o formato suposto — e por isso
# os testes passavam enquanto o parser descartava todo bonafide do arquivo
# verdadeiro. As duas formas ficam aqui de propósito.
LINHAS = """\
LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval
LA_0009 LA_E_1000001 alaw ita_tx bonafide bonafide notrim eval
LA_0010 LA_E_1000002 opus ita_tx A10 spoof notrim eval
LA_0010 LA_E_1000003 opus ita_tx bonafide bonafide notrim progress
LA_0011 LA_E_1000004 nocodec nocodec A12 spoof notrim eval
LA_0011 LA_E_1000005 nocodec nocodec - bonafide notrim eval
LA_0012 LA_E_1000006 gsm pstn A19 spoof notrim progress
"""


@pytest.fixture
def metadata(tmp_path):
    p = tmp_path / "trial_metadata.txt"
    p.write_text(LINHAS, encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# A regressão que motiva o módulo
# --------------------------------------------------------------------------- #
def test_parser_de_2019_leria_o_arquivo_de_2021_errado_e_em_silencio(metadata):
    """Sem conversão, o parser de 2019 devolve lixo plausível, sem erro.

    No formato real do 2021 o bonafide tem `bonafide` na 5ª coluna, que é onde o
    parser de 2019 procura a chave. Então ele ACEITA as linhas bonafide — com o
    canal (`ita_tx`) no lugar do ataque — e descarta todos os spoof. O resultado
    é um protocolo de uma classe só, que parece válido.
    """
    lidos = parse_protocol_with_systems(metadata)
    assert lidos, "se isto voltar a ser vazio, o formato do fixture mudou"
    assert all(label == 0 for _, label, _ in lidos), "só bonafide sobrevive"
    assert {sistema for _, _, sistema in lidos} <= {"ita_tx", "nocodec", "pstn", "-"}, \
        "o 'ataque' lido é na verdade o canal"


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


# --------------------------------------------------------------------------- #
# Config derivado — a regressão que motiva a função
#
# Os artefatos do evaluate.py levam o nome `experiment.name` + partição. O
# procedimento anterior mandava copiar o config do modelo e trocar só os
# caminhos do eval: a avaliação do 2021 gravaria POR CIMA dos resultados do
# eval de 2019 (métricas, scores reaproveitados) e recriaria o cache do eval.
# --------------------------------------------------------------------------- #
def _base():
    import yaml
    return yaml.safe_load(Path("configs/fusion_v4.yaml").read_text(encoding="utf-8"))


def test_config_derivado_nao_colide_com_os_artefatos_de_2019():
    from src.config import config_derivado, output_name
    from src.scores import scores_path

    base = _base()
    novo = config_derivado(base, "2021_opus_n10000", "p.txt", "flac/")

    assert output_name(novo) != output_name(base)
    assert scores_path("outputs", output_name(novo), "eval") != \
        scores_path("outputs", output_name(base), "eval"), \
        "o _eval_scores.npz de 2019 seria sobrescrito"


def test_config_derivado_desliga_o_cache():
    """O cache é indexado pela partição: ligado, apagaria o do eval de 2019."""
    from src.config import config_derivado

    assert config_derivado(_base(), "x", "p", "a")["train"]["cache_features"] is False


def test_config_derivado_aponta_o_eval_para_o_audio_novo():
    from src.config import config_derivado

    cfg = config_derivado(_base(), "x", "outputs/p.txt", "data/2021/flac")
    assert cfg["data"]["protocols"]["eval"] == "outputs/p.txt"
    assert cfg["data"]["audio_dir"]["eval"] == "data/2021/flac"


def test_config_derivado_nao_altera_o_original():
    from src.config import config_derivado

    base = _base()
    antes = (base["experiment"]["name"], base["data"]["protocols"]["eval"],
             base["train"]["cache_features"])
    config_derivado(base, "x", "p", "a")
    assert (base["experiment"]["name"], base["data"]["protocols"]["eval"],
            base["train"]["cache_features"]) == antes


def test_config_derivado_preserva_audio_features_e_modelo():
    """Tem que extrair as MESMAS features com que o modelo foi treinado."""
    from src.config import config_derivado

    base = _base()
    cfg = config_derivado(base, "x", "p", "a")
    for secao in ("audio", "features", "model"):
        assert cfg[secao] == base[secao]


def test_sufixo_com_caracteres_de_caminho_e_saneado():
    from src.config import config_derivado

    nome = config_derivado(_base(), "2021/opus ita_tx", "p", "a")["experiment"]["name"]
    assert "/" not in nome and " " not in nome


# --------------------------------------------------------------------------- #
# Subamostragem estratificada
# --------------------------------------------------------------------------- #
def _muitos_trials():
    from src.data.asvspoof2021 import Trial

    trials = [Trial("S", f"B{i}", "opus", "tx", "-", "bonafide") for i in range(1000)]
    for a in ("A07", "A10", "A12", "A19"):
        trials += [Trial("S", f"{a}_{i}", "opus", "tx", a, "spoof") for i in range(2250)]
    return trials                                    # 10.000, 10% bonafide


def test_subamostra_preserva_a_proporcao_das_classes():
    from src.data.asvspoof2021 import subamostrar

    amostra = subamostrar(_muitos_trials(), 1000, seed=1)
    bona = sum(1 for t in amostra if t.chave == "bonafide")
    assert abs(bona / len(amostra) - 0.10) < 0.01


def test_subamostra_nao_perde_nenhum_ataque():
    """Amostra simples poderia deixar A10 ou A12 de fora — os que mais importam."""
    from src.data.asvspoof2021 import subamostrar

    ataques = {t.ataque for t in subamostrar(_muitos_trials(), 50, seed=3)}
    assert {"A07", "A10", "A12", "A19"} <= ataques


def test_mesma_semente_da_mesma_amostra():
    from src.data.asvspoof2021 import subamostrar

    a = [t.arquivo for t in subamostrar(_muitos_trials(), 500, seed=7)]
    b = [t.arquivo for t in subamostrar(_muitos_trials(), 500, seed=7)]
    assert a == b


def test_amostra_zero_ou_maior_que_a_base_devolve_tudo():
    from src.data.asvspoof2021 import subamostrar

    trials = _muitos_trials()
    assert len(subamostrar(trials, 0)) == len(trials)
    assert len(subamostrar(trials, 10**9)) == len(trials)


# --------------------------------------------------------------------------- #
# CLI com config derivado
# --------------------------------------------------------------------------- #
def test_cli_gera_config_seguro_e_carregavel(metadata, tmp_path, capsys):
    from scripts.importar_asvspoof2021 import main
    from src.config import load_config

    destino = tmp_path / "opus.txt"
    codigo = main(["--metadata", str(metadata), "--codec", "opus",
                   "--saida", str(destino),
                   "--config-base", "configs/fusion_v4.yaml",
                   "--audio-dir", "data/2021/flac"])
    assert codigo == 0

    cfg = load_config(destino.with_suffix(".yaml"))
    assert cfg["experiment"]["name"] == "fusion_lcnn_v4__2021_opus"
    assert cfg["data"]["protocols"]["eval"] == str(destino)
    assert cfg["train"]["cache_features"] is False
    assert "evaluate.py --config" in capsys.readouterr().out


def test_cli_config_base_sem_audio_dir_e_recusado(metadata, capsys):
    from scripts.importar_asvspoof2021 import main

    assert main(["--metadata", str(metadata), "--codec", "opus",
                 "--config-base", "configs/fusion_v4.yaml"]) == 1
    assert "--audio-dir" in capsys.readouterr().out


def test_cli_sem_config_base_avisa_para_nao_copiar_a_mao(metadata, tmp_path, capsys):
    from scripts.importar_asvspoof2021 import main

    main(["--metadata", str(metadata), "--codec", "opus",
          "--saida", str(tmp_path / "p.txt")])
    assert "sobrescreve os resultados" in capsys.readouterr().out



# --------------------------------------------------------------------------- #
# Regressão: o formato real do bonafide
#
# Rodado no arquivo verdadeiro, o `--listar` devolveu 163.114 trials e ZERO
# bonafide. As linhas bonafide têm `bonafide` duas vezes (coluna de ataque e
# coluna de chave), e o parser exigia uma ocorrência só.
# --------------------------------------------------------------------------- #
def test_bonafide_no_formato_real_e_lido(tmp_path):
    p = tmp_path / "m.txt"
    p.write_text("LA_0007 LA_E_5932896 alaw ita_tx bonafide bonafide notrim eval\n"
                 "LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval\n",
                 encoding="utf-8")
    trials = ler_metadata(p)
    assert [t.chave for t in trials] == ["bonafide", "spoof"]
    assert trials[0].ataque == "-", "o bonafide não tem ataque"


def test_fixture_tem_as_duas_classes_nas_condicoes_certas(metadata):
    """O `--listar` real mostrou 0 bonafide em TODA condição; aqui não pode."""
    achadas = {n: (b, s) for n, b, s in condicoes(ler_metadata(metadata))}
    assert achadas["alaw/ita_tx"][0] == 1
    assert achadas["opus/ita_tx"][0] == 1


def test_descarte_de_linha_e_relatado(tmp_path):
    from src.data.asvspoof2021 import ler_metadata as ler

    p = tmp_path / "m.txt"
    p.write_text("LA_0009 LA_E_1 alaw ita_tx A07 spoof notrim eval\n"
                 "linha quebrada sem chave nenhuma aqui\n", encoding="utf-8")
    relatorio: dict = {}
    ler(p, relatorio)
    assert relatorio["ignoradas"] == 1
    assert "linha quebrada" in relatorio["exemplos"][0]


def test_cli_para_quando_uma_classe_inteira_some(tmp_path, capsys):
    """Era o sintoma do defeito: seguir adiante com zero bonafide."""
    from scripts.importar_asvspoof2021 import main

    p = tmp_path / "m.txt"
    p.write_text("".join(f"LA_0009 LA_E_{i} alaw ita_tx A07 spoof notrim eval\n"
                         for i in range(5)), encoding="utf-8")
    assert main(["--metadata", str(p), "--listar"]) == 1
    assert "uma classe só" in capsys.readouterr().out


def test_listar_mostra_as_fases(metadata, capsys):
    """O Müller reporta a fase de progresso; a divisão precisa estar visível."""
    from scripts.importar_asvspoof2021 import main

    assert main(["--metadata", str(metadata), "--listar"]) == 0
    saida = capsys.readouterr().out
    assert "progress" in saida and "eval" in saida


def test_fases_conta_por_classe(metadata):
    from src.data.asvspoof2021 import fases

    contagem = fases(ler_metadata(metadata))
    assert contagem["eval"] == (2, 3)
    assert contagem["progress"] == (1, 1)
