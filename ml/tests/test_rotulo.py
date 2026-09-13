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


# --------------------------------------------------------------------------- #
# Listagem de exemplos — existe para montar as figuras sem caçar IDs no
# protocolo de 71.237 linhas.
# --------------------------------------------------------------------------- #
@pytest.fixture
def base_grande(tmp_path):
    linhas = [f"LA_0039 LA_E_{i:07d} - - bonafide" for i in range(1000, 1020)]
    for a in ("A07", "A10", "A12", "A17"):
        linhas += [f"LA_0040 LA_E_{a[1:]}{i:05d} - {a} spoof" for i in range(30)]
    proto = tmp_path / "eval.txt"
    proto.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"data": {
        "protocols": {"eval": str(proto)},
        "audio_dir": {"eval": "data/LA/eval/flac"}}}), encoding="utf-8")
    return cfg


def test_exemplos_trazem_as_duas_classes(base_grande, capsys):
    sys.argv = ["x", "--config", str(base_grande), "--exemplos", "3"]
    assert main() == 0
    saida = capsys.readouterr().out
    assert saida.count("bonafide") >= 3
    assert saida.count("spoof") >= 3


def test_exemplos_espalham_os_spoof_entre_ataques(base_grande, capsys):
    """Pegar os primeiros do protocolo daria todos do mesmo algoritmo."""
    sys.argv = ["x", "--config", str(base_grande), "--exemplos", "4"]
    main()
    saida = capsys.readouterr().out
    for ataque in ("A07", "A10", "A12", "A17"):
        assert ataque in saida, f"{ataque} ficou de fora da amostra"


def test_exemplos_mostram_onde_estao_os_arquivos(base_grande, capsys):
    sys.argv = ["x", "--config", str(base_grande), "--exemplos", "2"]
    main()
    saida = capsys.readouterr().out
    assert "data/LA/eval/flac" in saida
    assert "BAIXO = voz humana" in saida


def test_exemplos_nao_quebram_com_n_maior_que_a_base(base_grande, capsys):
    sys.argv = ["x", "--config", str(base_grande), "--exemplos", "500"]
    assert main() == 0


def test_sem_alvo_e_sem_exemplos_explica_o_uso(base_grande):
    sys.argv = ["x", "--config", str(base_grande)]
    with pytest.raises(SystemExit):
        main()


# --------------------------------------------------------------------------- #
# Rótulo verdadeiro dentro do monitor
#
# Sem ele, rodar o monitor num arquivo mostra que o sistema opera e nada mais.
# Com ele, a mesma execução vira verificação — e foi a falta disso que deixou
# um score de 0,048 ser lido como "o programa disse que é sintético".
# --------------------------------------------------------------------------- #
def test_busca_encontra_nas_tres_particoes(projeto):
    from src.config import load_config
    from src.data.dataset import buscar_rotulo

    cfg = load_config(str(projeto))
    assert buscar_rotulo(cfg, "LA_E_1000147") == ("eval", 0, "-")
    assert buscar_rotulo(cfg, "LA_E_1000148") == ("eval", 1, "A10")
    assert buscar_rotulo(cfg, "LA_T_1000002") == ("train", 1, "A01")


def test_busca_devolve_none_para_id_de_fora(projeto):
    from src.config import load_config
    from src.data.dataset import buscar_rotulo

    assert buscar_rotulo(load_config(str(projeto)), "LA_E_A9898607") is None


def test_busca_ignora_protocolo_ausente_sem_quebrar(tmp_path):
    from src.data.dataset import buscar_rotulo

    cfg = {"data": {"protocols": {"eval": str(tmp_path / "sumiu.txt")}}}
    assert buscar_rotulo(cfg, "qualquer") is None


def test_busca_sem_secao_de_dados(tmp_path):
    from src.data.dataset import buscar_rotulo

    assert buscar_rotulo({}, "x") is None


def test_resumo_confronta_o_score_com_o_rotulo():
    import io
    from contextlib import redirect_stdout

    from monitor import relatar
    from src.capture.analyzer import Agregador, Leitura

    def _saida(score, verdade):
        ag = Agregador()
        ag.adicionar(Leitura(indice=0, instante=0.0, score=score, rms=0.1))
        buf = io.StringIO()
        with redirect_stdout(buf):
            relatar(ag, None, limiar=0.7173, verdade=verdade)
        return buf.getvalue()

    assert "COERENTE" in _saida(0.048, "bonafide")
    assert "COERENTE" in _saida(0.910, "spoof")
    assert "DIVERGENTE" in _saida(0.048, "spoof")
    assert "DIVERGENTE" in _saida(0.910, "bonafide")


def test_resumo_avisa_que_um_caso_nao_e_taxa_de_erro():
    """O confronto é ilustrativo; a métrica sai do evaluate.py."""
    import io
    from contextlib import redirect_stdout

    from monitor import relatar
    from src.capture.analyzer import Agregador, Leitura

    ag = Agregador()
    ag.adicionar(Leitura(indice=0, instante=0.0, score=0.048, rms=0.1))
    buf = io.StringIO()
    with redirect_stdout(buf):
        relatar(ag, None, limiar=0.7173, verdade="bonafide")
    assert "não mede taxa de erro" in buf.getvalue()


def test_sem_rotulo_o_resumo_nao_inventa_veredito():
    import io
    from contextlib import redirect_stdout

    from monitor import relatar
    from src.capture.analyzer import Agregador, Leitura

    ag = Agregador()
    ag.adicionar(Leitura(indice=0, instante=0.0, score=0.048, rms=0.1))
    buf = io.StringIO()
    with redirect_stdout(buf):
        relatar(ag, None, limiar=0.7173)
    saida = buf.getvalue()
    assert "COERENTE" not in saida and "DIVERGENTE" not in saida
