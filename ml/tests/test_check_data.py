"""Testes da verificação da base (scripts/check_data.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.check_data import CONTAGENS_OFICIAIS, _check_sample, check_partition  # noqa: E402


def _escreve_flac(destino: Path, segundos: float = 4.0, sr: int = 16000) -> Path:
    t = np.arange(int(sr * segundos)) / sr
    sf.write(destino, (0.4 * np.sin(2 * np.pi * 180 * t)).astype("float32"), sr)
    return destino


def _trunca(caminho: Path, fracao: float) -> None:
    """Simula um download interrompido: mantém só os primeiros bytes."""
    dados = caminho.read_bytes()
    caminho.write_bytes(dados[: int(len(dados) * fracao)])


# --------------------------------------------------------------------------- #
# Regressão: `sf.info` lê só o cabeçalho, que sobrevive à truncagem. Um FLAC
# cortado pela metade reporta a duração original e passaria como íntegro —
# derrubando o treino horas depois, com uma mensagem que não diz qual arquivo.
# --------------------------------------------------------------------------- #
def test_modo_raso_aprova_flac_truncado(tmp_path, capsys):
    _escreve_flac(tmp_path / "u0.flac")
    _trunca(tmp_path / "u0.flac", 0.1)

    ok = _check_sample([("u0", 1)], tmp_path, sample=1, file_ext=".flac", deep=False)

    assert ok, "o modo raso deveria (infelizmente) aprovar — é o que motiva o --deep"
    assert "cabeçalho" in capsys.readouterr().out


def test_deep_reprova_flac_truncado(tmp_path, capsys):
    _escreve_flac(tmp_path / "u0.flac")
    _trunca(tmp_path / "u0.flac", 0.1)

    ok = _check_sample([("u0", 1)], tmp_path, sample=1, file_ext=".flac", deep=True)

    assert not ok
    saida = capsys.readouterr().out
    assert "u0.flac" in saida, "a mensagem precisa dizer QUAL arquivo falhou"


def test_deep_aprova_arquivo_integro(tmp_path):
    _escreve_flac(tmp_path / "u0.flac")
    assert _check_sample([("u0", 1)], tmp_path, sample=1, file_ext=".flac", deep=True)


def test_deep_percorre_todos_ignorando_sample(tmp_path, capsys):
    """Com --deep, `--sample` não pode limitar a varredura."""
    for i in range(5):
        _escreve_flac(tmp_path / f"u{i}.flac")
    _trunca(tmp_path / "u4.flac", 0.1)   # só o último está quebrado

    itens = [(f"u{i}", 1) for i in range(5)]
    assert not _check_sample(itens, tmp_path, sample=1, file_ext=".flac", deep=True)
    assert "u4.flac" in capsys.readouterr().out


def test_arquivo_faltando_e_reportado_pelo_nome(tmp_path, capsys):
    ok = _check_sample([("sumiu", 1)], tmp_path, sample=1, file_ext=".flac", deep=True)
    assert not ok
    assert "sumiu.flac" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Regressão: o parser descarta linhas malformadas em silêncio, então um
# protocolo truncado gera uma partição menor sem nenhum sinal.
# --------------------------------------------------------------------------- #
def _protocolo(destino: Path, n: int) -> Path:
    linhas = [f"LA_0001 u{i:05d} - {'A07' if i % 2 else '-'} "
              f"{'spoof' if i % 2 else 'bonafide'}" for i in range(n)]
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return destino


def test_avisa_quando_a_contagem_diverge_da_oficial(tmp_path, capsys):
    proto = _protocolo(tmp_path / "p.txt", 100)      # muito menos que os 25.380
    audio = tmp_path / "flac"; audio.mkdir()
    for i in range(100):
        _escreve_flac(audio / f"u{i:05d}.flac", segundos=0.2)

    ok = check_partition("train", str(proto), str(audio), sample=2,
                         file_ext=".flac", deep=False)

    saida = capsys.readouterr().out
    assert not ok
    assert str(CONTAGENS_OFICIAIS["train"]) in saida
    assert "truncado" in saida


def test_nao_avisa_quando_a_contagem_bate(tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(CONTAGENS_OFICIAIS, "train", 4)
    proto = _protocolo(tmp_path / "p.txt", 4)
    audio = tmp_path / "flac"; audio.mkdir()
    for i in range(4):
        _escreve_flac(audio / f"u{i:05d}.flac", segundos=0.2)

    ok = check_partition("train", str(proto), str(audio), sample=4,
                         file_ext=".flac", deep=True)

    assert ok
    assert "truncado" not in capsys.readouterr().out


@pytest.mark.parametrize("particao,esperado", CONTAGENS_OFICIAIS.items())
def test_contagens_oficiais_do_asvspoof(particao, esperado):
    """Trava os números para que ninguém os altere sem intenção."""
    assert CONTAGENS_OFICIAIS[particao] == esperado
    assert sum(CONTAGENS_OFICIAIS.values()) == 121461


def test_deep_da_sinal_de_vida_em_bases_grandes(tmp_path, capsys):
    """Sem progresso, uma varredura de minutos é indistinguível de um travamento.

    Os arquivos nem precisam existir: o contador é impresso antes da leitura,
    que é justamente o que garante o sinal de vida mesmo em disco lento.
    """
    itens = [(f"u{i}", 1) for i in range(12000)]

    _check_sample(itens, tmp_path, sample=20, file_ext=".flac", deep=True)

    saida = capsys.readouterr().out
    assert "5000/12000 decodificados" in saida
    assert "10000/12000 decodificados" in saida


def test_modo_raso_nao_imprime_progresso(tmp_path, capsys):
    """A varredura rasa é instantânea; progresso ali seria só ruído no log."""
    _check_sample([(f"u{i}", 1) for i in range(12000)], tmp_path,
                  sample=20, file_ext=".flac", deep=False)
    assert "decodificados" not in capsys.readouterr().out
