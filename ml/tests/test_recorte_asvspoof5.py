"""Recorte da ASVspoof 5 a partir dos .tar (scripts/montar_recorte_asvspoof5.py).

Os .tar aqui são sintéticos: FLAC de 16 kHz com silêncio antes da "fala"
(ruído), no mesmo arranjo de pastas dos do Zenodo (flac_T/<id>.flac).
"""

import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.data.asvspoof5 as asv5  # noqa: E402
from scripts.montar_recorte_asvspoof5 import main  # noqa: E402
from src.data.dataset import parse_protocol_with_systems  # noqa: E402

SR = 16000

PROTOCOLO = """\
T_0001 T_0000000000 F - - - - bonafide bonafide -
T_0001 T_0000000001 F - - - AC1 A01 spoof -
T_0002 T_0000000002 M - - - AC1 A01 spoof -
T_0002 T_0000000003 M - - - AC2 A02 spoof -
T_0003 T_0000000004 F - - - - bonafide bonafide -
"""


def _audio(silencio_s, fala_s, seed):
    rng = np.random.default_rng(seed)
    wav = np.zeros(int((silencio_s + fala_s + 1.0) * SR), dtype=np.float32)
    i = int(silencio_s * SR)
    wav[i:i + int(fala_s * SR)] = rng.standard_normal(int(fala_s * SR)) * 0.2
    return wav


#: 9 s com a fala começando em 2 s, e 3 s (menos que o recorte de 6 s)
AUDIOS = {"T_0000000000": _audio(2.0, 6.0, 0), "T_0000000001": _audio(2.0, 6.0, 1),
          "T_0000000002": _audio(0.5, 1.5, 2), "T_0000000003": _audio(2.0, 6.0, 3),
          "T_0000000004": _audio(0.5, 1.5, 4)}


def _tar(pasta, nome, ids):
    origem = pasta / "origem"
    origem.mkdir(exist_ok=True)
    caminho = pasta / nome
    with tarfile.open(caminho, "w") as tar:
        for i in ids:
            arq = origem / f"{i}.flac"
            sf.write(arq, AUDIOS[i], SR, subtype="PCM_16")
            tar.add(arq, arcname=f"flac_T/{i}.flac")
    return caminho


@pytest.fixture
def base(tmp_path):
    (tmp_path / "proto.tsv").write_text(PROTOCOLO, encoding="utf-8")
    _tar(tmp_path, "t1.tar", ["T_0000000000", "T_0000000001", "T_0000000002"])
    _tar(tmp_path, "t2.tar", ["T_0000000003", "T_0000000004"])
    return tmp_path


def _rodar(base, tars, n=1):
    return main(["--protocolo", str(base / "proto.tsv"), "--tars", *tars,
                 "--saida", str(base / "recorte"), "--falsos-por-ataque", str(n),
                 "--lista", str(base / "lista.txt")])


def test_corte_comeca_meio_segundo_antes_da_fala():
    """A fala começa em 2 s; o split do librosa a acha por quadros (salto de 512
    amostras, janela de 2048), um pouco antes: a margem fica entre 0,5 e ~0,6 s."""
    wav = AUDIOS["T_0000000000"]
    trecho = asv5.cortar_trecho(wav, SR)
    assert len(trecho) == 6 * SR
    inicio = 2 * SR - int(np.flatnonzero(trecho)[0])
    assert 1.4 * SR <= inicio <= 1.5 * SR
    np.testing.assert_array_equal(trecho, wav[inicio:inicio + 6 * SR])


def test_corte_de_audio_curto_nao_muda_e_fala_tardia_puxa_a_janela():
    curto = AUDIOS["T_0000000002"]
    assert asv5.cortar_trecho(curto, SR) is curto
    tardio = _audio(8.0, 1.0, 5)                 # 10 s, fala em 8–9 s
    trecho = asv5.cortar_trecho(tardio, SR)
    np.testing.assert_array_equal(trecho, tardio[-6 * SR:])


def test_monta_o_recorte_e_retoma_com_os_tar_que_faltavam(base, capsys):
    assert _rodar(base, [str(base / "t1.tar")]) == 0
    saida = capsys.readouterr().out
    assert "faltam: 2" in saida                  # os dois do t2.tar
    parcial = parse_protocol_with_systems(base / "recorte" / "protocolo.txt")
    assert len(parcial) == 2

    # padrão com curinga, como no PowerShell; o que já existe é pulado
    assert _rodar(base, [str(base / "t*.tar")]) == 0
    saida = capsys.readouterr().out
    assert "Escritos agora: 2; já existiam: 2; faltam: 0" in saida

    protocolo = parse_protocol_with_systems(base / "recorte" / "protocolo.txt")
    assert [r[0] for r in protocolo] == [r[0] for r in parse_protocol_with_systems(base / "lista.txt")]
    assert {r[2] for r in protocolo} == {"-", "A01", "A02"}
    assert sum(r[1] == 0 for r in protocolo) == 2       # todos os bonafide
    for nome, _, _ in protocolo:
        info = sf.info(base / "recorte" / "flac" / f"{nome}.flac")
        assert info.samplerate == SR and info.subtype == "PCM_16"
        assert info.frames <= 6 * SR
    assert not list((base / "recorte" / "flac").glob("*.parcial"))


def test_protocolo_oficial_truncado_e_recusado(tmp_path, capsys):
    (tmp_path / "ASVspoof5.train.tsv").write_text(PROTOCOLO, encoding="utf-8")
    rc = main(["--protocolo", str(tmp_path / "ASVspoof5.train.tsv"), "--tars", "x.tar",
               "--saida", str(tmp_path / "r"), "--lista", str(tmp_path / "l.txt")])
    assert rc == 1 and "182357" in capsys.readouterr().out


def test_confere_tamanho_e_md5(tmp_path, monkeypatch):
    import hashlib

    incompleto = tmp_path / "flac_T_aa.tar"
    incompleto.write_bytes(b"0" * 10)
    assert "download não terminou" in asv5.conferir_arquivo(incompleto)
    assert asv5.conferir_arquivo(tmp_path / "outro.tar") is None   # fora da tabela

    arq = tmp_path / "teste.tar"
    arq.write_bytes(b"abc" * 1000)
    certo = hashlib.md5(arq.read_bytes()).hexdigest()
    monkeypatch.setitem(asv5.ARQUIVOS_OFICIAIS, "teste.tar", (3000, certo))
    assert asv5.conferir_arquivo(arq, md5=True) is None
    monkeypatch.setitem(asv5.ARQUIVOS_OFICIAIS, "teste.tar", (3000, "0" * 32))
    assert "Baixe de novo" in asv5.conferir_arquivo(arq, md5=True)
    assert asv5.conferir_arquivo(arq) is None                     # sem md5, só o tamanho


def test_le_o_tar_em_fluxo_so_os_pedidos(base):
    with open(base / "t1.tar", "rb") as fluxo:
        lidos = list(asv5.ler_tar(fluxo, {"T_0000000001"}))
    assert [(n, t) for n, _, t in lidos] == [("T_0000000001", SR)]
    np.testing.assert_allclose(lidos[0][1], AUDIOS["T_0000000001"], atol=1 / 32768)


def test_amostra_estratificada_para_o_dev(base, capsys):
    rc = main(["--protocolo", str(base / "proto.tsv"), "--tars", str(base / "t*.tar"),
               "--saida", str(base / "dev"), "--amostra", "3",
               "--lista", str(base / "lista_dev.txt")])
    assert rc == 0
    protocolo = parse_protocol_with_systems(base / "dev" / "protocolo.txt")
    # um de cada estrato (bonafide, A01, A02), na ordem do protocolo
    assert sorted(r[2] for r in protocolo) == ["-", "A01", "A02"]
    assert [r[0] for r in protocolo] == sorted(r[0] for r in protocolo)
    assert len(list((base / "dev" / "flac").glob("*.flac"))) == 3


def test_amostra_padrao_e_a_do_dev_ja_montado():
    """Com (chave, ataque), a função nova dá a mesma seleção da subamostrar do 2021."""
    base = [asv5.Enunciado(f"T_{i % 7}", f"T_{i:010d}", "F", ["-", "C01", "C02"][i % 3], "-", "-",
                           "-" if i % 5 == 0 else "AC1", "-" if i % 5 == 0 else f"A{1 + i % 4:02d}",
                           "bonafide" if i % 5 == 0 else "spoof") for i in range(300)]
    assert asv5.amostra_estratificada(base, 60, 42) == asv5.subamostrar(base, 60, 42)
    por_codec = asv5.amostra_estratificada(base, 30, 42, ("chave", "ataque", "codec"))
    estratos = {(e.chave, e.ataque, e.codec) for e in base}
    assert {(e.chave, e.ataque, e.codec) for e in por_codec} == estratos   # nenhum estrato perdido


def test_formato_original_volta_igual_e_guarda_o_codec(tmp_path):
    e = [asv5.Enunciado("E_0001", "E_0000000004", "M", "C05", "2", "E_0000000009", "AC3", "A20", "spoof"),
         asv5.Enunciado("E_0002", "E_0000000005", "F", "-", "-", "-", "-", "-", "bonafide")]
    destino = tmp_path / "sel.tsv"
    assert asv5.escrever_original(e, destino) == 2
    assert asv5.ler_protocolo(destino) == e


def test_eval_sem_corte_e_estratificado_por_codec(base, capsys):
    rc = main(["--protocolo", str(base / "proto.tsv"), "--tars", str(base / "t*.tar"),
               "--saida", str(base / "eval"), "--amostra", "3", "--segundos", "0",
               "--estratos", "chave,ataque,codec", "--lista", str(base / "lista_eval.txt")])
    assert rc == 0
    for nome, _, _ in parse_protocol_with_systems(base / "eval" / "protocolo.txt"):
        assert sf.info(base / "eval" / "flac" / f"{nome}.flac").frames == len(AUDIOS[nome])
    assert (base / "lista_eval.tsv").is_file()                   # a seleção com o codec


def test_tar_do_eval_e_conferido():
    assert asv5.ARQUIVOS_OFICIAIS["flac_E_aj.tar"][0] == 8_623_206_400
    assert sum(1 for n in asv5.ARQUIVOS_OFICIAIS if n.startswith("flac_E_")) == 10
