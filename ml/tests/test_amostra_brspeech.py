"""Amostra do BRSpeech-DF (scripts/montar_amostra_brspeech.py) com Parquet sintético."""

import io

import numpy as np
import pytest
import soundfile as sf

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from scripts.montar_amostra_brspeech import main, selecionar  # noqa: E402


def _flac(seed):
    b = io.BytesIO()
    sf.write(b, (np.random.default_rng(seed).standard_normal(2400) * 0.1).astype(np.float32), 24000, format="FLAC")
    return b.getvalue()


def _base(raiz):
    (raiz / "bonafide").mkdir(parents=True)
    (raiz / "spoof").mkdir()
    reais = pa.table({"audio": [{"bytes": _flac(i), "path": f"5739_1_{i:06d}-0001.flac"} for i in range(4)],
                      "label": [0] * 4})
    pq.write_table(reais, raiz / "bonafide" / "test-00000-of-00001.parquet")
    pq.write_table(pa.table({"audio": [{"bytes": _flac(9), "path": "1_2_000000-0001.flac"}], "label": [0]}),
                   raiz / "bonafide" / "train-00000-of-00001.parquet")
    modelos = ["xtts"] * 5 + ["f5tts"] * 3
    clones = pa.table({"audio": [{"bytes": _flac(10 + i), "path": None} for i in range(8)],
                       "label": [1] * 8, "model": modelos})
    pq.write_table(clones, raiz / "spoof" / "test-00000-of-00001.parquet")


def test_selecao_por_modelo_e_todas_as_reais_da_parte(tmp_path):
    _base(tmp_path)
    e = selecionar(tmp_path, "test", por_modelo=2)
    assert len(e["bonafide"]) == 4, "só a parte pedida: o treino fica de fora"
    assert len(e["xtts"]) == 2 and len(e["f5tts"]) == 2
    assert set(e["f5tts"]) <= {5, 6, 7}
    assert e == selecionar(tmp_path, "test", por_modelo=2)


def test_exporta_flac_legivel_e_lista(tmp_path):
    _base(tmp_path / "base")
    lista = tmp_path / "lista.tsv"
    assert main(["--pasta", str(tmp_path / "base"), "--parte", "test", "--por-modelo", "3",
                 "--saida", str(tmp_path / "amostra"), "--lista", str(lista)]) == 0
    reais = sorted((tmp_path / "amostra/test/bonafide").glob("*.flac"))
    assert [r.name for r in reais][0] == "5739_1_000000-0001.flac" and len(reais) == 4
    assert len(list((tmp_path / "amostra/test/xtts").glob("*.flac"))) == 3
    w, sr = sf.read(reais[0])
    assert sr == 24000 and len(w) == 2400
    linhas = lista.read_text(encoding="utf-8").splitlines()
    assert linhas[0] == "arquivo\tclasse" and len(linhas) == 1 + 4 + 3 + 3


def test_teto_de_reais(tmp_path):
    _base(tmp_path)
    e = selecionar(tmp_path, "test", por_modelo=1, max_reais=2)
    assert len(e["bonafide"]) == 2 and set(e["bonafide"]) <= {0, 1, 2, 3}
