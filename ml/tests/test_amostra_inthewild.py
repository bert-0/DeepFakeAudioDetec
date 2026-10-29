"""Amostra do In-the-Wild (scripts/montar_amostra_inthewild.py), com um meta.csv falso."""

import csv

import pytest

from scripts.montar_amostra_inthewild import main, selecionar


def _meta(pasta):
    pasta.mkdir()
    linhas = ([{"file": f"{i}.wav", "speaker": "Pessoa A", "label": "bona-fide"} for i in range(10)]
              + [{"file": f"{i}.wav", "speaker": "Pessoa A", "label": "spoof"} for i in range(10, 13)]
              + [{"file": f"{i}.wav", "speaker": "Pessoa B", "label": "spoof"} for i in range(13, 20)])
    with open(pasta / "meta.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "speaker", "label"])
        w.writeheader()
        w.writerows(linhas)
    return linhas


def test_ate_n_por_pessoa_e_classe_e_reprodutivel(tmp_path):
    linhas = _meta(tmp_path / "itw")
    sel = selecionar(linhas, 4, semente=42)
    contagem = {}
    for r in sel:
        contagem[(r["speaker"], r["label"])] = contagem.get((r["speaker"], r["label"]), 0) + 1
    assert contagem == {("Pessoa A", "bona-fide"): 4, ("Pessoa A", "spoof"): 3, ("Pessoa B", "spoof"): 4}
    assert sel == selecionar(linhas, 4, semente=42)


def test_lista_no_formato_do_estudo_ssl(tmp_path):
    from scripts.estudo_ssl import ler_lista

    _meta(tmp_path / "itw")
    lista = tmp_path / "l.tsv"
    assert main(["--pasta", str(tmp_path / "itw"), "--por-classe", "2", "--lista", str(lista)]) == 0
    itens = ler_lista(lista)
    assert len(itens) == 6 and {c for _, c, _ in itens} == {"bonafide", "spoof"}
    assert all((rot == 0) == (c == "bonafide") for _, c, rot in itens)


def test_recusa_rotulo_desconhecido(tmp_path):
    pasta = tmp_path / "itw"
    pasta.mkdir()
    (pasta / "meta.csv").write_text("file,speaker,label\n0.wav,A,fake\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--pasta", str(pasta)])
