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


def test_divisao_por_pessoa_sem_misturar_e_reprodutivel():
    from scripts.montar_amostra_inthewild import dividir_pessoas

    pessoas = [f"P{i:02d}" for i in range(12)] * 3
    partes = dividir_pessoas(pessoas, 6, 2, semente=42)
    assert [len(partes[k]) for k in ("treino", "dev", "regua")] == [6, 2, 4]
    assert set().union(*map(set, partes.values())) == set(pessoas)
    assert not set(partes["treino"]) & set(partes["regua"]) and not set(partes["dev"]) & set(partes["regua"])
    assert partes == dividir_pessoas(pessoas, 6, 2, semente=42)
    with pytest.raises(ValueError):
        dividir_pessoas(pessoas, 10, 2)


def test_separa_o_npz_pela_pessoa(tmp_path):
    import numpy as np

    from scripts.montar_amostra_inthewild import COLUNAS

    pessoas = ["A", "B", "C", "D", "E"]
    amostra = tmp_path / "amostra.tsv"
    linhas = [(f"data/InTheWild/{i}.wav", "bonafide" if i % 2 else "spoof", str(1 - i % 2), pessoas[i % 5])
              for i in range(20)]
    amostra.write_text("\t".join(COLUNAS) + "\n" + "".join("\t".join(x) + "\n" for x in linhas), encoding="utf-8")
    npz = tmp_path / "itw.npz"
    np.savez(npz, arquivo=np.array([f"data\\InTheWild\\{i}.wav" for i in range(20)]),
             rotulo=np.array([int(x[2]) for x in linhas], np.int8), camadas=np.arange(20 * 6).reshape(20, 3, 2),
             nomes_medidas=np.array(["a", "b"]))
    lista = tmp_path / "pessoas.tsv"
    assert main(["--dividir", "2", "1", "--amostra", str(amostra), "--npz", str(npz),
                 "--lista-pessoas", str(lista)]) == 0
    with open(lista, encoding="utf-8", newline="") as f:
        parte_de = {r["pessoa"]: r["parte"] for r in csv.DictReader(f, delimiter="\t")}
    assert sorted(parte_de.values()) == ["dev", "regua", "regua", "treino", "treino"]
    total = 0
    for parte in ("treino", "dev", "regua"):
        with np.load(tmp_path / f"itw_{parte}.npz") as d:
            donos = {pessoas[int(a.rsplit("\\", 1)[-1][:-4]) % 5] for a in d["arquivo"]}
            assert {parte_de[p] for p in donos} == {parte}
            assert d["camadas"].shape[1:] == (3, 2) and list(d["nomes_medidas"]) == ["a", "b"]
            total += len(d["arquivo"])
    assert total == 20


def test_recusa_rotulo_desconhecido(tmp_path):
    pasta = tmp_path / "itw"
    pasta.mkdir()
    (pasta / "meta.csv").write_text("file,speaker,label\n0.wav,A,fake\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--pasta", str(pasta)])
