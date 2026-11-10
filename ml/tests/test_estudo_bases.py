"""Parte 5 nas bases ASVspoof (scripts/estudo_bases.py), com uma base falsa."""

import json

import numpy as np
import soundfile as sf

from scripts.estudo_bases import SR, ler_corte, ler_protocolo, main


def _som(tipo, seed, segundos=3.5):
    rng = np.random.default_rng(seed)
    x = np.convolve(rng.standard_normal(int(SR * segundos)), [1.0, 0.6, 0.2], "same") * 0.05
    t = np.arange(x.size) / SR
    if tipo == "tom":       # linha fixa, como as do XTTS e do Chatterbox
        x += 0.004 * np.sin(2 * np.pi * 6000 * t + seed)
    elif tipo == "quadros":  # um pulso a cada 320 amostras: pente de 50 Hz
        x[::320] += 0.4
    return x.astype(np.float32)


def _base(tmp_path, n=4):
    pasta = tmp_path / "flac"
    pasta.mkdir()
    linhas = []
    for classe, tipo in (("bonafide", None), ("A01", "tom"), ("A02", "quadros")):
        for i in range(n):
            nome = f"{classe}_{i}"
            sf.write(pasta / f"{nome}.flac", _som(tipo, seed=100 * len(linhas) + i), SR)
            chave = "bonafide" if classe == "bonafide" else "spoof"
            linhas.append(f"LA_0001 {nome} - {'-' if chave == 'bonafide' else classe} {chave}")
    sf.write(pasta / "curto.flac", _som(None, 9, segundos=2.0), SR)
    linhas.append("LA_0001 curto - - bonafide")
    (tmp_path / "prot.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return tmp_path / "prot.txt", pasta


def test_protocolo_nos_dois_formatos(tmp_path):
    (tmp_path / "p.txt").write_text(
        "LA_0079 LA_T_1 - - bonafide\nLA_0079 LA_T_2 - A01 spoof\n"
        "T_2896 T_0000000005 - A05 spoof\n", encoding="utf-8")
    assert ler_protocolo(tmp_path / "p.txt") == {"bonafide": ["LA_T_1"], "A01": ["LA_T_2"],
                                                  "A05": ["T_0000000005"]}


def test_arquivo_curto_fica_de_fora(tmp_path):
    sf.write(tmp_path / "c.flac", _som(None, 1, segundos=2.0), SR)
    sf.write(tmp_path / "l.flac", _som(None, 1, segundos=3.2), SR)
    assert ler_corte(tmp_path / "c.flac") is None
    assert len(ler_corte(tmp_path / "l.flac")) == 3 * SR


def test_caminho_completo(tmp_path, capsys):
    prot, pasta = _base(tmp_path)
    saida = tmp_path / "b.json"
    assert main(["--protocolo", str(prot), "--pasta", str(pasta), "--rotulo", "falsa",
                 "--n", "4", "--json", str(saida)]) == 0
    c = json.loads(saida.read_text(encoding="utf-8"))["classes"]
    assert "curto" not in c["bonafide"]["arquivos"]
    linha = c["A01"]["linhas"][0]
    assert abs(linha["hz"] - 6000) <= 2
    assert linha["metade_b"] == [2, 2] and linha["humanos"] == [0, 2]
    assert c["A01"]["auc_linhas"] == 1.0
    assert c["A02"]["auc_pente"] == 1.0
    assert "auc_pente" not in c["bonafide"]
    assert "Linhas fixas de cada classe" in capsys.readouterr().out
