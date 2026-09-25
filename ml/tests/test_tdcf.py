"""Testes do cálculo de min t-DCF (scripts/tdcf.py)."""

import numpy as np
import pytest

from scripts.tdcf import (
    CUSTOS_2019,
    curva_det,
    eer_e_limiar,
    exportar_formato_oficial,
    ler_scores_asv,
    min_tdcf,
    postos_medios,
    score_bonafide,
)


def _asv(rng):
    return {
        "target": rng.normal(3.0, 1.0, 400),
        "nontarget": rng.normal(-3.0, 1.0, 400),
        "spoof": rng.normal(1.0, 1.5, 400),
    }


def _forca_bruta(bona, spoof, asv, c=CUSTOS_2019):
    """Mesma definição, escrita da forma mais literal possível."""
    _, lim_asv = eer_e_limiar(asv["target"], asv["nontarget"])
    pfa_asv = np.mean(asv["nontarget"] >= lim_asv)
    pmiss_asv = np.mean(asv["target"] < lim_asv)
    pmiss_spoof = np.mean(asv["spoof"] < lim_asv)
    c1 = c["Ptar"] * (c["Cmiss_cm"] - c["Cmiss_asv"] * pmiss_asv) - c["Pnon"] * c["Cfa_asv"] * pfa_asv
    c2 = c["Cfa_cm"] * c["Pspoof"] * (1 - pmiss_spoof)
    melhor = np.inf
    for t in list(np.unique(np.concatenate([bona, spoof]))) + [np.inf]:
        pmiss_cm = np.mean(bona < t)
        pfa_cm = np.mean(spoof >= t)
        melhor = min(melhor, (c1 * pmiss_cm + c2 * pfa_cm) / min(c1, c2))
    return melhor


def test_confere_com_forca_bruta():
    rng = np.random.default_rng(0)
    asv = _asv(rng)
    bona = rng.normal(1.0, 1.0, 300)
    spoof = rng.normal(-1.0, 1.0, 900)
    assert min_tdcf(bona, spoof, asv)["min_tdcf"] == pytest.approx(_forca_bruta(bona, spoof, asv))


def test_cm_perfeita_da_zero():
    rng = np.random.default_rng(1)
    r = min_tdcf(rng.uniform(5, 6, 100), rng.uniform(0, 1, 100), _asv(rng))
    assert r["min_tdcf"] == 0.0
    assert r["eer_cm"] == 0.0


def test_cm_sem_informacao_da_um():
    """Score constante: só há 'aceita tudo' e 'rejeita tudo', e o normalizado vale 1."""
    rng = np.random.default_rng(2)
    r = min_tdcf(np.zeros(100), np.zeros(300), _asv(rng))
    assert r["min_tdcf"] == pytest.approx(1.0)


def test_nunca_passa_de_um():
    rng = np.random.default_rng(3)
    asv = _asv(rng)
    # CM invertida (pior que o acaso): o mínimo ainda é o de um dos extremos.
    r = min_tdcf(rng.normal(-2, 1, 200), rng.normal(2, 1, 200), asv)
    assert r["min_tdcf"] <= 1.0 + 1e-12


def test_invariante_a_transformacao_monotonica():
    rng = np.random.default_rng(4)
    asv = _asv(rng)
    bona, spoof = rng.normal(1, 1, 200), rng.normal(-1, 1, 600)
    a = min_tdcf(bona, spoof, asv)["min_tdcf"]
    b = min_tdcf(np.exp(bona), np.exp(spoof), asv)["min_tdcf"]
    assert a == pytest.approx(b)


def test_empates_nao_criam_pontos_de_operacao():
    """Com score saturado (vários spoof e bonafide em exatamente -1,0), a curva
    não pode separar áudios empatados — nem depender da ordem do array."""
    bona = np.array([-1.0, -1.0, 0.5, 0.9])
    spoof = np.array([-1.0, -1.0, -1.0, -1.0])
    pmiss, pfa, _ = curva_det(bona, spoof)
    # limiares distintos: -1.0, 0.5, 0.9, +inf
    assert pmiss.tolist() == [0.0, 0.5, 0.75, 1.0]
    assert pfa.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_sentido_do_score():
    """O .npz guarda p(spoof); o t-DCF quer maior = bonafide."""
    p = np.array([0.1, 0.9])
    s = score_bonafide(p)
    assert s[0] > s[1]


def test_le_asv_com_3_e_4_colunas(tmp_path):
    arq = tmp_path / "asv.txt"
    arq.write_text(
        "bonafide target 2.5\n"
        "bonafide nontarget -3.0\n"
        "A07 spoof 0.4\n"
        "LA_0001 bonafide target 1.5\n"
        "LA_0001 A08 spoof -0.2\n",
        encoding="utf-8",
    )
    asv = ler_scores_asv(arq)
    assert asv["target"].tolist() == [2.5, 1.5]
    assert asv["nontarget"].tolist() == [-3.0]
    assert asv["spoof"].tolist() == [0.4, -0.2]


def test_asv_sem_uma_das_chaves_falha(tmp_path):
    arq = tmp_path / "asv.txt"
    arq.write_text("bonafide target 2.5\nA07 spoof 0.4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nontarget"):
        ler_scores_asv(arq)


def test_postos_medios_empate():
    r = postos_medios(np.array([3.0, 1.0, 1.0, 2.0]))
    assert r.tolist() == pytest.approx([1.0, 1 / 6, 1 / 6, 2 / 3])


def test_exporta_formato_oficial(tmp_path):
    arq = tmp_path / "cm.txt"
    exportar_formato_oficial(arq, ["LA_E_1", "LA_E_2"], ["-", "A07"], [0, 1],
                             score_bonafide(np.array([0.2, 0.8])))
    linhas = [l.split() for l in arq.read_text(encoding="utf-8").splitlines()]
    assert linhas[0][:3] == ["LA_E_1", "-", "bonafide"]
    assert linhas[1][:3] == ["LA_E_2", "A07", "spoof"]
    assert float(linhas[0][3]) > float(linhas[1][3])


def test_linha_de_comando_ponta_a_ponta(tmp_path, monkeypatch, capsys):
    """Dois .npz no formato do evaluate.py + arquivo ASV → tabela com a fusão."""
    import sys

    from scripts import tdcf

    rng = np.random.default_rng(5)
    labels = np.array([0] * 100 + [1] * 400)
    ids = np.array([f"LA_E_{i}" for i in range(labels.size)])
    systems = np.array(["-"] * 100 + ["A07"] * 400)
    for nome, desvio in [("m1", 1.0), ("m2", 2.0)]:
        p = 1 / (1 + np.exp(-(labels * 2 - 1) * 2 + rng.normal(0, desvio, labels.size)))
        np.savez_compressed(tmp_path / f"{nome}_eval_scores.npz", version=1, ids=ids,
                            labels=labels, scores=p, systems=systems,
                            fingerprint="x", partition="eval")
    asv = tmp_path / "asv.txt"
    asv.write_text("".join(
        [f"bonafide target {v}\n" for v in rng.normal(3, 1, 50)]
        + [f"bonafide nontarget {v}\n" for v in rng.normal(-3, 1, 50)]
        + [f"A07 spoof {v}\n" for v in rng.normal(1, 1, 50)]), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "tdcf.py", "--asv-scores", str(asv), "--fundir", "--exportar", str(tmp_path / "of"),
        "--scores", str(tmp_path / "m1_eval_scores.npz"), str(tmp_path / "m2_eval_scores.npz")])
    tdcf.main()
    saida = capsys.readouterr().out
    assert "m1 " in saida and "m2 " in saida and "(postos)" in saida
    assert (tmp_path / "of" / "m1_cm_scores_oficial.txt").exists()
