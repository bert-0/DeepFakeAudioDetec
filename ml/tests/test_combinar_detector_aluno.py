"""Aluno + detector (scripts/combinar_detector_aluno.py), com scores sintéticos."""

import numpy as np

from scripts.combinar_detector_aluno import combinar, criterio, juntar, medir_tudo


def _conjunto(nome, n, seed, ataques=("A01", "A02")):
    """O detector só vê o A01; o aluno só vê o A02: somados, veem os dois."""
    rng = np.random.default_rng(seed)
    classes = ["bonafide"] * n + [ataques[0]] * n + [ataques[1]] * n
    rotulo = np.array([0] * n + [1] * 2 * n, dtype=np.int8)
    det = rng.standard_normal(3 * n)
    alu = rng.standard_normal(3 * n)
    det[n:2 * n] += 3
    alu[2 * n:] += 3
    arquivos = [f"{nome}/f{i}.flac" for i in range(3 * n)]
    return arquivos, np.array(classes), rotulo, det, alu


def _entradas(seeds={"dev": 0, "asv19_train": 1, "ensaios": 2, "asv5_eval": 3}):
    npz, detector, aluno = {}, {"scores": {}}, {"sondas": {"nova": {"scores": {}, "arquivos": {}}}}
    nomes_aluno = {"dev": "dev", "asv19_train": "asv19_train", "ensaios": "ensaios", "asv5_eval": "asv5_eval_aluno"}
    for nome, s in seeds.items():
        arquivos, classes, rotulo, det, alu = _conjunto(nome, 40, s)
        npz[nome] = {"arquivo": np.array(arquivos), "classe": classes, "rotulo": rotulo}
        detector["scores"][nome] = det.tolist()
        aluno["sondas"]["nova"]["scores"][nomes_aluno[nome]] = alu[::-1].tolist()   # outra ordem
        aluno["sondas"]["nova"]["arquivos"][nomes_aluno[nome]] = arquivos[::-1]
    return aluno, detector, npz


def test_juntar_pelo_nome_do_arquivo():
    aluno, detector, npz = _entradas()
    conj = juntar(aluno, [detector], npz)
    assert set(conj) == {"dev", "asv19_train", "ensaios", "asv5_eval"}
    c = conj["dev"]
    assert len(c["arquivo"]) == 120 and np.allclose(c["detector"], detector["scores"]["dev"])
    assert np.allclose(c["aluno"], aluno["sondas"]["nova"]["scores"]["dev"][::-1])   # realinhado


def test_soma_ve_o_que_cada_um_ve_e_o_criterio():
    aluno, detector, npz = _entradas()
    conj = juntar(aluno, [detector], npz)
    scores, info = combinar(conj)
    m = medir_tudo(conj, scores)
    ev = m["asv5_eval"]
    assert ev["soma"]["auc"] > max(ev["detector"]["auc"], ev["aluno"]["auc"]) + 0.05
    assert ev["detector"]["classes"]["A01"]["auc"] > 0.9 and ev["detector"]["classes"]["A02"]["auc"] < 0.7
    assert all(w > 0 for w in info["pesos_logistica"])
    assert criterio(m)["passa"]
