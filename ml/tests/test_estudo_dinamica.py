"""Parte 8: dinâmica da voz no tempo (scripts/estudo_dinamica.py), com sinais sintéticos."""

import json

import numpy as np
import soundfile as sf

from scripts.estudo_dinamica import SALTO, SR, criterio, dinamica, extrair, main, sondar


def _vogal(f0_por_quadro, seed=0):
    """Sinal harmônico seguindo um contorno de F0 dado a cada 10 ms."""
    f0 = np.repeat(f0_por_quadro, SALTO)
    fase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * fase) / k for k in range(1, 15))
    return (x / np.abs(x).max() * 0.5 + np.random.default_rng(seed).standard_normal(len(x)) * 1e-3)


def test_contorno_tremido_tem_mais_microvariacao():
    t = np.arange(200)
    suave = 120 + 20 * np.sin(2 * np.pi * t / 200)                       # 2 s, sobe e desce
    tremido = suave * 2 ** (np.random.default_rng(1).standard_normal(200) * 0.6 / 12)
    ms, mt = dinamica(_vogal(suave)), dinamica(_vogal(tremido))
    assert ms["vozeada_fracao"] > 0.8 and mt["vozeada_fracao"] > 0.8
    assert mt["f0_micro"] > 2 * ms["f0_micro"]
    assert ms["f0_suavidade"] > mt["f0_suavidade"]


def test_ruido_quase_nao_e_vozeado():
    ruido = np.random.default_rng(2).standard_normal(2 * SR) * 0.1
    assert dinamica(ruido)["vozeada_fracao"] < 0.3


def test_contas_com_contorno_dado():
    n = 150
    wav = _vogal(np.full(n, 100.0))[: (n - 1) * SALTO]
    m = dinamica(wav, f0=np.full(n, 100.0), vozeado=np.ones(n, bool), prob=np.ones(n))
    assert m["f0_desvio_st"] == 0 and m["f0_delta_abs"] == 0 and m["transicoes_por_s"] == 0
    assert m["seg_vozeado_media_s"] == n * SALTO / SR and m["prob_voz_incerta"] == 0
    meio = np.ones(n, bool)
    meio[50:60] = False
    m2 = dinamica(wav, f0=np.where(meio, 100.0, np.nan), vozeado=meio, prob=meio.astype(float))
    assert m2["transicoes_por_s"] == 2 / (n * SALTO / SR)


def _conjunto(n, seed, classes=("bonafide", "A01"), informativo=True):
    rng = np.random.default_rng(seed)
    rotulo = np.array([0] * n + [1] * n, dtype=np.int8)
    din = rng.standard_normal((2 * n, 3)).astype(np.float32)
    if informativo:
        din[n:, 0] += 2.0
    din[0, 1] = np.nan  # medida que faltou num arquivo
    return {"arquivo": np.array([f"f{i}" for i in range(2 * n)]), "classe": np.array([classes[0]] * n + [classes[1]] * n),
            "rotulo": rotulo, "lfcc": rng.standard_normal((2 * n, 4)).astype(np.float32),
            "dinamica": din, "nomes": np.array(["a", "b", "c"])}


def test_sondar_e_criterio():
    r = sondar(_conjunto(60, 1), _conjunto(40, 2, ("bonafide", "A09")),
               {"asv19_train": _conjunto(40, 3, ("bonafide", "A01"))})
    assert r["dinamica"]["auc_media_dev"] > 0.9 and r["lfcc"]["auc_media_dev"] < 0.75
    assert abs(r["dinamica"]["pesos"]["a"]) > abs(r["dinamica"]["pesos"]["c"])
    c = criterio(r, "asv19_train")
    assert c["dinamica"]["passa"] and c["lfcc+dinamica"]["passa"]


def test_extrair_e_sondar_pela_linha_de_comando(tmp_path, capsys):
    arquivos = []
    for i, f0 in enumerate((110.0, 180.0)):
        caminho = tmp_path / f"v{i}.wav"
        sf.write(caminho, _vogal(np.full(250, f0), seed=i).astype(np.float32), SR)
        arquivos.append(str(caminho))
    origem = tmp_path / "o.npz"
    np.savez(origem, arquivo=np.array(arquivos), classe=np.array(["bonafide", "A01"]),
             rotulo=np.array([0, 1], dtype=np.int8), lfcc=np.zeros((2, 120), np.float32))
    destino = tmp_path / "d.npz"
    extrair(origem, destino, processos=1)
    with np.load(destino) as d:
        assert d["dinamica"].shape[0] == 2 and "f0_micro" in list(d["nomes"])
        assert d["lfcc"].shape == (2, 120)
    np.savez(tmp_path / "t.npz", **_conjunto(60, 1))
    np.savez(tmp_path / "dv.npz", **_conjunto(40, 2, ("bonafide", "A09")))
    np.savez(tmp_path / "asv19_train.npz", **_conjunto(40, 3))
    saida = tmp_path / "s.json"
    assert main(["sondar", "--treino", str(tmp_path / "t.npz"), "--dev", str(tmp_path / "dv.npz"),
                 "--avaliar", str(tmp_path / "asv19_train.npz"), "--json", str(saida)]) == 0
    assert json.loads(saida.read_text(encoding="utf-8"))["criterio"]["dinamica"]["passa"]
    assert "Critério" in capsys.readouterr().out
