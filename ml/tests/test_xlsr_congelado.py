"""XLS-R congelado com pesos por camada (scripts/xlsr_congelado.py), com camadas sintéticas."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _npz(caminho, n=120, seed=0, camada_sinal=3, so_reais=False, classe_real="bonafide"):
    """A diferença entre as classes mora numa camada só: a cabeça deve dar peso a ela. O LFCC
    é constante: ruído puro em 120 dimensões, com tão poucas amostras, só seria decorado."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, np.int8) if so_reais else (np.arange(n) % 2).astype(np.int8)
    camadas = rng.standard_normal((n, 25, 16)).astype(np.float16)
    camadas[:, camada_sinal, :] += (y[:, None] * 1.5).astype(np.float16)
    np.savez(caminho, arquivo=np.array([f"a{i}.wav" for i in range(n)]),
             classe=np.array([classe_real if v == 0 else "A01" for v in y]), rotulo=y,
             cnn=np.zeros((n, 8), np.float16), camadas=camadas, lfcc=np.zeros((n, 120), np.float32),
             medidas=np.zeros((n, 2), np.float32), nomes_medidas=np.array(["a", "b"]))
    return caminho


def test_pesos_por_dominio_igualam_dominios_e_classes():
    from scripts.xlsr_congelado import pesos_por_dominio

    a = {"rotulo": np.array([0, 0, 0, 1])}
    b = {"rotulo": np.array([0, 0])}
    wa, wb = pesos_por_dominio([a, b])
    assert wa.sum() == pytest.approx(1) and wb.sum() == pytest.approx(1)
    assert wa[3] == pytest.approx(0.5) and wa[:3].sum() == pytest.approx(0.5)


def test_treinar_acha_a_camada_do_sinal_e_avaliar_calibra(tmp_path, monkeypatch):
    from scripts.xlsr_congelado import main

    # 240 amostras em lotes de 256 dariam um passo por época; nos dados reais são ~80.
    monkeypatch.setattr("scripts.xlsr_congelado.LOTE", 16)

    treino_a = _npz(tmp_path / "a.npz", seed=0)
    treino_cv = _npz(tmp_path / "cv.npz", seed=1, so_reais=True)
    dev = _npz(tmp_path / "dev.npz", seed=2)
    ensaio = _npz(tmp_path / "ens.npz", n=40, seed=3, classe_real="L01_real")
    cal = _npz(tmp_path / "cal.npz", n=10, seed=4, so_reais=True, classe_real="L02_real")
    ck = tmp_path / "m.pt"
    assert main(["treinar", "--treino", f"asv5={treino_a}", "--treino", f"cv={treino_cv}", "--dev", f"dev={dev}",
                 "--saida", str(ck), "--dispositivo", "cpu", "--epocas", "40"]) == 0
    hist = json.loads(ck.with_suffix(".json").read_text())["historico"]
    assert hist[-1]["camadas_mais_pesadas"][0] == 3, "a soma ponderada aprende onde está o sinal"
    saida = tmp_path / "r.json"
    assert main(["avaliar", "--modelo", str(ck), "--dev", str(dev), "--regua", f"ensaio5={ensaio}",
                 "--calibrar", str(cal), "--dispositivo", "cpu", "--json", str(saida)]) == 0
    r = json.loads(saida.read_text(encoding="utf-8"))
    assert r["reguas"]["ensaio5"]["auc"] > 0.9
    assert set(r["calibrado"]) == {"limiar", "L01_real", "A01"} and len(r["pesos_camadas"]) == 25


def test_veredito_calibrado_pela_maior_voz_real():
    from scripts.xlsr_congelado import veredito_calibrado

    r = veredito_calibrado(np.array([0.1, 0.5, 0.9, 2.0]), np.array(["real", "real", "x", "x"]),
                           np.array([0.2, 0.6]))
    assert r == {"limiar": 0.6, "real": 2, "x": 2}
