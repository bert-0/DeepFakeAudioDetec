"""Efeitos do caminho do som (scripts/simular_caminho.py)."""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.simular_caminho import ida_e_volta, main, passa_baixa  # noqa: E402

SR = 16000


def _banda_db(original, processado, lo, hi):
    from scipy.signal import welch

    f, px = welch(original, SR, nperseg=2048)
    _, py = welch(processado[: len(original)], SR, nperseg=2048)
    m = (f >= lo) & (f < hi)
    return 10 * np.log10(py[m].mean() / px[m].mean())


def test_ida_e_volta_apaga_o_topo_da_banda():
    """Preserva até 7,3 kHz e apaga de 7,7 a 8 kHz."""
    x = np.random.default_rng(0).standard_normal(SR * 10).astype(np.float32) * 0.1
    y = ida_e_volta(x, SR)
    assert abs(len(y) - len(x)) <= 2
    assert abs(_banda_db(x, y, 6000, 7300)) < 0.5
    assert _banda_db(x, y, 7700, 7900) < -20


def test_passa_baixa_corta_onde_pedido_sem_atraso():
    x = np.random.default_rng(1).standard_normal(SR * 10).astype(np.float32) * 0.1
    y = passa_baixa(x, SR, 6000)
    assert abs(_banda_db(x, y, 3000, 5000)) < 0.5
    assert _banda_db(x, y, 7000, 8000) < -30
    # Fase zero: o sinal filtrado continua alinhado ao original.
    assert np.argmax(np.correlate(y[1000:5000], x[1000:5000], "full")) == 3999


def test_cli_gera_sessao_com_o_efeito(tmp_path, capsys):
    origem = tmp_path / "canal" / "limpo"
    (origem / "capturado").mkdir(parents=True)
    x = np.random.default_rng(2).standard_normal(SR * 2).astype(np.float32) * 0.1
    sf.write(origem / "capturado" / "LA_E_0.flac", x, SR)
    (origem / "protocolo_canal_real.txt").write_text("LA_0099 LA_E_0 - - bonafide\n",
                                                     encoding="utf-8")
    assert main(["--pasta", str(tmp_path / "canal"), "--destino", "pb",
                 "--efeito", "passa_baixa", "--hz", "5000"]) == 0
    y, _ = sf.read(tmp_path / "canal" / "pb" / "capturado" / "LA_E_0.flac", dtype="float32")
    assert _banda_db(x, y, 6000, 8000) < -30
    assert "evaluate.py" in capsys.readouterr().out


def test_efeitos_encadeados_na_ordem(tmp_path):
    """reamostragem + piso: o topo some E o piso sobe — o controle medido."""
    from scripts.comparar_espectro import comparar

    origem = tmp_path / "canal" / "limpo"
    (origem / "capturado").mkdir(parents=True)
    rng = np.random.default_rng(3)
    fala = rng.standard_normal(SR).astype(np.float32) * 0.3
    x = np.concatenate([fala, np.zeros(SR // 2, np.float32), fala])
    sf.write(origem / "capturado" / "LA_E_0.flac", x, SR)
    (origem / "protocolo_canal_real.txt").write_text("LA_0099 LA_E_0 - - bonafide\n",
                                                     encoding="utf-8")
    assert main(["--pasta", str(tmp_path / "canal"), "--destino", "ambos",
                 "--efeito", "reamostragem", "piso", "--db", "-31"]) == 0
    _, dif, pa, pb = comparar(tmp_path / "canal", "limpo", "ambos")
    assert np.median(pb - pa) > 30
