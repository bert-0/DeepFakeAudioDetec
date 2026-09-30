"""Diferença de espectro entre sessões (scripts/comparar_espectro.py)."""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.comparar_espectro import FAIXAS, comparar  # noqa: E402
from scripts.simular_caminho import ida_e_volta  # noqa: E402

SR = 16000


def _sessao(pasta, nome, audios):
    (pasta / nome / "capturado").mkdir(parents=True)
    linhas = []
    for aid, wav in audios.items():
        sf.write(pasta / nome / "capturado" / f"{aid}.flac", wav, SR)
        linhas.append(f"LA_0099 {aid} - - bonafide")
    (pasta / nome / "protocolo_canal_real.txt").write_text("\n".join(linhas) + "\n",
                                                           encoding="utf-8")


def _fala_com_pausa(rng):
    fala = rng.standard_normal(SR).astype(np.float32) * 0.3
    return np.concatenate([fala, np.zeros(SR // 2, np.float32), fala])


def test_reamostragem_aparece_so_no_topo(tmp_path):
    rng = np.random.default_rng(0)
    a = {f"LA_E_{i}": _fala_com_pausa(rng) for i in range(3)}
    _sessao(tmp_path, "limpo", a)
    _sessao(tmp_path, "b", {k: ida_e_volta(v, SR) for k, v in a.items()})
    _, dif, _, _ = comparar(tmp_path, "limpo", "b")
    med = np.median(dif, axis=0)
    topo = FAIXAS.index((7800, 8000))
    assert med[topo] < -20
    assert np.all(np.abs(med[: FAIXAS.index((7000, 7500))]) < 1.0)


def test_ruido_somado_sobe_o_piso(tmp_path):
    rng = np.random.default_rng(1)
    a = {f"LA_E_{i}": _fala_com_pausa(rng) for i in range(3)}
    _sessao(tmp_path, "limpo", a)
    _sessao(tmp_path, "b", {k: v + rng.standard_normal(len(v)).astype(np.float32) * 1e-3
                            for k, v in a.items()})
    _, _, pa, pb = comparar(tmp_path, "limpo", "b")
    assert np.median(pb - pa) > 40
