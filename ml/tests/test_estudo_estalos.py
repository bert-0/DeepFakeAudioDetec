"""Estalos curtos (scripts/estudo_estalos.py), com sinais sintéticos."""

import numpy as np
import soundfile as sf

from scripts.estudo_estalos import SR, estalos, main


def _com_estalos(instantes_s, segundos=3.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(SR * segundos)) * 0.01
    for t in instantes_s:
        i = int(t * SR)
        x[i:i + 8] += rng.standard_normal(8) * 0.5        # ~0,5 ms de banda larga
    return x


def test_acha_os_estalos_e_ignora_o_ruido():
    t = estalos(_com_estalos([0.5, 0.61, 1.7, 2.2]))
    assert len(t) == 4 and np.allclose(t, [0.5, 0.61, 1.7, 2.2], atol=0.003)
    assert len(estalos(_com_estalos([]))) == 0
    assert len(estalos(_com_estalos([], segundos=60.0, seed=7))) == 0   # nenhum falso em 1 minuto de ruído


def _fala_e_pausas(x, seed=0):
    """'Fala' forte de 0,3 s a cada segundo (no começo de cada segundo); o resto é pausa."""
    rng = np.random.default_rng(seed)
    t = np.arange(len(x)) / SR
    return x + rng.standard_normal(len(x)) * np.where((t % 1.0) < 0.3, 0.3, 0.0)


def test_relatorio_por_grupo(tmp_path):
    import json

    (tmp_path / "m").mkdir()
    sf.write(tmp_path / "m" / "f01.wav", _fala_e_pausas(_com_estalos([], segundos=3.0)).astype(np.float32), SR)
    saida = tmp_path / "e.json"
    assert main(["--grupo", f"mic={tmp_path / 'm'}", "--json", str(saida)]) == 0
    r = json.loads(saida.read_text(encoding="utf-8"))["mic"]["resumo"]
    assert set(r) == {"arquivos", "pausa_s", "pente_9hz", "fundo_pausa_dbfs", "saltos_por_s"}


def test_portao_de_ruido_derruba_o_fundo_e_cria_saltos(tmp_path):
    from scripts.estudo_estalos import medir_arquivo

    rng = np.random.default_rng(2)
    t = np.arange(SR * 4) / SR
    fala = (t % 1.0) < 0.3
    natural = rng.standard_normal(len(t)) * np.where(fala, 0.3, 0.003)
    portao = natural * np.where(fala, 1.0, 0.01)            # o portão corta 40 dB nas pausas
    sf.write(tmp_path / "n.wav", natural.astype(np.float32), SR)
    sf.write(tmp_path / "p.wav", portao.astype(np.float32), SR)
    n, p = medir_arquivo(tmp_path / "n.wav"), medir_arquivo(tmp_path / "p.wav")
    assert p["fundo_pausa_dbfs"] < n["fundo_pausa_dbfs"] - 30
    assert p["saltos_por_s"] > n["saltos_por_s"]
