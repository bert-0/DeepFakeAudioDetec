"""Estudo do relógio do gerador (scripts/estudo_relogio.py), com sinais sintéticos."""

import json

import numpy as np
import soundfile as sf

from scripts.estudo_relogio import SR, main, modulacao, pente, picos


def _ruido_modulado(hz=None, segundos=6.0, profundidade=0.5, seed=0):
    """Ruído (como fala sem relógio) e, se pedido, com a energia oscilando em `hz`."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(segundos * SR)) / SR
    x = rng.standard_normal(t.size) * 0.1
    if hz:
        x *= 1 + profundidade * np.sin(2 * np.pi * hz * t)
    return x


def test_acha_o_relogio_de_25_hz():
    """Como os 25 passos por segundo do Chatterbox."""
    x = _ruido_modulado(25)
    freqs, esp, n = modulacao(x, [(0, len(x))])
    assert n == 5
    melhor = picos(freqs, esp)[0]
    assert abs(melhor[0] - 25) <= 1 and melhor[1] > 10


def test_sem_relogio_nao_ha_pico_forte():
    x = _ruido_modulado(None)
    freqs, esp, _ = modulacao(x, [(0, len(x))])
    assert all(e < 10 for _, e in picos(freqs, esp))


def test_pente_de_estalos_a_cada_110_ms():
    """Estalos regulares (como os do microfone, §10.2.16) viram um pente de ~9 Hz."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(SR * 12) * 0.05
    x[::int(0.110 * SR)] += 1.0
    freqs, esp, _ = modulacao(x, [(0, len(x))])
    p = pente(freqs, esp)
    assert p["espacamento_hz"] == 9 and p["autocorrelacao"] > 0.5
    limpo = rng.standard_normal(SR * 12) * 0.05
    assert pente(*modulacao(limpo, [(0, len(limpo))])[:2])["autocorrelacao"] < 0.5


def test_trecho_curto_nao_entra():
    x = _ruido_modulado(25, segundos=0.8)
    _, _, n = modulacao(x, [(0, len(x))])
    assert n == 0


def test_caminho_completo(tmp_path, capsys):
    for nome, hz in (("real", None), ("clone", 25)):
        (tmp_path / nome).mkdir()
        sf.write(tmp_path / nome / "f01.wav", _ruido_modulado(hz, seed=1).astype(np.float32), SR)
    saida = tmp_path / "r.json"
    assert main(["--grupo", f"real={tmp_path / 'real'}", "--grupo", f"clone={tmp_path / 'clone'}",
                 "--json", str(saida), "--grafico", str(tmp_path / "g.png")]) == 0
    r = json.loads(saida.read_text(encoding="utf-8"))
    assert abs(r["clone"]["picos"][0][0] - 25) <= 1
    assert (tmp_path / "g.png").is_file()
    assert "picos acima do contorno" in capsys.readouterr().out
