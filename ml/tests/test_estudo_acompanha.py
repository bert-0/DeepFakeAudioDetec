"""Parte 7: a linha acompanha a fala? (scripts/estudo_acompanha.py), com sinais sintéticos."""

import json

import numpy as np
import soundfile as sf

from scripts.estudo_acompanha import acompanha, main, veredito

SR = 24000


def _fala_com_pausas(seed, tom_na_fala=False, tom_constante=False, hz=7850.0):
    """Rajadas de "fala" (ruído forte) alternadas com pausas (ruído fraco)."""
    rng = np.random.default_rng(seed)
    t = np.arange(SR * 6) / SR
    fala = (np.sin(2 * np.pi * 1.0 * t) > 0).astype(float)    # 0,5 s de fala, 0,5 s de pausa
    x = rng.standard_normal(t.size) * (0.002 + 0.1 * fala)
    tom = np.sin(2 * np.pi * hz * t)
    if tom_na_fala:      # o gerador: o tom sobe e desce com o som
        x += 0.02 * fala * tom
    if tom_constante:    # a sala: o tom não muda
        x += 0.001 * tom
    return x


def test_tom_do_gerador_acompanha_a_fala():
    m = acompanha(_fala_com_pausas(0, tom_na_fala=True), SR, 7850)
    assert m["diferenca"] > 3


def test_tom_da_sala_fica_nas_pausas():
    m = acompanha(_fala_com_pausas(0, tom_constante=True), SR, 7850)
    assert m["diferenca"] < -3


def test_veredito():
    assert veredito({"n": 10, "positiva": 7, "negativa": 3}) == "acompanha a fala"
    assert veredito({"n": 10, "positiva": 2, "negativa": 8}) == "fica nas pausas"
    assert veredito({"n": 10, "positiva": 5, "negativa": 5}) == "indefinida"


def test_caminho_completo(tmp_path, capsys):
    for nome, kw in (("clone", {"tom_na_fala": True}), ("real", {"tom_constante": True, "hz": 655.0})):
        (tmp_path / nome).mkdir()
        for s in range(3):
            sf.write(tmp_path / nome / f"f{s}.wav", _fala_com_pausas(s, **kw).astype(np.float32), SR)
    saida = tmp_path / "a.json"
    assert main(["--grupo", f"clone={tmp_path / 'clone'}", "--grupo", f"real={tmp_path / 'real'}",
                 "--linha", "clone=7850", "--linha", "real=655", "--json", str(saida)]) == 0
    r = json.loads(saida.read_text(encoding="utf-8"))
    assert r["clone"]["7850"]["positiva"] == 3 and r["real"]["655"]["negativa"] == 3
    out = capsys.readouterr().out
    assert "acompanha a fala" in out and "fica nas pausas" in out
