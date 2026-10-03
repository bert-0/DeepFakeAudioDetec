"""Teste do microfone (scripts/testar_microfone.py) com gravações sintéticas.

Cada cenário imita um caminho: igual nos dois modos, supressão de ruído no
hardware, aprimoramentos do Windows, ganho automático e conversão de taxa.
"""

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from scripts.testar_microfone import comparar, gravar, main, medir
from test_microfone_wasapi import _sounddevice_falso

SR = 48000


def _cena(ruido=0.003, fala=0.1, segundos=12.0, ruido_depois=None, seed=0):
    """Ruído constante o tempo todo e "fala" de 3 s a 9 s. `ruido_depois`: o nível
    do ruído a partir de 1 s, como uma supressão que converge."""
    rng = np.random.default_rng(seed)
    n = int(segundos * SR)
    nivel = np.full(n, ruido)
    if ruido_depois is not None:
        nivel[SR:] = ruido_depois
    wav = rng.standard_normal(n) * nivel
    wav[3 * SR:9 * SR] += rng.standard_normal(6 * SR) * fala
    return wav.astype(np.float32)


def test_mesmo_audio_nos_dois_modos():
    r = comparar(medir(_cena(), SR), medir(_cena(seed=1), SR))
    assert len(r) == 1 and r[0].startswith("nenhuma diferença")


def test_supressao_no_hardware_aparece_no_exclusivo():
    r = comparar(medir(_cena(ruido_depois=0.0003), SR), medir(_cena(seed=1), SR))
    assert any(x.startswith("exclusivo: o ruído constante caiu 20 dB") for x in r), r


def test_aprimoramentos_do_windows_tiram_ruido_no_compartilhado():
    r = comparar(medir(_cena(), SR), medir(_cena(ruido=0.001, seed=1), SR))
    assert any("aprimoramentos do Windows" in x for x in r), r


def test_ganho_automatico_muda_o_nivel_da_fala():
    r = comparar(medir(_cena(), SR), medir(_cena(fala=0.03, seed=1), SR))
    assert any("ganho ou AGC" in x for x in r), r


def test_conversao_de_taxa_perde_a_banda_alta():
    sos = butter(12, 7000, fs=SR, output="sos")
    compartilhado = sosfilt(sos, _cena(seed=1)).astype(np.float32)
    r = comparar(medir(_cena(), SR), medir(compartilhado, SR))
    assert any("7,6–8 kHz" in x for x in r), r


def test_sem_ruido_de_fundo_avisa_que_o_teste_nao_decide():
    r = comparar(medir(_cena(ruido=1e-5), SR), medir(_cena(ruido=1e-5, seed=1), SR))
    assert any("confira se o ruído constante estava tocando" in x for x in r), r


def test_grava_a_48_khz_sem_passar_pelo_fir(monkeypatch):
    sinal = np.random.default_rng(4).standard_normal(SR * 2) * 0.1
    _sounddevice_falso(monkeypatch, sinal)
    wav, modo, nome = gravar(1.0, True, None)
    assert modo == "exclusivo" and len(wav) == SR
    np.testing.assert_allclose(wav, sinal[:SR], atol=1e-6)


def test_analisa_gravacoes_ja_feitas(tmp_path, capsys):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(a, _cena(), SR)
    sf.write(b, _cena(ruido=0.001, seed=1), SR)
    assert main(["--analisar", str(a), str(b)]) == 0
    saida = capsys.readouterr().out
    assert "ruído no fim" in saida and "aprimoramentos do Windows" in saida
