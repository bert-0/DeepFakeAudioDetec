"""Teste do microfone (scripts/testar_microfone.py) com gravações sintéticas.

Cada cenário imita um caminho: igual nos dois modos, supressão de ruído no
hardware, aprimoramentos do Windows, ganho automático e conversão de taxa.
"""

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from scripts.testar_microfone import comparar, gravar, main, medir, salvar
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


def test_supressao_do_windows_afasta_o_ruido_da_fala():
    r = comparar(medir(_cena(), SR), medir(_cena(ruido=0.001, seed=1), SR))
    assert any("supressão de ruído no Windows" in x for x in r), r


def test_ganho_fixo_sobe_fala_e_ruido_juntos():
    """O caso medido no microfone do notebook: +17 a +20 dB em tudo."""
    r = comparar(medir(_cena(ruido=0.0003, fala=0.01), SR), medir(_cena(seed=1), SR))
    assert any("ganho fixo de +20 dB" in x for x in r), r
    assert not any("AGC" in x or "espectro" in x for x in r), r


def test_ganho_automatico_muda_so_o_nivel_da_fala():
    r = comparar(medir(_cena(), SR), medir(_cena(fala=0.03, seed=1), SR))
    assert any("ganho automático (AGC)" in x for x in r), r


def test_supressao_que_tira_os_graves_do_ruido():
    sos = butter(8, 1000, btype="highpass", fs=SR, output="sos")
    compartilhado = _cena(seed=1)
    compartilhado[10 * SR:] = sosfilt(sos, compartilhado[10 * SR:])
    r = comparar(medir(_cena(), SR), medir(compartilhado, SR))
    assert any("formato do espectro do ruído" in x and "50–250 Hz" in x for x in r), r


def test_supressao_adaptativa_so_no_compartilhado():
    sos = butter(8, 1000, btype="highpass", fs=SR, output="sos")
    compartilhado = _cena(seed=1)
    compartilhado[9 * SR:] = sosfilt(sos, compartilhado[9 * SR:])
    r = comparar(medir(_cena(), SR), medir(compartilhado, SR))
    assert any(x.startswith("compartilhado: o ruído perde") and "adaptativa do Windows" in x
               for x in r), r


def test_mesma_faixa_some_nos_dois_modos():
    sos = butter(8, 1000, btype="highpass", fs=SR, output="sos")
    cenas = []
    for seed in (0, 1):
        x = _cena(seed=seed)
        x[9 * SR:] = sosfilt(sos, x[9 * SR:])
        cenas.append(medir(x, SR))
    r = comparar(*cenas)
    assert any(x.startswith("nos dois modos uma faixa do ruído some") for x in r), r


def test_conversao_de_taxa_perde_a_banda_alta():
    sos = butter(12, 7000, fs=SR, output="sos")
    compartilhado = sosfilt(sos, _cena(seed=1)).astype(np.float32)
    r = comparar(medir(_cena(), SR), medir(compartilhado, SR))
    assert any("7,6–8 kHz" in x for x in r), r


def test_sem_ruido_de_fundo_avisa_que_o_teste_nao_decide():
    r = comparar(medir(_cena(ruido=1e-5), SR), medir(_cena(ruido=1e-5, seed=1), SR))
    assert any("confira se ele estava tocando" in x for x in r), r


def test_grava_a_48_khz_sem_o_fir_e_sem_o_clique_do_inicio(monkeypatch):
    sinal = np.random.default_rng(4).standard_normal(SR * 2) * 0.1
    _sounddevice_falso(monkeypatch, sinal)
    wav, modo, nome, taxa = gravar(1.0, True, None)
    assert (modo, taxa) == ("exclusivo", SR) and len(wav) == SR
    np.testing.assert_allclose(wav, sinal[SR // 2:SR // 2 + SR], atol=1e-6)


def test_grava_na_taxa_do_dispositivo(monkeypatch):
    sinal = np.random.default_rng(6).standard_normal(96000 * 2) * 0.1
    _sounddevice_falso(monkeypatch, sinal, taxas=(96000,))
    wav, _, _, taxa = gravar(1.0, True, None)
    assert taxa == 96000 and len(wav) == 96000


def test_salva_em_24_bits(tmp_path):
    destino = tmp_path / "x.wav"
    salvar(destino, np.random.default_rng(5).standard_normal(SR).astype(np.float32) * 0.01, SR)
    assert sf.info(destino).subtype == "PCM_24"


def test_analisa_gravacoes_ja_feitas(tmp_path, capsys):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(a, _cena(), SR)
    sf.write(b, _cena(ruido=0.001, seed=1), SR)
    assert main(["--analisar", str(a), str(b)]) == 0
    saida = capsys.readouterr().out
    assert "ruído no fim" in saida and "Nível por meio segundo" in saida
    assert "supressão de ruído no Windows" in saida
