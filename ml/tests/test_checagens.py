"""Checagens depois de gravar (src/capture/checagens.py), com sinais sintéticos.

"Fala" aqui é ruído branco: tem energia em todas as faixas, como a fala tem
acima de 4 kHz nas fricativas.
"""

import numpy as np
from scipy.signal import butter, sosfilt

from src.capture.checagens import checar, energia_por_faixa

SR = 48000


def _gravacao(fala_s=5.0, silencio_s=1.0, ganho=0.1, ruido=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    n_sil, n_fala = int(silencio_s * SR), int(fala_s * SR)
    wav = rng.standard_normal(2 * n_sil + n_fala) * ruido
    wav[n_sil:n_sil + n_fala] += rng.standard_normal(n_fala) * ganho
    return wav.astype(np.float32)


def _por_nome(checagens):
    return {c.nome: c for c in checagens}


def test_gravacao_boa_passa_nas_quatro():
    r = _por_nome(checar(_gravacao(), SR))
    assert list(r) == ["saturação", "fala", "banda", "piso de ruído"]
    assert all(c.ok for c in r.values()), r
    assert r["fala"].valor == "5.0 s"


def test_ganho_alto_satura():
    r = _por_nome(checar(np.clip(_gravacao(ganho=2.0), -1, 1), SR))
    assert not r["saturação"].ok and "baixe o ganho" in r["saturação"].motivo


def test_pico_isolado_no_limite_nao_e_saturacao():
    wav = _gravacao()
    wav[SR * 2] = 1.0
    assert _por_nome(checar(wav, SR))["saturação"].ok


def test_frase_curta_nao_ocupa_uma_janela():
    r = _por_nome(checar(_gravacao(fala_s=2.5), SR))
    assert not r["fala"].ok and "janela" in r["fala"].motivo


def test_frase_cortada_no_inicio():
    r = _por_nome(checar(_gravacao(fala_s=6.0, silencio_s=0.1), SR))
    assert not r["fala"].ok and "encosta" in r["fala"].motivo


def test_banda_estreita():
    sos = butter(10, 3400, fs=SR, output="sos")
    r = _por_nome(checar(sosfilt(sos, _gravacao()).astype(np.float32), SR))
    assert not r["banda"].ok and r["fala"].ok


def test_ruido_nas_pausas_acima_do_corte_de_silencio():
    r = _por_nome(checar(_gravacao(ruido=0.005), SR))
    assert not r["piso de ruído"].ok and "corte de silêncio" in r["piso de ruído"].motivo
    assert _por_nome(checar(_gravacao(ruido=0.0005), SR))["piso de ruído"].ok


def test_silencio_total():
    r = checar(np.zeros(SR * 3, dtype=np.float32), SR)
    assert [c.ok for c in r] == [True, False, False, False]


def test_energia_por_faixa_acha_o_tom_em_7_8_khz():
    t = np.arange(SR * 2) / SR
    e = energia_por_faixa(np.sin(2 * np.pi * 7800 * t), SR)
    assert list(e) == ["0–4 kHz", "4–7,6 kHz", "7,6–8 kHz", "8–24 kHz"]
    assert e["7,6–8 kHz"] > 0.95
    assert abs(sum(e.values()) - 1) < 1e-6
    assert list(energia_por_faixa(np.zeros(16000), 16000)) == ["0–4 kHz", "4–7,6 kHz", "7,6–8 kHz"]


def test_ruido_a_20_db_mede_a_fala_e_nao_a_gravacao_inteira():
    """O microfone embutido na sala (24/10/2026): ruído 16–21 dB abaixo da fala.
    Pelo corte de 30 dB do pico, a gravação inteira contava como fala e encostava
    nas bordas; acima do ruído, a fala tem a duração certa e as bordas ficam livres."""
    wav = _gravacao(fala_s=6.0, silencio_s=1.5, ruido=0.01)
    r = _por_nome(checar(wav, SR))
    assert r["fala"].ok and abs(float(r["fala"].valor.split()[0]) - 6.0) < 0.3
    assert not r["piso de ruído"].ok and "corte de silêncio" in r["piso de ruído"].motivo


def test_piso_minimo_do_teste_com_a_voz_propria():
    wav = _gravacao(fala_s=6.0, silencio_s=1.5, ruido=0.01)
    r = _por_nome(checar(wav, SR, piso_minimo=15))
    assert r["piso de ruído"].ok
    ruim = _por_nome(checar(_gravacao(fala_s=6.0, silencio_s=1.5, ruido=0.05), SR, piso_minimo=15))
    assert not ruim["piso de ruído"].ok
    assert ruim["piso de ruído"].motivo == "ruído a menos de 15 dB da fala"


def test_fala_que_vai_ate_o_fim_da_gravacao_com_ruido():
    """As frases de 10 s terminavam coladas no fim: a checagem precisa acusar."""
    rng = np.random.default_rng(3)
    wav = (rng.standard_normal(SR * 8) * 0.01).astype(np.float32)
    wav[int(1.5 * SR):] += (rng.standard_normal(SR * 8 - int(1.5 * SR)) * 0.1).astype(np.float32)
    r = _por_nome(checar(wav, SR, piso_minimo=15))
    assert not r["fala"].ok and "encosta" in r["fala"].motivo
