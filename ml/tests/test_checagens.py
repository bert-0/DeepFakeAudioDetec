"""Checagens depois de gravar (src/capture/checagens.py), com sinais sintéticos.

"Fala" aqui é ruído branco: tem energia em todas as faixas, como a fala tem
acima de 4 kHz nas fricativas.
"""

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from src.capture.checagens import checar, checar_copias, energia_por_faixa

SR = 48000


def _gravacao(fala_s=5.0, silencio_s=1.0, ganho=0.1, ruido=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    n_sil, n_fala = int(silencio_s * SR), int(fala_s * SR)
    wav = rng.standard_normal(2 * n_sil + n_fala) * ruido
    wav[n_sil:n_sil + n_fala] += rng.standard_normal(n_fala) * ganho
    return wav.astype(np.float32)


def _por_nome(checagens):
    return {c.nome: c for c in checagens}


def test_gravacao_boa_passa_em_todas():
    r = _por_nome(checar(_gravacao(), SR))
    assert list(r) == ["saturação", "cópias", "fala", "banda", "piso de ruído"]
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
    rng = np.random.default_rng(4)
    wav = (rng.standard_normal(SR * 9) * 1e-4).astype(np.float32)
    wav[int(0.1 * SR):int(6.1 * SR)] += (rng.standard_normal(SR * 6) * 0.1).astype(np.float32)
    r = _por_nome(checar(wav, SR))
    assert not r["fala"].ok and "encosta" in r["fala"].motivo
    # Quase toda fala, sem silêncio para medir o ruído: o modelo não acha frase.
    assert not _por_nome(checar(_gravacao(fala_s=6.0, silencio_s=0.1), SR))["fala"].ok


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
    assert [c.ok for c in r] == [True, True, False, False, False]


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


def test_fim_da_frase_mais_baixo_ainda_e_fala():
    """06/11/2026, sala do trabalho: a leitura começou forte e terminou ~15 dB mais
    baixa, a uns 10 dB do ruído. A régua antiga contava só a parte forte (3,8 s) e
    pedia para regravar; o modelo usava a frase inteira. Agora a checagem vê o que o
    modelo vê."""
    rng = np.random.default_rng(5)
    wav = (rng.standard_normal(SR * 12) * 0.005).astype(np.float32)
    wav[int(1.5 * SR):int(4.5 * SR)] += (rng.standard_normal(SR * 3) * 0.1).astype(np.float32)
    wav[int(4.5 * SR):int(8.5 * SR)] += (rng.standard_normal(SR * 4) * 0.015).astype(np.float32)
    r = _por_nome(checar(wav, SR, piso_minimo=15))
    assert r["fala"].ok, r["fala"].motivo
    assert abs(float(r["fala"].valor.split()[0]) - 7.0) < 0.3


def test_duas_frases_pedem_regravar():
    """Uma pausa longa no meio, ou alguém falando depois da leitura, vira outra frase."""
    rng = np.random.default_rng(6)
    wav = (rng.standard_normal(SR * 12) * 1e-3).astype(np.float32)
    for a, b in ((1.5, 5.0), (7.0, 10.0)):
        wav[int(a * SR):int(b * SR)] += (rng.standard_normal(int((b - a) * SR)) * 0.1).astype(np.float32)
    r = _por_nome(checar(wav, SR, piso_minimo=15))
    assert not r["fala"].ok and "2 frases" in r["fala"].motivo and "em 2 frases" in r["fala"].valor
    # Clone tocado pelo ar: o XTTS faz pausas longas no meio, e regravar não muda o clone.
    assert _por_nome(checar(wav, SR, piso_minimo=15, uma_frase=False))["fala"].ok


def test_a_checagem_mede_as_frases_que_o_modelo_analisa():
    """Mesmo detector: as frases medidas são as que o ao vivo e o limiar usam."""
    from src.capture.frases import frases_de, medir_frases
    from src.preprocess.reamostragem import para_taxa
    from src.xlsr_final import Pontuacao

    class _Nulo:
        def pontuar_wav(self, wav, sample_rate=16000):
            return Pontuacao(logodds=0.0, duracao_s=len(wav) / sample_rate, silencio_s=0.0)

    rng = np.random.default_rng(7)
    wav = (rng.standard_normal(SR * 12) * 1e-3).astype(np.float32)
    for a, b in ((1.5, 4.0), (6.0, 9.5)):
        wav[int(a * SR):int(b * SR)] += (rng.standard_normal(int((b - a) * SR)) * 0.1).astype(np.float32)
    medidas = medir_frases(wav, SR)
    leituras = frases_de(_Nulo(), para_taxa(wav, SR, 16000).astype(np.float32), 16000)
    assert len(medidas) == len(leituras) == 2
    for (inicio, fim, fala), leitura in zip(medidas, leituras, strict=True):
        assert inicio - leitura.instante == pytest.approx(0.3, abs=0.03), "a margem antes da fala"
        assert fim > inicio and fala <= fim - inicio + 0.02


def test_pancada_do_teclado_na_borda_nao_e_fala():
    """A referência gravada em 24/10/2026: Enter e mão no notebook viram pancadas
    graves e curtas (~150 ms, metade da energia abaixo de 150 Hz) no microfone
    embutido, e a checagem acusava fala encostada no início sem ninguém falar."""
    wav = _gravacao(fala_s=6.0, silencio_s=1.5, ruido=0.01)
    t = np.arange(int(0.15 * SR)) / SR
    pancada = (0.08 * np.sin(2 * np.pi * 60 * t) * np.hanning(t.size)).astype(np.float32)
    wav[int(0.2 * SR):int(0.2 * SR) + t.size] += pancada
    wav[-int(0.4 * SR):-int(0.4 * SR) + t.size] += pancada
    r = _por_nome(checar(wav, SR, piso_minimo=15))
    assert r["fala"].ok, r["fala"].motivo
    assert abs(float(r["fala"].valor.split()[0]) - 6.0) < 0.3


def _com_copias(wav, a_cada=3, distancia=11, pacote=480):
    """Como o modo exclusivo (RESUMO_TCC §10.2.34): a cada `a_cada` trechos de 10 ms
    novos, insere a cópia do trecho `distancia` atrás."""
    blocos = list(wav[:len(wav) // pacote * pacote].reshape(-1, pacote))
    saida = []
    for i, b in enumerate(blocos):
        saida.append(b)
        if (i + 1) % a_cada == 0 and len(saida) > distancia:
            saida.append(saida[-distancia])
    return np.concatenate(saida)


def test_trechos_repetidos_pedem_regravar():
    r = _por_nome(checar(_com_copias(_gravacao()), SR))
    assert not r["cópias"].ok and "modo compartilhado" in r["cópias"].motivo
    assert r["cópias"].valor.startswith("230 de 930 "), "700 trechos novos, 230 cópias"
    assert r["fala"].ok, "o resto da gravação não acusa nada"


def test_uma_copia_so_ja_acusa():
    wav = _gravacao()
    wav[480 * 300:480 * 301] = wav[480 * 200:480 * 201]
    assert not _por_nome(checar(wav, SR))["cópias"].ok


def test_enchimento_de_um_degrau_de_16_bits_nao_e_copia():
    """O fim dos clones do Chatterbox: trechos iguais de no máximo 1 degrau de 16
    bits (RESUMO_TCC §10.2.34). Não é a captura repetindo áudio."""
    wav = _gravacao()
    rng = np.random.default_rng(9)
    enchimento = np.tile(rng.integers(-1, 2, 480) / 32768, 20).astype(np.float32)
    r = _por_nome(checar(np.concatenate([wav, enchimento]), SR))
    assert r["cópias"].ok, r["cópias"].valor


def test_copias_a_96_khz_em_trechos_de_10_ms():
    sr = 96000
    rng = np.random.default_rng(10)
    wav = (rng.standard_normal(sr * 7) * 0.1).astype(np.float32)
    assert checar_copias(wav, sr).ok
    assert not checar_copias(_com_copias(wav, pacote=960), sr).ok
