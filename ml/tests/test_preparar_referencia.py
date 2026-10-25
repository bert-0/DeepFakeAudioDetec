"""Preparo da referência para a clonagem (scripts/preparar_referencia.py).

O mesmo que foi feito à mão na referência dos ensaios 1 e 2: trecho com fala,
pico a −3 dBFS, 16 bits.
"""

import numpy as np
import soundfile as sf

from scripts.preparar_referencia import main, preparar

SR = 48000


def _gravacao(antes=2.0, fala=25.0, depois=3.0, seed=0):
    """Ruído baixo o tempo todo e "fala" no meio, no nível do microfone embutido."""
    rng = np.random.default_rng(seed)
    n = int((antes + fala + depois) * SR)
    wav = rng.standard_normal(n) * 0.003
    i, f = int(antes * SR), int((antes + fala) * SR)
    wav[i:f] += rng.standard_normal(f - i) * 0.02
    return wav.astype(np.float32)


def test_corta_o_trecho_com_fala_e_leva_o_pico_a_menos_3_dbfs():
    pronto, ganho = preparar(_gravacao(), SR)
    assert abs(len(pronto) / SR - 26.0) < 0.2, "25 s de fala + 0,5 s de folga em cada ponta"
    assert abs(20 * np.log10(np.abs(pronto).max()) + 3.0) < 0.01
    assert ganho > 10, "a gravação do microfone embutido chega baixa"


def test_grava_em_16_bits_ao_lado_da_entrada(tmp_path, capsys):
    entrada = tmp_path / "f01.wav"
    sf.write(entrada, _gravacao(), SR, subtype="PCM_24")
    assert main([str(entrada)]) == 0
    info = sf.info(tmp_path / "referencia_envio.wav")
    assert info.subtype == "PCM_16" and info.samplerate == SR
    assert "ganho de" in capsys.readouterr().out


def test_avisa_referencia_curta(tmp_path, capsys):
    entrada = tmp_path / "curta.wav"
    sf.write(entrada, _gravacao(fala=8.0), SR)
    assert main([str(entrada), "--saida", str(tmp_path / "r.wav")]) == 0
    assert "menos de 20 s de fala" in capsys.readouterr().out


def test_sem_fala_ou_sem_arquivo(tmp_path, capsys):
    vazio = tmp_path / "vazio.wav"
    sf.write(vazio, np.zeros(SR * 3, np.float32), SR)
    assert main([str(vazio)]) == 1
    assert main([str(tmp_path / "nada.wav")]) == 1
    saida = capsys.readouterr().out
    assert "nenhuma fala" in saida and "não encontrada" in saida
