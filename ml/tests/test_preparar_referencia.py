"""Preparo da referência para a clonagem (scripts/preparar_referencia.py).

O mesmo que foi feito à mão na referência dos ensaios 1 e 2: trecho com fala,
pico a −3 dBFS, 16 bits.
"""

import numpy as np
import soundfile as sf

from scripts.preparar_referencia import juntar_frases, main, preparar

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


def _frase(inicio=1.5, fala=4.0, total=12.0, seed=0):
    """Uma frase do gravador: 12 s, a fala no começo e o resto só ruído da sala."""
    rng = np.random.default_rng(seed)
    wav = rng.standard_normal(int(total * SR)) * 1e-3
    i, f = int(inicio * SR), int((inicio + fala) * SR)
    wav[i:f] += rng.standard_normal(f - i) * 0.03
    return wav.astype(np.float32)


def test_varias_frases_juntam_so_a_fala_de_cada_uma():
    """As frases da calibração têm uns 5 s de ruído da sala depois da fala."""
    frases = [_frase(seed=1), _frase(inicio=2.0, fala=5.0, seed=2), _frase(fala=4.5, seed=3)]
    pronto, ganho = juntar_frases(frases, SR)
    esperado = (4.0 + 1.0) + (5.0 + 1.0) + (4.5 + 1.0)
    assert abs(len(pronto) / SR - esperado) < 0.6, "fala + 0,5 s de folga em cada ponta, por frase"
    assert abs(20 * np.log10(np.abs(pronto).max()) + 3.0) < 0.01
    assert ganho > 10


def test_varias_frases_pedem_saida_e_a_mesma_taxa(tmp_path, capsys):
    a, b = tmp_path / "f01.wav", tmp_path / "f02.wav"
    sf.write(a, _frase(seed=1), SR)
    sf.write(b, _frase(seed=2), SR)
    assert main([str(a), str(b)]) == 1, "ao lado das frases, viraria mais uma frase da pasta"
    assert not (tmp_path / "referencia_envio.wav").exists()
    c = tmp_path / "f03.wav"
    sf.write(c, _frase(seed=3)[::3], SR // 3)
    assert main([str(a), str(c), "--saida", str(tmp_path / "r.wav")]) == 1
    saida = capsys.readouterr().out
    assert "--saida" in saida and "taxas diferentes" in saida


def test_varias_frases_gravam_onde_pedido(tmp_path, capsys):
    frases = []
    for i in range(1, 4):
        frases.append(tmp_path / f"f{i:02d}.wav")
        sf.write(frases[-1], _frase(seed=i), SR, subtype="PCM_24")
    saida = tmp_path / "referencias" / "L03.wav"
    assert main([*map(str, frases), "--saida", str(saida)]) == 0
    info = sf.info(saida)
    assert info.subtype == "PCM_16" and info.samplerate == SR
    assert abs(info.duration - 15.0) < 0.6
    assert "menos de 20 s de fala" in capsys.readouterr().out


def test_frase_sem_fala_entre_varias(tmp_path, capsys):
    a, b = tmp_path / "f01.wav", tmp_path / "f02.wav"
    sf.write(a, _frase(seed=1), SR)
    sf.write(b, (np.random.default_rng(5).standard_normal(SR * 12) * 1e-3).astype(np.float32), SR)
    assert main([str(a), str(b), "--saida", str(tmp_path / "r.wav")]) == 1
    assert "nenhuma fala encontrada na gravação 2" in capsys.readouterr().out
