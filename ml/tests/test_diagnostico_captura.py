"""Pacotes repetidos na captura (scripts/diagnostico_captura.py), sem microfone."""

import numpy as np

from scripts.diagnostico_captura import contar_repetidos


def test_acha_pacote_repetido_e_a_distancia():
    x = np.random.default_rng(0).standard_normal(480 * 30) * 1e-3
    x[480 * 13:480 * 14] = x[480 * 2:480 * 3]          # o pacote 13 é cópia do 2: 11 para trás
    x[480 * 20:480 * 21] = x[480 * 18:480 * 19]        # o 20 é cópia do 18: 2 para trás
    r = contar_repetidos(x)
    assert r["repetidos"] == 2 and r["distancias"] == {11: 1, 2: 1}


def test_ruido_e_silencio_digital_nao_contam():
    x = np.random.default_rng(1).standard_normal(480 * 50) * 1e-3
    x[:480 * 10] = 0.0                                  # silêncio digital repetido não é defeito
    r = contar_repetidos(x)
    assert r["repetidos"] == 0 and r["pacotes"] == 40


def _com_copias_inseridas(original, a_cada=3, distancia=2, pacote=480):
    """Como a captura defeituosa: depois de cada `a_cada` pacotes, repete o de `distancia` atrás."""
    blocos = list(original.reshape(-1, pacote))
    saida = []
    for i, b in enumerate(blocos):
        saida.append(b)
        if (i + 1) % a_cada == 0:
            saida.append(saida[-distancia])
    return np.concatenate(saida)


def test_reparar_devolve_o_sinal_original():
    from scripts.diagnostico_captura import remover_repetidos

    original = np.random.default_rng(2).standard_normal(480 * 30) * 1e-2
    com_copias = _com_copias_inseridas(original)
    assert len(com_copias) == 480 * 40
    assert np.array_equal(remover_repetidos(com_copias), original)


def test_reparar_em_estereo_e_pela_linha_de_comando(tmp_path):
    import soundfile as sf

    from scripts.diagnostico_captura import main

    rng = np.random.default_rng(3)
    original = (rng.standard_normal((480 * 12, 2)) * 1e-2).astype(np.float32)
    canal0 = _com_copias_inseridas(original[:, 0], a_cada=4, distancia=1)
    canal1 = _com_copias_inseridas(original[:, 1], a_cada=4, distancia=1)
    (tmp_path / "a").mkdir()
    sf.write(tmp_path / "a" / "f01.wav", np.stack([canal0, canal1], axis=1), 48000, subtype="FLOAT")
    assert main(["reparar", "--pasta", str(tmp_path / "a"), "--destino", str(tmp_path / "b")]) == 0
    y, sr = sf.read(tmp_path / "b" / "f01.wav", dtype="float32")
    assert sr == 48000 and np.array_equal(y, original)
