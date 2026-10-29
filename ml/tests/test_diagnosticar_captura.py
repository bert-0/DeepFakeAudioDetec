"""Diagnóstico de deformação temporal (scripts/diagnosticar_captura.py)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.diagnosticar_captura import analisar, diagnosticar  # noqa: E402
from src.capture.alinhamento import Trecho, montar_referencia  # noqa: E402
from test_alinhamento import SR  # noqa: E402


def _rajadas(rng, segundos):
    """Rajadas de ruído de duração e volume aleatórios: envelope único, como o
    da fala real (um tom periódico casaria em qualquer ponto da playlist)."""
    partes, total = [], 0
    while total < segundos * SR:
        n = int(rng.uniform(0.05, 0.35) * SR)
        vol = rng.uniform(0.0, 0.6) if rng.random() < 0.8 else 0.0
        partes.append((rng.standard_normal(n) * vol).astype(np.float32))
        total += n
    return np.concatenate(partes)


def _ref():
    rng = np.random.default_rng(0)
    audios = [(Trecho(f"t{i}", "bonafide", "-", 0, 0),
               _rajadas(rng, float(rng.uniform(2.0, 4.0)))) for i in range(25)]
    return montar_referencia(audios, SR, 1.0)[0]


def _laudo(ref, cap):
    return " ".join(diagnosticar(*analisar(ref, cap, SR), len(ref) / SR, len(cap) / SR))


def test_so_atraso():
    ref = _ref()
    cap = np.concatenate([np.zeros(SR * 2, np.float32), ref])
    laudo = _laudo(ref, cap)
    assert "só atraso" in laudo, laudo


def test_taxa_errada_da_o_fator():
    import librosa

    ref = _ref()
    # Tocado a 16 kHz, mas tratado como se fosse 44,1 -> 48: tudo estica 8,8%.
    cap = librosa.resample(ref, orig_sr=44100, target_sr=48000)
    laudo = _laudo(ref, np.concatenate([np.zeros(SR, np.float32), cap]))
    assert "TAXA DE AMOSTRAGEM" in laudo and "1.08" in laudo, laudo
    assert "DEGRAUS" not in laudo, laudo


def test_buracos_viram_degraus():
    ref = _ref()
    corte = [len(ref) // 4, len(ref) // 2, 3 * len(ref) // 4]
    partes, ini = [], 0
    for c in corte:
        partes += [ref[ini:c], np.zeros(int(0.3 * SR), np.float32)]   # 300 ms inseridos
        ini = c
    partes.append(ref[ini:])
    laudo = _laudo(ref, np.concatenate(partes))
    assert "DEGRAUS" in laudo and "3 vez" in laudo, laudo
    assert "TAXA" not in laudo, laudo


def test_gravacao_sem_a_playlist():
    ref = _ref()
    ruido = np.random.default_rng(3).standard_normal(len(ref)).astype(np.float32) * 0.1
    assert "não é a playlist" in _laudo(ref, ruido)


def test_gravacao_comecou_depois():
    """O caso real do primeiro controle: 7 s do começo ficaram de fora."""
    ref = _ref()
    laudo = _laudo(ref, ref[7 * SR:])
    assert "COMEÇOU 7.0 s DEPOIS" in laudo, laudo
    assert "não é a playlist" not in laudo and "TAXA" not in laudo, laudo
