"""Testes do portão de qualidade do canal.

Contexto medido (`robustness_eval.py`, eval completo): o codec Opus custa +1,90
pp de EER ao `fusion_v4`, mas perder a banda alta custa +5,35 pp. A segunda
condição é detectável no próprio áudio, e é isso que este módulo faz.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture.qualidade import (  # noqa: E402
    CORTE_HZ,
    FRACAO_MINIMA,
    avaliar,
    fracao_energia_alta,
)
from src.preprocess.channel import limitar_banda, opus_roundtrip  # noqa: E402

SR = 16000


@pytest.fixture
def banda_larga():
    """Ruído rosa: energia em todas as bandas, como a fala."""
    rng = np.random.default_rng(0)
    branco = rng.standard_normal(SR * 4)
    espectro = np.fft.rfft(branco)
    f = np.fft.rfftfreq(len(branco), 1 / SR)
    f[0] = f[1]
    sinal = np.fft.irfft(espectro / np.sqrt(f)).astype(np.float32)
    return (0.3 * sinal / np.abs(sinal).max()).astype(np.float32)


# --------------------------------------------------------------------------- #
# O caso que o portão existe para pegar
# --------------------------------------------------------------------------- #
def test_banda_estreita_e_recusada(banda_larga):
    q = avaliar(limitar_banda(banda_larga, SR, 8000), SR)
    assert not q.avaliavel
    assert q.fracao_alta < 0.01


def test_banda_larga_e_aceita(banda_larga):
    q = avaliar(banda_larga, SR)
    assert q.avaliavel


@pytest.mark.parametrize("nivel", [0.90, 0.92, 0.96])
def test_opus_na_faixa_do_teams_continua_avaliavel(banda_larga, nivel):
    """O codec custa ~2 pp de EER; não é motivo para recusar a janela."""
    assert avaliar(opus_roundtrip(banda_larga, SR, nivel), SR).avaliavel


def test_banda_estreita_com_codec_tambem_e_recusada(banda_larga):
    """O caso realista de rede ruim: o Teams cai para banda estreita E comprime."""
    degradado = opus_roundtrip(limitar_banda(banda_larga, SR, 8000), SR, 0.92)
    assert not avaliar(degradado, SR).avaliavel


# --------------------------------------------------------------------------- #
# A separação é ampla, não apertada — é o que torna o limiar defensável
# --------------------------------------------------------------------------- #
def test_a_separacao_e_ampla_nos_dois_lados(banda_larga):
    """Margens medidas, com o corte em 2%:

        limpo            11,21%  = 5,61x o corte
        opus 25 kbps      6,39%  = 3,19x
        opus 15 kbps      8,33%  = 4,16x
        banda estreita     0,00%  = 0,00x

    O limiar não está espremido entre os casos: sobra fator 3 de um lado e a
    banda estreita zera do outro. É isso que o torna defensável.
    """
    larga = fracao_energia_alta(banda_larga, SR)
    estreita = fracao_energia_alta(limitar_banda(banda_larga, SR, 8000), SR)
    assert larga > 5 * FRACAO_MINIMA, "banda larga fica bem acima do corte"
    assert estreita < FRACAO_MINIMA / 100, "banda estreita zera"


def test_pior_caso_de_banda_larga_ainda_folga_do_corte(banda_larga):
    """O Opus a 25 kbps é o que menos preserva alta frequência entre os casos
    aceitáveis — 3,19x o corte. Se ele encostasse no limiar, o portão recusaria
    uma chamada perfeitamente normal do Teams."""
    pior = fracao_energia_alta(opus_roundtrip(banda_larga, SR, 0.92), SR)
    assert pior > 3 * FRACAO_MINIMA


# --------------------------------------------------------------------------- #
# Bordas
# --------------------------------------------------------------------------- #
def test_silencio_nao_e_avaliavel():
    assert not avaliar(np.zeros(SR, dtype=np.float32), SR).avaliavel


def test_sinal_vazio_nao_quebra():
    q = avaliar(np.zeros(0, dtype=np.float32), SR)
    assert q.fracao_alta == 0.0
    assert not q.avaliavel


def test_janela_curta_nao_quebra(banda_larga):
    """A janela pode ser menor que o nperseg padrão do Welch."""
    assert 0.0 <= fracao_energia_alta(banda_larga[:100], SR) <= 1.0


def test_fracao_sempre_entre_zero_e_um(banda_larga):
    for x in (banda_larga, limitar_banda(banda_larga, SR, 8000),
              np.zeros(SR, dtype=np.float32)):
        assert 0.0 <= fracao_energia_alta(x, SR) <= 1.0


def test_corte_bate_com_a_nyquist_da_banda_estreita():
    """8 kHz de amostragem => nada acima de 4 kHz. O corte precisa ser esse."""
    assert CORTE_HZ == 8000 / 2


def test_descricao_diz_qual_e_o_problema(banda_larga):
    assert "BANDA ESTREITA" in avaliar(limitar_banda(banda_larga, SR, 8000), SR).descricao()
    assert "banda larga" in avaliar(banda_larga, SR).descricao()


def test_qualidade_e_imutavel(banda_larga):
    """Um diagnóstico não pode ser alterado depois de emitido."""
    q = avaliar(banda_larga, SR)
    with pytest.raises(Exception):
        q.banda_larga = False
