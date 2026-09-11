"""Testes do portão de qualidade do canal.

Contexto medido (`robustness_eval.py`, eval completo): o codec Opus custa +1,90
pp de EER ao `fusion_v4`, mas perder a banda alta custa +5,35 pp.

O ponto central destes testes é a **assimetria**: presença de alta frequência
prova que o canal a transmite; ausência não prova nada, porque uma vogal
sustentada também não tem alta frequência.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture.qualidade import (  # noqa: E402
    CORTE_HZ,
    FRACAO_MINIMA,
    EstadoDoCanal,
    avaliar,
    fracao_energia_alta,
)
from src.preprocess.channel import limitar_banda  # noqa: E402

SR = 16000
T = np.arange(SR * 4) / SR


def _vogal(seed=0):
    """Vogal sustentada: harmônicos com formantes e queda espectral.
    Não tem energia acima de 4 kHz — como um canal de banda estreita."""
    rng = np.random.default_rng(seed)
    s = np.zeros_like(T)
    for n in range(1, 60):
        f = 120 * n
        if f >= SR / 2:
            break
        env = sum(np.exp(-((f - fc) / bw) ** 2)
                  for fc, bw in ((730, 80), (1090, 90), (2440, 120)))
        s += (env + 0.05 * min(1.0, (500 / max(f, 1)) ** 2)) * \
            np.sin(2 * np.pi * f * T + rng.uniform(0, 6.28))
    return (0.3 * s / np.abs(s).max()).astype(np.float32)


def _fricativa(seed=0):
    """/s/: energia concentrada acima de 3 kHz."""
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(SR * 4)
    X = np.fft.rfft(b)
    f = np.fft.rfftfreq(len(b), 1 / SR)
    X[f < 3000] *= 0.05
    w = np.fft.irfft(X).astype(np.float32)
    return (0.3 * w / np.abs(w).max()).astype(np.float32)


def _fala(seed=0):
    """Fala: vogais na maior parte, fricativas em trechos curtos."""
    rng = np.random.default_rng(seed)
    v, fr = _vogal(seed), _fricativa(seed)
    m = np.zeros_like(T)
    for i in rng.choice(len(T) - 3200, 6, replace=False):
        m[i:i + 3200] = 1.0
    return ((1 - m) * v + m * fr).astype(np.float32)


# --------------------------------------------------------------------------- #
# A medição por janela: p90 dos quadros, não a média
#
# Regressão: a primeira versão usava a média da janela. Fala real dava 5,40% e
# vogal sustentada 0,00% — indistinguível de banda estreita. O p90 leva a fala
# real a 81%, porque pergunta "algum quadro teve alta frequência?".
# --------------------------------------------------------------------------- #
def test_fala_real_mostra_alta_frequencia():
    assert fracao_energia_alta(_fala(), SR) > 10 * FRACAO_MINIMA


def test_fricativa_e_quase_toda_alta_frequencia():
    assert fracao_energia_alta(_fricativa(), SR) > 0.5


def test_vogal_sustentada_nao_mostra_alta_frequencia():
    """E isso é normal — não é defeito do canal."""
    assert fracao_energia_alta(_vogal(), SR) < FRACAO_MINIMA


def test_banda_estreita_zera():
    assert fracao_energia_alta(limitar_banda(_fala(), SR, 8000), SR) < FRACAO_MINIMA / 10


def test_vogal_e_banda_estreita_sao_indistinguiveis_numa_janela():
    """A justificativa do desenho: por isso o veredito é de sessão."""
    vogal = fracao_energia_alta(_vogal(), SR)
    estreita = fracao_energia_alta(limitar_banda(_fala(), SR, 8000), SR)
    assert vogal < FRACAO_MINIMA and estreita < FRACAO_MINIMA


# --------------------------------------------------------------------------- #
# O veredito de sessão
# --------------------------------------------------------------------------- #
def _rodar(janelas):
    estado = EstadoDoCanal()
    for w in janelas:
        estado.observar(avaliar(w, SR))
    return estado


def test_fala_normal_confirma_banda_larga():
    e = _rodar([_fala(i) for i in range(3)])
    assert e.veredito == EstadoDoCanal.LARGA
    assert e.avaliavel


def test_banda_estreita_e_concluida_depois_de_varias_janelas():
    e = _rodar([limitar_banda(_fala(i), SR, 8000) for i in range(7)])
    assert e.veredito == EstadoDoCanal.ESTREITA
    assert not e.avaliavel


def test_comeca_indeterminado_e_nao_acusa_cedo_demais():
    """Sem evidência, o sistema não afirma banda estreita — só não sabe."""
    e = _rodar([_vogal(seed=i) for i in range(3)])
    assert e.veredito == EstadoDoCanal.INDETERMINADO
    assert e.avaliavel, "na dúvida, continua medindo"


def test_uma_fricativa_tardia_recupera_o_veredito():
    """Regressão: a primeira versão reprovaria as vogais iniciais como banda
    estreita, e o score dessas janelas seria descartado sem motivo."""
    e = _rodar([_vogal(seed=i) for i in range(3)] + [_fala(0)])
    assert e.veredito == EstadoDoCanal.LARGA


def test_banda_larga_provada_nao_volta_atras():
    """O canal não muda a cada 2 s; uma sequência de vogais não o desqualifica."""
    e = _rodar([_fala(0)] + [_vogal(seed=i) for i in range(10)])
    assert e.veredito == EstadoDoCanal.LARGA


def test_estado_novo_e_indeterminado():
    e = EstadoDoCanal()
    assert e.veredito == EstadoDoCanal.INDETERMINADO
    assert e.avaliavel
    assert e.observadas == 0


def test_a_descricao_diz_em_que_pe_esta():
    assert "banda larga" in _rodar([_fala(0)]).descricao()
    assert "indeterminado" in _rodar([_vogal()]).descricao()
    assert "BANDA ESTREITA" in _rodar(
        [limitar_banda(_fala(i), SR, 8000) for i in range(7)]).descricao()


# --------------------------------------------------------------------------- #
# Bordas
# --------------------------------------------------------------------------- #
def test_sinal_vazio_nao_quebra():
    assert fracao_energia_alta(np.zeros(0, dtype=np.float32), SR) == 0.0
    assert fracao_energia_alta(np.zeros(1, dtype=np.float32), SR) == 0.0


def test_silencio_absoluto_nao_quebra():
    """Quadros sem energia nenhuma ficariam 0/0; precisam ser ignorados."""
    assert fracao_energia_alta(np.zeros(SR, dtype=np.float32), SR) == 0.0


def test_janela_curta_nao_quebra():
    assert 0.0 <= fracao_energia_alta(_fala()[:100], SR) <= 1.0


def test_fracao_sempre_entre_zero_e_um():
    for x in (_fala(), _vogal(), limitar_banda(_fala(), SR, 8000)):
        assert 0.0 <= fracao_energia_alta(x, SR) <= 1.0


def test_corte_bate_com_a_nyquist_da_banda_estreita():
    assert CORTE_HZ == 8000 / 2


def test_qualidade_e_imutavel():
    q = avaliar(_fala(), SR)
    with pytest.raises(Exception):
        q.fracao_alta = 0.0
