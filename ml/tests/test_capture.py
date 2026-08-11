"""Testes da captura ao vivo.

A parte que fala com o WASAPI só roda no Windows, então tudo o mais — janela,
agregação, análise — foi separado para ser testável em qualquer máquina. Se
esses testes precisassem de placa de som, o monitor não teria teste nenhum.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture import CaptureError, FileSource, JanelaDeslizante  # noqa: E402
from src.capture.analyzer import (  # noqa: E402
    SILENCIO_RMS,
    Agregador,
    Leitura,
)
from src.capture.sources import resample_mono  # noqa: E402


# --------------------------------------------------------------------------- #
# Janela deslizante
# --------------------------------------------------------------------------- #
def test_emite_janelas_com_sobreposicao():
    j = JanelaDeslizante(tamanho=4, passo=2)
    janelas = list(j.alimentar(np.arange(6, dtype=np.float32)))
    assert len(janelas) == 2
    assert np.array_equal(janelas[0], [0, 1, 2, 3])
    assert np.array_equal(janelas[1], [2, 3, 4, 5])


def test_junta_blocos_pequenos():
    """A placa entrega ~100 ms por vez; a janela precisa de 4 s."""
    j = JanelaDeslizante(tamanho=10, passo=10)
    assert list(j.alimentar(np.ones(3, dtype=np.float32))) == []
    assert list(j.alimentar(np.ones(3, dtype=np.float32))) == []
    assert len(list(j.alimentar(np.ones(4, dtype=np.float32)))) == 1


def test_passo_igual_ao_tamanho_nao_sobrepoe():
    j = JanelaDeslizante(tamanho=3, passo=3)
    janelas = list(j.alimentar(np.arange(6, dtype=np.float32)))
    assert np.array_equal(janelas[0], [0, 1, 2])
    assert np.array_equal(janelas[1], [3, 4, 5])


def test_resto_guarda_o_incompleto():
    j = JanelaDeslizante(tamanho=4, passo=4)
    list(j.alimentar(np.arange(6, dtype=np.float32)))
    assert np.array_equal(j.resto(), [4, 5])


def test_instante_acompanha_o_passo():
    j = JanelaDeslizante(tamanho=16000 * 4, passo=16000 * 2)
    assert j.instante_da_janela(0, 16000) == 0.0
    assert j.instante_da_janela(3, 16000) == 6.0


@pytest.mark.parametrize("tamanho,passo", [(0, 1), (-1, 1), (4, 0), (4, 5), (4, -2)])
def test_parametros_invalidos(tamanho, passo):
    with pytest.raises(ValueError):
        JanelaDeslizante(tamanho=tamanho, passo=passo)


def test_bloco_vazio_nao_quebra():
    j = JanelaDeslizante(tamanho=4, passo=2)
    assert list(j.alimentar(np.zeros(0, dtype=np.float32))) == []


# --------------------------------------------------------------------------- #
# Fonte de arquivo — é ela que torna uma captura ao vivo reproduzível
# --------------------------------------------------------------------------- #
def _wav(destino: Path, segundos=1.0, sr=16000):
    t = np.arange(int(sr * segundos)) / sr
    sf.write(destino, (0.4 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    return destino


def test_file_source_devolve_o_sinal_inteiro(tmp_path):
    fonte = FileSource(_wav(tmp_path / "a.wav", 1.0), sample_rate=16000, bloco=1000)
    total = np.concatenate(list(fonte.blocos()))
    assert abs(len(total) - 16000) <= 1
    assert total.dtype == np.float32


def test_file_source_reamostra_para_a_taxa_do_modelo(tmp_path):
    _wav(tmp_path / "a.wav", segundos=1.0, sr=44100)
    total = np.concatenate(list(FileSource(tmp_path / "a.wav", 16000).blocos()))
    assert abs(len(total) - 16000) <= 100


def test_file_source_arquivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        FileSource(tmp_path / "nao_existe.wav", 16000)


def test_file_source_arquivo_corrompido_da_erro_de_captura(tmp_path):
    ruim = tmp_path / "ruim.wav"
    ruim.write_bytes(b"isto nao e um wav")
    with pytest.raises(CaptureError, match="ruim.wav"):
        FileSource(ruim, 16000)


def test_context_manager_fecha(tmp_path):
    with FileSource(_wav(tmp_path / "a.wav"), 16000) as fonte:
        assert fonte.sample_rate == 16000


# --------------------------------------------------------------------------- #
# Reamostragem
# --------------------------------------------------------------------------- #
def test_resample_sem_efeito_quando_as_taxas_batem():
    wav = np.arange(10, dtype=np.float32)
    assert np.array_equal(resample_mono(wav, 16000, 16000), wav)


def test_resample_48k_para_16k_reduz_a_um_terco():
    wav = np.random.default_rng(0).standard_normal(48000).astype(np.float32)
    assert abs(len(resample_mono(wav, 48000, 16000)) - 16000) <= 100


def test_resample_de_vazio_nao_quebra():
    assert resample_mono(np.zeros(0, dtype=np.float32), 48000, 16000).size == 0


# --------------------------------------------------------------------------- #
# Agregação
#
# O silêncio precisa ficar de fora: o modelo nunca viu silêncio rotulado, e
# incluí-lo faria a média do resumo perder o sentido numa chamada real, onde a
# maior parte do tempo ninguém está falando.
# --------------------------------------------------------------------------- #
def _leitura(i, score, rms=0.1):
    return Leitura(indice=i, instante=float(i), score=score, rms=rms)


def test_silencio_e_identificado_pelo_rms():
    assert _leitura(0, 0.5, rms=SILENCIO_RMS / 10).silencio
    assert not _leitura(0, 0.5, rms=0.1).silencio


def test_media_ignora_o_silencio():
    ag = Agregador(janela_media=10)
    ag.adicionar(_leitura(0, 0.9))
    ag.adicionar(_leitura(1, 0.0, rms=0.0))   # silêncio: não pode puxar a média
    ag.adicionar(_leitura(2, 0.7))
    assert ag.media_movel() == pytest.approx(0.8)


def test_media_movel_usa_so_as_ultimas():
    ag = Agregador(janela_media=2)
    for i, s in enumerate([0.1, 0.2, 0.9, 0.9]):
        ag.adicionar(_leitura(i, s))
    assert ag.media_movel() == pytest.approx(0.9)


def test_media_e_none_sem_audio_util():
    ag = Agregador()
    ag.adicionar(_leitura(0, 0.0, rms=0.0))
    assert ag.media_movel() is None


def test_resumo_conta_silencio_separado():
    ag = Agregador()
    ag.adicionar(_leitura(0, 0.8))
    ag.adicionar(_leitura(1, 0.0, rms=0.0))
    r = ag.resumo()
    assert r["janelas_total"] == 2
    assert r["janelas_uteis"] == 1
    assert r["janelas_silencio"] == 1
    assert r["score_medio"] == pytest.approx(0.8)


def test_resumo_sem_leituras_nao_quebra():
    r = Agregador().resumo()
    assert r["janelas_total"] == 0
    assert r["score_medio"] is None
