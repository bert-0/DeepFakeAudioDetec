"""Escolha automática do limiar (src/limiares.py)."""

import numpy as np
import soundfile as sf
import torch

from src.limiares import (
    Limiares,
    caminho_captura,
    escolher_limiar,
    limiares_do_checkpoint,
    taxa_nativa,
)

L = Limiares(original=0.65, captura=0.98)


def test_nativo_16k_usa_o_original():
    e = escolher_limiar(L, 16000, taxa_arquivo=16000)
    assert e.limiar == 0.65 and e.aviso is None


def test_gravado_acima_de_16k_usa_o_recalibrado():
    for taxa in (22050, 44100, 48000):
        assert escolher_limiar(L, 16000, taxa_arquivo=taxa).limiar == 0.98


def test_ao_vivo_usa_o_recalibrado():
    assert escolher_limiar(L, 16000, ao_vivo=True).limiar == 0.98


def test_taxa_desconhecida_e_tratada_como_mundo_real():
    assert escolher_limiar(L, 16000, taxa_arquivo=None).limiar == 0.98


def test_telefonia_usa_o_original_com_aviso():
    e = escolher_limiar(L, 16000, taxa_arquivo=8000)
    assert e.limiar == 0.65 and "nenhum limiar" in e.aviso


def test_sem_recalibrado_avisa_em_vez_de_inventar():
    e = escolher_limiar(Limiares(0.65, None), 16000, ao_vivo=True)
    assert e.limiar == 0.65 and "calibrar_captura" in e.aviso


def test_limiares_da_copia_recalibrada():
    dados = {"threshold": 0.98, "calibracao": {"threshold_original": 0.65}}
    assert limiares_do_checkpoint("x_captura.pt", dados) == L


def _estado(semente):
    torch.manual_seed(semente)
    return {"w": torch.randn(3, 3)}


def test_acha_a_copia_ao_lado_so_com_os_mesmos_pesos(tmp_path):
    original = tmp_path / "m.pt"
    estado = _estado(0)
    dados = {"model_state": estado, "threshold": 0.65}
    torch.save(dados, original)
    assert limiares_do_checkpoint(original, dados).captura is None

    torch.save({"model_state": estado, "threshold": 0.98,
                "calibracao": {"threshold_original": 0.65}}, caminho_captura(original))
    assert limiares_do_checkpoint(original, dados) == L

    # Cópia de outro treino: pesos diferentes, limiar de outro modelo — recusada.
    torch.save({"model_state": _estado(1), "threshold": 0.98,
                "calibracao": {"threshold_original": 0.6}}, caminho_captura(original))
    assert limiares_do_checkpoint(original, dados).captura is None


def test_taxa_nativa_le_o_cabecalho(tmp_path):
    arq = tmp_path / "a.wav"
    sf.write(arq, np.zeros(4410, np.float32), 44100)
    assert taxa_nativa(arq) == 44100
    assert taxa_nativa(tmp_path / "nao_existe.wav") is None
