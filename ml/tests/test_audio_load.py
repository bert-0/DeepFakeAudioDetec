"""Testes da leitura de áudio — o ponto onde um arquivo ruim derruba o treino."""

import warnings
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.preprocess import AudioLoadError, load_audio


def _escreve_flac(destino: Path, segundos: float = 1.0, sr: int = 16000) -> Path:
    t = np.arange(int(sr * segundos)) / sr
    sf.write(destino, (0.4 * np.sin(2 * np.pi * 180 * t)).astype("float32"), sr)
    return destino


def _trunca(caminho: Path, fracao: float = 0.1) -> Path:
    dados = caminho.read_bytes()
    caminho.write_bytes(dados[: int(len(dados) * fracao)])
    return caminho


def test_arquivo_integro_e_lido_normalmente(tmp_path):
    """O caminho feliz não pode ter mudado."""
    caminho = _escreve_flac(tmp_path / "bom.flac", segundos=1.0)
    wav = load_audio(caminho, 16000)
    assert wav.dtype == np.float32
    assert abs(len(wav) - 16000) <= 1


def test_reamostragem_continua_valendo(tmp_path):
    caminho = _escreve_flac(tmp_path / "bom.flac", segundos=1.0, sr=16000)
    assert abs(len(load_audio(caminho, 8000)) - 8000) <= 1


# --------------------------------------------------------------------------- #
# Regressão: sem o tratamento, um .flac truncado entre os 121.461 da base
# derruba um treino de horas com `NoBackendError` de mensagem VAZIA — o log
# não diz qual arquivo quebrou, e o erro real do libsndfile é descartado.
# --------------------------------------------------------------------------- #
def test_arquivo_truncado_levanta_erro_com_o_caminho(tmp_path):
    caminho = _trunca(_escreve_flac(tmp_path / "LA_E_1234567.flac"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(AudioLoadError) as exc:
            load_audio(caminho, 16000)

    mensagem = str(exc.value)
    assert "LA_E_1234567.flac" in mensagem, "a mensagem precisa dizer QUAL arquivo"
    assert mensagem.strip(), "mensagem vazia é exatamente o defeito original"


def test_a_mensagem_aponta_para_a_ferramenta_de_diagnostico(tmp_path):
    """Quem lê o log às 3h da manhã precisa saber o próximo comando."""
    caminho = _trunca(_escreve_flac(tmp_path / "ruim.flac"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(AudioLoadError) as exc:
            load_audio(caminho, 16000)
    assert "--deep" in str(exc.value)


def test_a_excecao_original_fica_encadeada(tmp_path):
    """`--deep` recupera a causa real; aqui garantimos que ela não se perde."""
    caminho = _trunca(_escreve_flac(tmp_path / "ruim.flac"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(AudioLoadError) as exc:
            load_audio(caminho, 16000)
    assert exc.value.__cause__ is not None


def test_erro_de_mensagem_vazia_ainda_identifica_o_tipo(tmp_path, monkeypatch):
    """O caso real: `NoBackendError()` — `str(erro)` é `''`.

    Sem o recuo para `type(erro).__name__`, a mensagem terminaria em
    "falha ao ler o áudio X: ." e o motivo sumiria por completo.
    """
    class ErroSemMensagem(Exception):
        pass

    import src.preprocess.audio as mod
    monkeypatch.setattr(mod.librosa, "load",
                        lambda *a, **k: (_ for _ in ()).throw(ErroSemMensagem()))

    with pytest.raises(AudioLoadError, match="ErroSemMensagem"):
        load_audio(tmp_path / "qualquer.flac", 16000)


def test_arquivo_inexistente_continua_sendo_filenotfound(tmp_path):
    """`FileNotFoundError` já é inequívoco e é tratado em outros pontos;
    embrulhá-lo mudaria o comportamento de quem o captura."""
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "nao_existe.flac", 16000)


def test_audio_load_error_e_capturavel_como_runtimeerror(tmp_path):
    """Herda de RuntimeError para não quebrar `except RuntimeError` existente."""
    assert issubclass(AudioLoadError, RuntimeError)
