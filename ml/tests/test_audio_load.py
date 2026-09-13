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


# --------------------------------------------------------------------------- #
# Regressão de plataforma
#
# Os três testes de truncagem acima passam no Linux porque o libsndfile recusa
# o arquivo. No Windows o `audioread` decodifica o pedaço que existe e devolve
# áudio parcial **sem erro** — foi assim que apareceram, rodando a suíte lá.
#
# Um treino nesse estado consome meio enunciado como se fosse inteiro. O
# `check_data.py --deep` não cobre o buraco: ele usa `sf.read()` direto, então
# protege a base antes do treino, mas não a leitura durante ele.
#
# Estes testes simulam o decodificador permissivo, para que a proteção seja
# verificada nos dois sistemas e no CI — que roda Linux.
# --------------------------------------------------------------------------- #
def _decodificador_permissivo(fracao):
    """Imita o audioread do Windows: devolve só parte do sinal, sem reclamar."""
    def load(caminho, sr=None, mono=True, **kwargs):
        import soundfile as sf
        info = sf.info(str(caminho))
        taxa = sr or info.samplerate
        n = int(info.frames / info.samplerate * taxa * fracao)
        return np.zeros(n, dtype=np.float32), taxa
    return load


def test_audio_parcial_e_recusado_mesmo_sem_erro_do_decodificador(tmp_path, monkeypatch):
    caminho = _escreve_flac(tmp_path / "LA_E_1234567.flac", segundos=4.0)
    monkeypatch.setattr("librosa.load", _decodificador_permissivo(0.1))

    with pytest.raises(AudioLoadError) as exc:
        load_audio(caminho, 16000)

    assert "LA_E_1234567.flac" in str(exc.value)
    assert "4.000s" in str(exc.value), "a mensagem precisa dizer o esperado"
    assert "0.400s" in str(exc.value), "e o que de fato saiu"


def test_a_mensagem_do_truncado_aponta_o_check_data(tmp_path, monkeypatch):
    caminho = _escreve_flac(tmp_path / "ruim.flac", segundos=2.0)
    monkeypatch.setattr("librosa.load", _decodificador_permissivo(0.5))
    with pytest.raises(AudioLoadError) as exc:
        load_audio(caminho, 16000)
    assert "check_data.py" in str(exc.value) and "--deep" in str(exc.value)


def test_diferenca_de_arredondamento_nao_e_confundida_com_truncagem(tmp_path, monkeypatch):
    """Reamostrar muda o comprimento em alguns quadros; isso não é defeito."""
    caminho = _escreve_flac(tmp_path / "bom.flac", segundos=3.0)
    monkeypatch.setattr("librosa.load", _decodificador_permissivo(0.999))
    load_audio(caminho, 16000)   # não deve levantar


def test_audio_mais_longo_que_o_cabecalho_nao_levanta(tmp_path, monkeypatch):
    """A checagem é unilateral: sobra não é sinal de corrupção."""
    caminho = _escreve_flac(tmp_path / "bom.flac", segundos=1.0)
    monkeypatch.setattr("librosa.load", _decodificador_permissivo(1.5))
    load_audio(caminho, 16000)


def test_formato_sem_cabecalho_legivel_ainda_carrega(tmp_path, monkeypatch):
    """Se o libsndfile não abre o formato, não há o que comparar — não trava."""
    caminho = _escreve_flac(tmp_path / "bom.flac", segundos=1.0)
    monkeypatch.setattr("src.preprocess.audio._duracao_do_cabecalho",
                        lambda p: None)
    monkeypatch.setattr("librosa.load", _decodificador_permissivo(0.1))
    assert len(load_audio(caminho, 16000)) > 0


def test_reamostragem_real_nao_dispara_o_alarme(tmp_path):
    """Caminho real, sem mock: 16k -> 8k precisa continuar passando."""
    caminho = _escreve_flac(tmp_path / "bom.flac", segundos=2.0, sr=16000)
    assert abs(len(load_audio(caminho, 8000)) - 16000) <= 2
