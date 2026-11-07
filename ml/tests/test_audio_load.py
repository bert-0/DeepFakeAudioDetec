"""Testes da leitura de áudio e dos erros para arquivos ruins."""

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


def _decodificador_permissivo(fracao):
    """Imita o audioread do Windows: devolve só parte do sinal, sem reclamar."""
    def load(caminho, sr=None, mono=True, **kwargs):
        import soundfile as sf
        info = sf.info(str(caminho))
        taxa = sr or info.samplerate
        n = int(info.frames / info.samplerate * taxa * fracao)
        return np.zeros(n, dtype=np.float32), taxa
    return load


def test_arquivo_integro_e_lido_normalmente(tmp_path):
    """Arquivo íntegro continua sendo lido normalmente."""
    caminho = _escreve_flac(tmp_path / "bom.flac", segundos=1.0)
    wav = load_audio(caminho, 16000)
    assert wav.dtype == np.float32
    assert abs(len(wav) - 16000) <= 1


def test_reamostragem_continua_valendo(tmp_path):
    caminho = _escreve_flac(tmp_path / "bom.flac", segundos=1.0, sr=16000)
    assert abs(len(load_audio(caminho, 8000)) - 8000) <= 1


# --------------------------------------------------------------------------- #
# Regressão: .flac truncado dava `NoBackendError` sem mensagem nem caminho
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
    """A mensagem sugere o próximo comando (`--deep`)."""
    caminho = _trunca(_escreve_flac(tmp_path / "ruim.flac"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(AudioLoadError) as exc:
            load_audio(caminho, 16000)
    assert "--deep" in str(exc.value)


def test_a_excecao_original_fica_encadeada(tmp_path, monkeypatch):
    """Quando o decodificador levanta (libsndfile no Linux), a exceção original
    fica encadeada. O decodificador é simulado para o teste valer no Windows."""
    caminho = _escreve_flac(tmp_path / "ruim.flac")

    def decodificador_que_falha(*args, **kwargs):
        raise RuntimeError("flac decoder lost sync")

    monkeypatch.setattr("librosa.load", decodificador_que_falha)
    with pytest.raises(AudioLoadError) as exc:
        load_audio(caminho, 16000)

    assert exc.value.__cause__ is not None
    assert "lost sync" in str(exc.value), "o erro real do libsndfile some"


def test_erro_sem_causa_encadeada_ainda_diz_o_que_houve(tmp_path, monkeypatch):
    """Sem exceção original (áudio parcial, caso do Windows), a mensagem diz
    a duração esperada, a obtida e o próximo comando."""
    caminho = _escreve_flac(tmp_path / "ruim.flac", segundos=4.0)
    monkeypatch.setattr("librosa.load", _decodificador_permissivo(0.25))

    with pytest.raises(AudioLoadError) as exc:
        load_audio(caminho, 16000)

    assert exc.value.__cause__ is None, "não há exceção original neste caminho"
    mensagem = str(exc.value)
    assert "ruim.flac" in mensagem
    assert "4.000s" in mensagem and "1.000s" in mensagem
    assert "--deep" in mensagem


def test_erro_de_mensagem_vazia_ainda_identifica_o_tipo(tmp_path, monkeypatch):
    """Erro com `str(erro) == ''` (como `NoBackendError()`) aparece pelo nome do tipo."""
    class ErroSemMensagem(Exception):
        pass

    import src.preprocess.audio as mod
    monkeypatch.setattr(mod.librosa, "load",
                        lambda *a, **k: (_ for _ in ()).throw(ErroSemMensagem()))

    with pytest.raises(AudioLoadError, match="ErroSemMensagem"):
        load_audio(tmp_path / "qualquer.flac", 16000)


def test_arquivo_inexistente_continua_sendo_filenotfound(tmp_path):
    """`FileNotFoundError` não é embrulhado, para não mudar quem o captura."""
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "nao_existe.flac", 16000)


def test_audio_load_error_e_capturavel_como_runtimeerror(tmp_path):
    """Herda de RuntimeError para não quebrar `except RuntimeError` existente."""
    assert issubclass(AudioLoadError, RuntimeError)


# --------------------------------------------------------------------------- #
# Áudio parcial sem erro (audioread no Windows), com decodificador simulado
# --------------------------------------------------------------------------- #
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
    """Diferença de poucos quadros pela reamostragem não conta como truncagem."""
    caminho = _escreve_flac(tmp_path / "bom.flac", segundos=3.0)
    monkeypatch.setattr("librosa.load", _decodificador_permissivo(0.999))
    load_audio(caminho, 16000)   # não deve levantar


def test_audio_mais_longo_que_o_cabecalho_nao_levanta(tmp_path, monkeypatch):
    """Áudio mais longo que o cabeçalho não é tratado como corrupção."""
    caminho = _escreve_flac(tmp_path / "bom.flac", segundos=1.0)
    monkeypatch.setattr("librosa.load", _decodificador_permissivo(1.5))
    load_audio(caminho, 16000)


def test_formato_sem_cabecalho_legivel_ainda_carrega(tmp_path, monkeypatch):
    """Sem cabeçalho legível não há o que comparar, e o áudio carrega."""
    caminho = _escreve_flac(tmp_path / "bom.flac", segundos=1.0)
    monkeypatch.setattr("src.preprocess.audio._duracao_do_cabecalho",
                        lambda p: None)
    monkeypatch.setattr("librosa.load", _decodificador_permissivo(0.1))
    assert len(load_audio(caminho, 16000)) > 0


def test_reamostragem_real_nao_dispara_o_alarme(tmp_path):
    """Reamostragem real 16k -> 8k (sem mock) não dispara a checagem."""
    caminho = _escreve_flac(tmp_path / "bom.flac", segundos=2.0, sr=16000)
    assert abs(len(load_audio(caminho, 8000)) - 16000) <= 2
