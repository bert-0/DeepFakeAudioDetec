"""Teste ao vivo sem VB-Cable nem VLC (subcomandos `chrome` e `tocar`)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import scripts.canal_real as canal_real  # noqa: E402
from test_alinhamento import SR, _base_falsa  # noqa: E402


def _playlist(tmp_path):
    proto = _base_falsa(tmp_path / "base")
    cfg = {
        "audio": {"sample_rate": SR, "duration": 4.0, "trim_silence": False,
                  "top_db": 30, "peak_normalize": False},
        "data": {"protocols": {"eval": str(proto)},
                 "audio_dir": {"eval": str(tmp_path / "base" / "flac")}},
    }
    caminho = tmp_path / "c.yaml"
    caminho.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    saida = tmp_path / "canal"
    assert canal_real.cmd_preparar(argparse.Namespace(
        config=str(caminho), n_por_classe=4, seed=1, saida=str(saida))) == 0
    return saida


def test_arquivo_do_chrome_tem_o_formato_e_a_espera(tmp_path):
    ref = np.sin(np.linspace(0, 200, SR * 2)).astype(np.float32) * 0.5
    destino = canal_real.arquivo_para_chrome(ref, SR, tmp_path / "c.wav", espera_s=3.0)
    info = sf.info(destino)
    assert info.samplerate == 48000
    assert info.subtype == "PCM_16"
    assert info.channels == 1
    wav, _ = sf.read(destino)
    assert np.abs(wav[: 3 * 48000 - 10]).max() == 0.0, "a espera tem de ser silêncio"
    final = int(canal_real.SILENCIO_FINAL_S * 48000)
    assert abs(len(wav) - (3 + 2) * 48000 - final) <= 2
    # Com %noloop o Chrome repete o último bloco depois do fim: tem de ser silêncio.
    assert np.abs(wav[-final:]).max() == 0.0


def test_espera_do_chrome_nao_atrapalha_o_alinhamento(tmp_path, capsys):
    """O caminho inteiro: o que o Chrome tocaria, voltando para 16 kHz com a
    espera no começo, tem de alinhar como uma gravação comum."""
    from scipy.signal import resample_poly

    saida = _playlist(tmp_path)
    ref, _ = sf.read(saida / "referencia.wav", dtype="float32")
    chrome = canal_real.arquivo_para_chrome(ref, SR, saida / "referencia_chrome.wav",
                                            espera_s=20.0)
    tocado, taxa = sf.read(chrome, dtype="float32")
    capturado = resample_poly(tocado, 1, taxa // SR).astype(np.float32)
    gravacao = tmp_path / "chamada.wav"
    sf.write(gravacao, np.concatenate([np.zeros(SR, np.float32), capturado]), SR)

    assert canal_real.cmd_alinhar(argparse.Namespace(
        pasta=str(saida), gravacao=str(gravacao), sessao="chamada")) == 0
    assert "Recuperados: 8/8" in capsys.readouterr().out


def test_comando_do_chrome_forca_instancia_nova_e_arquivo_sem_repeticao(tmp_path):
    cmd = canal_real.comando_chrome("chrome.exe", tmp_path / "a b.wav",
                                    tmp_path / "perfil", "https://meet.google.com")
    assert cmd[0] == "chrome.exe"
    assert "--use-fake-device-for-media-stream" in cmd
    audio = [c for c in cmd if c.startswith("--use-file-for-fake-audio-capture=")]
    assert len(audio) == 1 and audio[0].endswith("a b.wav%noloop")
    assert any(c.startswith("--user-data-dir=") for c in cmd), \
        "sem perfil próprio, um Chrome já aberto ignoraria as opções"
    assert cmd[-1] == "https://meet.google.com"


def test_navegador_explicito_inexistente_nao_e_aceito(tmp_path):
    assert canal_real.achar_navegador(str(tmp_path / "nao_existe.exe")) is None
    falso = tmp_path / "chrome.exe"
    falso.write_bytes(b"")
    assert canal_real.achar_navegador(str(falso)) == str(falso)


def test_chrome_so_mostrar_nao_abre_nada(tmp_path, monkeypatch, capsys):
    saida = _playlist(tmp_path)
    falso = tmp_path / "chrome.exe"
    falso.write_bytes(b"")
    import subprocess

    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("abriu")))
    assert canal_real.cmd_chrome(argparse.Namespace(
        pasta=str(saida), url="https://meet.google.com", espera=5.0,
        navegador=str(falso), so_mostrar=True)) == 0
    assert "%noloop" in capsys.readouterr().out
    assert (saida / "referencia_chrome.wav").is_file()


def test_tocar_usa_a_referencia_e_o_dispositivo(tmp_path, monkeypatch):
    saida = _playlist(tmp_path)
    chamadas = []
    monkeypatch.setattr(canal_real, "tocar_audio",
                        lambda wav, sr, disp: chamadas.append((len(wav), sr, disp)))
    assert canal_real.cmd_tocar(argparse.Namespace(
        pasta=str(saida), dispositivo="CABLE Input", sem_pausa=True)) == 0
    ref, sr = sf.read(saida / "referencia.wav")
    assert chamadas == [(len(ref), sr, "CABLE Input")]
