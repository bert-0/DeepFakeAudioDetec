"""Testes da conversão para WAV (scripts/converter_para_wav.py).

Motivação medida: nos .flac do ASVspoof 2021, o libsndfile falhou em 38 de 50
arquivos ("unknown error in flac decoder"), e o fallback do librosa abre um
FFmpeg por arquivo no Windows. A conversão faz esse custo uma vez só.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.converter_para_wav import (  # noqa: E402
    ALTERNATIVO,
    PULADO,
    RAPIDO,
    converter,
    converter_um,
    ids_dos_protocolos,
    main,
)

SR = 16000


def _flac(pasta: Path, nome: str, segundos: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Valores exatamente representáveis em 16 bits: a conversão tem que ser exata.
    wav = (rng.integers(-20000, 20000, int(SR * segundos)) / 32768.0).astype(np.float32)
    sf.write(pasta / f"{nome}.flac", wav, SR, subtype="PCM_16")
    return wav


@pytest.fixture
def origem(tmp_path):
    pasta = tmp_path / "flac"
    pasta.mkdir()
    return pasta


def test_conversao_e_exata_amostra_a_amostra(origem, tmp_path):
    """Origem de 16 bits gravada em WAV de 16 bits: nenhuma amostra pode mudar."""
    original = _flac(origem, "a")
    destino = tmp_path / "wav"
    destino.mkdir()

    audio_id, status = converter_um(("a", str(origem / "a.flac"), str(destino / "a.wav")))

    lido, sr = sf.read(destino / "a.wav", dtype="float32")
    assert status == RAPIDO and sr == SR
    assert np.array_equal(lido, original)


def test_usa_o_leitor_alternativo_quando_o_libsndfile_falha(origem, tmp_path, monkeypatch):
    """Simula o defeito real: sf.read falha no FLAC, o librosa decodifica."""
    original = _flac(origem, "a")
    destino = tmp_path / "wav"
    destino.mkdir()
    ler_de_verdade = sf.read

    def read_que_falha_no_flac(caminho, *a, **k):
        if str(caminho).endswith(".flac"):
            raise RuntimeError("Error : unknown error in flac decoder.")
        return ler_de_verdade(caminho, *a, **k)

    monkeypatch.setattr(sf, "read", read_que_falha_no_flac)
    monkeypatch.setattr("librosa.load", lambda caminho, sr=None, mono=True: (original, SR))

    _, status = converter_um(("a", str(origem / "a.flac"), str(destino / "a.wav")))

    assert status == ALTERNATIVO
    assert np.array_equal(ler_de_verdade(destino / "a.wav", dtype="float32")[0], original)


def test_arquivo_ilegivel_nao_derruba_o_lote(origem, tmp_path):
    _flac(origem, "bom")
    (origem / "ruim.flac").write_bytes(b"nao e audio")

    r = converter(["bom", "ruim"], origem, tmp_path / "wav", workers=1)

    assert r[RAPIDO] == ["bom"]
    assert r["falhou"] == ["ruim"]
    assert r["erros"] and r["erros"][0].startswith("ruim:")


def test_e_retomavel(origem, tmp_path):
    """Rodar de novo pula o que já foi convertido — interrupção não custa refazer."""
    for i in range(3):
        _flac(origem, f"a{i}", seed=i)
    destino = tmp_path / "wav"

    converter([f"a{i}" for i in range(3)], origem, destino, workers=1)
    r = converter([f"a{i}" for i in range(3)], origem, destino, workers=1)

    assert len(r[PULADO]) == 3 and not r[RAPIDO]


def test_nao_deixa_temporario_para_tras(origem, tmp_path):
    _flac(origem, "a")
    destino = tmp_path / "wav"
    converter(["a"], origem, destino, workers=1)
    assert [p.name for p in destino.iterdir()] == ["a.wav"]


def test_arquivo_vazio_no_destino_e_refeito(origem, tmp_path):
    """Um WAV de 0 bytes não conta como pronto."""
    _flac(origem, "a")
    destino = tmp_path / "wav"
    destino.mkdir()
    (destino / "a.wav").write_bytes(b"")

    r = converter(["a"], origem, destino, workers=1)
    assert r[RAPIDO] == ["a"]


def test_paralelo_da_o_mesmo_resultado(origem, tmp_path):
    for i in range(6):
        _flac(origem, f"a{i}", seed=i)
    ids = [f"a{i}" for i in range(6)]

    r = converter(ids, origem, tmp_path / "wav", workers=2)

    assert sorted(r[RAPIDO]) == sorted(ids)
    for i in ids:
        assert (tmp_path / "wav" / f"{i}.wav").is_file()


def test_ids_de_varios_protocolos_sem_repeticao(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("S x1 - - bonafide\nS x2 - A07 spoof\n", encoding="utf-8")
    b.write_text("S x2 - A07 spoof\nS x3 - A10 spoof\n", encoding="utf-8")
    assert ids_dos_protocolos([a, b]) == ["x1", "x2", "x3"]


def test_cli_ponta_a_ponta(origem, tmp_path, capsys):
    for i in range(3):
        _flac(origem, f"LA_E_{i}", seed=i)
    proto = tmp_path / "p.txt"
    proto.write_text("".join(f"S LA_E_{i} - - bonafide\n" for i in range(3)),
                     encoding="utf-8")

    codigo = main(["--protocolo", str(proto), "--origem", str(origem),
                   "--destino", str(tmp_path / "wav"), "--workers", "1"])

    assert codigo == 0
    saida = capsys.readouterr().out
    assert "--audio-ext .wav" in saida


# --------------------------------------------------------------------------- #
# O resto do pipeline precisa ler os WAV
# --------------------------------------------------------------------------- #
def test_dataset_le_a_extensao_do_config(tmp_path):
    """Sem `data.file_ext`, o dataset procuraria .flac e não acharia nada."""
    import yaml

    from src.config import config_derivado
    from src.data.dataset import build_dataset
    from src.features import FeatureExtractor

    pasta = tmp_path / "wav"
    pasta.mkdir()
    sf.write(pasta / "LA_E_1.wav", np.zeros(SR, dtype=np.float32) + 0.01, SR)
    proto = tmp_path / "p.txt"
    proto.write_text("S LA_E_1 - - bonafide\n", encoding="utf-8")

    base = yaml.safe_load(Path("configs/baseline_v2.yaml").read_text(encoding="utf-8"))
    cfg = config_derivado(base, "t", proto, pasta, ext=".wav")
    ex = FeatureExtractor(cfg["audio"], cfg["features"])

    ds = build_dataset(cfg, "eval", ex, smoke=False)
    features, rotulo = ds[0]
    assert rotulo == 0 and features


def test_config_derivado_so_grava_extensao_quando_nao_e_flac():
    import yaml

    from src.config import config_derivado

    base = yaml.safe_load(Path("configs/baseline_v2.yaml").read_text(encoding="utf-8"))
    assert "file_ext" not in config_derivado(base, "t", "p", "a")["data"]
    assert config_derivado(base, "t", "p", "a", ext=".wav")["data"]["file_ext"] == ".wav"
