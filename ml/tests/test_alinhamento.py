"""Testes do alinhamento de uma gravação de chamada real (camada 2).

O alinhamento é o que torna a camada 2 mensurável: sem ele a gravação é um
bloco de minutos sem rótulo. Um alinhamento errado não falha de forma visível —
ele produz recortes com o rótulo do vizinho e um EER que *parece* resultado.
Por isso os testes exercitam o canal real (codec, banda, ruído, AGC, deriva de
relógio) e não só o caminho limpo.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture.alinhamento import (  # noqa: E402
    CORRELACAO_MINIMA,
    Trecho,
    alinhar,
    envelope,
    montar_referencia,
    recortar,
)

SR = 16000
RNG = np.random.default_rng(7)


def _fala(segundos: float, f0: float = 150.0) -> np.ndarray:
    """Sinal com envelope silábico — é o envelope que o alinhamento usa."""
    t = np.arange(int(SR * segundos)) / SR
    silabas = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)      # ~4 sílabas/s
    harm = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3))
    return (0.3 * silabas * harm).astype(np.float32)


def _playlist(n: int = 4) -> tuple[np.ndarray, list[Trecho]]:
    audios = []
    for i in range(n):
        rotulo = "bonafide" if i % 2 == 0 else "spoof"
        sistema = "-" if rotulo == "bonafide" else "A10"
        t = Trecho(f"LA_E_{i:05d}", rotulo, sistema, 0, 0)
        audios.append((t, _fala(2.0 + 0.5 * i, f0=120 + 30 * i)))
    return montar_referencia(audios, SR, gap_s=1.0)


def _atrasar(wav: np.ndarray, segundos: float) -> np.ndarray:
    """A gravação começa antes da reprodução: sobra silêncio na frente."""
    return np.concatenate([np.zeros(int(segundos * SR), dtype=np.float32), wav])


# --------------------------------------------------------------------------- #
# Caminho limpo
# --------------------------------------------------------------------------- #
def test_acha_todos_os_trechos_sem_degradacao():
    ref, mapa = _playlist()
    captura = _atrasar(ref, 0.7)

    encaixes = alinhar(mapa, ref, captura, SR)

    assert all(e.confiavel for e in encaixes)
    for e in encaixes:
        erro_ms = abs(e.inicio_capturado - (e.trecho.inicio + int(0.7 * SR))) / SR * 1000
        assert erro_ms < 50, f"{e.trecho.id} errou {erro_ms:.0f} ms"


def test_recorte_preserva_o_rotulo_e_a_duracao():
    ref, mapa = _playlist()
    captura = _atrasar(ref, 0.35)

    pedacos = recortar(captura, alinhar(mapa, ref, captura, SR))

    assert [t.id for t, _ in pedacos] == [t.id for t in mapa]
    assert [t.rotulo for t, _ in pedacos] == ["bonafide", "spoof"] * 2
    for t, wav in pedacos:
        assert len(wav) == t.n


# --------------------------------------------------------------------------- #
# Canal real: é aqui que um marcador por bipe falharia
# --------------------------------------------------------------------------- #
def test_sobrevive_ao_codec_e_a_banda_estreita():
    from src.preprocess.channel import limitar_banda, opus_roundtrip

    ref, mapa = _playlist()
    degradado = opus_roundtrip(limitar_banda(ref, SR, 8000), SR, compression_level=0.92)
    captura = _atrasar(degradado[: len(ref)], 0.5)

    encaixes = alinhar(mapa, ref, captura, SR)

    assert all(e.confiavel for e in encaixes), \
        [f"{e.trecho.id}={e.correlacao:.2f}" for e in encaixes]


def test_sobrevive_ao_ganho_automatico():
    """O AGC muda a amplitude ao longo da chamada; a normalização absorve isso."""
    ref, mapa = _playlist()
    captura = _atrasar(ref, 0.4).copy()
    rampa = np.linspace(0.2, 3.0, len(captura)).astype(np.float32)
    captura *= rampa

    assert all(e.confiavel for e in alinhar(mapa, ref, captura, SR))


def test_sobrevive_a_ruido_de_fundo():
    ref, mapa = _playlist()
    captura = _atrasar(ref, 0.6) + 0.05 * RNG.standard_normal(
        len(ref) + int(0.6 * SR)).astype(np.float32)

    assert all(e.confiavel for e in alinhar(mapa, ref, captura, SR))


def test_absorve_deriva_de_relogio():
    """Duas placas de som não andam exatamente na mesma taxa.

    0,1% em 12 s dá 12 ms — pouco. O que mata é o acúmulo: o ajuste é
    sequencial justamente para que o erro não cresça trecho a trecho.
    """
    import scipy.signal as sps

    ref, mapa = _playlist(n=6)
    n_esticado = int(len(ref) * 1.001)                 # +0,1% de deriva
    esticado = sps.resample(ref, n_esticado).astype(np.float32)
    captura = _atrasar(esticado, 0.5)

    encaixes = alinhar(mapa, ref, captura, SR)

    assert all(e.confiavel for e in encaixes)
    ultimo = encaixes[-1]
    previsto_sem_deriva = ultimo.trecho.inicio + int(0.5 * SR)
    achado = ultimo.inicio_capturado
    assert achado > previsto_sem_deriva, "a deriva não foi seguida"
    esperado = int(ultimo.trecho.inicio * 1.001) + int(0.5 * SR)
    assert abs(achado - esperado) / SR < 0.10


# --------------------------------------------------------------------------- #
# O portão: um alinhamento ruim precisa se declarar, não emitir lixo rotulado
# --------------------------------------------------------------------------- #
def test_descarta_trecho_que_nao_foi_tocado():
    """Se a chamada cai no meio, o resto não existe na gravação."""
    ref, mapa = _playlist(n=4)
    metade = len(ref) // 2
    captura = _atrasar(ref[:metade], 0.3)

    pedacos = recortar(captura, alinhar(mapa, ref, captura, SR))

    assert 0 < len(pedacos) < 4, "a queda da chamada precisa reduzir o conjunto"


def test_ruido_puro_nao_produz_recorte_confiavel():
    """Gravar o microfone errado não pode virar um EER."""
    ref, mapa = _playlist()
    captura = 0.2 * RNG.standard_normal(len(ref) + 8000).astype(np.float32)

    encaixes = alinhar(mapa, ref, captura, SR)

    assert sum(e.confiavel for e in encaixes) <= 1, \
        [f"{e.correlacao:.2f}" for e in encaixes]


def test_correlacao_de_trecho_identico_e_um():
    x = _fala(2.0)
    from src.capture.alinhamento import correlacao_maxima
    _, corr = correlacao_maxima(envelope(x, SR), envelope(x, SR))
    assert corr == pytest.approx(1.0, abs=1e-6)


def test_limiar_esta_entre_o_canal_real_e_o_ruido():
    """Trava a margem medida: o canal degradado fica bem acima do limiar."""
    from src.preprocess.channel import limitar_banda, opus_roundtrip

    ref, mapa = _playlist()
    degradado = opus_roundtrip(limitar_banda(ref, SR, 8000), SR, compression_level=0.92)
    pior = min(e.correlacao for e in alinhar(mapa, ref, _atrasar(degradado[:len(ref)], 0.5), SR))
    assert pior > CORRELACAO_MINIMA * 1.3, f"margem apertada demais: {pior:.2f}"


# --------------------------------------------------------------------------- #
# Montagem da playlist
# --------------------------------------------------------------------------- #
def test_referencia_tem_silencio_entre_os_audios():
    ref, mapa = _playlist(n=3)
    for anterior, seguinte in zip(mapa, mapa[1:]):
        fim = anterior.inicio + anterior.n
        assert np.abs(ref[fim:seguinte.inicio]).max() == 0.0
        assert (seguinte.inicio - fim) == SR      # gap_s=1.0


def test_mapa_sobrevive_ao_disco(tmp_path):
    from src.capture.alinhamento import carregar_mapa, salvar_mapa

    _, mapa = _playlist()
    salvar_mapa(mapa, tmp_path / "mapa.json", SR)
    lido, sr = carregar_mapa(tmp_path / "mapa.json")
    assert sr == SR and lido == mapa


def test_linha_de_protocolo_e_lida_pelo_parser(tmp_path):
    """O que sai daqui precisa entrar no evaluate.py sem conversão."""
    from src.capture.alinhamento import linha_de_protocolo
    from src.data.dataset import parse_protocol_with_systems

    _, mapa = _playlist()
    proto = tmp_path / "p.txt"
    proto.write_text("\n".join(linha_de_protocolo(t) for t in mapa) + "\n",
                     encoding="utf-8")

    lidos = parse_protocol_with_systems(proto)
    assert [n for n, _, _ in lidos] == [t.id for t in mapa]
    assert [s for _, _, s in lidos] == [t.sistema for t in mapa]
    assert [lab for _, lab, _ in lidos] == [0, 1, 0, 1]   # bonafide=0, spoof=1


# --------------------------------------------------------------------------- #
# O fluxo inteiro do scripts/canal_real.py, com um canal simulado no meio.
#
# É o ensaio do procedimento da camada 2: sem isso, o primeiro teste de verdade
# seria também o primeiro teste do código — com uma chamada real aberta e duas
# pessoas esperando.
# --------------------------------------------------------------------------- #
def _base_falsa(raiz: Path, n: int = 8) -> Path:
    import soundfile as sf

    audio = raiz / "flac"; audio.mkdir(parents=True)
    linhas = []
    for i in range(n):
        spoof = i % 2 == 1
        nome = f"LA_E_{i:05d}"
        sf.write(audio / f"{nome}.flac", _fala(3.0, f0=110 + 20 * i), SR)
        linhas.append(f"LA_0099 {nome} - {'A10' if spoof else '-'} "
                      f"{'spoof' if spoof else 'bonafide'}")
    proto = raiz / "eval.txt"
    proto.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return proto


def test_fluxo_completo_preparar_e_alinhar(tmp_path, monkeypatch, capsys):
    import soundfile as sf
    import yaml

    from scripts.canal_real import cmd_alinhar, cmd_preparar
    from src.preprocess.channel import limitar_banda, opus_roundtrip

    proto = _base_falsa(tmp_path / "base")
    cfg = {
        "audio": {"sample_rate": SR, "duration": 4.0, "trim_silence": False,
                  "top_db": 30, "peak_normalize": False},
        "data": {"protocols": {"eval": str(proto)},
                 "audio_dir": {"eval": str(tmp_path / "base" / "flac")}},
    }
    caminho_cfg = tmp_path / "c.yaml"
    caminho_cfg.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    saida = tmp_path / "canal"
    assert cmd_preparar(argparse.Namespace(
        config=str(caminho_cfg), n_por_classe=4, seed=1, saida=str(saida))) == 0
    assert (saida / "referencia.wav").is_file()

    # A "chamada": codec, banda estreita, AGC e o atraso de quem grava antes.
    ref, _ = sf.read(saida / "referencia.wav", dtype="float32")
    passou = opus_roundtrip(limitar_banda(ref, SR, 8000), SR, 0.92)[: len(ref)]
    passou = passou * np.linspace(0.5, 1.8, len(passou)).astype(np.float32)
    gravacao = tmp_path / "chamada.wav"
    sf.write(gravacao, _atrasar(passou, 1.3), SR)

    assert cmd_alinhar(argparse.Namespace(
        pasta=str(saida), gravacao=str(gravacao))) == 0

    saida_txt = capsys.readouterr().out
    assert "Recuperados: 8/8" in saida_txt, saida_txt
    assert (saida / "protocolo_canal_real.txt").is_file()
    assert len(list((saida / "capturado").glob("*.flac"))) == 8


def test_preparar_cobre_varios_ataques(tmp_path):
    """Uma playlist pequena não pode sortear só um ataque."""
    import yaml

    from scripts.canal_real import sortear
    from src.data.dataset import parse_protocol_with_systems

    linhas = [f"LA_0099 b{i:04d} - - bonafide" for i in range(200)]
    for a in ("A07", "A10", "A12", "A17"):
        linhas += [f"LA_0099 {a}_{i:04d} - {a} spoof" for i in range(200)]
    proto = tmp_path / "p.txt"
    proto.write_text("\n".join(linhas) + "\n", encoding="utf-8")

    escolhidos = sortear(parse_protocol_with_systems(proto), 8, seed=3)
    ataques = {s for _, lab, s in escolhidos if lab == 1}
    assert ataques == {"A07", "A10", "A12", "A17"}, ataques
    assert sum(1 for _, lab, _ in escolhidos if lab == 0) == 8
