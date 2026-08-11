"""Testes das degradações de canal (codec e banda estreita)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess.channel import (  # noqa: E402
    ChannelChain,
    ChannelDegradation,
    limitar_banda,
    medir_taxa_kbps,
    opus_roundtrip,
)

SR = 16000


@pytest.fixture
def wav():
    """Ruído rosa: energia em todas as bandas, como a fala. Um tom puro
    comprimiria a quase nada e não exercitaria o codec."""
    rng = np.random.default_rng(0)
    branco = rng.standard_normal(SR * 2)
    espectro = np.fft.rfft(branco)
    f = np.fft.rfftfreq(len(branco), 1 / SR)
    f[0] = f[1]
    sinal = np.fft.irfft(espectro / np.sqrt(f)).astype(np.float32)
    return (0.3 * sinal / np.abs(sinal).max()).astype(np.float32)


# --------------------------------------------------------------------------- #
# Comprimento — a perturbação roda DEPOIS do preprocess_waveform, quando o sinal
# já tem o tamanho que define o shape das features. Mudá-lo derrubaria a
# inferência no meio de uma avaliação de horas.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind,level", [("opus", 0.5), ("opus", 1.0), ("band", 8000)])
def test_comprimento_preservado(wav, kind, level):
    saida = ChannelDegradation(kind, level, SR)(wav)
    assert saida.size == wav.size


def test_comprimento_forcado_mesmo_se_o_codec_mudar(monkeypatch, wav):
    """Se um dia o codec devolver quadros a mais, o ajuste tem de segurar."""
    import src.preprocess.channel as mod
    monkeypatch.setattr(mod, "opus_roundtrip",
                        lambda w, sr, lvl: np.concatenate([w, w[:500]]))
    assert ChannelDegradation("opus", 0.5, SR)(wav).size == wav.size


def test_comprimento_forcado_quando_vem_curto(monkeypatch, wav):
    import src.preprocess.channel as mod
    monkeypatch.setattr(mod, "opus_roundtrip", lambda w, sr, lvl: w[:-500])
    assert ChannelDegradation("opus", 0.5, SR)(wav).size == wav.size


@pytest.mark.parametrize("kind,level", [("opus", 0.5), ("band", 8000)])
def test_dtype_float32(wav, kind, level):
    assert ChannelDegradation(kind, level, SR)(wav).dtype == np.float32


# --------------------------------------------------------------------------- #
# Opus
# --------------------------------------------------------------------------- #
def test_opus_altera_o_sinal(wav):
    assert not np.array_equal(opus_roundtrip(wav, SR, 0.9), wav)


def test_opus_preserva_a_forma_geral(wav):
    """Degradar não é destruir: a 0.5 o sinal ainda tem de ser reconhecível."""
    saida = opus_roundtrip(wav, SR, 0.5)
    n = min(len(saida), len(wav))
    assert np.corrcoef(wav[:n], saida[:n])[0, 1] > 0.5


def test_mais_compressao_menos_bits(wav):
    alta = medir_taxa_kbps(wav, SR, 0.2)
    baixa = medir_taxa_kbps(wav, SR, 1.0)
    assert alta > baixa, "compression_level maior tem de gerar menos bits"
    assert baixa < 40, "o nível mais alto deveria chegar à faixa de voz do Teams"


def test_opus_e_deterministico(wav):
    """Sem isso, duas execuções da robustez dariam EERs diferentes."""
    assert np.array_equal(opus_roundtrip(wav, SR, 0.8),
                          opus_roundtrip(wav, SR, 0.8))


def test_opus_recusa_taxa_invalida(wav):
    with pytest.raises(ValueError, match="44100"):
        opus_roundtrip(wav, 44100, 0.5)


# --------------------------------------------------------------------------- #
# Banda estreita
# --------------------------------------------------------------------------- #
def test_banda_estreita_remove_a_alta_frequencia(wav):
    from scipy.signal import welch

    saida = limitar_banda(wav, SR, 8000)
    f, p_antes = welch(wav, SR, nperseg=1024)
    _, p_depois = welch(saida, SR, nperseg=1024)
    alta = (f >= 5000)
    razao = p_depois[alta].mean() / p_antes[alta].mean()
    assert razao < 0.01, f"acima de 5 kHz deveria sumir, sobrou {razao:.4f}"


def test_banda_estreita_preserva_a_baixa_frequencia(wav):
    from scipy.signal import welch

    saida = limitar_banda(wav, SR, 8000)
    f, p_antes = welch(wav, SR, nperseg=1024)
    _, p_depois = welch(saida, SR, nperseg=1024)
    baixa = (f < 2000)
    assert 0.5 < p_depois[baixa].mean() / p_antes[baixa].mean() < 2.0


def test_banda_sem_efeito_quando_nao_reduz(wav):
    assert np.array_equal(limitar_banda(wav, SR, SR), wav)


# --------------------------------------------------------------------------- #
# Cadeia e interface
# --------------------------------------------------------------------------- #
def test_cadeia_aplica_as_duas_etapas(wav):
    cadeia = ChannelChain([ChannelDegradation("band", 8000, SR),
                           ChannelDegradation("opus", 0.9, SR)])
    saida = cadeia(wav)
    assert saida.size == wav.size
    assert not np.array_equal(saida, ChannelDegradation("band", 8000, SR)(wav))


def test_cadeia_vazia_e_erro():
    with pytest.raises(ValueError):
        ChannelChain([])


def test_kind_desconhecido():
    with pytest.raises(ValueError, match="mp3"):
        ChannelDegradation("mp3", 0.5, SR)


def test_ignora_o_rng(wav):
    """O canal é determinístico; o `rng` existe só para casar com a interface."""
    d = ChannelDegradation("opus", 0.8, SR)
    assert np.array_equal(d(wav, rng=np.random.default_rng(1)),
                          d(wav, rng=np.random.default_rng(99)))


# --------------------------------------------------------------------------- #
# Picklable — no Windows o dataset inteiro é serializado para cada worker
# (`spawn`). Uma closure prenderia a robustez a num_workers=0, que já foi um
# problema real neste projeto.
# --------------------------------------------------------------------------- #
def test_degradacao_e_picklavel(wav):
    import pickle

    d = pickle.loads(pickle.dumps(ChannelDegradation("opus", 0.8, SR)))
    assert d(wav).size == wav.size


def test_cadeia_e_picklavel(wav):
    import pickle

    c = ChannelChain([ChannelDegradation("band", 8000, SR),
                      ChannelDegradation("opus", 0.9, SR)])
    assert pickle.loads(pickle.dumps(c))(wav).size == wav.size


# --------------------------------------------------------------------------- #
# Fronteira com o treino: este módulo não pode alcançar o caminho de treino.
# --------------------------------------------------------------------------- #
def test_channel_nao_e_importado_pelo_caminho_de_treino():
    import ast

    ml = Path(__file__).resolve().parent.parent
    for arquivo in ("train.py", "evaluate.py", "src/preprocess/augment.py",
                    "src/data/dataset.py", "scripts/score_fusion.py"):
        arvore = ast.parse((ml / arquivo).read_text(encoding="utf-8"))
        for no in ast.walk(arvore):
            modulo = getattr(no, "module", "") or ""
            assert "channel" not in modulo, (
                f"{arquivo} importa o módulo de canal — a fronteira que impede "
                "a medição de robustez de afetar treino/avaliação foi rompida")
