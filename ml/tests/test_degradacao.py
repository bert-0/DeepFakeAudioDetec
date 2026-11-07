"""Gancho de degradação antes do pré-processamento (src/preprocess/degradacao.py).

O ponto do gancho: a degradação recebe o áudio como foi lido, com as pausas e
sem normalizar, e não o trecho de 4 s já cortado que o Augmenter recebe.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.data.dataset import ASVspoofDataset, SmokeDataset, _config_fingerprint  # noqa: E402
from src.features import FeatureExtractor  # noqa: E402
from src.preprocess import load_audio  # noqa: E402
from src.preprocess.degradacao import Degradacao, construir  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
SR = 16000


def _fala_com_pausas():
    """1 s de silêncio, 2 s de tom, 1 s de silêncio."""
    t = np.arange(2 * SR) / SR
    return np.concatenate([np.zeros(SR), 0.3 * np.sin(2 * np.pi * 220 * t), np.zeros(SR)]).astype(np.float32)


def _cfg(*etapas):
    return {"enabled": True, "etapas": list(etapas)}


def test_desligada_ou_ausente_nao_constroi_nada():
    assert construir(None, SR) is None
    assert construir({"enabled": False, "etapas": [{"tipo": "ruido", "nivel": 20}]}, SR) is None


@pytest.mark.parametrize("etapa, erro", [
    ({"tipo": "mp3", "nivel": 1}, "desconhecida"),
    ({"tipo": "opus", "nivel": 0.5, "faixa": [0.1, 0.9]}, "exatamente um"),
    ({"tipo": "opus"}, "exatamente um"),
    ({"tipo": "ruido", "faixa": [10]}, "dois valores"),
    ({"tipo": "band", "opcoes": []}, "vazia"),
    ({"tipo": "ruido", "prob": 1.5, "nivel": 20}, "fora de"),
])
def test_config_invalida_e_recusada(etapa, erro):
    with pytest.raises(ValueError, match=erro):
        Degradacao(_cfg(etapa), SR)


def test_sem_etapas_e_recusada():
    with pytest.raises(ValueError, match="nenhuma etapa"):
        construir({"enabled": True, "etapas": []}, SR)


def test_ruido_no_audio_bruto_chega_as_pausas():
    wav = _fala_com_pausas()
    saida = Degradacao(_cfg({"tipo": "ruido", "prob": 1.0, "nivel": 20}), SR)(
        wav, np.random.default_rng(0))
    assert saida.shape == wav.shape and saida.dtype == np.float32
    assert np.sqrt(np.mean(saida[:SR] ** 2)) > 1e-3        # a pausa agora tem ruído


def test_prob_zero_nao_mexe_e_a_mesma_semente_repete():
    wav = _fala_com_pausas()
    nada = Degradacao(_cfg({"tipo": "ruido", "prob": 0.0, "nivel": 20}), SR)
    np.testing.assert_array_equal(nada(wav, np.random.default_rng(0)), wav)
    d = Degradacao(_cfg({"tipo": "ruido", "prob": 1.0, "faixa": [10, 30]}), SR)
    a, b = d(wav, np.random.default_rng(5)), d(wav, np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, d(wav, np.random.default_rng(6)))


def test_parametro_sorteado_fica_na_faixa_ou_nas_opcoes():
    d = Degradacao(_cfg({"tipo": "opus", "faixa": [0.3, 0.9]},
                        {"tipo": "band", "opcoes": [8000, 11025]}), SR)
    rng = np.random.default_rng(1)
    faixa = [d.etapas[0].parametro(rng) for _ in range(200)]
    opcoes = {d.etapas[1].parametro(rng) for _ in range(200)}
    assert 0.3 <= min(faixa) and max(faixa) <= 0.9
    assert opcoes == {8000.0, 11025.0}
    assert d.descricao() == "opus 100%, band 100%"


def test_degradacoes_de_canal_mantem_o_comprimento():
    wav = _fala_com_pausas()
    d = Degradacao(_cfg({"tipo": "captura_fir", "nivel": 48000}, {"tipo": "opus", "nivel": 0.5},
                        {"tipo": "band", "nivel": 8000}), SR)
    assert d(wav, np.random.default_rng(0)).shape == wav.shape


class _Espia:
    """Registra o que recebe e o primeiro sorteio do rng; devolve o áudio igual."""

    def __init__(self):
        self.recebido, self.sorteios = [], []

    def __call__(self, wav, rng):
        self.recebido.append(wav.copy())
        self.sorteios.append(rng.random())
        return wav


@pytest.fixture
def base(tmp_path):
    config = load_config(RAIZ / "configs" / "baseline_v2.yaml")
    wav = 0.5 * _fala_com_pausas()                     # pico 0,15: não normalizado
    sf.write(tmp_path / "LA_T_0000001.flac", wav, SR)
    (tmp_path / "proto.txt").write_text("LA_0001 LA_T_0000001 - - bonafide\n", encoding="utf-8")
    return config, tmp_path


def _dataset(config, pasta, **kw):
    return ASVspoofDataset(pasta / "proto.txt", pasta, config["audio"],
                           FeatureExtractor(config["audio"], config["features"]),
                           cache_dir=pasta / "cache", **kw)


def test_dataset_aplica_antes_do_preprocessamento(base):
    config, pasta = base
    espia = _Espia()
    ds = _dataset(config, pasta, degradacao=espia)
    ds[0]
    np.testing.assert_array_equal(espia.recebido[0], load_audio(pasta / "LA_T_0000001.flac", SR))
    assert len(espia.recebido[0]) == 4 * SR            # com as pausas, sem cortar
    assert np.abs(espia.recebido[0]).max() < 0.2       # antes da normalização pelo pico
    assert ds.cache is None                            # aleatório por época: sem cache


def test_sorteio_muda_por_epoca_sem_mexer_no_recorte_aleatorio(base):
    config, pasta = base
    espia = _Espia()
    ds = _dataset(config, pasta, degradacao=espia, random_crop=True)
    ds[0]
    ds.set_epoch(1)
    ds[0]
    assert espia.sorteios[0] != espia.sorteios[1]
    # o fluxo do recorte é (semente, época, índice); o da degradação é outro
    assert espia.sorteios[1] != ds._sample_rng(0).random()


def test_dataset_de_fumaca_tambem_aplica(base):
    config, _ = base
    espia = _Espia()
    SmokeDataset(2, config["audio"], FeatureExtractor(config["audio"], config["features"]),
                 degradacao=espia)[1]
    assert len(espia.recebido) == 1


def test_ligar_a_degradacao_nao_muda_o_cache_do_dev(base):
    config, _ = base
    extrator = FeatureExtractor(config["audio"], config["features"])
    com = dict(config["audio"], degradacao=_cfg({"tipo": "ruido", "nivel": 20}))
    assert _config_fingerprint(com, extrator) == _config_fingerprint(config["audio"], extrator)
