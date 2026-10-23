"""Config do fusion_v4 no recorte da ASVspoof 5 (Incremento 4, desenho 2x2).

Só pode diferir do fusion_v4 no nome, nos workers, nos dados e na degradação; o
cache não pode cair na pasta do da 2019.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess.channel import ocluir_banda_alta  # noqa: E402
from src.preprocess.degradacao import construir  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent


def _carrega(nome):
    return yaml.safe_load((RAIZ / "configs" / nome).read_text(encoding="utf-8"))


def test_difere_do_fusion_v4_so_no_nome_nos_dados_e_na_degradacao():
    base = copy.deepcopy(_carrega("fusion_v4.yaml"))
    cfg = _carrega("fusion_v4_asv5.yaml")
    assert cfg["experiment"]["name"] == "fusion_lcnn_v4_asv5"
    assert cfg["train"]["num_workers"] == 6
    base["experiment"]["name"] = cfg["experiment"]["name"]
    base["train"]["num_workers"] = 6
    base["data"] = cfg["data"]
    base["audio"]["degradacao"] = cfg["audio"]["degradacao"]
    assert cfg == base


def test_treino_e_dev_no_recorte_eval_na_2019_e_cache_proprio():
    d = _carrega("fusion_v4_asv5.yaml")["data"]
    assert d["protocols"]["train"] == "data/ASVspoof5_recorte/train/protocolo.txt"
    assert d["audio_dir"]["dev"] == "data/ASVspoof5_recorte/dev/flac"
    assert d["protocols"]["eval"] == _carrega("fusion_v4.yaml")["data"]["protocols"]["eval"]
    assert d["cache_dir"] == "data/cache_asvspoof5"


def test_degradacao_com_as_etapas_do_plano():
    cfg = _carrega("fusion_v4_asv5.yaml")
    d = construir(cfg["audio"]["degradacao"], cfg["audio"]["sample_rate"])
    assert [e.tipo for e in d.etapas] == ["rawboost", "captura_fir", "codec", "oclusao"]
    oclusao = d.etapas[-1]
    assert (oclusao.prob, oclusao.valor) == (0.5, [4000, 7900])   # o descarte decidido
    x = np.random.default_rng(0).standard_normal(16000 * 3).astype(np.float32) * 0.1
    y = d(x, np.random.default_rng(1))
    assert y.shape == x.shape and np.isfinite(y).all()


def test_cache_dir_do_config_e_usado(monkeypatch, tmp_path):
    from src.data import dataset

    vistos = {}

    class _Falso:
        def __init__(self, **kw):
            vistos.update(kw)

    monkeypatch.setattr(dataset, "ASVspoofDataset", _Falso)
    cfg = _carrega("fusion_v4_asv5.yaml")
    dataset.build_dataset(cfg, "dev", extractor=None, smoke=False)
    assert vistos["cache_dir"] == Path("data/cache_asvspoof5")
    sem = _carrega("fusion_v4.yaml")
    dataset.build_dataset(sem, "dev", extractor=None, smoke=False)
    assert vistos["cache_dir"] == Path("data/cache")              # o de antes


def test_oclusao_em_comprimento_ruim_continua_zerando_acima_do_corte():
    sr = 16000
    t = np.arange(96001) / sr
    x = (np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 7000 * t)).astype(np.float32) * 0.1
    y = ocluir_banda_alta(x, sr, 6000)
    assert y.shape == x.shape
    f = np.fft.rfftfreq(x.size, 1 / sr)
    potencia = np.abs(np.fft.rfft(y)) ** 2
    assert potencia[f >= 6050].sum() / potencia.sum() < 1e-4


def test_oclusao_em_comprimento_rapido_e_a_de_antes():
    x = np.random.default_rng(2).standard_normal(64000)
    espectro = np.fft.rfft(x)
    espectro[np.fft.rfftfreq(64000, 1 / 16000) >= 7600] = 0
    antiga = np.fft.irfft(espectro, n=64000).astype(np.float32)
    np.testing.assert_array_equal(ocluir_banda_alta(x, 16000, 7600), antiga)
