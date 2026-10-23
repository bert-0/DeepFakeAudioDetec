"""Treino curto da banda alta (RESUMO_TCC §10.2.11): LFCC com f_min/f_max, a
oclusão no gancho e os três configs, que só podem diferir do baseline_v2 no que
cada um testa.
"""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.lfcc import compute_lfcc, linear_filterbank  # noqa: E402
from src.preprocess.degradacao import Degradacao  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
SR = 16000
CFG_LFCC = {"n_lfcc": 20, "n_fft": 512, "win_length": 400, "hop_length": 160,
            "n_filter": 20, "deltas": True}


def _carrega(nome):
    return yaml.safe_load((RAIZ / "configs" / nome).read_text(encoding="utf-8"))


def test_sem_f_max_o_banco_e_o_de_antes():
    """A fórmula antiga, 0 a Nyquist: os configs atuais não mudam nem a chave do cache."""
    n_filter, n_fft = 20, 512
    bins = np.linspace(0.0, SR / 2, n_fft // 2 + 1)
    pts = np.linspace(0.0, SR / 2, n_filter + 2)
    esq, c, d = pts[:-2, None], pts[1:-1, None], pts[2:, None]
    antigo = np.clip(np.minimum((bins - esq) / (c - esq), (d - bins) / (d - c)), 0, None)
    np.testing.assert_allclose(linear_filterbank(n_filter, n_fft, SR), antigo, atol=1e-6)


def test_f_max_zera_os_filtros_acima_do_corte():
    fb = linear_filterbank(20, 512, SR, 0.0, 4000.0)
    bins = np.linspace(0.0, SR / 2, 257)
    assert fb.shape == (20, 257)
    assert np.all(fb[:, bins > 4000] == 0)
    assert np.all(fb.sum(axis=1) > 0)          # nenhum filtro vazio


@pytest.mark.parametrize("f_min, f_max", [(4000, 4000), (5000, 4000), (-1, 4000), (0, 9000)])
def test_faixa_invalida(f_min, f_max):
    with pytest.raises(ValueError, match="f_min"):
        linear_filterbank(20, 512, SR, float(f_min), float(f_max))


def test_lfcc_ate_4_khz_nao_ve_um_tom_de_6_khz():
    rng = np.random.default_rng(0)
    t = np.arange(2 * SR) / SR
    base = (0.1 * np.sin(2 * np.pi * 500 * t) + 0.01 * rng.standard_normal(t.size)).astype(np.float32)
    com_tom = base + 0.1 * np.sin(2 * np.pi * 6000 * t).astype(np.float32)
    ate_4k = dict(CFG_LFCC, f_max=4000)
    assert compute_lfcc(base, SR, ate_4k).shape == compute_lfcc(base, SR, CFG_LFCC).shape
    # Fora as pontas: o tom começa e termina de golpe no arquivo, o que espalha
    # energia por todas as frequências nos quadros da borda (e os deltas a levam
    # alguns quadros adiante).
    miolo = slice(10, -10)
    np.testing.assert_allclose(compute_lfcc(com_tom, SR, ate_4k)[:, miolo],
                               compute_lfcc(base, SR, ate_4k)[:, miolo], atol=1e-3)
    assert np.abs(compute_lfcc(com_tom, SR, CFG_LFCC) - compute_lfcc(base, SR, CFG_LFCC)).max() > 0.1


def test_oclusao_no_gancho_zera_a_banda_alta():
    t = np.arange(SR) / SR
    x = (np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 7000 * t)).astype(np.float32) * 0.1
    d = Degradacao({"enabled": True, "etapas": [{"tipo": "oclusao", "prob": 1.0, "faixa": [4000, 4001]}]}, SR)
    y = d(x, np.random.default_rng(0))
    f = np.fft.rfftfreq(SR, 1 / SR)
    Y = np.abs(np.fft.rfft(y))
    assert Y[f >= 4001].max() < 1e-3 * Y.max()


# Cada config só pode mudar estas chaves em relação ao baseline_v2.
VARIANTES = {
    "banda_controle.yaml": {},
    "banda_lfcc4k.yaml": {("features", "lfcc", "f_max"): 4000},
    "banda_descarte.yaml": {("audio", "degradacao"): {
        "enabled": True, "etapas": [{"tipo": "oclusao", "prob": 0.5, "faixa": [4000, 7900]}]}},
}


@pytest.mark.parametrize("nome", sorted(VARIANTES))
def test_config_difere_do_baseline_v2_so_no_que_testa(nome):
    base = copy.deepcopy(_carrega("baseline_v2.yaml"))
    cfg = _carrega(nome)
    assert cfg["experiment"]["name"] == nome.removesuffix(".yaml") + "_v2"
    assert cfg["train"]["epochs"] == 10
    base["experiment"]["name"] = cfg["experiment"]["name"]
    base["train"]["epochs"] = 10
    for caminho, valor in VARIANTES[nome].items():
        alvo = base
        for chave in caminho[:-1]:
            alvo = alvo[chave]
        alvo[caminho[-1]] = valor
    assert cfg == base


def test_nenhum_config_repete_o_nome_de_experimento():
    """O nome vira o do checkpoint: repetir sobrescreveria um modelo treinado."""
    nomes = [yaml.safe_load(p.read_text(encoding="utf-8"))["experiment"]["name"]
             for p in sorted((RAIZ / "configs").glob("*.yaml"))]
    repetidos = {n for n in nomes if nomes.count(n) > 1}
    assert not repetidos, repetidos
