"""Todo config precisa das chaves que os scripts leem sem `.get()`.

Motivo: uma edição de config removeu `train.device` por acidente. Os 278 testes
unitários passaram e só o smoke ponta a ponta pegou — `KeyError: 'device'` na
linha 290 do train.py, depois de todo o carregamento de dados.
"""

from pathlib import Path

import pytest
import yaml

CONFIGS = sorted((Path(__file__).resolve().parent.parent / "configs").glob("*.yaml"))

# Lidas com indexação direta em train.py / evaluate.py — a falta derruba a execução.
TRAIN_OBRIGATORIAS = {"epochs", "batch_size", "lr", "weight_decay",
                      "num_workers", "device"}
BLOCOS_OBRIGATORIOS = {"experiment", "audio", "features", "model", "train"}


def _carrega(caminho):
    return yaml.safe_load(caminho.read_text(encoding="utf-8"))


def test_ha_configs_para_verificar():
    assert len(CONFIGS) >= 10, f"achei só {len(CONFIGS)} configs"


@pytest.mark.parametrize("caminho", CONFIGS, ids=lambda p: p.name)
def test_blocos_de_topo(caminho):
    faltando = BLOCOS_OBRIGATORIOS - set(_carrega(caminho))
    assert not faltando, f"{caminho.name}: faltam os blocos {faltando}"


@pytest.mark.parametrize("caminho", CONFIGS, ids=lambda p: p.name)
def test_chaves_de_train(caminho):
    faltando = TRAIN_OBRIGATORIAS - set(_carrega(caminho)["train"])
    assert not faltando, f"{caminho.name}: faltam em `train:` {faltando}"


@pytest.mark.parametrize("caminho", CONFIGS, ids=lambda p: p.name)
def test_features_declaradas_tem_bloco_proprio(caminho):
    cfg = _carrega(caminho)
    for tipo in cfg["features"]["types"]:
        assert tipo in cfg["features"], \
            f"{caminho.name}: '{tipo}' está em types mas não tem bloco de config"


@pytest.mark.parametrize("caminho", CONFIGS, ids=lambda p: p.name)
def test_workers_nao_negativos(caminho):
    train = _carrega(caminho)["train"]
    assert train["num_workers"] >= 0
    assert train.get("dev_num_workers", 0) >= 0
