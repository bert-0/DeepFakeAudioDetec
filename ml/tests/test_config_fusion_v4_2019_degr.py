"""Config da outra célula do 2x2: fusion_v4 na 2019 LA com a degradação nova.

Só pode diferir do fusion_v4 no nome, nos workers e na degradação, e a degradação
tem de ser a mesma do fusion_v4_asv5; senão a comparação entre os dois mediria
mais de uma coisa.
"""

import copy
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent


def _carrega(nome):
    return yaml.safe_load((RAIZ / "configs" / nome).read_text(encoding="utf-8"))


def test_difere_do_fusion_v4_so_no_nome_nos_workers_e_na_degradacao():
    base = copy.deepcopy(_carrega("fusion_v4.yaml"))
    cfg = _carrega("fusion_v4_2019_degr.yaml")
    assert cfg["experiment"]["name"] == "fusion_lcnn_v4_2019_degr"
    base["experiment"]["name"] = cfg["experiment"]["name"]
    base["train"]["num_workers"] = 6
    base["audio"]["degradacao"] = cfg["audio"]["degradacao"]
    assert cfg == base


def test_mesma_degradacao_e_mesmo_treino_que_o_do_recorte():
    cfg = _carrega("fusion_v4_2019_degr.yaml")
    asv5 = _carrega("fusion_v4_asv5.yaml")
    assert cfg["audio"] == asv5["audio"]
    assert cfg["features"] == asv5["features"] and cfg["model"] == asv5["model"]
    assert cfg["train"] == asv5["train"]
    # o que muda entre as duas células é só a base de treino e de validação
    assert cfg["data"]["protocols"]["train"] != asv5["data"]["protocols"]["train"]
    assert cfg["data"]["protocols"]["eval"] == asv5["data"]["protocols"]["eval"]
