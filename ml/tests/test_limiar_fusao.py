"""Limiar da fusão escolhido nos devs (scripts/limiar_fusao.py)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from scripts.limiar_fusao import (  # noqa: E402
    MODELOS_PADRAO, caminho_scores, comandos, escolher, main, preparar, rotulo_do_dev)
from src.config import load_config  # noqa: E402
from src.models import build_model  # noqa: E402
from src.scores import checkpoint_fingerprint  # noqa: E402


def _dados(scores, labels, prefixo="LA_D_"):
    scores = np.asarray(scores, dtype=float)
    p = np.clip(scores, 1e-6, 1 - 1e-6)
    return {"ids": np.array([f"{prefixo}{i:07d}" for i in range(len(scores))]),
            "systems": np.array(["-" if x == 0 else "A01" for x in labels]),
            "labels": np.asarray(labels), "scores": scores,
            "logodds": np.log(p) - np.log1p(-p), "threshold": np.array(np.nan)}


def test_rotulos_dos_devs_sem_dev_no_nome():
    assert rotulo_do_dev("data/LA/ASVspoof2019_LA_cm_protocols/x.dev.trl.txt") == "calib_2019"
    assert rotulo_do_dev("data/ASVspoof5_recorte/dev/protocolo.txt") == "calib_asv5"
    assert rotulo_do_dev("data/outra/protocolo_dev.txt") == "calib_protocolo-dev"


def test_caminho_dos_scores_igual_ao_do_robustness_eval():
    assert caminho_scores("m__calib_2019", 10000, "captura_48k_fir") == Path(
        "outputs/m__calib_2019_amostra10000_captura_48k_fir_eval_captura_48k_fir_scores.npz")


def test_preparar_avalia_cada_modelo_nos_dois_devs(tmp_path):
    preparo = preparar(MODELOS_PADRAO, tmp_path)
    assert set(preparo["devs"]) == {"calib_2019", "calib_asv5"}
    devs = {"calib_2019": load_config(MODELOS_PADRAO[0][0])["data"],
            "calib_asv5": load_config(MODELOS_PADRAO[1][0])["data"]}
    for rotulo, d in preparo["devs"].items():
        assert [e["checkpoint"] for e in d["execucoes"]] == [k for _, k in MODELOS_PADRAO]
        for e in d["execucoes"]:
            cfg = yaml.safe_load(Path(e["config"]).read_text(encoding="utf-8"))
            assert cfg["data"]["protocols"]["eval"] == devs[rotulo]["protocols"]["dev"]
            assert cfg["data"]["audio_dir"]["eval"] == devs[rotulo]["audio_dir"]["dev"]
            assert cfg["train"]["cache_features"] is False
            assert cfg["experiment"]["name"].endswith("__" + rotulo)
            assert Path(e["scores"]).name.startswith(cfg["experiment"]["name"] + "_amostra10000")
    assert json.loads((tmp_path / "preparo.json").read_text(encoding="utf-8")) == preparo
    linhas = comandos(preparo)
    assert len(linhas) == 4
    assert all("--amostra 10000 --so captura_48k_fir" in c for c in linhas)


def test_limiar_com_os_humanos_dos_dois_devs():
    rng = np.random.default_rng(0)
    pares = {}
    for rotulo, n_h, desloc in (("calib_2019", 300, 0.0), ("calib_asv5", 700, 0.2)):
        labels = np.r_[np.zeros(n_h, int), np.ones(500, int)]
        base = np.r_[rng.uniform(0, 0.6, n_h) + desloc, rng.uniform(0.4, 1.0, 500)].clip(0, 1)
        a = _dados(base, labels, rotulo)
        b = _dados((base + rng.normal(0, 0.05, len(base))).clip(0, 1), labels, rotulo)
        pares[rotulo] = (a, b)
    r = escolher(pares, 0.10)
    assert r["humanos"] == 1000
    assert r["humanos_acima"] <= 0.10 and r["humanos_acima"] > 0.09
    # o dev com humanos mais altos fica acima do alvo; o outro, abaixo
    assert r["devs"]["calib_asv5"]["humanos_acima"] > 0.10 > r["devs"]["calib_2019"]["humanos_acima"]
    assert set(r["devs"]["calib_2019"]) >= {"eer_modelos", "eer_fusao", "sinteticos_passam"}


def test_logodds_usam_o_modelo_de_probabilidades_minusculas():
    """O caso do modelo do recorte: separa bem, mas na escala de 1e-5 a 1e-2."""
    rng = np.random.default_rng(1)
    labels = np.r_[np.zeros(400, int), np.ones(400, int)]
    a = _dados(rng.uniform(0.01, 0.6, 800), labels)                     # não separa
    b = _dados(np.r_[10 ** rng.uniform(-6, -5, 400), 10 ** rng.uniform(-3, -2, 400)], labels)
    prob = escolher({"calib_2019": (a, b)}, 0.10, "probabilidade")
    lo = escolher({"calib_2019": (a, b)}, 0.10, "logodds")
    assert prob["devs"]["calib_2019"]["eer_fusao"] > 0.3
    assert lo["devs"]["calib_2019"]["eer_fusao"] < 0.15
    assert lo["regra"] == "logodds" and lo["limiar"] < 0
    assert lo["limiar_probabilidade"] == pytest.approx(1 / (1 + np.exp(-lo["limiar"])))
    assert prob["limiar_probabilidade"] == prob["limiar"]


def test_recusa_rotulos_diferentes():
    a = _dados([0.1, 0.9], [0, 1])
    b = _dados([0.1, 0.9], [1, 0])
    with pytest.raises(ValueError, match="rótulos diferentes"):
        escolher({"calib_2019": (a, b)}, 0.1)


def _checkpoint(caminho, seed):
    config = load_config("configs/baseline.yaml")
    torch.manual_seed(seed)
    estado = build_model(config["model"]).state_dict()
    torch.save({"model_state": estado, "config": config, "threshold": 0.5}, caminho)
    return checkpoint_fingerprint(estado)


def test_escolher_grava_o_limiar_com_a_impressao_dos_pesos(tmp_path, capsys):
    impressoes = [_checkpoint(tmp_path / "a.pt", 0), _checkpoint(tmp_path / "b.pt", 1)]
    labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    preparo = {"amostra": 10, "condicao": "captura_48k_fir",
               "modelos": [{"config": "configs/baseline.yaml", "checkpoint": str(tmp_path / k)}
                           for k in ("a.pt", "b.pt")], "devs": {}}
    for rotulo in ("calib_2019", "calib_asv5"):
        execucoes = []
        for k, desloc in (("a", 0.0), ("b", 0.05)):
            caminho = tmp_path / f"{k}_{rotulo}.npz"
            np.savez(caminho, **_dados(np.r_[np.linspace(0.05, 0.5, 10), [0.9, 0.95]] + desloc,
                                       labels))
            execucoes.append({"config": "-", "checkpoint": str(tmp_path / f"{k}.pt"),
                              "scores": str(caminho)})
        preparo["devs"][rotulo] = {"execucoes": execucoes}
    (tmp_path / "preparo.json").write_text(json.dumps(preparo), encoding="utf-8")

    saida = tmp_path / "limiar.json"
    assert main(["escolher", "--pasta", str(tmp_path), "--saida", str(saida)]) == 0
    r = json.loads(saida.read_text(encoding="utf-8"))
    assert [m["impressao"] for m in r["modelos"]] == impressoes
    assert r["alvo"] == 0.10 and r["humanos_acima"] == 0.10 and r["condicao"] == "captura_48k_fir"
    assert r["regra"] == "logodds", "a regra escolhida nos devs em 24/10/2026"
    assert "Limiar da fusão" in capsys.readouterr().out


def test_escolher_sem_os_scores_diz_quais_faltam(tmp_path, capsys):
    # Os scores apontam para tmp_path: os de outputs/ podem existir (calibração real).
    preparo = preparar(MODELOS_PADRAO, tmp_path)
    for d in preparo["devs"].values():
        for e in d["execucoes"]:
            e["scores"] = str(tmp_path / Path(e["scores"]).name)
    (tmp_path / "preparo.json").write_text(json.dumps(preparo), encoding="utf-8")
    saida = tmp_path / "limiar.json"
    assert main(["escolher", "--pasta", str(tmp_path), "--saida", str(saida)]) == 1
    assert capsys.readouterr().out.count("_scores.npz") == 4
    assert not saida.exists()
