"""Fusão ao vivo (scripts/fusao_ao_vivo.py) e os scores por áudio do robustness_eval."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from scripts.fusao_ao_vivo import (  # noqa: E402
    alinhar, carregar, fundir, limiar_por_alarme_falso, main, ponto_de_operacao, por_que_passam)


def _npz(pasta, nome, scores, labels, limiar, ids=None):
    scores = np.asarray(scores, dtype=float)
    ids = np.array(ids if ids is not None else [f"LA_E_{i:07d}" for i in range(len(scores))])
    caminho = pasta / f"{nome}_amostra10_captura_48k_fir_eval_captura_48k_fir_scores.npz"
    np.savez(caminho, ids=ids, systems=np.array(["-" if l == 0 else "A07" for l in labels]),
             labels=np.asarray(labels), scores=scores,
             logodds=np.log(np.clip(scores, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(scores, 1e-6, 1 - 1e-6)),
             threshold=limiar)
    return caminho


LABELS = [0, 0, 0, 0, 1, 1, 1, 1]
# Cada modelo erra um áudio diferente; a média acerta os dois.
A = [0.1, 0.2, 0.8, 0.1, 0.8, 0.9, 0.95, 0.3]
B = [0.1, 0.1, 0.1, 0.85, 0.9, 0.2, 0.95, 0.9]


def test_fusao_pela_media_corrige_erros_que_os_modelos_nao_dividem(tmp_path):
    r = fundir(carregar(_npz(tmp_path, "baseline_lfcc_cnn_v2", A, LABELS, 0.5)),
               carregar(_npz(tmp_path, "fusion_lcnn_v4", B, LABELS, 0.5)))
    assert r["modelos"]["baseline_lfcc_cnn_v2"]["eer"] > 0
    assert r["modelos"]["fusion_lcnn_v4"]["eer"] > 0
    assert r["fusao"]["eer_media"] == 0.0
    ponto = r["fusao"]["pontos"]["média dos dois limiares"]
    assert ponto["humanos_acima"] == 0.0 and ponto["sinteticos_passam"] == 0.0


def test_ponto_de_operacao():
    p = ponto_de_operacao(np.array(LABELS), np.array(A), 0.5)
    assert p["humanos_acima"] == 0.25 and p["sinteticos_passam"] == 0.25


def test_alinha_pela_identidade_do_audio(tmp_path):
    ids = [f"LA_E_{i:07d}" for i in range(8)]
    a = carregar(_npz(tmp_path, "a", A, LABELS, 0.5, ids))
    ordem = [7, 6, 5, 4, 3, 2, 1, 0]
    b = carregar(_npz(tmp_path, "b", np.array(B)[ordem], np.array(LABELS)[ordem], 0.5,
                      np.array(ids)[ordem]))
    a2, b2 = alinhar(a, b)
    assert (a2["ids"] == b2["ids"]).all() and (a2["labels"] == b2["labels"]).all()
    assert fundir(a, b)["fusao"]["eer_media"] == 0.0


def test_recusa_amostras_diferentes(tmp_path):
    a = carregar(_npz(tmp_path, "a", A, LABELS, 0.5))
    b = carregar(_npz(tmp_path, "b", B, LABELS, 0.5, [f"outro_{i}" for i in range(8)]))
    with pytest.raises(ValueError, match="mesma --amostra"):
        fundir(a, b)


def test_limiar_calibrado_no_dev_e_limiares_pedidos(tmp_path, capsys):
    dev = tmp_path / "dev"
    dev.mkdir()
    da = _npz(dev, "a", A, LABELS, 0.5)
    db = _npz(dev, "b", B, LABELS, 0.5)
    rc = main([str(_npz(tmp_path, "a", A, LABELS, 0.5)), str(_npz(tmp_path, "b", B, LABELS, 0.7)),
               "--dev", str(da), str(db), "--limiar", "0.4", "--json", str(tmp_path / "r.json")])
    saida = capsys.readouterr().out
    assert rc == 0
    assert "calibrado no dev" in saida and "pedido (0.4000)" in saida
    assert "o que o monitor usa hoje" in saida
    assert (tmp_path / "r.json").is_file()


def test_limiar_por_alarme_falso_deixa_no_maximo_o_alvo_de_humanos_acima():
    labels = np.array([0, 0, 0, 0, 1])
    scores = np.array([0.1, 0.45, 0.475, 0.15, 0.9])
    t = limiar_por_alarme_falso(labels, scores, 0.25)
    assert ponto_de_operacao(labels, scores, t)["humanos_acima"] == 0.25
    assert 0.45 < t <= 0.475
    assert ponto_de_operacao(labels, scores, limiar_por_alarme_falso(labels, scores, 0.0))[
        "humanos_acima"] == 0.0


def test_acima_de_meio_a_media_exige_que_os_dois_concordem():
    labels = np.array([0, 1, 1])
    a = np.array([0.0, 1.0, 1.0])
    b = np.array([0.0, 0.0, 0.9])
    # o primeiro sintético, que só o A pega, não chega a 0,6 nem com o A em 1,0
    q = por_que_passam(labels, a, b, 0.5, 0.5, 0.6)
    assert q == {"so_um_pega_e_passa": 0.5, "passam_qualquer_que_seja_o_outro": 0.5}
    assert por_que_passam(labels, a, b, 0.5, 0.5, 0.4)["so_um_pega_e_passa"] == 0.0


def test_mesmo_alarme_falso_no_dev_que_o_primeiro_modelo(tmp_path, capsys):
    dev = tmp_path / "dev"
    dev.mkdir()
    da, db = _npz(dev, "a", A, LABELS, 0.5), _npz(dev, "b", B, LABELS, 0.5)
    a, b = _npz(tmp_path, "a", A, LABELS, 0.5), _npz(tmp_path, "b", B, LABELS, 0.5)
    r = fundir(carregar(a), carregar(b), (carregar(da), carregar(db)), alvos=(0.5,))
    nome = r["mesmo_alarme_falso"]["limiar"]
    # o A deixa 1 de 4 humanos acima de 0,5 no dev; a fusão é ajustada para o mesmo
    assert nome == "alarme falso de a no dev (25.0%)"
    assert r["fusao"]["pontos_dev"][nome]["humanos_acima"] == 0.25
    assert r["fusao"]["pontos_dev"]["alarme falso de 50.0% no dev"]["humanos_acima"] == 0.5
    assert set(r["fusao"]["por_que_passam"]) == set(r["fusao"]["pontos"])

    assert main([str(a), str(b), "--dev", str(da), str(db)]) == 0
    assert "Com o mesmo alarme falso no dev que a" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([str(a), str(b), "--alarme-falso", "0.1"])


def test_robustness_eval_salva_os_scores_por_audio(tmp_path):
    from src.config import load_config
    from src.models import build_model

    config = load_config(RAIZ / "configs" / "baseline_v2.yaml")
    ckpt = tmp_path / "m.pt"
    torch.save({"model_state": build_model(config["model"]).state_dict(), "config": config,
                "threshold": 0.5}, ckpt)
    r = subprocess.run([sys.executable, str(RAIZ / "scripts" / "robustness_eval.py"),
                        "--config", str(RAIZ / "configs" / "baseline_v2.yaml"), "--checkpoint", str(ckpt),
                        "--smoke", "--so", "clean", "captura_48k_fir", "--device", "cpu"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    arquivos = sorted((tmp_path / "outputs").glob("*_scores.npz"))
    assert [a.name.rsplit("_eval_", 1)[1] for a in arquivos] == [
        "captura_48k_fir_scores.npz", "clean_scores.npz"]
    d = np.load(arquivos[1])
    assert len(d["ids"]) == len(d["scores"]) == len(d["labels"]) == len(d["systems"]) > 0
    assert float(d["threshold"]) == 0.5
