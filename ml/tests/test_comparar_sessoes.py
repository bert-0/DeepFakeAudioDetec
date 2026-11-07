"""Comparação das sessões do teste ao vivo (scripts/comparar_sessoes.py)."""

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.comparar_sessoes as comparar  # noqa: E402
from src.scores import save_scores, scores_path  # noqa: E402

IDS = ["LA_E_1", "LA_E_2", "LA_E_3", "LA_E_4"]
LABELS = [0, 0, 1, 1]


def _sessao(tmp_path, nome, ids, scores, saida):
    pasta = tmp_path / "canal" / nome
    pasta.mkdir(parents=True)
    experimento = f"baseline_lfcc_cnn_v2__canal_real_canal_{nome}"
    cfg = pasta / "config_canal_real.yaml"
    cfg.write_text(yaml.safe_dump({"experiment": {"name": experimento}}), encoding="utf-8")
    labels = [LABELS[IDS.index(i)] for i in ids]
    save_scores(scores_path(saida, experimento, "eval"), ids=ids, labels=labels,
                scores=scores, systems=["-" if lab == 0 else "A10" for lab in labels],
                fingerprint="x", partition="eval", logodds=np.log(np.array(scores) / (1 - np.array(scores))))
    (saida / f"{experimento}_eval_metrics.json").write_text('{"threshold": 0.5}', encoding="utf-8")
    return cfg


def test_compara_so_os_audios_comuns(tmp_path, monkeypatch, capsys):
    saida = tmp_path / "outputs"
    saida.mkdir()
    monkeypatch.setattr(comparar, "OUTPUT_DIR", saida)
    limpo = _sessao(tmp_path, "limpo", IDS, [0.1, 0.2, 0.8, 0.9], saida)
    # A chamada perdeu um áudio no alinhamento e inverteu um bonafide.
    chamada = _sessao(tmp_path, "chamada", IDS[1:], [0.95, 0.7, 0.9], saida)

    csv = tmp_path / "t.csv"
    assert comparar.main([str(limpo), str(chamada), "--csv", str(csv)]) == 0
    out = capsys.readouterr().out
    assert "Áudios em comum: 3" in out
    assert "LA_E_1" not in out.split("Áudios em comum")[1].split("\n", 2)[2]
    linhas = csv.read_text(encoding="utf-8").splitlines()
    assert linhas[0] == "id,rotulo,ataque,limpo,chamada"
    assert len(linhas) == 4
    assert "3/3" in out and "2/3" in out


def test_sessao_sem_evaluate_explica_o_que_rodar(tmp_path, monkeypatch, capsys):
    saida = tmp_path / "outputs"
    saida.mkdir()
    monkeypatch.setattr(comparar, "OUTPUT_DIR", saida)
    pasta = tmp_path / "canal" / "chamada"
    pasta.mkdir(parents=True)
    cfg = pasta / "config_canal_real.yaml"
    cfg.write_text(yaml.safe_dump({"experiment": {"name": "x"}}), encoding="utf-8")
    assert comparar.main([str(cfg)]) == 1
    assert "evaluate.py" in capsys.readouterr().out
