"""Testes do encadeador de experimentos."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pipeline import build_steps, collect_results, experiment_name, fmt_dur  # noqa: E402


def _args(**kw):
    base = dict(skip_train=False, skip_check=False, robustness=False, smoke=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_experiment_name_comes_from_config():
    assert experiment_name("configs/fusion_v4.yaml") == "fusion_lcnn_v4"


def test_default_sequence_has_all_stages():
    labels = [l for l, _ in build_steps("c.yaml", "exp", _args())]
    assert labels == ["verificação da base", "treino", "avaliação (eval)", "EER por ataque"]


def test_robustness_is_opt_in():
    sem = [l for l, _ in build_steps("c.yaml", "exp", _args())]
    com = [l for l, _ in build_steps("c.yaml", "exp", _args(robustness=True))]
    assert "robustez" not in sem
    assert "robustez" in com


def test_skip_train_removes_only_training():
    labels = [l for l, _ in build_steps("c.yaml", "exp", _args(skip_train=True))]
    assert "treino" not in labels
    assert "avaliação (eval)" in labels


def test_smoke_skips_data_check_and_propagates_flag():
    steps = build_steps("c.yaml", "exp", _args(smoke=True))
    labels = [l for l, _ in steps]
    # check_data exige o dataset real; não faz sentido no modo sintético.
    assert "verificação da base" not in labels
    for _, cmd in steps:
        assert "--smoke" in cmd


def test_all_stages_point_to_the_same_checkpoint():
    """Se o nome do checkpoint divergir, as análises avaliariam outro modelo."""
    steps = build_steps("c.yaml", "meu_exp", _args())
    usados = {cmd[cmd.index("--checkpoint") + 1] for _, cmd in steps if "--checkpoint" in cmd}
    assert usados == {"checkpoints/meu_exp.pt"}


def test_eval_stage_uses_eval_partition():
    """A tabela final precisa vir do conjunto de teste, não do dev."""
    for label, cmd in build_steps("c.yaml", "exp", _args()):
        if label == "avaliação (eval)":
            assert cmd[cmd.index("--partition") + 1] == "eval"
            return
    raise AssertionError("etapa de avaliação não encontrada")


def test_fmt_dur():
    assert fmt_dur(45) == "45s"
    assert fmt_dur(125) == "2m05s"
    assert fmt_dur(3725) == "1h02m05s"


def test_collect_results_reads_jsons(tmp_path, monkeypatch):
    import scripts.run_pipeline as rp

    monkeypatch.setattr(rp, "OUTPUT_DIR", tmp_path)
    (tmp_path / "exp_eval_metrics.json").write_text(json.dumps({"eer": 0.15, "f1": 0.9}))
    (tmp_path / "exp_eval_per_attack.json").write_text(json.dumps(
        {"global_eer": 0.15,
         "per_attack": {"A07": {"eer": 0.1}, "A12": {"eer": 0.5}, "A13": {"eer": 0.3}}}))

    r = collect_results("exp")
    assert r["eer"] == 0.15
    assert abs(r["eer_medio_por_ataque"] - 0.3) < 1e-9
    assert r["piores_ataques"].startswith("A12=50.0%")   # ordenado do pior para o melhor


def test_collect_results_tolerates_missing_files(tmp_path, monkeypatch):
    """Se uma etapa falhou, o resumo ainda precisa ser gerado."""
    import scripts.run_pipeline as rp

    monkeypatch.setattr(rp, "OUTPUT_DIR", tmp_path)
    assert collect_results("inexistente") == {"experimento": "inexistente"}
