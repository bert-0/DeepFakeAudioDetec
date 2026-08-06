"""Testes da consolidação de resultados para o relatório.

O risco aqui não é o script quebrar — é ele montar uma tabela *plausível* com o
número errado. Um EER trocado de coluna, um experimento faltando em silêncio ou
um `_` não escapado que quebra o LaTeX só aparecem na leitura do documento.
"""

import json
import sys

import pytest

from scripts.make_report import (
    as_latex,
    as_markdown,
    discover,
    duracao,
    tabela_comparativa,
    tabela_por_ataque,
    tabela_robustez,
    training_summary,
)


def escrever(outputs, nome, partition="eval", *, metrics=None, per_attack=None,
             robustness=None, history=None):
    outputs.mkdir(parents=True, exist_ok=True)
    def dump(sufixo, dados):
        if dados is not None:
            (outputs / f"{nome}{sufixo}").write_text(json.dumps(dados), encoding="utf-8")
    dump(f"_{partition}_metrics.json", metrics if metrics is not None else {"eer": 0.1})
    dump(f"_{partition}_per_attack.json", per_attack)
    dump(f"_{partition}_robustness.json", robustness)
    dump("_history.json", history)


# --------------------------------------------------------------------------- #
# Descoberta
# --------------------------------------------------------------------------- #
def test_discover_finds_experiments(tmp_path):
    escrever(tmp_path, "baseline_v4", metrics={"eer": 0.18, "f1": 0.9})
    escrever(tmp_path, "fusion_v4", metrics={"eer": 0.15, "f1": 0.92})
    achados = discover(tmp_path, "eval", None, include_smoke=False)
    assert [e["nome"] for e in achados] == ["baseline_v4", "fusion_v4"]


def test_discover_skips_smoke_runs(tmp_path):
    """Um smoke de 30 s não pode entrar na tabela de resultados do TCC."""
    escrever(tmp_path, "baseline_v4")
    escrever(tmp_path, "baseline_v4_smoke")
    assert [e["nome"] for e in discover(tmp_path, "eval", None, False)] == ["baseline_v4"]
    assert len(discover(tmp_path, "eval", None, True)) == 2


def test_discover_filters_by_name(tmp_path):
    escrever(tmp_path, "baseline_v3")
    escrever(tmp_path, "baseline_v4")
    escrever(tmp_path, "fusion_v4")
    achados = discover(tmp_path, "eval", ["v4"], False)
    assert [e["nome"] for e in achados] == ["baseline_v4", "fusion_v4"]


def test_discover_separates_partitions(tmp_path):
    escrever(tmp_path, "exp", partition="eval", metrics={"eer": 0.2})
    escrever(tmp_path, "exp", partition="dev", metrics={"eer": 0.05})
    assert discover(tmp_path, "dev", None, False)[0]["metrics"]["eer"] == 0.05
    assert discover(tmp_path, "eval", None, False)[0]["metrics"]["eer"] == 0.2


def test_discover_survives_corrupt_json(tmp_path):
    escrever(tmp_path, "exp")
    (tmp_path / "exp_eval_per_attack.json").write_text("{ quebrado", encoding="utf-8")
    achados = discover(tmp_path, "eval", None, False)
    assert achados[0]["per_attack"] is None


# --------------------------------------------------------------------------- #
# Resumo do treino
# --------------------------------------------------------------------------- #
def test_training_summary_picks_the_best_epoch():
    history = [{"epoch": 1, "dev_eer": 0.30, "t_epoch": 100},
               {"epoch": 2, "dev_eer": 0.12, "t_epoch": 90},
               {"epoch": 3, "dev_eer": 0.20, "t_epoch": 90}]
    resumo = training_summary(history)
    assert resumo == {"epocas": 3, "melhor_epoca": 2, "tempo_s": 280}


def test_training_summary_ignores_nan_eer():
    """Uma época que divergiu (NaN) não pode ser eleita a melhor."""
    history = [{"epoch": 1, "dev_eer": 0.4}, {"epoch": 2, "dev_eer": float("nan")}]
    assert training_summary(history)["melhor_epoca"] == 1


def test_training_summary_handles_history_without_timings():
    """Históricos de treinos anteriores à instrumentação ainda precisam servir."""
    resumo = training_summary([{"epoch": 1, "dev_eer": 0.4}])
    assert resumo["epocas"] == 1
    assert "tempo_s" not in resumo


def test_training_summary_of_nothing():
    assert training_summary(None) == {}


@pytest.mark.parametrize("segundos,esperado", [
    (0, "—"), (None, "—"), (90, "1m"), (3600, "1h00m"), (5400, "1h30m")])
def test_duracao(segundos, esperado):
    assert duracao(segundos) == esperado


# --------------------------------------------------------------------------- #
# Tabelas
# --------------------------------------------------------------------------- #
def test_tabela_comparativa_converts_eer_to_percent():
    exp = [{"nome": "v4", "metrics": {"eer": 0.1584, "f1": 0.9123}, "history": None}]
    _, linhas = tabela_comparativa(exp)
    assert linhas[0][1] == "15.84"
    assert linhas[0][5] == "0.9123"


def test_tabela_comparativa_marks_missing_values():
    exp = [{"nome": "v4", "metrics": {}, "history": None}]
    _, linhas = tabela_comparativa(exp)
    assert linhas[0][1:] == ["—"] * 9


def test_tabela_por_ataque_unions_attacks_across_experiments():
    """Modelos avaliados em conjuntos diferentes de ataques ainda comparam."""
    exp = [
        {"nome": "a", "per_attack": {"global_eer": 0.2,
                                     "per_attack": {"A07": {"eer": 0.1}}}},
        {"nome": "b", "per_attack": {"global_eer": 0.3,
                                     "per_attack": {"A10": {"eer": 0.5}}}},
    ]
    cab, linhas = tabela_por_ataque(exp)
    assert cab == ["Ataque", "a", "b"]
    assert linhas[0] == ["A07", "10.00", "—"]
    assert linhas[1] == ["A10", "—", "50.00"]
    assert linhas[-1] == ["GLOBAL", "20.00", "30.00"]


def test_tabela_por_ataque_is_none_without_data():
    assert tabela_por_ataque([{"nome": "a", "per_attack": None}]) == (None, None)


def test_tabela_robustez_keeps_condition_order():
    """A ordem das condições é a do experimento (clean primeiro), não alfabética."""
    exp = [{"nome": "a", "robustness": {"clean": {"eer": 0.1},
                                        "noise_10dB": {"eer": 0.3}}},
           {"nome": "b", "robustness": {"clean": {"eer": 0.2},
                                        "gain_+6dB": {"eer": 0.25}}}]
    cab, linhas = tabela_robustez(exp)
    assert cab == ["Condição", "a", "b"]
    assert [l[0] for l in linhas] == ["clean", "noise_10dB", "gain_+6dB"]
    assert linhas[1] == ["noise_10dB", "30.00", "—"]


# --------------------------------------------------------------------------- #
# Formatos
# --------------------------------------------------------------------------- #
def test_markdown_has_a_separator_row():
    saida = as_markdown(["A", "B"], [["1", "2"]])
    linhas = saida.strip().split("\n")
    assert len(linhas) == 3
    assert set(linhas[1]) <= {"|", "-"}


def test_markdown_of_an_empty_table():
    assert "Experimento" in as_markdown(["Experimento"], [])


def test_latex_escapes_underscores():
    """`baseline_lcnn_v4` sem escape vira subscrito e quebra a compilação."""
    saida = as_latex(["Experimento"], [["baseline_lcnn_v4"]], "legenda", "tab:x")
    assert r"baseline\_lcnn\_v4" in saida
    assert "baseline_lcnn_v4" not in saida


def test_latex_escapes_percent_in_header():
    saida = as_latex(["EER (%)"], [["15.84"]], "legenda", "tab:x")
    assert r"EER (\%)" in saida


def test_latex_converts_the_empty_marker():
    """O travessão Unicode não compila em LaTeX pdflatex sem pacote extra."""
    saida = as_latex(["A"], [["—"]], "legenda", "tab:x")
    assert "—" not in saida
    assert "--" in saida


def test_latex_column_alignment_matches_header():
    saida = as_latex(["A", "B", "C"], [["1", "2", "3"]], "legenda", "tab:x")
    assert r"\begin{tabular}{lrr}" in saida


def test_latex_carries_caption_and_label():
    saida = as_latex(["A"], [["1"]], "Minha legenda", "tab:comparativo")
    assert r"\caption{Minha legenda}" in saida
    assert r"\label{tab:comparativo}" in saida


# --------------------------------------------------------------------------- #
# Ponta a ponta
# --------------------------------------------------------------------------- #
def test_main_writes_every_format(tmp_path, monkeypatch):
    from scripts import make_report

    escrever(tmp_path, "baseline_v4",
             metrics={"eer": 0.18, "accuracy": 0.9, "precision": 0.9,
                      "recall": 0.9, "f1": 0.9, "threshold": 0.5},
             per_attack={"global_eer": 0.18, "per_attack": {"A07": {"eer": 0.1}}},
             history=[{"epoch": 1, "dev_eer": 0.2, "train_loss": 0.5, "t_epoch": 60}])
    escrever(tmp_path, "fusion_v4", metrics={"eer": 0.15})

    monkeypatch.setattr(sys, "argv", ["make_report.py", "--outputs", str(tmp_path)])
    assert make_report.main() == 0

    dest = tmp_path / "report"
    for esperado in ("comparativo_eval.md", "comparativo_eval.tex",
                     "comparativo_eval.csv", "por_ataque_eval.md",
                     "curvas_comparadas.png", "eer_por_ataque.png"):
        assert (dest / esperado).exists(), esperado

    # Melhor EER primeiro: é a ordem de leitura da tabela no texto.
    md = (dest / "comparativo_eval.md").read_text(encoding="utf-8")
    assert md.index("fusion_v4") < md.index("baseline_v4")


def test_main_reports_when_there_is_nothing(tmp_path, monkeypatch, capsys):
    from scripts import make_report

    monkeypatch.setattr(sys, "argv", ["make_report.py", "--outputs", str(tmp_path)])
    assert make_report.main() == 1
    assert "Nenhum experimento" in capsys.readouterr().out
