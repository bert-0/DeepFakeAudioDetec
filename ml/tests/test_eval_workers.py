"""Todo DataLoader do projeto precisa de `worker_init_fn=seed_worker`.

Sem ele, cada worker abre uma thread BLAS por núcleo e o estágio de dados fica
3,3x mais lento (ver `src/config.py`).
"""

import ast
from pathlib import Path

import pytest

ML = Path(__file__).resolve().parent.parent
SCRIPTS = ["train.py", "evaluate.py", "infer.py",
           "scripts/per_attack_eval.py", "scripts/robustness_eval.py",
           "scripts/score_fusion.py"]


def _dataloaders(caminho: Path):
    """Devolve (linha, {argumentos}) de cada construção de DataLoader."""
    arvore = ast.parse(caminho.read_text(encoding="utf-8"))
    for no in ast.walk(arvore):
        if isinstance(no, ast.Call) and getattr(no.func, "id", None) == "DataLoader":
            yield no.lineno, {kw.arg for kw in no.keywords}


@pytest.mark.parametrize("script", SCRIPTS)
def test_todo_dataloader_limita_threads(script):
    caminho = ML / script
    if not caminho.exists():
        pytest.skip(f"{script} não existe")
    faltando = [linha for linha, args in _dataloaders(caminho)
                if "worker_init_fn" not in args]
    assert not faltando, (
        f"{script}: DataLoader sem worker_init_fn na(s) linha(s) {faltando} — "
        "os workers vão disputar CPU entre si")


def test_o_teste_enxerga_algum_dataloader():
    """Guarda contra o teste virar vacuamente verdadeiro."""
    total = sum(len(list(_dataloaders(ML / s))) for s in SCRIPTS if (ML / s).exists())
    assert total >= 6, f"esperava encontrar vários DataLoaders, achei {total}"
