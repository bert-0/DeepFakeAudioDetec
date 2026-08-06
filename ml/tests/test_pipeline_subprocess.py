"""Testes do ambiente que o pipeline entrega aos subprocessos.

São testes de *execução*, não de lógica: rodam um Python filho de verdade,
porque o defeito que eles cobrem só existe na fronteira pai/filho.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pipeline import child_env, run_step  # noqa: E402


class LogCronometrado:
    """Arquivo de log falso que anota o instante de chegada de cada linha."""

    def __init__(self):
        self.inicio = time.perf_counter()
        self.linhas: list[tuple[float, str]] = []

    def write(self, texto: str) -> None:
        self.linhas.append((time.perf_counter() - self.inicio, texto))

    def flush(self) -> None:
        pass

    @property
    def texto(self) -> str:
        return "".join(t for _, t in self.linhas)

    def instante_de(self, trecho: str) -> float:
        for t, linha in self.linhas:
            if trecho in linha:
                return t
        raise AssertionError(f"{trecho!r} não apareceu no log:\n{self.texto}")


def _py(codigo: str) -> list[str]:
    return [sys.executable, "-c", codigo]


# --------------------------------------------------------------------------- #
# child_env
# --------------------------------------------------------------------------- #
def test_child_env_define_as_duas_variaveis():
    env = child_env()
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_child_env_preserva_o_resto_do_ambiente(monkeypatch):
    """O filho precisa do PATH, do CUDA_VISIBLE_DEVICES, do conda etc."""
    monkeypatch.setenv("UMA_VARIAVEL_QUALQUER", "valor")
    assert child_env()["UMA_VARIAVEL_QUALQUER"] == "valor"
    assert "PATH" in child_env()


def test_child_env_sobrepoe_um_locale_conflitante(monkeypatch):
    """No Windows o ambiente pode já trazer cp1252 — a nossa escolha vence."""
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    assert child_env()["PYTHONIOENCODING"] == "utf-8"


# --------------------------------------------------------------------------- #
# run_step: o env chega mesmo ao processo filho
# --------------------------------------------------------------------------- #
def test_run_step_entrega_o_env_ao_processo_filho(monkeypatch):
    """Regressão: sem `env=` no Popen o filho herda o padrão do sistema.

    O filho imprime o que ele próprio vê, então o teste falha se alguém
    remover o argumento `env` — nenhum mock envolvido.
    """
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)

    log = LogCronometrado()
    ok, _ = run_step("eco do ambiente", _py(
        "import os,sys;"
        "print('IOENC=' + os.environ.get('PYTHONIOENCODING','<vazio>'));"
        "print('UNBUF=' + os.environ.get('PYTHONUNBUFFERED','<vazio>'));"
        "print('STDOUT=' + sys.stdout.encoding.lower().replace('-',''))"
    ), log)

    assert ok
    assert "IOENC=utf-8" in log.texto
    assert "UNBUF=1" in log.texto
    # É o sys.stdout do filho que decide a codificação da saída, não o Popen.
    assert "STDOUT=utf8" in log.texto


def test_run_step_preserva_acentos_no_log():
    """`época`/`acurácia` precisam sobreviver à viagem até o log.

    Sem PYTHONIOENCODING no Windows o filho emite cp1252, o pai decodifica
    UTF-8 e cada acento vira U+FFFD — perda irreversível, já que o log grava
    o caractere de substituição.
    """
    log = LogCronometrado()
    ok, _ = run_step("acentos", _py(
        "print('Época 3 | dev: acurácia 0,97 — atenção')"
    ), log)

    assert ok
    assert "Época 3 | dev: acurácia 0,97 — atenção" in log.texto
    assert "�" not in log.texto, "houve substituição de caractere"


def test_run_step_transmite_a_saida_progressivamente(monkeypatch):
    """Regressão: um treino de horas não pode ficar mudo até o fim.

    `bufsize=1` no Popen configura o *pai*. Quem decide se o filho segura a
    saída é o buffer dele: com stdout em pipe, o Python usa buffer de bloco
    (~8 KB) e nada sai até o processo terminar.
    """
    monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)

    log = LogCronometrado()
    ok, dur = run_step("saída em conta-gotas", _py(
        "import time\n"
        "for i in range(3):\n"
        "    print('marco', i)\n"
        "    time.sleep(0.6)\n"
    ), log)

    assert ok
    assert dur > 1.5, "o filho precisa mesmo demorar, senão o teste não prova nada"
    # Sem a correção as três linhas chegariam juntas, no fim (~1,8 s).
    assert log.instante_de("marco 0") < 0.4
    assert log.instante_de("marco 2") > log.instante_de("marco 0")


def test_run_step_relata_falha_do_filho():
    log = LogCronometrado()
    ok, _ = run_step("saída 3", _py("raise SystemExit(3)"), log)
    assert not ok
    assert "FALHOU (código 3)" in log.texto


# --------------------------------------------------------------------------- #
# Prova de que o defeito é real: o mesmo filho, sem o env, segura a saída.
# --------------------------------------------------------------------------- #
def test_o_python_realmente_bufferiza_sem_a_variavel(monkeypatch):
    """Sem isto, os testes acima poderiam estar passando por acidente."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
    inicio = time.perf_counter()
    proc = subprocess.Popen(
        _py("import time\nfor i in range(3):\n    print('m', i)\n    time.sleep(0.6)\n"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    primeira = None
    for _ in proc.stdout:
        primeira = time.perf_counter() - inicio
        break
    proc.wait()
    assert primeira is not None and primeira > 1.0, (
        "esperava-se que a primeira linha só chegasse no fim; se este teste "
        "falhar, o ambiente já força saída sem buffer e os testes de "
        "progressividade acima perdem o valor")
