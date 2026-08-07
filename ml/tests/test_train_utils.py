"""Testes de utilidades do script de treino."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from train import archive_previous_checkpoints, class_weights_from  # noqa: E402


def test_archive_renames_existing_checkpoint(tmp_path):
    """Um treino novo não pode destruir o melhor modelo do treino anterior."""
    ckpt = tmp_path / "exp.pt"
    ckpt.write_text("modelo-bom")

    archive_previous_checkpoints(ckpt)

    assert not ckpt.exists()
    assert (tmp_path / "exp_prev.pt").read_text() == "modelo-bom"


def test_archive_is_noop_when_missing(tmp_path):
    archive_previous_checkpoints(tmp_path / "inexistente.pt")  # não deve levantar


def test_archive_overwrites_older_backup(tmp_path):
    """O backup guarda a execução imediatamente anterior, não acumula arquivos."""
    ckpt = tmp_path / "exp.pt"
    (tmp_path / "exp_prev.pt").write_text("antigo")
    ckpt.write_text("recente")

    archive_previous_checkpoints(ckpt)

    assert (tmp_path / "exp_prev.pt").read_text() == "recente"


def test_archive_handles_multiple_paths(tmp_path):
    best = tmp_path / "exp.pt"
    last = tmp_path / "exp_last.pt"
    best.write_text("b")
    last.write_text("l")

    archive_previous_checkpoints(best, last)

    assert (tmp_path / "exp_prev.pt").read_text() == "b"
    assert (tmp_path / "exp_last_prev.pt").read_text() == "l"


def test_class_weights_auto_is_inverse_frequency():
    import torch

    labels = [0] * 10 + [1] * 30  # 25% bonafide, 75% spoof
    w = class_weights_from(labels, 2, torch.device("cpu"), mode="auto")
    assert abs(float(w[0]) - 2.0) < 1e-6      # 40 / (2*10)
    assert abs(float(w[1]) - 0.6667) < 1e-3   # 40 / (2*30)


def test_class_weights_sqrt_is_milder():
    import math

    import torch

    labels = [0] * 10 + [1] * 30
    auto = class_weights_from(labels, 2, torch.device("cpu"), mode="auto")
    sqrt = class_weights_from(labels, 2, torch.device("cpu"), mode="sqrt")
    assert abs(float(sqrt[0]) - math.sqrt(float(auto[0]))) < 1e-5
    # A razão entre as classes fica menor (compensação mais suave).
    assert float(sqrt[0]) / float(sqrt[1]) < float(auto[0]) / float(auto[1])


# --------------------------------------------------------------------------- #
# Cronômetro por época
#
# A separação entre "espera por dados" e "cálculo" é o que decide qual
# otimização vale a pena. Se ela medir errado, aponta para o lado errado.
# --------------------------------------------------------------------------- #
def test_timer_separates_waiting_from_computing():
    import time

    from train import EpochTimer

    ESPERA, CALCULO, N = 0.05, 0.10, 3

    def loader_lento():
        for _ in range(N):
            time.sleep(ESPERA)   # tempo do "DataLoader"
            yield "lote"

    cronometro = EpochTimer()
    for _ in cronometro.batches(loader_lento()):
        with cronometro.medindo("calculo"):
            time.sleep(CALCULO)  # tempo da "GPU"

    # Limites só por baixo, com folga por cima: `time.sleep` garante dormir *no
    # mínimo* o pedido, e no Windows a granularidade do timer é de ~15 ms. Uma
    # tolerância simétrica apertada torna o teste intermitente sob carga — o que
    # de fato aconteceu aqui. O que importa é a *separação* das duas parcelas.
    assert cronometro.dados >= N * ESPERA * 0.9
    assert cronometro.calculo >= N * CALCULO * 0.9
    assert cronometro.dados < N * ESPERA + 0.5
    assert cronometro.calculo < N * CALCULO + 0.5
    # A relação entre elas é o que o cronômetro existe para mostrar.
    assert cronometro.calculo > cronometro.dados


def test_timer_records_time_of_skipped_batches():
    """`continue` dentro do bloco medido ainda precisa fechar a medição."""
    import time

    from train import EpochTimer

    cronometro = EpochTimer()
    for _ in cronometro.batches(range(3)):
        with cronometro.medindo("calculo"):
            time.sleep(0.01)
            continue   # imita o descarte de um lote com loss NaN

    assert cronometro.calculo > 0.02


def test_timer_history_fields():
    from train import EpochTimer

    campos = EpochTimer().as_dict()
    assert set(campos) == {"t_epoch", "t_data", "t_compute", "t_dev"}


def test_time_report_points_at_the_data_stage(capsys):
    from train import print_time_report

    history = [{"t_epoch": 100, "t_data": 70, "t_compute": 20, "t_dev": 10}] * 3
    print_time_report(history, num_workers=2)
    saida = capsys.readouterr().out
    assert "espera por dados" in saida
    assert "num_workers" in saida


def test_time_report_points_at_the_gpu(capsys):
    from train import print_time_report

    history = [{"t_epoch": 100, "t_data": 5, "t_compute": 85, "t_dev": 10}] * 3
    print_time_report(history, num_workers=2)
    saida = capsys.readouterr().out
    assert "GPU" in saida
    assert "não vai ajudar" in saida


def test_time_report_ignores_the_first_epoch(capsys):
    """A 1ª época enche o cache e mede algoritmos do cuDNN — não é o regime."""
    from train import print_time_report

    history = [{"t_epoch": 900, "t_data": 880, "t_compute": 15, "t_dev": 5},
               {"t_epoch": 100, "t_data": 5, "t_compute": 85, "t_dev": 10},
               {"t_epoch": 100, "t_data": 5, "t_compute": 85, "t_dev": 10}]
    print_time_report(history, num_workers=4)
    saida = capsys.readouterr().out
    assert "espera por dados" not in saida, "a 1ª época contaminou a média"


def test_time_report_survives_a_single_epoch(capsys):
    from train import print_time_report

    print_time_report([{"t_epoch": 10, "t_data": 3, "t_compute": 5, "t_dev": 2}], 4)
    assert "Tempo de treino" in capsys.readouterr().out
