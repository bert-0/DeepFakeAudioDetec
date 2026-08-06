"""Testes das proteções de memória e de queda do treino.

Contexto: num notebook de 16 GB o treino travava a máquina a ponto de exigir
desligamento no botão. A causa era o loader de dev herdar `num_workers` e
`persistent_workers` do treino — 8 processos de ~512 MB vivos ao mesmo tempo,
4 deles ociosos durante todo o treino.
"""

import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (  # noqa: E402
    RAM_POR_WORKER_GB,
    estimativa_ram_gb,
    memoria_total_gb,
)
from train import aviso_de_memoria, save_checkpoint  # noqa: E402

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


# --------------------------------------------------------------------------- #
# Estimativa de RAM
# --------------------------------------------------------------------------- #
def test_workers_de_dev_entram_no_pico():
    """Regressão: o custo dos workers de dev era invisível justamente por eles
    ficarem ociosos — mas ocupam RAM o tempo todo mesmo assim."""
    com_dev = estimativa_ram_gb(4, 4)["pico"]
    sem_dev = estimativa_ram_gb(4, 0)["pico"]
    assert com_dev - sem_dev == 4 * RAM_POR_WORKER_GB


def test_a_configuracao_antiga_estourava_16gb():
    """4+4 workers: ~8,5 GB. Com navegador aberto, não cabe em 16 GB."""
    assert estimativa_ram_gb(4, 4)["pico"] > 8.0


def test_a_configuracao_nova_cabe_com_folga():
    assert estimativa_ram_gb(2, 0)["pico"] < 6.0


def test_memoria_total_e_plausivel_ou_none():
    total = memoria_total_gb()
    assert total is None or 0.5 < total < 4096


def test_aviso_dispara_quando_nao_cabe(capsys, monkeypatch):
    monkeypatch.setattr("train.memoria_total_gb", lambda: 8.0)
    assert aviso_de_memoria(4, 4) is True
    saida = capsys.readouterr().out
    assert "[AVISO]" in saida
    assert "dev_num_workers" in saida, "o aviso precisa dizer o que ajustar"


def test_aviso_silencioso_quando_cabe(capsys, monkeypatch):
    monkeypatch.setattr("train.memoria_total_gb", lambda: 32.0)
    assert aviso_de_memoria(2, 0) is False
    assert "[AVISO]" not in capsys.readouterr().out


def test_sem_leitura_de_ram_nao_avisa(capsys, monkeypatch):
    """Não dá para avisar sobre o que não se mediu."""
    monkeypatch.setattr("train.memoria_total_gb", lambda: None)
    assert aviso_de_memoria(8, 8) is False
    assert "[AVISO]" not in capsys.readouterr().out


def test_o_pico_sempre_aparece_na_saida(capsys, monkeypatch):
    monkeypatch.setattr("train.memoria_total_gb", lambda: 32.0)
    aviso_de_memoria(3, 1)
    saida = capsys.readouterr().out
    assert "3 treino + 1 dev" in saida
    assert "RAM estimada no pico" in saida


# --------------------------------------------------------------------------- #
# Checkpoint atômico
# --------------------------------------------------------------------------- #
def test_checkpoint_e_gravado_e_relido(tmp_path):
    destino = tmp_path / "m.pt"
    save_checkpoint({"model_state": {"w": torch.ones(3)}, "threshold": 0.5}, destino)
    lido = torch.load(destino, weights_only=False)
    assert torch.equal(lido["model_state"]["w"], torch.ones(3))
    assert lido["threshold"] == 0.5


def test_nenhum_tmp_sobra_apos_gravar(tmp_path):
    save_checkpoint({"a": 1}, tmp_path / "m.pt")
    assert list(tmp_path.iterdir()) == [tmp_path / "m.pt"]


def test_queda_no_meio_da_escrita_preserva_o_checkpoint_anterior(tmp_path, monkeypatch):
    """Regressão: `torch.save` direto no destino deixa um .pt truncado.

    É o cenário real deste projeto — a máquina travava por falta de RAM e o
    usuário desligava no botão. Perder o `best.pt` custa o treino inteiro.
    """
    destino = tmp_path / "m.pt"
    save_checkpoint({"epoca": 1}, destino)

    def morre_no_meio(payload, caminho, *a, **k):
        """Escreve parte dos bytes e some — é o que a queda de energia faz."""
        Path(caminho).write_bytes(b"PK\x03\x04 bytes pela metade")
        raise OSError("a máquina foi desligada no botão")

    monkeypatch.setattr("train.torch.save", morre_no_meio)
    with pytest.raises(OSError):
        save_checkpoint({"epoca": 2}, destino)

    # Sem a gravação atômica, `destino` agora seriam os bytes truncados acima e
    # o torch.load abaixo estouraria — perdendo o treino inteiro.
    assert torch.load(destino, weights_only=False) == {"epoca": 1}, \
        "o checkpoint bom foi destruído por uma escrita interrompida"


def test_cria_o_diretorio_se_faltar(tmp_path):
    destino = tmp_path / "novo" / "sub" / "m.pt"
    save_checkpoint({"a": 1}, destino)
    assert destino.is_file()


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #
def test_configs_v4_cabem_em_16gb():
    """Os dois configs que serão treinados precisam caber na máquina real."""
    for nome in ("fusion_v4.yaml", "attention_v4.yaml"):
        cfg = yaml.safe_load((CONFIGS / nome).read_text(encoding="utf-8"))["train"]
        pico = estimativa_ram_gb(cfg["num_workers"], cfg.get("dev_num_workers", 0))["pico"]
        assert pico <= 16.0 - 2.0, f"{nome}: pico de {pico:.1f} GB não deixa folga"
