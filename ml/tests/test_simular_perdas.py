"""Simulação de descontinuidades (scripts/simular_perdas.py)."""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.simular_perdas import main, remover_trechos  # noqa: E402

SR = 16000


def test_remove_exatamente_o_pedido_longe_das_pontas():
    wav = np.arange(SR * 3, dtype=np.float32)
    saida = remover_trechos(wav, SR, n=3, ms=10, rng=np.random.default_rng(0))
    assert len(wav) - len(saida) == 3 * 160
    faltando = np.setdiff1d(wav, saida)
    assert faltando.min() >= 0.1 * len(wav) and faltando.max() < 0.9 * len(wav)
    assert np.all(np.diff(saida) > 0), "a ordem das amostras restantes se mantém"


def test_zero_perdas_nao_mexe():
    wav = np.random.default_rng(1).standard_normal(SR).astype(np.float32)
    np.testing.assert_array_equal(remover_trechos(wav, SR, 0, 10, np.random.default_rng(0)), wav)


def test_gera_sessao_avaliavel(tmp_path, capsys):
    origem = tmp_path / "canal" / "limpo"
    (origem / "capturado").mkdir(parents=True)
    linhas = []
    for i, rot in enumerate(["bonafide", "spoof"]):
        aid = f"LA_E_{i}"
        sf.write(origem / "capturado" / f"{aid}.flac",
                 np.random.default_rng(i).standard_normal(SR * 2).astype(np.float32) * 0.1, SR)
        linhas.append(f"LA_0099 {aid} - {'-' if rot == 'bonafide' else 'A10'} {rot}")
    (origem / "protocolo_canal_real.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")

    assert main(["--pasta", str(tmp_path / "canal"), "--destino", "perdas",
                 "--ms", "20", "--n", "2"]) == 0
    destino = tmp_path / "canal" / "perdas"
    for i in range(2):
        info = sf.info(destino / "capturado" / f"LA_E_{i}.flac")
        assert info.frames == SR * 2 - 2 * 320
    cfg = yaml.safe_load((destino / "config_canal_real.yaml").read_text(encoding="utf-8"))
    assert cfg["experiment"]["name"].endswith("canal_real_canal_perdas")
    assert "evaluate.py" in capsys.readouterr().out
