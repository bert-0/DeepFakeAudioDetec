"""Placar do teste com a voz própria (scripts/teste_voz_propria.py).

O placar usa análises montadas à mão; o caminho completo roda com dois modelos
pequenos sem treino e ruído no lugar das gravações.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from scripts.teste_voz_propria import (  # noqa: E402
    Analise, frases_sem_par, main, minimo_de_acertos, placar)
from src.config import load_config  # noqa: E402
from src.models import build_model  # noqa: E402
from src.scores import checkpoint_fingerprint  # noqa: E402

LIMIAR = 0.5


def _analises(pasta, scores):
    return [Analise(pasta=pasta, frase=f"f{i:02d}", score=s, por_modelo=[s, s],
                    janelas_uteis=0 if s is None else 3, taxa_nativa=48000, duracao_s=8.0)
            for i, s in enumerate(scores, start=1)]


def test_minimo_de_acertos_e_oito_de_dez():
    assert [minimo_de_acertos(n) for n in (10, 5, 9, 1)] == [8, 4, 8, 1]


def test_meta_por_caminho_exige_as_duas_classes():
    reais = _analises("real", [0.1] * 8 + [0.7, 0.9])            # 8 de 10 humanos
    clones = _analises("clone", [0.9] * 9 + [0.2])               # 9 de 10 sintéticos
    pelo_ar = _analises("clone_ar", [0.9] * 7 + [0.1, 0.2, 0.3])  # 7 de 10
    r = placar(reais + clones + pelo_ar, LIMIAR)
    assert r["arquivo"]["real"]["acertos"] == 8 and r["arquivo"]["clone"]["acertos"] == 9
    assert r["arquivo"]["meta"] is True
    assert r["microfone"]["clone_ar"]["acertos"] == 7 and r["microfone"]["meta"] is False


def test_no_limiar_e_sintetico_e_sem_fala_conta_como_erro():
    r = placar(_analises("real", [LIMIAR, None]) + _analises("clone", [LIMIAR, None]), LIMIAR)
    assert r["arquivo"]["real"]["acertos"] == 0
    assert r["arquivo"]["clone"]["acertos"] == 1


def test_comparacao_por_frase_e_so_descritiva():
    reais = _analises("real", [0.8, 0.8, 0.1])
    clones = _analises("clone", [0.9, 0.3, 0.2])
    r = placar(reais + clones, LIMIAR)
    assert r["arquivo"]["falso_acima_do_real"] == {"vezes": 2, "pares": 3}
    assert r["arquivo"]["meta"] is False
    assert r["microfone"]["clone_ar"]["n"] == 0 and r["microfone"]["meta"] is False


def test_avisa_frases_que_faltam_numa_pasta():
    a = _analises("real", [0.1, 0.1, 0.1]) + _analises("clone", [0.9, 0.9])
    assert frases_sem_par(a) == {"clone": ["f03"]}


def _checkpoint(caminho, seed):
    config = load_config("configs/baseline.yaml")
    torch.manual_seed(seed)
    estado = build_model(config["model"]).state_dict()
    torch.save({"model_state": estado, "config": config, "threshold": 0.5}, caminho)
    return {"config": "configs/baseline.yaml", "checkpoint": str(caminho),
            "impressao": checkpoint_fingerprint(estado)}


def _gravacoes(pasta):
    rng = np.random.default_rng(0)
    for nome, taxa in (("real", 48000), ("clone", 24000), ("clone_ar", 48000)):
        (pasta / nome).mkdir(parents=True)
        for i in (1, 2):
            sf.write(pasta / nome / f"f{i:02d}.wav",
                     (rng.standard_normal(taxa * 5) * 0.1).astype(np.float32), taxa)


def test_caminho_completo_com_dois_modelos(tmp_path, capsys):
    modelos = [_checkpoint(tmp_path / "a.pt", 0), _checkpoint(tmp_path / "b.pt", 1)]
    limiar = tmp_path / "limiar.json"
    limiar.write_text(json.dumps({"limiar": 0.5, "modelos": modelos}), encoding="utf-8")
    _gravacoes(tmp_path / "vozes")
    saida = tmp_path / "r.json"

    assert main(["--pasta", str(tmp_path / "vozes"), "--limiar", str(limiar),
                 "--json", str(saida)]) == 0
    r = json.loads(saida.read_text(encoding="utf-8"))
    assert len(r["arquivos"]) == 6
    for a in r["arquivos"]:
        assert a["score"] == pytest.approx(np.mean(a["por_modelo"]), abs=1e-6), \
            "a fusão é a média dos dois, com os mesmos pesos por janela"
    assert {a["taxa_nativa"] for a in r["arquivos"] if a["pasta"] == "clone"} == {24000}
    assert r["placar"]["arquivo"]["real"]["n"] == 2
    texto = capsys.readouterr().out
    assert "Caminho arquivo" in texto and "Caminho microfone" in texto


def test_recusa_limiar_de_outros_pesos(tmp_path):
    m = _checkpoint(tmp_path / "a.pt", 0)
    m["impressao"] = "0" * 16
    limiar = tmp_path / "limiar.json"
    limiar.write_text(json.dumps({"limiar": 0.5, "modelos": [m, m]}), encoding="utf-8")
    _gravacoes(tmp_path / "vozes")
    with pytest.raises(SystemExit, match="mudou desde a escolha do limiar"):
        main(["--pasta", str(tmp_path / "vozes"), "--limiar", str(limiar),
              "--json", str(tmp_path / "r.json")])


def test_sem_gravacoes(tmp_path, capsys):
    assert main(["--pasta", str(tmp_path), "--limiar", str(tmp_path / "x.json")]) == 1
    assert "faltam gravações" in capsys.readouterr().out
