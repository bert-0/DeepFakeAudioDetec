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
    assert r["microfone"]["clone_ar"]["n"] == 0 and r["microfone"]["meta"] is None, \
        "sem gravações o caminho não foi medido; não é meta não atingida"


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


def test_caminho_completo_nos_logodds(tmp_path):
    modelos = [_checkpoint(tmp_path / "a.pt", 0), _checkpoint(tmp_path / "b.pt", 1)]
    limiar = tmp_path / "limiar.json"
    limiar.write_text(json.dumps({"regra": "logodds", "limiar": 0.0, "modelos": modelos}),
                      encoding="utf-8")
    _gravacoes(tmp_path / "vozes")
    saida = tmp_path / "r.json"

    assert main(["--pasta", str(tmp_path / "vozes"), "--limiar", str(limiar),
                 "--json", str(saida)]) == 0
    r = json.loads(saida.read_text(encoding="utf-8"))
    assert r["regra"] == "logodds" and r["limiar_score"] == 0.5
    for a in r["arquivos"]:
        assert a["logodds"] == pytest.approx(np.mean(a["por_modelo"]), abs=1e-6), \
            "cada modelo aparece nos log-odds; a fusão é a média deles"
        assert a["score"] == pytest.approx(1 / (1 + np.exp(-a["logodds"])), abs=1e-6)


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


def test_sem_taxa_comum_le_como_o_file_source(tmp_path):
    from scripts.teste_voz_propria import ler_na_taxa_do_modelo
    from src.capture import FileSource

    arq = tmp_path / "x.wav"
    sf.write(arq, (np.random.default_rng(2).standard_normal(48000 * 2) * 0.1).astype(np.float32), 48000)
    with FileSource(arq, 16000) as fonte:
        esperado = fonte._wav
    np.testing.assert_array_equal(ler_na_taxa_do_modelo(arq, 16000), esperado)


def test_taxa_comum_leva_o_clone_de_24_khz_pelo_caminho_da_gravacao(tmp_path):
    """Ensaio 2 (RESUMO_TCC §10.2.13): 24 -> 48 kHz e depois a conversão de 48 kHz
    que a gravação faz, em vez do 24 -> 32 -> 16 kHz direto."""
    from scripts.teste_voz_propria import ler_na_taxa_do_modelo
    from src.preprocess.reamostragem import para_taxa

    x = (np.random.default_rng(3).standard_normal(24000 * 2) * 0.1).astype(np.float32)
    arq = tmp_path / "clone.wav"
    sf.write(arq, x, 24000, subtype="FLOAT")
    comum = ler_na_taxa_do_modelo(arq, 16000, taxa_comum=48000)
    np.testing.assert_allclose(comum, para_taxa(para_taxa(x, 24000, 48000), 48000, 16000),
                               atol=1e-6)
    direto = ler_na_taxa_do_modelo(arq, 16000)
    assert len(comum) == len(direto) == 32000
    assert not np.allclose(comum, direto, atol=1e-4), "as duas conversões não são a mesma"


def test_ensaio_so_do_arquivo_com_um_modelo(tmp_path, capsys):
    modelo = _checkpoint(tmp_path / "a.pt", 0)
    limiar = tmp_path / "limiar.json"
    limiar.write_text(json.dumps({"regra": "logodds", "limiar": 0.0, "modelos": [modelo]}),
                      encoding="utf-8")
    _gravacoes(tmp_path / "vozes")
    import shutil

    shutil.rmtree(tmp_path / "vozes" / "clone_ar")
    saida = tmp_path / "r.json"
    assert main(["--pasta", str(tmp_path / "vozes"), "--limiar", str(limiar),
                 "--json", str(saida), "--taxa-comum", "48000"]) == 0
    r = json.loads(saida.read_text(encoding="utf-8"))
    assert r["taxa_comum"] == 48000 and r["placar"]["microfone"]["meta"] is None
    assert all(len(a["por_modelo"]) == 1 for a in r["arquivos"])
    assert "Caminho microfone: não medido" in capsys.readouterr().out


def test_outra_pasta_de_clones(tmp_path):
    """Ensaio 4: as mesmas gravações reais contra os clones de outra ferramenta."""
    modelo = _checkpoint(tmp_path / "a.pt", 0)
    limiar = tmp_path / "limiar.json"
    limiar.write_text(json.dumps({"regra": "logodds", "limiar": 0.0, "modelos": [modelo]}),
                      encoding="utf-8")
    _gravacoes(tmp_path / "vozes")
    (tmp_path / "vozes" / "clone").rename(tmp_path / "vozes" / "clone_chatterbox")
    saida = tmp_path / "r.json"
    assert main(["--pasta", str(tmp_path / "vozes"), "--limiar", str(limiar),
                 "--json", str(saida), "--clones", "clone_chatterbox"]) == 0
    r = json.loads(saida.read_text(encoding="utf-8"))
    assert r["clones"] == "clone_chatterbox" and r["placar"]["arquivo"]["clone"]["n"] == 2
