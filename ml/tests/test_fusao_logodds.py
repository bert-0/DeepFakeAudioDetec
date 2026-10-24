"""Fusão pela média dos log-odds no caminho ao vivo (src/capture/analyzer.py).

Escolhida nos devs em 24/10/2026 (RESUMO_TCC §10.2.12): com a média das
probabilidades, um modelo de probabilidades minúsculas quase não pesa.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.capture.analyzer import AnalisadorContinuo, Agregador, Leitura, sigmoide  # noqa: E402
from src.capture.qualidade import EstadoDoCanal  # noqa: E402
from src.limiares import LimiarFusao, ler_limiar_fusao  # noqa: E402
from src.models import build_model  # noqa: E402
from src.scores import checkpoint_fingerprint  # noqa: E402


def _checkpoint(tmp_path, nome_config, arquivo, seed):
    cfg = yaml.safe_load((RAIZ / "configs" / nome_config).read_text(encoding="utf-8"))
    torch.manual_seed(seed)
    estado = build_model(cfg["model"]).state_dict()
    destino = tmp_path / arquivo
    torch.save({"model_state": estado, "config": cfg, "threshold": 0.5}, destino)
    return str(destino), cfg, checkpoint_fingerprint(estado)


def _ruido(segundos=10.0, sr=16000):
    return (0.3 * np.random.default_rng(0).standard_normal(int(sr * segundos))).astype(np.float32)


@pytest.fixture
def dois_modelos(tmp_path):
    a, cfg, _ = _checkpoint(tmp_path, "fusion_v4.yaml", "a.pt", 0)
    b, _, _ = _checkpoint(tmp_path, "baseline_v2.yaml", "b.pt", 1)
    return [a, b], cfg


def test_score_e_a_sigmoide_da_media_dos_logodds(dois_modelos):
    caminhos, cfg = dois_modelos
    a = AnalisadorContinuo(cfg, caminhos, torch.device("cpu"), regra="logodds")
    leituras = [x for x in a.processar(_ruido()) if not x.silencio]
    assert leituras
    for x in leituras:
        assert len(x.logodds_por_modelo) == 2
        assert x.logodds == pytest.approx(float(np.mean(x.logodds_por_modelo)))
        assert x.score == pytest.approx(sigmoide(x.logodds))
        # o log-odds de cada modelo é o mesmo ponto da probabilidade dele
        for p, lo in zip(x.por_modelo, x.logodds_por_modelo, strict=True):
            assert sigmoide(lo) == pytest.approx(p, abs=1e-5)


def test_regra_padrao_continua_a_media_das_probabilidades(dois_modelos):
    caminhos, cfg = dois_modelos
    a = AnalisadorContinuo(cfg, caminhos, torch.device("cpu"))
    for x in a.processar(_ruido()):
        if not x.silencio:
            assert x.score == pytest.approx(float(np.mean(x.por_modelo)))
            assert x.logodds is not None, "o log-odds sai nas duas regras"


def test_regra_desconhecida(dois_modelos):
    caminhos, cfg = dois_modelos
    with pytest.raises(ValueError, match="regra de fusão"):
        AnalisadorContinuo(cfg, caminhos, torch.device("cpu"), regra="max")


def test_sigmoide_sem_estouro():
    assert sigmoide(0.0) == 0.5
    assert sigmoide(1000.0) == 1.0 and sigmoide(-1000.0) == 0.0


def _leitura(i, logodds, fracao=1.0):
    return Leitura(indice=i, instante=2.0 * i, score=sigmoide(logodds), rms=0.1,
                   canal=EstadoDoCanal.LARGA, fracao_fala=fracao, logodds=logodds)


def test_agregador_faz_a_media_nos_logodds():
    """Uma janela muito alta não arrasta a média como faria nas probabilidades."""
    ag = Agregador(regra="logodds")
    for i, lo in enumerate((-8.0, -8.0, -8.0, 4.0)):
        ag.adicionar(_leitura(i, lo))
    r = ag.resumo()
    assert r["logodds_medio"] == pytest.approx(-5.0)
    assert r["score_medio"] == pytest.approx(sigmoide(-5.0))
    assert ag.media_movel() == pytest.approx(sigmoide(-5.0))
    nas_probabilidades = Agregador()
    for x in ag.leituras:
        nas_probabilidades.adicionar(x)
    assert nas_probabilidades.resumo()["score_medio"] > 20 * r["score_medio"]


def test_agregador_pondera_pela_fala_nos_logodds():
    ag = Agregador(regra="logodds")
    ag.adicionar(_leitura(0, -6.0, 1.0))
    ag.adicionar(_leitura(1, 6.0, 0.5))
    assert ag.resumo()["logodds_medio"] == pytest.approx((-6.0 + 3.0) / 1.5)
    assert ag.resumo()["score_medio_simples"] == pytest.approx(0.5)


def _json(tmp_path, modelos, **extra):
    caminho = tmp_path / "limiar.json"
    caminho.write_text(json.dumps({"limiar": -4.0, "modelos": modelos, **extra}), encoding="utf-8")
    return caminho


def test_le_o_limiar_da_fusao_e_converte_para_o_score(tmp_path):
    a, _, ia = _checkpoint(tmp_path, "baseline_v2.yaml", "a.pt", 0)
    b, _, ib = _checkpoint(tmp_path, "baseline_v2.yaml", "b.pt", 1)
    modelos = [{"config": "configs/baseline_v2.yaml", "checkpoint": a, "impressao": ia},
               {"config": "configs/baseline_v2.yaml", "checkpoint": b, "impressao": ib}]
    lf = ler_limiar_fusao(_json(tmp_path, modelos, regra="logodds", alvo=0.1,
                                condicao="captura_48k_fir"))
    assert lf.checkpoints == (a, b) and lf.regra == "logodds"
    assert lf.probabilidade == pytest.approx(sigmoide(-4.0))
    assert "log-odds" in lf.descricao() and "10%" in lf.descricao()
    # JSON de antes da opção de regra: a média das probabilidades
    antigo = ler_limiar_fusao(_json(tmp_path, modelos))
    assert antigo.regra == "probabilidade" and antigo.probabilidade == -4.0


def test_recusa_checkpoint_com_outros_pesos(tmp_path):
    a, _, _ = _checkpoint(tmp_path, "baseline_v2.yaml", "a.pt", 0)
    modelos = [{"config": "configs/baseline_v2.yaml", "checkpoint": a, "impressao": "0" * 16}]
    with pytest.raises(ValueError, match="mudou desde a escolha"):
        ler_limiar_fusao(_json(tmp_path, modelos))


def test_limiar_em_probabilidade_fica_como_esta():
    assert LimiarFusao("c", ("a",), "probabilidade", 0.43, 0.1, "x").probabilidade == 0.43


def test_monitor_usa_modelos_regra_e_limiar_do_json(tmp_path, capsys):
    import monitor

    a, _, ia = _checkpoint(tmp_path, "baseline_v2.yaml", "a.pt", 0)
    b, _, ib = _checkpoint(tmp_path, "baseline_v2.yaml", "b.pt", 1)
    modelos = [{"config": "configs/baseline_v2.yaml", "checkpoint": a, "impressao": ia},
               {"config": "configs/baseline_v2.yaml", "checkpoint": b, "impressao": ib}]
    caminho = _json(tmp_path, modelos, regra="logodds", alvo=0.1, condicao="captura_48k_fir")
    wav = tmp_path / "x.wav"
    sf.write(wav, _ruido(6.0, 48000), 48000)
    historico = tmp_path / "h.json"

    assert monitor.main(["--limiar-fusao", str(caminho), "--arquivo", str(wav),
                         "--device", "cpu", "--json", str(historico)]) == 0
    saida = capsys.readouterr().out
    assert "2 em fusão (média dos log-odds)" in saida
    assert f"Limiar: {sigmoide(-4.0):.4f}" in saida
    assert json.loads(historico.read_text(encoding="utf-8"))["limiar"] == pytest.approx(
        sigmoide(-4.0))


def test_monitor_recusa_json_e_checkpoint_juntos(tmp_path, capsys):
    import monitor

    assert monitor.main(["--limiar-fusao", "x.json", "--checkpoint", "a.pt"]) == 1
    assert "não os dois" in capsys.readouterr().out
