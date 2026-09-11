"""Testes da fusão de scores e da agregação no caminho ao vivo."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture.analyzer import AnalisadorContinuo, Agregador, Leitura  # noqa: E402
from src.capture.qualidade import EstadoDoCanal  # noqa: E402
from src.models import build_model  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent


def _checkpoint(tmp_path, nome_config, arquivo):
    """Checkpoint com pesos aleatórios — basta para exercitar o caminho."""
    cfg = yaml.safe_load((RAIZ / "configs" / nome_config).read_text(encoding="utf-8"))
    destino = tmp_path / arquivo
    torch.save({"model_state": build_model(cfg["model"]).state_dict(),
                "config": cfg, "threshold": 0.5}, destino)
    return str(destino), cfg


@pytest.fixture
def dois_modelos(tmp_path):
    a, cfg = _checkpoint(tmp_path, "fusion_v4.yaml", "a.pt")
    b, _ = _checkpoint(tmp_path, "baseline_v2.yaml", "b.pt")
    return [a, b], cfg


def _ruido(segundos=10.0, sr=16000):
    return (0.3 * np.random.default_rng(0).standard_normal(
        int(sr * segundos))).astype(np.float32)


# --------------------------------------------------------------------------- #
# Fusão
# --------------------------------------------------------------------------- #
def test_um_checkpoint_continua_funcionando(dois_modelos):
    """Compatibilidade: a assinatura antiga aceita uma string."""
    caminhos, cfg = dois_modelos
    a = AnalisadorContinuo(cfg, caminhos[0], torch.device("cpu"))
    assert a.n_modelos == 1


def test_lista_carrega_os_dois(dois_modelos):
    caminhos, cfg = dois_modelos
    assert AnalisadorContinuo(cfg, caminhos, torch.device("cpu")).n_modelos == 2


def test_o_score_fundido_e_a_media_dos_individuais(dois_modelos):
    """A regra é `mean` e não `rank`: rank precisa do conjunto todo de scores,
    que não existe num fluxo ao vivo."""
    caminhos, cfg = dois_modelos
    a = AnalisadorContinuo(cfg, caminhos, torch.device("cpu"))
    leituras = [x for x in a.processar(_ruido()) if not x.silencio]
    assert leituras
    for x in leituras:
        assert len(x.por_modelo) == 2
        assert x.score == pytest.approx(float(np.mean(x.por_modelo)))


def test_modelos_com_features_diferentes_convivem(dois_modelos):
    """O `fusion_v4` usa n_filter 70 + espectrograma; o v2, n_filter 20 e só
    LFCC. Cada um tem o seu extractor sobre o mesmo waveform."""
    caminhos, cfg = dois_modelos
    a = AnalisadorContinuo(cfg, caminhos, torch.device("cpu"))
    assert a.modelos[0].config["features"]["types"] != \
        a.modelos[1].config["features"]["types"]
    assert any(not x.silencio for x in a.processar(_ruido()))


def test_lista_vazia_e_erro(dois_modelos):
    _, cfg = dois_modelos
    with pytest.raises(ValueError, match="ao menos um"):
        AnalisadorContinuo(cfg, [], torch.device("cpu"))


def test_janelas_incompativeis_sao_recusadas(tmp_path, dois_modelos):
    """Fundir modelos treinados com janelas diferentes compararia trechos de
    comprimentos distintos do mesmo áudio."""
    caminhos, cfg = dois_modelos
    outro = yaml.safe_load((RAIZ / "configs" / "baseline_v2.yaml").read_text(encoding="utf-8"))
    outro["audio"]["duration"] = 2.0
    destino = tmp_path / "curto.pt"
    torch.save({"model_state": build_model(outro["model"]).state_dict(),
                "config": outro, "threshold": 0.5}, destino)
    with pytest.raises(ValueError, match="duration"):
        AnalisadorContinuo(cfg, [caminhos[0], str(destino)], torch.device("cpu"))


# --------------------------------------------------------------------------- #
# Janelas independentes
#
# Regressão: o resumo contava janelas sobrepostas como se fossem observações
# independentes. Com janela 4 s e passo 2 s, 5 janelas cobrem 12 s — o
# equivalente a 3 janelas sem sobreposição.
# --------------------------------------------------------------------------- #
def _leitura(i, score=0.5, canal=EstadoDoCanal.LARGA):
    return Leitura(indice=i, instante=2.0 * i, score=score, rms=0.1, canal=canal)


def test_cinco_janelas_valem_tres_independentes():
    ag = Agregador(janela_s=4.0, passo_s=2.0)
    assert ag.efetivas(5) == pytest.approx(3.0)


def test_sem_sobreposicao_a_contagem_e_exata():
    ag = Agregador(janela_s=4.0, passo_s=4.0)
    assert ag.efetivas(5) == pytest.approx(5.0)


def test_mais_sobreposicao_infla_mais():
    muita = Agregador(janela_s=4.0, passo_s=1.0)
    pouca = Agregador(janela_s=4.0, passo_s=2.0)
    assert muita.efetivas(9) < pouca.efetivas(9)


def test_zero_janelas_da_zero():
    assert Agregador().efetivas(0) == 0.0


def test_o_resumo_reporta_as_independentes():
    ag = Agregador(janela_s=4.0, passo_s=2.0)
    for i in range(5):
        ag.adicionar(_leitura(i))
    r = ag.resumo()
    assert r["janelas_uteis"] == 5
    assert r["janelas_independentes"] == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# O portão exclui das médias
# --------------------------------------------------------------------------- #
def test_banda_estreita_fica_fora_da_media():
    ag = Agregador()
    ag.adicionar(_leitura(0, score=0.1))
    ag.adicionar(_leitura(1, score=0.9, canal=EstadoDoCanal.ESTREITA))
    assert ag.media_movel() == pytest.approx(0.1)
    assert ag.resumo()["janelas_canal_ruim"] == 1


def test_indeterminado_entra_na_media():
    """Na dúvida o sistema continua medindo — só não afirma banda estreita."""
    ag = Agregador()
    ag.adicionar(_leitura(0, score=0.2, canal=EstadoDoCanal.INDETERMINADO))
    assert ag.media_movel() == pytest.approx(0.2)


def test_silencio_e_canal_ruim_sao_contados_separados():
    ag = Agregador()
    ag.adicionar(_leitura(0))
    ag.adicionar(Leitura(indice=1, instante=2.0, score=0.0, rms=0.0))
    ag.adicionar(_leitura(2, canal=EstadoDoCanal.ESTREITA))
    r = ag.resumo()
    assert (r["janelas_uteis"], r["janelas_silencio"], r["janelas_canal_ruim"]) == (1, 1, 1)
