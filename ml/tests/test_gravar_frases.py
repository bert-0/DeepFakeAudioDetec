"""Gravador de frases (scripts/gravar_frases.py) com o microfone falso.

O falso entrega o mesmo sinal a cada gravação, como o `sounddevice` do
test_microfone_wasapi; no modo pelo-ar, o `play` só registra o que tocaria.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.gravar_frases import DESCARTE_S, MARGEM_S, gravar, ler_frases, main
from test_microfone_wasapi import _sounddevice_falso

SR = 48000
RAIZ = Path(__file__).resolve().parent.parent


def _cena(segundos, seed=0):
    """Ruído baixo o tempo todo e "fala" entre as margens."""
    rng = np.random.default_rng(seed)
    n = int((segundos + DESCARTE_S + 0.5) * SR)
    wav = rng.standard_normal(n) * 0.002
    ini, fim = int((DESCARTE_S + MARGEM_S) * SR), int((DESCARTE_S + segundos - MARGEM_S) * SR)
    wav[ini:fim] += rng.standard_normal(fim - ini) * 0.1
    return wav.astype(np.float32)


def _frases(tmp_path, n=2):
    arq = tmp_path / "frases.txt"
    arq.write_text("\n".join(f"Frase de teste {i}." for i in range(1, n + 1)) + "\n\n",
                   encoding="utf-8")
    return arq


def test_le_uma_frase_por_linha_sem_as_vazias(tmp_path):
    assert ler_frases(_frases(tmp_path, 3)) == ["Frase de teste 1.", "Frase de teste 2.",
                                               "Frase de teste 3."]


@pytest.mark.parametrize("arquivo", ["frases_teste_voz_propria.txt", "frases_ensaio2.txt",
                                     "frases_ensaio3.txt"])
def test_frases_do_teste_fora_do_texto_de_referencia(arquivo):
    """O clone sai do texto de referência; as frases do teste não podem estar nele."""
    frases = ler_frases(RAIZ / "docs/etica" / arquivo)
    referencia = (RAIZ / "docs/etica/texto_referencia_clonagem.txt").read_text(encoding="utf-8")
    assert len(frases) == 10 and len(set(frases)) == 10
    assert not any(f.rstrip(".") in referencia for f in frases)


@pytest.mark.parametrize("novo", ["frases_ensaio2.txt", "frases_ensaio3.txt"])
def test_cada_ensaio_tem_frases_novas(novo):
    """Cada ensaio confirma o anterior com dados novos (RESUMO_TCC §10.2.13)."""
    conjuntos = ["frases_teste_voz_propria.txt", "frases_ensaio2.txt", "frases_ensaio3.txt"]
    anteriores = set().union(*(ler_frases(RAIZ / "docs/etica" / c)
                               for c in conjuntos[:conjuntos.index(novo)]))
    frases = ler_frases(RAIZ / "docs/etica" / novo)
    assert not anteriores & set(frases)
    palavras = [len(f.split()) for f in frases]
    assert min(palavras) >= 15 and max(palavras) <= 22, "mesmo tamanho do ensaio 1 (~7 s)"


def test_grava_sem_o_comeco_e_avisa_no_instante_certo(monkeypatch):
    sinal = _cena(3.0)
    _sounddevice_falso(monkeypatch, sinal)
    instantes = []
    wav, taxa, modo, _ = gravar(3.0, avisos=((1.0, lambda: instantes.append(1.0)),))
    assert (taxa, modo, len(wav)) == (SR, "compartilhado", 3 * SR)
    np.testing.assert_allclose(wav, sinal[SR // 2:SR // 2 + 3 * SR], atol=1e-6)
    assert instantes == [1.0]


def test_modo_voz_grava_cada_frase_em_24_bits_e_pula_as_feitas(monkeypatch, tmp_path, capsys):
    sd = _sounddevice_falso(monkeypatch, _cena(7.0))
    saida = tmp_path / "real"
    saida.mkdir()
    sf.write(saida / "f01.wav", np.zeros(SR, np.float32), SR)

    assert main(["voz", "--frases", str(_frases(tmp_path)), "--saida", str(saida),
                 "--segundos", "7", "--sem-pausa"]) == 0
    assert len(sd.fluxos) == 1, "a f01 já existia"
    info = sf.info(saida / "f02.wav")
    assert (info.samplerate, info.subtype, info.frames) == (SR, "PCM_24", 7 * SR)
    texto = capsys.readouterr().out
    assert "já existe" in texto and "leia agora" in texto
    assert "piso de ruído" in texto, "as checagens rodam depois de cada gravação"


def test_r_regrava_a_mesma_frase(monkeypatch, tmp_path):
    sd = _sounddevice_falso(monkeypatch, _cena(3.0))
    respostas = iter(["", "r", "", ""])
    monkeypatch.setattr("builtins.input", lambda *_: next(respostas))
    saida = tmp_path / "real"
    assert main(["voz", "--frases", str(_frases(tmp_path, 1)), "--saida", str(saida),
                 "--segundos", "3"]) == 0
    assert len(sd.fluxos) == 2 and (saida / "f01.wav").is_file()


def test_pelo_ar_toca_cada_clone_depois_da_margem(monkeypatch, tmp_path):
    sd = _sounddevice_falso(monkeypatch, _cena(2.0 + 2 * MARGEM_S))
    tocados = []
    sd.play = lambda x, taxa: tocados.append((len(x), taxa))
    sd.stop = lambda: None
    clones = tmp_path / "clone"
    clones.mkdir()
    sf.write(clones / "f01.wav", np.random.default_rng(1).standard_normal(2 * 24000) * 0.1, 24000)
    (clones / "notas.txt").write_text("não é áudio", encoding="utf-8")

    saida = tmp_path / "clone_ar"
    assert main(["pelo-ar", "--clones", str(clones), "--saida", str(saida), "--sem-pausa"]) == 0
    assert tocados == [(2 * 24000, 24000)]
    info = sf.info(saida / "f01.wav")
    assert (info.samplerate, info.frames) == (SR, int((2.0 + 2 * MARGEM_S) * SR))


def test_cada_modo_exige_a_sua_entrada(tmp_path):
    with pytest.raises(SystemExit):
        main(["voz", "--saida", str(tmp_path)])
    with pytest.raises(SystemExit):
        main(["pelo-ar", "--saida", str(tmp_path)])


def test_pasta_de_clones_sem_audio(monkeypatch, tmp_path, capsys):
    _sounddevice_falso(monkeypatch)
    assert main(["pelo-ar", "--clones", str(tmp_path), "--saida", str(tmp_path / "x"),
                 "--sem-pausa"]) == 1
    assert "nenhum áudio" in capsys.readouterr().out


def test_pasta_de_clones_que_ainda_nao_existe(monkeypatch, tmp_path, capsys):
    """24/10/2026: rodado antes de gerar os clones, parava com FileNotFoundError."""
    _sounddevice_falso(monkeypatch)
    assert main(["pelo-ar", "--clones", str(tmp_path / "clone"), "--saida", str(tmp_path / "x"),
                 "--sem-pausa"]) == 1
    assert "Gere os clones antes" in capsys.readouterr().out


def test_padrao_de_12_s_e_piso_minimo_do_teste(monkeypatch, tmp_path, capsys):
    """Fala a ~34 dB do ruído: falha no padrão de 35 dB e passa com o piso do teste."""
    _sounddevice_falso(monkeypatch, _cena(12.0))
    assert main(["voz", "--frases", str(_frases(tmp_path, 1)), "--saida", str(tmp_path / "a"),
                 "--sem-pausa"]) == 0
    assert sf.info(tmp_path / "a" / "f01.wav").frames == 12 * SR
    assert "REGRAVAR piso de ruído" in capsys.readouterr().out
    assert main(["voz", "--frases", str(_frases(tmp_path, 1)), "--saida", str(tmp_path / "b"),
                 "--sem-pausa", "--piso-minimo", "15"]) == 0
    assert "ok       piso de ruído" in capsys.readouterr().out


def test_bloco_fixo_para_diagnostico(monkeypatch, tmp_path):
    """--bloco-ms volta aos blocos fixos de antes, para comparar as duas capturas."""
    sd = _sounddevice_falso(monkeypatch, _cena(3.0))
    assert main(["voz", "--frases", str(_frases(tmp_path, 1)), "--saida", str(tmp_path / "a"),
                 "--segundos", "3", "--sem-pausa", "--bloco-ms", "100"]) == 0
    assert sd.fluxos[0].pedido == 4800
    sd = _sounddevice_falso(monkeypatch, _cena(3.0))
    assert main(["voz", "--frases", str(_frases(tmp_path, 1)), "--saida", str(tmp_path / "b"),
                 "--segundos", "3", "--sem-pausa"]) == 0
    assert sd.fluxos[0].pedido == 0


def test_compartilhado_por_padrao_e_exclusivo_so_pedido(monkeypatch, tmp_path, capsys):
    """26/10/2026: o exclusivo insere cópias de trechos neste notebook (RESUMO_TCC §10.2.34)."""
    sd = _sounddevice_falso(monkeypatch, _cena(3.0))
    assert main(["voz", "--frases", str(_frases(tmp_path, 1)), "--saida", str(tmp_path / "a"),
                 "--segundos", "3", "--sem-pausa"]) == 0
    assert not sd.fluxos[0].kw["extra_settings"].exclusive
    assert "WASAPI compartilhado" in capsys.readouterr().out
    sd = _sounddevice_falso(monkeypatch, _cena(3.0))
    assert main(["voz", "--frases", str(_frases(tmp_path, 1)), "--saida", str(tmp_path / "b"),
                 "--segundos", "3", "--sem-pausa", "--exclusivo"]) == 0
    assert sd.fluxos[0].kw["extra_settings"].exclusive
    assert "WASAPI exclusivo" in capsys.readouterr().out


def test_avisa_quando_a_captura_repete_trechos(monkeypatch, tmp_path, capsys):
    sinal = _cena(7.0)
    sinal[SR * 3:SR * 3 + 480] = sinal[SR * 2:SR * 2 + 480]
    _sounddevice_falso(monkeypatch, sinal, pacote=480)
    assert main(["voz", "--frases", str(_frases(tmp_path, 1)), "--saida", str(tmp_path / "a"),
                 "--segundos", "7", "--sem-pausa"]) == 0
    assert "REGRAVAR cópias" in capsys.readouterr().out
