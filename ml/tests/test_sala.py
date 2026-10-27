"""Sala simulada (src/preprocess/sala.py) com uma árvore falsa no formato do BUT ReverbDB."""

import numpy as np
import pytest
import soundfile as sf

from src.preprocess.sala import Sala, indexar, preparar_rir

SR = 16000


def _arvore(raiz, n_posicoes=2, atraso=200, com_silencio=True):
    rng = np.random.default_rng(0)
    for k in range(n_posicoes):
        pos = raiz / "SALA_X" / f"MicID0{k}" / "SpkID01" / "01"
        (pos / "RIR").mkdir(parents=True)
        h = np.zeros(SR)
        h[atraso] = 1.0
        h[atraso + 1:] += rng.standard_normal(SR - atraso - 1) * 0.05 * np.exp(-np.arange(SR - atraso - 1) / 800)
        sf.write(pos / "RIR" / "IR_sweep_FS16kHz.v00.wav", h.astype(np.float32), SR)
        if com_silencio:
            (pos / "silence").mkdir()
            sf.write(pos / "silence" / "silence_16kHz_60sec.v00.wav",
                     (rng.standard_normal(SR * 3) * 0.01).astype(np.float32), SR)
    (raiz / "SALA_X" / "sem_rir").mkdir()


def test_indexa_so_posicoes_com_resposta_ao_impulso(tmp_path):
    _arvore(tmp_path)
    pos = indexar(tmp_path)
    assert len(pos) == 2 and all(len(p.rirs) == 1 and len(p.silencios) == 1 for p in pos)
    with pytest.raises(ValueError, match="nenhuma pasta RIR"):
        Sala(tmp_path / "SALA_X" / "sem_rir")


def test_rir_comeca_no_som_direto_com_energia_1():
    h = np.zeros(SR)
    h[300] = 0.5
    h[310] = 0.2
    r = preparar_rir(h, SR)
    assert np.argmax(np.abs(r)) == 16, "1 ms antes do pico"
    assert np.sum(r ** 2) == pytest.approx(1.0)


def test_sala_mantem_comprimento_e_tempo_e_soma_o_ruido(tmp_path):
    _arvore(tmp_path)
    t = np.arange(SR * 2) / SR
    x = (np.sin(2 * np.pi * 300 * t) * np.hanning(t.size)).astype(np.float32)
    sem_ruido = Sala(tmp_path, SR, prob_ruido=0.0)(x, np.random.default_rng(1))
    assert sem_ruido.shape == x.shape and np.max(np.abs(sem_ruido)) == pytest.approx(1.0)
    atraso = np.argmax(np.correlate(sem_ruido[:4000], x[:4000], "full")) - 3999
    assert abs(atraso) <= 20, "a fala não atrasa: o som direto fica no início da resposta"
    com_ruido = Sala(tmp_path, SR, faixa_snr=(10, 10))(x, np.random.default_rng(1))
    assert not np.allclose(com_ruido, sem_ruido)
    np.testing.assert_array_equal(Sala(tmp_path, SR)(x, np.random.default_rng(5)),
                                  Sala(tmp_path, SR)(x, np.random.default_rng(5)))


def test_silencio_mais_curto_que_o_audio_e_repetido(tmp_path):
    _arvore(tmp_path)
    x = np.random.default_rng(2).standard_normal(SR * 5).astype(np.float32)   # 5 s; silêncio de 3 s
    assert Sala(tmp_path, SR)(x, np.random.default_rng(0)).shape == x.shape


def test_preparar_com_a_sala_muda_o_que_o_professor_ouve(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from scripts.destilacao import carregar_dados, preparar

    torch.manual_seed(0)
    cfg = transformers.Wav2Vec2Config(
        hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
        conv_dim=(8, 8), conv_kernel=(10, 3), conv_stride=(5, 2), num_conv_pos_embeddings=8,
        num_conv_pos_embedding_groups=2, do_stable_layer_norm=True, feat_extract_norm="layer")
    rede = transformers.Wav2Vec2Model(cfg).eval()
    _arvore(tmp_path / "but")
    sf.write(tmp_path / "a.wav", (np.random.default_rng(3).standard_normal(SR * 3) * 0.1).astype(np.float32), SR)
    origem = {"arquivo": np.array([str(tmp_path / "a.wav")]), "classe": np.array(["bonafide"]),
              "rotulo": np.zeros(1, dtype=np.int8)}
    preparar(origem, tmp_path / "limpo", rede, torch.device("cpu"), camada=1)
    preparar(origem, tmp_path / "sala", rede, torch.device("cpu"), camada=1, transformar=Sala(tmp_path / "but", SR))
    limpo, sala = carregar_dados(tmp_path / "limpo"), carregar_dados(tmp_path / "sala")
    assert limpo["entradas"].shape == sala["entradas"].shape
    assert not np.allclose(np.asarray(limpo["alvos"]), np.asarray(sala["alvos"]))
    assert not np.allclose(np.asarray(limpo["entradas"]), np.asarray(sala["entradas"]))


def test_sala_copia_entre_processos_e_vira_condicao_da_robustez(tmp_path):
    """Os workers do robustness_eval copiam a condição (pickle) para cada processo."""
    import pickle

    from scripts.robustness_eval import condicao_sala

    _arvore(tmp_path)
    nome, sala = condicao_sala(str(tmp_path), SR)
    x = np.random.default_rng(4).standard_normal(SR).astype(np.float32)
    antes = sala(x, rng=np.random.default_rng(9))
    copia = pickle.loads(pickle.dumps(sala))
    assert nome == "sala"
    np.testing.assert_array_equal(copia(x, rng=np.random.default_rng(9)), antes)
