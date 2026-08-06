"""Testes do cache de features em arquivo único (memory-map).

O cache é indexado pela POSIÇÃO da amostra no protocolo, não pelo nome do
arquivo. Isso é rápido, mas só é seguro enquanto a lista de ids for a mesma —
daí a maior parte dos testes aqui ser sobre invalidação.
"""

import json

import numpy as np
import pytest
import soundfile as sf
import torch

from src.data.cache import FeatureCache, ids_fingerprint, legacy_pt_files
from src.data.dataset import ASVspoofDataset
from src.features import FeatureExtractor


def make_features(seed: int, shapes) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {k: torch.rand(*v, generator=g) for k, v in shapes.items()}


SHAPES = {"lfcc": (1, 6, 5), "spectrogram": (1, 4, 5)}


def test_roundtrip_is_bit_exact(tmp_path):
    cache = FeatureCache(tmp_path, [f"utt{i}" for i in range(4)], SHAPES)
    original = make_features(0, SHAPES)
    cache.put(2, original)
    voltou = cache.get(2)
    for key in SHAPES:
        assert torch.equal(voltou[key], original[key]), key


def test_get_returns_none_before_put(tmp_path):
    cache = FeatureCache(tmp_path, ["a", "b"], SHAPES)
    assert cache.get(0) is None
    assert cache.get(1) is None


def test_tensors_are_independent_copies(tmp_path):
    """Escrever no tensor devolvido não pode contaminar o cache.

    Se `get` devolvesse uma vista do memmap, o `.to(device)` seguinte ainda
    funcionaria — mas qualquer operação in-place gravaria no arquivo, corrompendo
    o cache em silêncio para todas as épocas seguintes.
    """
    cache = FeatureCache(tmp_path, ["a"], SHAPES)
    cache.put(0, make_features(1, SHAPES))
    primeiro = cache.get(0)
    primeiro["lfcc"].add_(100.0)
    segundo = cache.get(0)
    assert not torch.allclose(primeiro["lfcc"], segundo["lfcc"])


def test_survives_reopening(tmp_path):
    ids = ["a", "b", "c"]
    original = make_features(3, SHAPES)
    FeatureCache(tmp_path, ids, SHAPES).put(1, original)

    outro = FeatureCache(tmp_path, ids, SHAPES)
    assert torch.equal(outro.get(1)["lfcc"], original["lfcc"])
    assert outro.get(0) is None


def test_rebuilt_when_id_list_changes(tmp_path):
    """Trocar o protocolo PRECISA invalidar: a linha 1 passa a ser outro áudio."""
    FeatureCache(tmp_path, ["a", "b", "c"], SHAPES).put(1, make_features(4, SHAPES))
    outro = FeatureCache(tmp_path, ["a", "z", "c"], SHAPES)
    assert outro.get(1) is None


def test_rebuilt_when_ids_are_reordered(tmp_path):
    FeatureCache(tmp_path, ["a", "b"], SHAPES).put(0, make_features(5, SHAPES))
    assert FeatureCache(tmp_path, ["b", "a"], SHAPES).get(0) is None


def test_rebuilt_when_shapes_change(tmp_path):
    ids = ["a", "b"]
    FeatureCache(tmp_path, ids, SHAPES).put(0, make_features(6, SHAPES))
    outras = {"lfcc": (1, 60, 5), "spectrogram": (1, 4, 5)}
    assert FeatureCache(tmp_path, ids, outras).get(0) is None


def test_rebuilt_when_meta_is_corrupt(tmp_path):
    ids = ["a", "b"]
    FeatureCache(tmp_path, ids, SHAPES).put(0, make_features(7, SHAPES))
    (tmp_path / "meta.json").write_text("{ isso nao e json", encoding="utf-8")
    assert FeatureCache(tmp_path, ids, SHAPES).get(0) is None


def test_rebuilt_when_data_file_is_missing(tmp_path):
    """meta.json sozinho não pode fazer o cache se dar por pronto."""
    ids = ["a", "b"]
    FeatureCache(tmp_path, ids, SHAPES).put(0, make_features(8, SHAPES))
    (tmp_path / "data.npy").unlink()
    assert FeatureCache(tmp_path, ids, SHAPES).get(0) is None


def test_put_rejects_unexpected_shape(tmp_path):
    cache = FeatureCache(tmp_path, ["a"], SHAPES)
    errado = {"lfcc": torch.zeros(1, 6, 99), "spectrogram": torch.zeros(1, 4, 5)}
    with pytest.raises(ValueError, match="shape constante"):
        cache.put(0, errado)


def test_put_is_idempotent(tmp_path):
    cache = FeatureCache(tmp_path, ["a"], SHAPES)
    primeiro = make_features(9, SHAPES)
    cache.put(0, primeiro)
    cache.put(0, make_features(10, SHAPES))  # ignorado: a linha já está gravada
    assert torch.equal(cache.get(0)["lfcc"], primeiro["lfcc"])


def test_survives_pickle_like_a_dataloader_worker(tmp_path):
    """O dataset é serializado para os workers; o memmap não pode ir junto."""
    import pickle

    ids = ["a", "b"]
    cache = FeatureCache(tmp_path, ids, SHAPES)
    original = make_features(11, SHAPES)
    cache.put(0, original)

    clone = pickle.loads(pickle.dumps(cache))
    assert clone._data is None, "o memmap foi serializado junto"
    assert torch.equal(clone.get(0)["lfcc"], original["lfcc"])


def test_meta_records_layout(tmp_path):
    FeatureCache(tmp_path, ["a", "b"], SHAPES)
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_items"] == 2
    assert meta["row_floats"] == 6 * 5 + 4 * 5
    assert meta["ids_fingerprint"] == ids_fingerprint(["a", "b"])


def test_n_filled_counts_written_rows(tmp_path):
    cache = FeatureCache(tmp_path, ["a", "b", "c"], SHAPES)
    assert cache.n_filled() == 0
    cache.put(0, make_features(12, SHAPES))
    cache.put(2, make_features(13, SHAPES))
    assert cache.n_filled() == 2


def test_legacy_pt_files_are_reported(tmp_path):
    assert legacy_pt_files(tmp_path) == (0, 0)
    (tmp_path / "LA_0001.pt").write_bytes(b"x" * 100)
    (tmp_path / "LA_0002.pt").write_bytes(b"x" * 50)
    (tmp_path / "meta.json").write_bytes(b"{}")
    assert legacy_pt_files(tmp_path) == (2, 150)


# --------------------------------------------------------------------------- #
# Integração com o dataset
# --------------------------------------------------------------------------- #
def build_fake_la(tmp_path, audio_cfg, n=4):
    """Monta um mini-ASVspoof em disco: protocolo + .flac."""
    audio_dir = tmp_path / "flac"
    audio_dir.mkdir()
    rng = np.random.default_rng(0)
    linhas = []
    for i in range(n):
        nome = f"LA_T_{i:04d}"
        dur = 1.0 + 0.5 * i  # durações diferentes -> exercita o fix_length
        wav = rng.standard_normal(int(audio_cfg["sample_rate"] * dur)).astype(np.float32)
        sf.write(audio_dir / f"{nome}.flac", wav * 0.5, audio_cfg["sample_rate"],
                 format="FLAC")
        chave = "bonafide" if i % 2 == 0 else "spoof"
        sistema = "-" if i % 2 == 0 else "A07"
        linhas.append(f"LA_00{i} {nome} - {sistema} {chave}")
    proto = tmp_path / "proto.txt"
    proto.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return proto, audio_dir


def test_dataset_cached_matches_uncached(tmp_path, audio_cfg, feat_cfg):
    """O cache não pode mudar UM BIT do que o modelo recebe."""
    proto, audio_dir = build_fake_la(tmp_path, audio_cfg)
    extractor = FeatureExtractor(audio_cfg, feat_cfg)

    sem = ASVspoofDataset(proto, audio_dir, audio_cfg, extractor, cache_dir=None)
    com = ASVspoofDataset(proto, audio_dir, audio_cfg, extractor,
                          cache_dir=tmp_path / "cache", partition="train")
    assert com.cache is not None

    for idx in range(len(sem)):
        esperado, _ = sem[idx]
        # 1a passada grava no cache, 2a lê de volta — ambas precisam bater.
        for _ in range(2):
            obtido, _ = com[idx]
            for key in esperado:
                assert torch.equal(obtido[key], esperado[key]), (idx, key)


def test_dataset_cache_is_reused_between_instances(tmp_path, audio_cfg, feat_cfg):
    proto, audio_dir = build_fake_la(tmp_path, audio_cfg)
    extractor = FeatureExtractor(audio_cfg, feat_cfg)
    cache_dir = tmp_path / "cache"

    primeiro = ASVspoofDataset(proto, audio_dir, audio_cfg, extractor,
                               cache_dir=cache_dir, partition="dev")
    for idx in range(len(primeiro)):
        primeiro[idx]
    assert primeiro.cache.n_filled() == len(primeiro)

    segundo = ASVspoofDataset(proto, audio_dir, audio_cfg, extractor,
                              cache_dir=cache_dir, partition="dev")
    assert segundo.cache.n_filled() == len(segundo)

    # Apagar os áudios prova que a leitura veio mesmo do cache.
    for flac in audio_dir.iterdir():
        flac.unlink()
    assert torch.equal(segundo[1][0]["lfcc"], primeiro[1][0]["lfcc"])


def test_partitions_do_not_share_rows(tmp_path, audio_cfg, feat_cfg):
    """train e dev têm protocolos diferentes: não podem cair na mesma matriz."""
    proto, audio_dir = build_fake_la(tmp_path, audio_cfg)
    extractor = FeatureExtractor(audio_cfg, feat_cfg)
    cache_dir = tmp_path / "cache"

    treino = ASVspoofDataset(proto, audio_dir, audio_cfg, extractor,
                             cache_dir=cache_dir, partition="train")
    validacao = ASVspoofDataset(proto, audio_dir, audio_cfg, extractor,
                                cache_dir=cache_dir, partition="dev")
    assert treino.cache.dir != validacao.cache.dir


def test_cache_disabled_when_augmentation_is_on(tmp_path, audio_cfg, feat_cfg):
    """Com aumentação/recorte aleatório o cache congelaria uma única versão."""
    from src.preprocess.augment import Augmenter

    proto, audio_dir = build_fake_la(tmp_path, audio_cfg)
    extractor = FeatureExtractor(audio_cfg, feat_cfg)
    aug = Augmenter({"enabled": True, "noise": {"prob": 1.0, "snr_db": [10, 30]}})

    for kwargs in ({"augmenter": aug}, {"random_crop": True}):
        ds = ASVspoofDataset(proto, audio_dir, audio_cfg, extractor,
                             cache_dir=tmp_path / "cache", partition="train", **kwargs)
        assert ds.cache is None, kwargs


def test_dataset_works_through_dataloader_workers(tmp_path, audio_cfg, feat_cfg):
    """Os workers gravam no mesmo memmap; nenhuma linha pode sair corrompida."""
    from torch.utils.data import DataLoader

    proto, audio_dir = build_fake_la(tmp_path, audio_cfg, n=8)
    extractor = FeatureExtractor(audio_cfg, feat_cfg)

    sem = ASVspoofDataset(proto, audio_dir, audio_cfg, extractor, cache_dir=None)
    esperado = torch.stack([sem[i][0]["lfcc"] for i in range(len(sem))])

    com = ASVspoofDataset(proto, audio_dir, audio_cfg, extractor,
                          cache_dir=tmp_path / "cache", partition="eval")
    loader = DataLoader(com, batch_size=2, shuffle=False, num_workers=2)
    for _ in range(2):  # 1a época grava, 2a lê
        obtido = torch.cat([lote["lfcc"] for lote, _ in loader])
        assert torch.equal(obtido, esperado)
    assert com.cache.n_filled() == len(com)
