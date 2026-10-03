"""Datasets: ASVspoof 2019 LA e áudios sintéticos para o modo --smoke.

Rótulos: 0 = bonafide, 1 = spoof (classe positiva).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..features import FeatureExtractor
from ..preprocess import load_audio, preprocess_waveform
from .cache import FeatureCache, legacy_pt_files

LABEL_MAP = {"bonafide": 0, "spoof": 1}


def _shared_epoch() -> torch.Tensor:
    """Contador de época em memória compartilhada entre processos.

    Com `persistent_workers=True` um `int` comum ficaria congelado na cópia de
    cada worker, e `set_epoch()` não os alcançaria.
    """
    return torch.zeros(1, dtype=torch.long).share_memory_()


# --------------------------------------------------------------------------- #
# ASVspoof 2019 LA
# --------------------------------------------------------------------------- #
def parse_protocol_with_systems(
    protocol_path: str | Path,
) -> list[tuple[str, int, str]]:
    """Lê o protocolo e devolve [(audio_file_name, label, system_id), ...].

    Formato: `SPEAKER FILE - SYSTEM_ID KEY`; `system_id` é A01…A19 ou "-".
    """
    items: list[tuple[str, int, str]] = []
    with open(protocol_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            file_name, system_id, key = parts[1], parts[3], parts[4]
            if key in LABEL_MAP:
                items.append((file_name, LABEL_MAP[key], system_id))
    return items


def parse_protocol(protocol_path: str | Path) -> list[tuple[str, int]]:
    """Lê o protocolo e devolve [(audio_file_name, label), ...]."""
    return [(name, label) for name, label, _ in parse_protocol_with_systems(protocol_path)]


def buscar_rotulo(config: dict, audio_id: str) -> tuple[str, int, str] | None:
    """Procura o rótulo verdadeiro de um áudio nas três partições.

    Devolve `(partição, label, system_id)`, ou `None` se o id não estiver em
    nenhum protocolo. Protocolos ausentes são pulados.
    """
    for particao, caminho in config.get("data", {}).get("protocols", {}).items():
        if not Path(caminho).is_file():
            continue
        for nome, label, sistema in parse_protocol_with_systems(caminho):
            if nome == audio_id:
                return particao, label, sistema
    return None


def protocol_ids_and_systems(config: dict, partition: str) -> tuple[list[str], list[str]]:
    """Ids e algoritmos de síntese de uma partição, sem abrir nenhum áudio."""
    registros = parse_protocol_with_systems(config["data"]["protocols"][partition])
    return [nome for nome, _, _ in registros], [sis for _, _, sis in registros]


class ASVspoofDataset(Dataset):
    def __init__(
        self,
        protocol_path: str | Path,
        audio_dir: str | Path,
        audio_cfg: dict,
        extractor: FeatureExtractor,
        cache_dir: str | Path | None = None,
        file_ext: str = ".flac",
        augmenter=None,
        random_crop: bool = False,
        seed: int = 0,
        partition: str = "train",
    ):
        records = parse_protocol_with_systems(protocol_path)
        self.items = [(name, label) for name, label, _ in records]
        self.labels = [label for _, label, _ in records]
        self.ids = [name for name, _, _ in records]
        self.system_ids = [system for _, _, system in records]
        self.audio_dir = Path(audio_dir)
        self.audio_cfg = audio_cfg
        self.extractor = extractor
        self.file_ext = file_ext
        self.augmenter = augmenter
        self.random_crop = random_crop
        self.seed = seed
        self.partition = partition
        self._epoch = _shared_epoch()
        # O cache guarda uma versão só; não combina com aleatoriedade por época.
        stochastic = augmenter is not None or random_crop
        self.cache_dir = None
        self.cache = None
        if cache_dir and not stochastic and self.items:
            # Pasta pelo fingerprint das features, não pelo experimento:
            # configs com features iguais compartilham o cache (~11 GB cada).
            self._cache_key = _config_fingerprint(audio_cfg, extractor)
            self.cache_dir = Path(cache_dir) / self._cache_key
            self.cache = self._open_cache()

    def _open_cache(self) -> FeatureCache | None:
        """Abre o cache da partição; extrai a amostra 0 para descobrir os shapes."""
        quantos, tamanho = legacy_pt_files(self.cache_dir)
        if quantos:
            print(f"[cache] {quantos} arquivos .pt do formato antigo em "
                  f"{self.cache_dir} (~{tamanho / 1e9:.1f} GB). O novo formato "
                  "não os usa — podem ser apagados com segurança.")
        try:
            features = self._extract(0)
            cache = FeatureCache(self.cache_dir / self.partition, self.ids,
                                 {k: tuple(v.shape) for k, v in features.items()})
            cache.put(0, features)
        except OSError as erro:
            # Sem espaço ou permissão, o treino segue sem cache.
            print(f"[cache] desativado para '{self.partition}': {erro}")
            return None
        return cache

    def set_epoch(self, epoch: int) -> None:
        """Varia a semente por época para que o recorte aleatório mude a cada uma."""
        self._epoch[0] = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        _, label = self.items[idx]
        return self._load_features(idx), label

    def _sample_rng(self, idx: int) -> np.random.Generator:
        """Gerador próprio de cada amostra, derivado de (semente, época, índice).

        Não guarda estado: um `rng` no objeto seria copiado igual para cada worker.
        """
        return np.random.default_rng((self.seed, int(self._epoch[0]), idx))

    def _load_features(self, idx: int) -> dict[str, torch.Tensor]:
        if self.cache is not None:
            cached = self.cache.get(idx)
            if cached is not None:
                return cached
        features = self._extract(idx)
        if self.cache is not None:
            self.cache.put(idx, features)
        return features

    def _extract(self, idx: int) -> dict[str, torch.Tensor]:
        """Caminho completo: lê o áudio, pré-processa, aumenta e extrai."""
        file_name, _ = self.items[idx]
        rng = self._sample_rng(idx) if (self.random_crop or self.augmenter) else None
        wav = load_audio(self.audio_dir / f"{file_name}{self.file_ext}",
                         self.audio_cfg["sample_rate"])
        wav = preprocess_waveform(wav, self.audio_cfg,
                                  rng=rng if self.random_crop else None)
        if self.augmenter is not None:
            wav = self.augmenter(wav, rng=rng)
        return self.extractor(wav)


# --------------------------------------------------------------------------- #
# Smoke — dados sintéticos
# --------------------------------------------------------------------------- #
class SmokeDataset(Dataset):
    """Áudios sintéticos separáveis, só para exercitar o pipeline.

    Bonafide: harmônicos + ruído baixo. Spoof: o mesmo + tom perto de Nyquist.
    """

    def __init__(self, n: int, audio_cfg: dict, extractor: FeatureExtractor,
                 seed: int = 0, augmenter=None, random_crop: bool = False):
        self.n = n
        self.audio_cfg = audio_cfg
        self.extractor = extractor
        self.seed = seed
        self.augmenter = augmenter
        self.random_crop = random_crop
        self._epoch = _shared_epoch()
        self.sr = audio_cfg["sample_rate"]
        self.n_samples = int(self.sr * audio_cfg["duration"])
        self.labels = [i % 2 for i in range(n)]
        self.ids = [f"smoke_{i:05d}" for i in range(n)]
        # Dois "ataques" fictícios, só para exercitar a análise por sistema.
        self.system_ids = ["-" if i % 2 == 0 else f"A{7 + (i // 2) % 2:02d}"
                           for i in range(n)]

    def set_epoch(self, epoch: int) -> None:
        self._epoch[0] = int(epoch)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed * 100_000 + idx)
        label = idx % 2  # metade bonafide, metade spoof
        # Mais longo que o alvo, para o recorte aleatório ter o que recortar.
        wav = self._synth(rng, spoof=bool(label),
                          n_samples=int(self.n_samples * 1.5) if self.random_crop
                          else self.n_samples)
        aug_rng = np.random.default_rng((self.seed, int(self._epoch[0]), idx))
        wav = preprocess_waveform(wav, self.audio_cfg,
                                  rng=aug_rng if self.random_crop else None)
        if self.augmenter is not None:
            wav = self.augmenter(wav, rng=aug_rng)
        return self.extractor(wav), label

    def _synth(self, rng: np.random.Generator, spoof: bool,
               n_samples: int | None = None) -> np.ndarray:
        n = self.n_samples if n_samples is None else n_samples
        t = np.arange(n) / self.sr
        f0 = rng.uniform(110, 220)  # frequência fundamental
        wav = np.zeros_like(t)
        for k in range(1, 6):  # harmônicos
            wav += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi))
        wav += 0.01 * rng.standard_normal(n)  # ruído de fundo baixo
        if spoof:
            # Artefato de alta frequência + ruído extra.
            wav += 0.3 * np.sin(2 * np.pi * (self.sr * 0.45) * t)
            wav += 0.03 * rng.standard_normal(n)
        return wav.astype(np.float32)


# --------------------------------------------------------------------------- #
# Fábrica
# --------------------------------------------------------------------------- #
def build_dataset(
    config: dict,
    partition: str,
    extractor: FeatureExtractor,
    smoke: bool,
    augmenter=None,
    random_crop: bool = False,
) -> Dataset:
    """Constrói o dataset de uma partição ('train' | 'dev' | 'eval').

    `augmenter` é um `wav -> wav` aplicado após o pré-processamento;
    `random_crop` só deve ser usado no treino.
    """
    if smoke:
        smoke_cfg = config["smoke"]
        n = smoke_cfg[f"n_{partition}"]
        seed = {"train": 1, "dev": 2, "eval": 3}[partition]
        return SmokeDataset(n, config["audio"], extractor, seed=seed,
                            augmenter=augmenter, random_crop=random_crop)

    cache_dir = None
    if config["train"].get("cache_features", False):
        # Base comum; o dataset cria uma subpasta por fingerprint.
        cache_dir = Path(config["data"]["root"]).parent / "cache"
    return ASVspoofDataset(
        protocol_path=config["data"]["protocols"][partition],
        audio_dir=config["data"]["audio_dir"][partition],
        audio_cfg=config["audio"],
        extractor=extractor,
        cache_dir=cache_dir,
        # Outra extensão só se os áudios foram convertidos (converter_para_wav.py).
        file_ext=config["data"].get("file_ext", ".flac"),
        augmenter=augmenter,
        random_crop=random_crop,
        seed=config["experiment"]["seed"],
        partition=partition,
    )


# Só controlam aleatoriedade por época, que já desliga o cache; fora do fingerprint.
_NAO_AFETAM_CACHE = ("augment", "random_crop")


def _config_fingerprint(audio_cfg: dict, extractor: FeatureExtractor) -> str:
    """Hash curto de áudio+features (com parâmetros) que identifica o cache.

    Chaves novas de `audio` entram por padrão; só `_NAO_AFETAM_CACHE` fica fora.
    """
    audio_relevante = {k: v for k, v in audio_cfg.items()
                       if k not in _NAO_AFETAM_CACHE}
    payload = json.dumps(
        {"audio": audio_relevante, "features": extractor.fingerprint()},
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:8]
