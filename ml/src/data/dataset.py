"""Datasets para o pipeline de detecção de deepfakes.

- ASVspoofDataset: lê os .flac e o protocolo da base ASVspoof 2019 LA.
- SmokeDataset: gera áudios sintéticos (bonafide vs. spoof) para validar o
  pipeline ponta-a-ponta sem precisar da base (modo --smoke).

Convenção de rótulos: 0 = bonafide (autêntico), 1 = spoof (sintético).
A classe positiva é `spoof`, alinhada ao objetivo de detecção.
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

LABEL_MAP = {"bonafide": 0, "spoof": 1}


# --------------------------------------------------------------------------- #
# ASVspoof 2019 LA
# --------------------------------------------------------------------------- #
def parse_protocol_with_systems(
    protocol_path: str | Path,
) -> list[tuple[str, int, str]]:
    """Lê o protocolo e devolve [(audio_file_name, label, system_id), ...].

    Formato esperado: `SPEAKER  FILE  -  SYSTEM_ID  KEY`, KEY ∈ {bonafide, spoof}.
    O `system_id` identifica o algoritmo de síntese (A01…A19) e vale "-" para
    áudios bonafide. É o que permite avaliar o desempenho por tipo de ataque.
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
        self._epoch = 0
        # Cache guarda uma única versão das features, então é incompatível com
        # qualquer aleatoriedade por época (aumentação ou recorte aleatório).
        stochastic = augmenter is not None or random_crop
        self.cache_dir = Path(cache_dir) if (cache_dir and not stochastic) else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_key = _config_fingerprint(audio_cfg, extractor)

    def set_epoch(self, epoch: int) -> None:
        """Varia a semente por época para que o recorte aleatório mude a cada uma."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        file_name, label = self.items[idx]
        features = self._load_features(file_name, idx)
        return features, label

    def _rng(self, idx: int) -> np.random.Generator | None:
        if not self.random_crop:
            return None
        return np.random.default_rng((self.seed, self._epoch, idx))

    def _load_features(self, file_name: str, idx: int) -> dict[str, torch.Tensor]:
        cache_path = None
        if self.cache_dir:
            cache_path = self.cache_dir / f"{file_name}_{self._cache_key}.pt"
            if cache_path.exists():
                return torch.load(cache_path)

        wav = load_audio(self.audio_dir / f"{file_name}{self.file_ext}",
                         self.audio_cfg["sample_rate"])
        wav = preprocess_waveform(wav, self.audio_cfg, rng=self._rng(idx))
        if self.augmenter is not None:
            wav = self.augmenter(wav)
        features = self.extractor(wav)

        if cache_path is not None:
            torch.save(features, cache_path)
        return features


# --------------------------------------------------------------------------- #
# Smoke — dados sintéticos
# --------------------------------------------------------------------------- #
class SmokeDataset(Dataset):
    """Gera áudios sintéticos separáveis para validar o pipeline.

    - bonafide: soma de harmônicos "limpos" + ruído baixo.
    - spoof: harmônicos + artefato de alta frequência + ruído (imita o tipo de
      inconsistência espectral que um detector deve aprender).

    Não é dado realista — serve apenas para exercitar o código ponta-a-ponta.
    """

    def __init__(self, n: int, audio_cfg: dict, extractor: FeatureExtractor,
                 seed: int = 0, augmenter=None):
        self.n = n
        self.audio_cfg = audio_cfg
        self.extractor = extractor
        self.seed = seed
        self.augmenter = augmenter
        self.sr = audio_cfg["sample_rate"]
        self.n_samples = int(self.sr * audio_cfg["duration"])
        self.labels = [i % 2 for i in range(n)]
        self.ids = [f"smoke_{i:05d}" for i in range(n)]
        # Dois "ataques" fictícios, só para exercitar a análise por sistema.
        self.system_ids = ["-" if i % 2 == 0 else f"A{7 + (i // 2) % 2:02d}"
                           for i in range(n)]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed * 100_000 + idx)
        label = idx % 2  # metade bonafide, metade spoof
        wav = self._synth(rng, spoof=bool(label))
        wav = preprocess_waveform(wav, self.audio_cfg)
        if self.augmenter is not None:
            wav = self.augmenter(wav)
        return self.extractor(wav), label

    def _synth(self, rng: np.random.Generator, spoof: bool) -> np.ndarray:
        t = np.arange(self.n_samples) / self.sr
        f0 = rng.uniform(110, 220)  # frequência fundamental
        wav = np.zeros_like(t)
        for k in range(1, 6):  # harmônicos
            wav += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi))
        wav += 0.01 * rng.standard_normal(self.n_samples)  # ruído de fundo baixo
        if spoof:
            # Artefato de alta frequência (tom puro perto de Nyquist) + ruído extra,
            # imitando a "assinatura" espectral de sinais sintéticos.
            wav += 0.3 * np.sin(2 * np.pi * (self.sr * 0.45) * t)
            wav += 0.03 * rng.standard_normal(self.n_samples)
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

    `augmenter`: callable opcional `wav -> wav` aplicado após o pré-processamento
    (aumentação no treino ou perturbação na avaliação de robustez).
    `random_crop`: recorte temporal aleatório — usar apenas no treino.
    """
    if smoke:
        smoke_cfg = config["smoke"]
        n = smoke_cfg[f"n_{partition}"]
        seed = {"train": 1, "dev": 2, "eval": 3}[partition]
        return SmokeDataset(n, config["audio"], extractor, seed=seed, augmenter=augmenter)

    cache_dir = None
    if config["train"].get("cache_features", False):
        cache_dir = Path(config["data"]["root"]).parent / "cache" / config["experiment"]["name"]
    return ASVspoofDataset(
        protocol_path=config["data"]["protocols"][partition],
        audio_dir=config["data"]["audio_dir"][partition],
        audio_cfg=config["audio"],
        extractor=extractor,
        cache_dir=cache_dir,
        augmenter=augmenter,
        random_crop=random_crop,
        seed=config["experiment"]["seed"],
    )


def _config_fingerprint(audio_cfg: dict, extractor: FeatureExtractor) -> str:
    """Hash curto de áudio+features para invalidar o cache quando a config muda.

    Inclui os *parâmetros* das features (n_filter, n_lfcc, n_mels, ...), não só
    os tipos — caso contrário, alterar `n_filter` reusaria features antigas do
    cache e o experimento seria silenciosamente inválido.
    """
    payload = json.dumps(
        {"audio": audio_cfg, "features": extractor.fingerprint()},
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:8]
