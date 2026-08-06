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
from .cache import FeatureCache, legacy_pt_files

LABEL_MAP = {"bonafide": 0, "spoof": 1}


def _shared_epoch() -> torch.Tensor:
    """Contador de época em memória compartilhada entre processos.

    Guardar a época como um `int` comum não funciona com
    `DataLoader(persistent_workers=True)`: os workers recebem uma *cópia* do
    dataset ao serem criados e nunca mais a atualizam, então `set_epoch()` no
    processo principal não os alcança — o recorte aleatório e a aumentação
    ficariam congelados na época em que o worker nasceu, repetindo o mesmo
    trecho de áudio e a mesma perturbação em todas as épocas seguintes.

    Um tensor em memória compartilhada é visto por todos os processos, e
    funciona tanto com `fork` (Linux) quanto com `spawn` (Windows).
    """
    return torch.zeros(1, dtype=torch.long).share_memory_()


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


def protocol_ids_and_systems(config: dict, partition: str) -> tuple[list[str], list[str]]:
    """Ids e algoritmos de síntese de uma partição, direto do protocolo.

    Permite conferir scores salvos em disco sem abrir um único áudio: se eles
    ainda valem para este modelo, o dataset (e o cache de features) nem chega a
    ser construído.
    """
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
        # Cache guarda uma única versão das features, então é incompatível com
        # qualquer aleatoriedade por época (aumentação ou recorte aleatório).
        stochastic = augmenter is not None or random_crop
        self.cache_dir = None
        self.cache = None
        if cache_dir and not stochastic and self.items:
            # O cache é indexado pelo *fingerprint* da configuração de features,
            # não pelo nome do experimento: assim experimentos que usam features
            # idênticas (ex.: v3a, v3 e v4, todos com n_filter=70) compartilham
            # os mesmos arquivos em vez de duplicá-los — cada cópia custa ~11 GB.
            self._cache_key = _config_fingerprint(audio_cfg, extractor)
            self.cache_dir = Path(cache_dir) / self._cache_key
            self.cache = self._open_cache()

    def _open_cache(self) -> FeatureCache | None:
        """Abre o cache da partição, descobrindo os shapes com uma extração.

        Os shapes só são conhecidos depois de extrair uma amostra — e como a
        matriz em disco precisa do tamanho da linha para ser criada, essa
        primeira extração acontece aqui, no processo principal. Ela não é
        desperdiçada: o resultado já é gravado na posição 0.
        """
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
            # Falta de espaço ou permissão não deve derrubar o treino: sem cache
            # ele continua, só recalculando as features a cada época.
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

        Deriva em vez de guardar estado: com `num_workers > 0` o dataset é
        copiado para cada worker, então um `rng` guardado no objeto daria a
        mesma sequência em todos eles — e se repetiria a cada época, quando os
        workers são recriados.
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
    """Gera áudios sintéticos separáveis para validar o pipeline.

    - bonafide: soma de harmônicos "limpos" + ruído baixo.
    - spoof: harmônicos + artefato de alta frequência + ruído (imita o tipo de
      inconsistência espectral que um detector deve aprender).

    Não é dado realista — serve apenas para exercitar o código ponta-a-ponta.
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
        # Sinal mais longo que o alvo quando há recorte aleatório, para que o
        # modo --smoke exercite de fato esse caminho.
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
            # Artefato de alta frequência (tom puro perto de Nyquist) + ruído extra,
            # imitando a "assinatura" espectral de sinais sintéticos.
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

    `augmenter`: callable opcional `wav -> wav` aplicado após o pré-processamento
    (aumentação no treino ou perturbação na avaliação de robustez).
    `random_crop`: recorte temporal aleatório — usar apenas no treino.
    """
    if smoke:
        smoke_cfg = config["smoke"]
        n = smoke_cfg[f"n_{partition}"]
        seed = {"train": 1, "dev": 2, "eval": 3}[partition]
        return SmokeDataset(n, config["audio"], extractor, seed=seed,
                            augmenter=augmenter, random_crop=random_crop)

    cache_dir = None
    if config["train"].get("cache_features", False):
        # Base comum a todos os experimentos; o dataset cria dentro dela uma
        # subpasta por fingerprint de features (ver ASVspoofDataset).
        cache_dir = Path(config["data"]["root"]).parent / "cache"
    return ASVspoofDataset(
        protocol_path=config["data"]["protocols"][partition],
        audio_dir=config["data"]["audio_dir"][partition],
        audio_cfg=config["audio"],
        extractor=extractor,
        cache_dir=cache_dir,
        augmenter=augmenter,
        random_crop=random_crop,
        seed=config["experiment"]["seed"],
        partition=partition,
    )


# Chaves de `audio` que só governam aleatoriedade por época. Não entram no
# fingerprint porque não podem afetar o que é gravado no cache: quando estão
# ativas, o cache é desativado para aquele dataset (ver `stochastic` acima), e
# dev/eval nunca as recebem. Incluí-las apenas duplicaria pastas idênticas.
_NAO_AFETAM_CACHE = ("augment", "random_crop")


def _config_fingerprint(audio_cfg: dict, extractor: FeatureExtractor) -> str:
    """Hash curto de áudio+features para invalidar o cache quando a config muda.

    Inclui os *parâmetros* das features (n_filter, n_lfcc, n_mels, ...), não só
    os tipos — caso contrário, alterar `n_filter` reusaria features antigas do
    cache e o experimento seria silenciosamente inválido.

    A lista de exclusões é curta e explícita de propósito: qualquer chave nova
    de `audio` entra no fingerprint por padrão. Errar para o lado conservador
    custa espaço; errar para o outro lado corromperia o experimento.
    """
    audio_relevante = {k: v for k, v in audio_cfg.items()
                       if k not in _NAO_AFETAM_CACHE}
    payload = json.dumps(
        {"audio": audio_relevante, "features": extractor.fingerprint()},
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:8]
