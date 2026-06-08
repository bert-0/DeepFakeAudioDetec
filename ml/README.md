# `ml/` — Pipeline de IA

Pipeline de detecção de deepfakes em áudio (pré-processamento → extração de
características → modelo → métricas), construído nos três incrementos descritos
no TC1.

## Instalação

```bash
cd ml
python -m venv .venv && source .venv/bin/activate   # opcional
pip install -r requirements.txt
```

> **GPU:** o pipeline detecta CUDA automaticamente e cai para CPU se não houver.
> Uma GTX 1650 (4 GB) roda os três incrementos sem problemas — use
> `cache_features: true` no config e `batch_size` 16–32.

## Modo `--smoke` (validação rápida, sem dataset)

Gera um punhado de áudios sintéticos (bonafide vs. spoof) e roda o pipeline
inteiro em segundos. Serve para validar o código antes de ter o ASVspoof.

```bash
python train.py    --config configs/baseline.yaml --smoke
python evaluate.py --config configs/baseline.yaml --smoke \
                   --checkpoint checkpoints/baseline_lfcc_cnn.pt
```

Funciona igual para `configs/fusion.yaml` e `configs/attention.yaml`.

## Treino real (ASVspoof 2019 LA)

1. Baixe a base e coloque em `ml/data/LA/` (veja [`data/README.md`](data/README.md)).
2. **Verifique a base antes de treinar** (confere protocolos, contagem
   bonafide/spoof e se os `.flac` existem e abrem):

```bash
python scripts/check_data.py --config configs/baseline.yaml
```

3. Treine cada incremento:

```bash
python train.py    --config configs/baseline.yaml
python train.py    --config configs/fusion.yaml
python train.py    --config configs/attention.yaml
```

4. Avalie no conjunto de teste (eval) com EER, F1 e matriz de confusão:

```bash
python evaluate.py --config configs/baseline.yaml \
                   --checkpoint checkpoints/baseline_lfcc_cnn.pt --partition eval
```

## Inferência em um único áudio (RF05/RF06/RF07)

```bash
python infer.py --config configs/baseline.yaml \
                --checkpoint checkpoints/baseline_lfcc_cnn.pt \
                --audio caminho/para/audio.wav
```

Saída: classificação (`bonafide`/`spoof`) e a probabilidade associada.

## Estrutura

```
ml/
├── configs/              # baseline.yaml, fusion.yaml, attention.yaml
├── src/
│   ├── config.py         # carga de YAML + utilidades (seed, device)
│   ├── metrics.py        # accuracy, precision, recall, F1, EER, matriz
│   ├── data/
│   │   └── dataset.py    # ASVspoofDataset + SmokeDataset
│   ├── preprocess/
│   │   └── audio.py      # PreProcessador: resample, mono, fix-length, silêncio
│   ├── features/
│   │   ├── lfcc.py       # LFCC + delta/delta-delta
│   │   ├── spectrogram.py# log-mel espectrograma
│   │   ├── deltas.py
│   │   └── extractor.py  # ExtratorCaracteristicas (orquestra as features)
│   └── models/
│       ├── blocks.py     # blocos CNN compartilhados
│       ├── baseline_cnn.py
│       ├── fusion.py     # Incremento 2
│       ├── attention.py  # Incremento 3
│       └── registry.py   # build_model(config)
├── scripts/
│   └── check_data.py     # valida a estrutura/conteúdo da base antes do treino
├── train.py
├── evaluate.py
└── infer.py
```
