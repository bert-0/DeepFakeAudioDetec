# `ml/` — Pipeline de IA

Pipeline de detecção de deepfakes em áudio (pré-processamento → extração de
características → modelo → métricas), construído nos três incrementos descritos
no TC1.

## Instalação

```bash
cd ml
python -m venv .venv && source .venv/bin/activate   # opcional
pip install -r requirements.txt
pip install -r requirements-dev.txt   # opcional: para rodar os testes
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

4. Avalie no conjunto de teste (eval) com EER, F1 e matriz de confusão
   (use `--score-file` para exportar os scores por utterance, estilo ASVspoof):

```bash
python evaluate.py --config configs/baseline.yaml \
                   --checkpoint checkpoints/baseline_lfcc_cnn.pt --partition eval \
                   --score-file outputs/baseline_scores.txt
```

Cada treino também salva, em `outputs/`: o **histórico por época** (`*_history.json`)
e as **curvas** de loss/EER/F1 (`*_curves.png`) — úteis para o relatório.

## Avaliação de robustez (TC1 §5.5)

Mede a degradação do modelo sob ruído de fundo e variações de ganho aplicados
ao conjunto de avaliação, gerando uma tabela e um gráfico de EER por condição:

```bash
python scripts/robustness_eval.py --config configs/attention.yaml \
                                  --checkpoint checkpoints/attention_fusion.pt
```

## Análise por tipo de ataque (TC1 §5.4)

O EER global esconde *onde* o modelo falha: 20% pode ser "20% em todos os
ataques" ou "0% em doze e 90% em um". Este script calcula o EER por algoritmo
de síntese (A07…A19), sempre contra todos os bonafide — o protocolo padrão de
reporte da ASVspoof:

```bash
python scripts/per_attack_eval.py --config configs/baseline_v3a.yaml \
                                  --checkpoint checkpoints/baseline_lfcc_cnn_v3a.pt
```

Gera uma tabela ordenada, um JSON e um gráfico de barras destacando os ataques
acima do EER global. Não exige retreinar o modelo.

## Aumentação de dados (treino)

Desativada por padrão (para uma comparação justa entre os incrementos). Para
ligar, edite `audio.augment.enabled: true` no config — adiciona ruído/ganho/shift
aleatórios no treino, melhorando a robustez. Quando ligada, o cache de features
do treino é automaticamente desativado.

## Versões dos configs (`v1` → `v2` → `v3`)

Cada incremento tem várias versões, formando uma **série de ablação** para o
relatório. Como os nomes de experimento diferem, nada se sobrescreve.

| Config | O que muda | Motivação |
|---|---|---|
| `baseline.yaml` (**v1**) | configuração inicial | referência |
| `baseline_v2.yaml` (**v2**) | 5 melhorias anti-overfitting | o v1 memorizava o treino |
| `baseline_v3a.yaml` (**v3a**) | v2 **+ apenas** `n_filter: 70` | isola o efeito da resolução espectral |
| `baseline_v3.yaml` (**v3**) | v3a + encoder de 4 blocos + `patience: 12` | mais capacidade, já com regularização |

Rodar `v3a` **e** `v3` permite separar quanto veio das *features* e quanto veio
da *arquitetura* — uma ablação limpa para a seção de resultados.

### Por que `n_filter: 20 → 70`

O banco de filtros do LFCC define a resolução espectral da feature:

| `n_filter` | Largura de cada filtro |
|---|---|
| 20 | 762 Hz (muito grosseiro) |
| 70 | 225 Hz |

Artefatos de síntese de voz vivem na estrutura fina do espectro, sobretudo em
alta frequência. Com filtros de 762 Hz essa informação é borrada antes de chegar
ao modelo. Além disso, com `n_filter == n_lfcc` a DCT guarda todos os
coeficientes e deixa de comprimir/decorrelacionar — o valor de referência da
literatura para o ASVspoof é ~70 filtros com 20 coeficientes.

> ⚠️ Mudar qualquer parâmetro de feature (`n_filter`, `n_lfcc`, `n_mels`, ...)
> invalida o cache automaticamente — as features são recalculadas na primeira
> época. Isso é intencional: reusar cache antigo tornaria o experimento inválido.

## Versões `v1` × `v2` dos configs

Cada incremento tem dois configs, para permitir a comparação "antes × depois":

- `configs/baseline.yaml` (**v1**) — configuração original, sem as melhorias.
- `configs/baseline_v2.yaml` (**v2**) — mesmas features e arquitetura, com as
  cinco melhorias anti-overfitting ligadas.

O mesmo vale para `fusion*` e `attention*`. Como os nomes de experimento diferem
(`..._v2`), os checkpoints e saídas não se sobrescrevem.

| Melhoria | Chave | v1 | v2 |
|---|---|---|---|
| Threshold calibrado no dev | `train.calibrate_threshold` | `false` | `true` |
| Recorte temporal aleatório | `audio.random_crop` | `false` | `true` |
| Aumentação no treino | `audio.augment.enabled` | `false` | `true` |
| Peso de classe suavizado | `train.class_weights` | `auto` | `sqrt` |
| Pooling estatístico | `model.pooling` | `avg` | `stats` |

Além dessas, `model.channels` controla a profundidade/largura do encoder
(`[16, 32, 64]` em v1/v2, `[32, 64, 128, 128]` em v3). Cada entrada da lista é um
bloco convolucional com aquele número de canais.

### Por que cada uma

- **`calibrate_threshold`** — reporta as métricas no ponto de corte do EER
  (medido no dev) em vez do 0,5 fixo. Sem isso, accuracy/F1 oscilam muito quando
  há desbalanceamento e pesos de classe. O threshold escolhido é salvo dentro do
  checkpoint e reutilizado por `evaluate.py` e `infer.py`.
- **`random_crop`** — cada época vê um trecho temporal diferente do mesmo áudio
  (antes era sempre o início), aumentando a diversidade dos dados.
- **`augment`** — ruído/ganho/deslocamento aleatórios, para generalizar melhor a
  ataques não vistos no treino.
- **`class_weights: sqrt`** — compensação mais suave (√ da razão) que `auto`, que
  no ASVspoof LA chega a ~8,8× e desloca demais o ponto de decisão.
- **`pooling: stats`** — concatena média **e desvio-padrão** temporais, preservando
  a variação do sinal que a média global descarta. No modelo de atenção isso vira
  *Attentive Statistics Pooling*.

> ⚠️ `random_crop` e `augment` desativam automaticamente o cache de features **do
> treino** (o cache congelaria uma única versão aleatória). Dev e eval seguem
> usando cache normalmente.

## Estabilidade numérica (AMP)

Os configs `v3` usam **`amp: false`** de propósito. Em GPUs sem Tensor Cores
(caso da GTX 1650) o ganho de velocidade do mixed precision é pequeno, e os
modelos aqui são de poucos MB — não há pressão de memória que compense o risco.

O risco é concreto: em fp16 a variância calculada pelo pooling estatístico é uma
soma de quadrados que ultrapassa o alcance do tipo (~65504), virando `inf` e
depois `NaN`. O NaN contamina as estatísticas do BatchNorm e o modelo passa a
emitir `NaN` para sempre em modo `eval`.

Proteções no código (valem mesmo com AMP ligado):

- O pooling estatístico calcula média/desvio sempre em **float32**.
- Batches com loss `inf`/`NaN` são descartados, com aviso ao fim da época.
- `grad_clip` limita a norma do gradiente.
- Se o dev produzir `NaN`, o treino **encerra de forma limpa** preservando o
  melhor checkpoint, em vez de estourar exceção e perder a execução.

## Proteção dos checkpoints

Ao iniciar um treino, o melhor EER ainda é infinito — então a primeira época
sempre salva por cima. Para que um treino novo não destrua o melhor modelo do
anterior, os arquivos existentes são renomeados antes de começar:

```
checkpoints/exp.pt        ->  checkpoints/exp_prev.pt
checkpoints/exp_last.pt   ->  checkpoints/exp_last_prev.pt
```

O backup guarda a execução **imediatamente anterior**. Para preservar um modelo
importante por mais tempo, copie-o com outro nome.

## Calibrando um modelo já treinado

Para corrigir o ponto de operação de um checkpoint antigo **sem retreinar**:

```bash
python evaluate.py --config configs/baseline.yaml \
                   --checkpoint checkpoints/baseline_lfcc_cnn.pt \
                   --partition eval --calibrate-on dev
```

O threshold é calculado no `dev` e aplicado ao `eval` — nunca calibrando na
partição de teste. Use `--threshold 0.42` para informar um valor manualmente.

## Principais ajustes do config (seção `train`)

| Chave | Efeito |
|-------|--------|
| `class_weights` | `auto` (compensação total), `sqrt` (suave) ou `none` |
| `calibrate_threshold` | reporta métricas no corte do EER em vez de 0,5 |
| `scheduler` | reduz o learning rate quando o EER de validação estagna |
| `early_stopping_patience` | interrompe o treino sem melhora por N épocas (0 = desligado) |
| `amp` | mixed precision em GPU — ver aviso abaixo |
| `grad_clip` | limita a norma do gradiente (`0` desliga); ajuda contra divergência |
| `cache_features` | salva features em disco para acelerar épocas seguintes |

## Testes

```bash
python -m pytest          # roda a suíte (rápida, ~3 s, não precisa do dataset)
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
│   │   ├── audio.py      # PreProcessador: resample, mono, fix-length, silêncio
│   │   └── augment.py    # aumentação (treino) e perturbações (robustez)
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
│   ├── check_data.py     # valida a estrutura/conteúdo da base antes do treino
│   └── robustness_eval.py# avaliação de robustez sob ruído/ganho (TC1 §5.5)
├── tests/                # suíte pytest (métricas, features, modelos, etc.)
├── train.py
├── evaluate.py
└── infer.py
```
