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

## Executando um experimento completo

Em vez de encadear treino, avaliação e análises manualmente, `run_pipeline.py`
roda a sequência inteira e imprime uma tabela comparativa ao final:

```bash
python scripts/run_pipeline.py --config configs/fusion_v4.yaml \
                               --config configs/attention_v4.yaml
```

Etapas: verificação da base → treino → avaliação no `eval` (com arquivo de
scores) → EER por ataque → robustez (com `--robustness`) → tabelas do relatório.

- Cada experimento gera `outputs/<nome>_pipeline.log` com toda a saída.
- Uma falha não derruba os experimentos seguintes; o que falhou é listado no fim.
- `--skip-train` reavalia modelos já treinados; `--dry-run` mostra o plano sem
  executar; `--smoke` testa o encadeamento em segundos.
- Ao final roda `make_report.py` e consolida todos os experimentos numa tabela
  comparativa (`--no-report` desliga).

## Tabelas e figuras do relatório

Cada experimento deixa em `outputs/` vários JSONs soltos. Copiá-los à mão para o
texto é onde o erro entra — um EER desatualizado numa tabela não dá nenhum sinal
de que está errado. `make_report.py` lê tudo o que existe e monta as tabelas:

```bash
python scripts/make_report.py                 # todos os experimentos avaliados
python scripts/make_report.py --only v4       # só os que têm "v4" no nome
```

Gera em `outputs/report/`, cada tabela em três formatos — `.md` para conferir,
`.tex` para colar no documento (já com `\caption` e `\label`), `.csv` para a
planilha:

| Arquivo | Conteúdo |
|---|---|
| `comparativo_eval.*` | EER, acurácia, precisão, recall, F1, limiar, épocas e tempo de treino de cada experimento |
| `por_ataque_eval.*` | matriz ataque × experimento — mostra *onde* cada modelo falha |
| `robustez_eval.*` | EER por condição de ruído/ganho |
| `curvas_comparadas.png` | EER de validação e loss por época, todos no mesmo eixo |
| `eer_por_ataque.png` | barras agrupadas por ataque |

Execuções `--smoke` ficam de fora por padrão (`--include-smoke` inclui): um teste
de 30 segundos com áudio sintético não pode entrar na tabela de resultados.

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

## Fusão de scores entre modelos (ensemble)

Diferente da *fusão de características* do Incremento 2 (dois ramos dentro de um
mesmo modelo), aqui combinam-se as **saídas** de modelos treinados separadamente
— prática padrão dos sistemas do ASVspoof. Não exige retreinar nada:

```bash
python scripts/score_fusion.py \
    --model configs/baseline_v2.yaml  checkpoints/baseline_lfcc_cnn_v2.pt \
    --model configs/baseline_v3a.yaml checkpoints/baseline_lfcc_cnn_v3a.pt \
    --model configs/baseline_v4.yaml  checkpoints/baseline_lcnn_v4.pt
```

Testa quatro regras de combinação e reporta o EER global e por ataque:

| Regra | Quando ajuda |
|---|---|
| `mean` | modelos com calibração parecida |
| `rank` | **usa só a ordenação** — imune a diferenças de escala entre modelos |
| `max` | prioriza detectar spoof (o modelo mais desconfiado decide) |
| `min` | prioriza não barrar áudio autêntico |

## Fusão multi-resolução (experimento extra)

`configs/fusion_multires.yaml` e `attention_multires.yaml` usam **dois ramos
LFCC** com resoluções diferentes (`n_filter` 20 e 70) em vez de LFCC +
espectrograma.

Motivação empírica: a análise por ataque mostrou que baixa resolução vence no
núcleo duro (A10/A12/A18) e alta resolução vence nos ataques semelhantes ao
treino. A fusão de *scores* entre essas duas resoluções levou o EER de 18,99%
para 15,84% — aqui a mesma complementaridade é explorada **dentro de um único
modelo**, treinado ponta a ponta.

Isso **complementa**, não substitui, o `fusion_v4`, que segue o TC1 com
LFCC + espectrograma.

O extrator aceita qualquer número de variantes da mesma feature: o nome do tipo
define o cálculo pelo prefixo (`lfcc*` ou `spectrogram*`) e o bloco de
configuração pelo nome completo. `model.branches` escolhe quais features
alimentam cada ramo:

```yaml
features:
  types: [lfcc, lfcc_hi]
  lfcc:    {n_filter: 20, ...}
  lfcc_hi: {n_filter: 70, ...}
model:
  name: fusion
  branches: [lfcc, lfcc_hi]   # padrão: [lfcc, spectrogram]
```

## Onde o tempo do treino vai

Cada época imprime a separação entre **espera por dados** e **cálculo na GPU**, e
ao final o treino diz qual dos dois é o gargalo:

```
Época   7 | lr=1.00e-03 | loss=0.2411 | dev: ... EER=16.82%
           tempo  138.4s  (dados  44.1s | GPU  78.3s | dev  16.0s)

Tempo de treino: 1h52m em 43 época(s)
Por época (média das 42 últimas): 137s = dados 44s + GPU 78s + dev 16s
Gargalo: cálculo na GPU (57% da época) — o carregamento já acompanha o treino.
```

É essa medida que decide qual otimização vale a pena, e a intuição costuma errar:
se o tempo é quase todo cálculo, aumentar `num_workers` ou ligar cache não muda
nada. Os tempos também vão para o `*_history.json`, então entram na tabela do
relatório.

Ordens de grandeza medidas neste projeto (por áudio, uma thread): decode do FLAC
1,3 ms, pré-processamento 0,6 ms, aumentação 0,6 ms, extração LFCC+deltas 4,0 ms
(6,2 ms com espectrograma). Com 25.380 áudios de treino, algumas horas por
experimento é o esperado — não é sintoma de nada errado.

> **AMP na GTX 1650.** Os configs `v3`/`v4` usam `amp: false` (ver *Estabilidade
> numérica* abaixo). Se o relatório de tempo apontar a GPU como gargalo, vale
> medir `amp: true` numa execução curta: a Turing faz fp16 em taxa dobrada, e as
> proteções contra `NaN` já estão no código. Compare o EER antes de adotar.

## Espaço em disco

O cache de features é o item mais pesado do projeto: cerca de **11 GB** por
configuração de features (train+dev+eval do ASVspoof LA), ou ~25 GB quando o
espectrograma também é extraído.

O cache é indexado pelo *fingerprint* da configuração de features, em
`ml/data/cache/<fingerprint>/<partição>/`. Experimentos com features idênticas
**compartilham** os mesmos arquivos — v3a, v3 e v4 (todos com `n_filter: 70`)
usam uma única pasta.

Cada partição são três arquivos (`data.npy`, `filled.npy`, `meta.json`) em vez de
um `.pt` por áudio. O formato antigo criava ~96 mil arquivos pequenos, reabertos
a cada época: medido, `torch.load` de um `.pt` custa 0,32 ms contra 0,01 ms de
uma linha do arquivo mapeado em memória. Se você tem caches do formato antigo, o
treino avisa quanto espaço dá para recuperar apagando-os.

Apagar `ml/data/cache/` é **sempre seguro**: ele é regenerado automaticamente na
próxima execução (só a primeira época fica mais lenta). Já `ml/checkpoints/`
contém os modelos treinados e `ml/outputs/` os resultados — apague com cuidado.

## Scores salvos e reaproveitados

`evaluate.py` grava os scores da partição em `outputs/<nome>_<partição>_scores.npz`
(precisão total). `per_attack_eval.py` e `score_fusion.py` os reaproveitam em vez
de repetir a inferência — no `eval` do LA, cada passada percorre 71.237 áudios.

O arquivo só é aceito se **a lista de áudios, a partição e os pesos do
checkpoint** baterem. Retreinou o modelo? O reuso é recusado e a inferência
refeita, com o motivo impresso. `--recompute` força o recálculo.

O `.txt` no estilo ASVspoof continua sendo gerado por `--score-file`: ele é o
artefato legível e a entrada do script oficial de t-DCF. O `.npz` existe à parte
porque o `.txt` arredonda em seis casas, o que criaria empates artificiais no
cálculo do EER.

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

### Encoder LCNN e preservação da frequência (v4)

Os configs `*_v4.yaml` trocam duas peças em relação ao v3a:

| Chave | v1–v3 | v4 |
|---|---|---|
| `model.encoder` | `cnn` | `lcnn` |
| `model.pooling` | `stats` | `freq_stats` |

**`encoder: lcnn`** — Light CNN com *Max-Feature-Map* (MFM). No lugar da ReLU,
a MFM divide os canais em duas metades e mantém o máximo elemento a elemento:
em vez de zerar valores negativos, ela faz uma *seleção* entre respostas
concorrentes. É a arquitetura de referência da literatura para detecção de
spoofing com features LFCC.

**`pooling: freq_stats`** — corrige uma perda de informação presente desde o
início: `avg` e `stats` calculavam a **média ao longo da frequência**, colapsando
os 60 bins do LFCC em um único valor antes da classificação. Como os artefatos
de síntese são específicos de faixas de frequência, isso descartava justamente o
sinal discriminante. O `freq_stats` reduz a frequência a um número fixo de faixas
(`model.freq_bins`, padrão 4) e as achata nos canais, preservando *onde* no
espectro cada padrão ocorreu.

Os três incrementos do TC1 continuam valendo — o encoder é ortogonal a eles:

```bash
python train.py --config configs/baseline_v4.yaml    # Incremento 1: LFCC + LCNN
python train.py --config configs/fusion_v4.yaml      # Incremento 2: + espectrograma
python train.py --config configs/attention_v4.yaml   # Incremento 3: + atenção
```

> O encoder `cnn` segue disponível e inalterado: ele é o baseline exigido pelo
> TC1 §4.7 e a referência dos resultados já medidos (v1–v3).

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
| `cudnn_benchmark` | deixa o cuDNN escolher o algoritmo de convolução mais rápido (padrão `true`) |

> **`cudnn_benchmark`** só ajuda porque `audio.duration` fixa o comprimento do
> sinal: as features chegam sempre com o mesmo shape, então o cuDNN mede os
> algoritmos disponíveis na primeira iteração e reusa a escolha no resto do
> treino. Em troca, a escolha pode variar entre máquinas, o que mexe nos últimos
> dígitos do resultado. Para uma execução bit-a-bit reprodutível, use `false`.

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

## Monitor de chamada ao vivo (`monitor.py`)

Analisa uma chamada **em andamento** capturando a saída de áudio do sistema
(loopback WASAPI no Windows). Como pega o que sai da caixa de som, funciona com
Microsoft Teams, Meet, Zoom ou qualquer outro — **sem publicar aplicativo em
tenant nenhum**, sem Azure e sem consentimento de administrador.

```bash
pip install soundcard                    # só para o modo ao vivo

python monitor.py --listar-dispositivos  # descobrir a saída a escutar

# durante uma chamada, gravando o que ouviu
python monitor.py --config configs/fusion_v4.yaml \
    --checkpoint checkpoints/fusion_lcnn_v4.pt \
    --gravar outputs/chamada.wav --json outputs/chamada.json

# reprocessar a gravação (mesmos scores, bit a bit)
python monitor.py --config ... --checkpoint ... --arquivo outputs/chamada.wav
```

O áudio é cortado em janelas do tamanho de `audio.duration` com metade de
sobreposição, e cada janela passa pelo **mesmo** caminho do `infer.py`:
`preprocess_waveform` -> `FeatureExtractor` -> modelo -> softmax. Janelas em
silêncio são marcadas e ficam fora das médias — o modelo nunca viu silêncio
rotulado, e incluí-lo tornaria o resumo sem sentido numa chamada real.

### Resolução temporal: o limite dos 4 segundos

A janela é de `audio.duration` segundos porque **foi assim que os modelos foram
treinados** — o `fix_length` força esse comprimento em toda amostra. Não é
parâmetro livre: mudá-la exige retreinar, e as métricas medidas deixam de valer.

A consequência é um limite duro de resolução:

| trecho sintético inserido | janelas 100% sintéticas | pior janela |
|---|---|---|
| 1 s | 0 | 25% sintética |
| 2 s | 0 | 50% |
| 3 s | 0 | 75% |
| **4 s** | **1** | 100% |
| 10 s | 4 | 100% |

**Abaixo de 4 s nenhuma janela é puramente sintética**: o modelo sempre vê uma
mistura, tendo sido treinado só em áudio homogêneo. Reduzir o passo não resolve —
é consequência da janela, não da taxa de atualização. Um atacante que insere
frases curtas de voz clonada está abaixo da resolução do sistema.

O **passo** é livre. Com 99,65% de folga de CPU, `--hop 1` dá reação mais rápida
sem custo relevante. O primeiro veredito sempre demora 4 s, porque é preciso
encher a janela.

### Janelas sobrepostas não são observações independentes

Com janela de 4 s e passo de 2 s, janelas vizinhas compartilham metade do áudio:

| N janelas | áudio coberto | independentes |
|---|---|---|
| 3 | 8 s | 2,0 |
| 5 | 12 s | 3,0 |
| 10 | 22 s | 5,5 |
| 20 | 42 s | 10,5 |

Uma média de 5 janelas parece 5 observações; são **3**. Por isso o resumo reporta
`janelas_independentes` além da contagem bruta — sem isso, o número sugeriria
mais solidez do que existe.

### O que ficou medido, e o que continua sendo limitação

**O áudio é o mix.** O loopback entrega a soma de todos os participantes. Não há
atribuição por pessoa — para isso só o bot de mídia do Teams serviria, que é
exatamente o caminho que este módulo evita. O resultado é sobre o *trecho*, não
sobre quem falou.

**O canal foi medido — e ele inverte a escolha do modelo.** Em áudio limpo o
`baseline_lfcc_cnn_v2` é o melhor modelo isolado (18,99% contra 20,18%). Sob as
condições de uma chamada, a ordem se inverte (`robustness_eval.py`, eval
completo, 71.237 áudios):

| condição | v2 | fusion_v4 | |
|---|---|---|---|
| limpo | **18,99%** | 20,18% | v2 |
| opus 25 kbps | **20,04%** | 22,08% | v2 |
| banda estreita (8 kHz) | 35,95% | **25,53%** | v4 por 10,4 pp |
| ruído 5 dB SNR | 42,41% | **27,42%** | v4 por 15,0 pp |
| **degradação máxima** | +23,42 pp | **+7,24 pp** | |

O codec Opus custa pouco (+1,2 a +2,8 pp até 15 kbps, abaixo do que o Teams
usa). Quem derruba é perder a banda alta e o ruído acústico do interlocutor. Por
isso o monitor usa o `fusion_lcnn_v4`: ele perde no benchmark limpo e ganha com
folga em tudo que se parece com uma chamada real.

**O ponto de operação não transfere.** O limiar gravado no checkpoint foi
calibrado em áudio limpo, e fora do domínio se comporta de forma imprevisível:
sob o mesmo Opus a 15 kbps o recall do v2 sobe (0,72 -> 0,86) e o do v4 cai
(0,55 -> 0,39). Mesma perturbação, direções contrárias. O monitor mostra
**score**, não veredito, e um ponto de operação confiável exige recalibração no
canal de destino.

É para isso que serve o `--gravar`: toque numa chamada real áudios do ASVspoof
com rótulo conhecido, capture o que chega do outro lado e avalie. Isso mede o
canal com o Opus real e o processamento real, em vez da simulação. O par
gravar/`--arquivo` garante que o resultado seja reproduzível.


## Estrutura

```
ml/
├── configs/              # baseline.yaml, fusion.yaml, attention.yaml
├── src/
│   ├── config.py         # carga de YAML + utilidades (seed, device)
│   ├── metrics.py        # accuracy, precision, recall, F1, EER, matriz
│   ├── scores.py         # scores salvos/reaproveitados entre as análises
│   ├── data/
│   │   ├── dataset.py    # ASVspoofDataset + SmokeDataset
│   │   └── cache.py      # cache de features em arquivo mapeado em memória
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
│   ├── run_pipeline.py   # encadeia treino + avaliação + análises
│   ├── per_attack_eval.py# EER por algoritmo de síntese (TC1 §5.4)
│   ├── score_fusion.py   # fusão de scores entre modelos treinados
│   ├── make_report.py    # tabelas e figuras do relatório
│   └── robustness_eval.py# avaliação de robustez sob ruído/ganho (TC1 §5.5)
├── tests/                # suíte pytest (métricas, features, modelos, etc.)
├── train.py
├── evaluate.py
└── infer.py
```
