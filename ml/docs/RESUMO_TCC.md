# Resumo consolidado para a escrita do TCC

Documento de referência com **tudo o que foi medido** no projeto. Serve tanto ao
TC1 (artigo científico) quanto à APS (engenharia de software).

> **Regra deste documento:** só entra número que saiu de execução registrada.
> Onde o número depende de uma rodada que precisa ser refeita, o lugar está
> marcado com `[REGENERAR]` e o comando que produz o valor. Nada é estimado.
>
> Versionado em `ml/docs/` de propósito: a versão anterior morava em
> `ml/outputs/`, que é gitignored, e se perdeu quando o ambiente foi reciclado.

Última atualização: setembro de 2026 · 455 testes automatizados passando.

---

## 1. O que foi construído

Um detector de deepfake de áudio com duas frentes:

- **TC1** — *Detecção de Deep Fakes em Áudio com Fusão de Características e
  Atenção: um estudo sobre robustez e generalização.* Mede o ganho de dois
  incrementos (fusão de características e atenção) e a robustez ao canal.
- **APS** — sistema para o cliente fictício *CyberShield*, com requisitos
  funcionais e não funcionais próprios.

O código cobre o ciclo inteiro: verificação da base, treino, avaliação,
avaliação por ataque, avaliação sob degradação de canal, fusão de scores,
inferência em arquivo único, monitor de chamada ao vivo e avaliação através de
uma chamada real.

---

## 2. Base e protocolo

**ASVspoof 2019, trilha Logical Access (LA).**

| partição | arquivos | ataques |
|---|---|---|
| train | 25.380 | A01–A06 |
| dev | 24.844 | A01–A06 |
| eval | 71.237 | **A07–A19 (13 ataques inéditos)** |
| **total** | **121.461** | |

O eval tem 7.355 bonafide e 63.882 spoof — **4.914 por ataque**, e **89,7% de
spoof**. Esse desequilíbrio volta a importar na Seção 8.

O ponto metodológico central: **os ataques do eval não aparecem no treino**.
Todo número de eval neste documento é generalização para ataque não visto, não
desempenho em distribuição conhecida. É por isso que os EERs são de dezenas de
pontos percentuais e não de unidades.

A integridade da base é verificada por `scripts/check_data.py --deep`, que
decodifica os arquivos em vez de ler só o cabeçalho — um FLAC truncado reporta a
duração original e passaria despercebido até derrubar o treino horas depois.

---

## 3. Resultados dos modelos isolados

EER no eval completo (71.237 áudios). Menor é melhor.

| modelo | EER eval |
|---|---|
| **baseline_lfcc_cnn_v2** | **18,99%** |
| fusion_lcnn_v4 | 20,18% |
| baseline_lfcc_cnn | 20,78% |
| baseline_lfcc_cnn_v3 | 21,10% |
| attention_lcnn_v4 | 21,13% |
| baseline_lfcc_cnn_v3a | 21,23% |
| baseline_lcnn_v4 | 21,56% |

**Os dois incrementos do TC1, isolados.** Cada incremento é comparado com o
modelo imediatamente anterior, de modo que só uma coisa muda por vez:

| incremento | contraste controlado | o que muda | efeito |
|---|---|---|---|
| fusão de características (LFCC + espectrograma) | `baseline_lcnn_v4` → `fusion_lcnn_v4` | acrescenta o ramo de espectrograma | **−1,38 pp** (21,56 → 20,18) |
| atenção no pooling | `fusion_lcnn_v4` → `attention_lcnn_v4` | só o pooling | **+0,95 pp** (20,18 → 21,13) |

> **Não escrever "21,56 → 21,13" para a atenção.** Essa diferença é −0,43 pp e
> mistura dois incrementos (ramo extra + atenção). O contraste que isola a
> atenção é contra o `fusion_lcnn_v4`: os dois configs diferem apenas em
> `experiment.name` e `model.name` (`diff configs/fusion_v4.yaml
> configs/attention_v4.yaml`). No texto, sempre **fusão → atenção**.

A conclusão honesta do TC1 é que **a fusão ajuda e a atenção não**. Reportar o
resultado negativo da atenção vale mais do que escondê-lo: ele foi medido com o
mesmo encoder, a mesma base e a mesma semente.

**Escopo do resultado negativo — escrever de forma restrita.** A atenção testada
é mínima (`AttentiveFreqStatsPool`, `src/models/blocks.py`): cada frame é
pontuado por **uma única projeção linear** (`Conv1d` de kernel 1), sem camada
oculta e sem `tanh`, seguida de softmax no tempo. São **258 parâmetros a mais**
(349.124 contra 348.866: 32 canais × 4 faixas + 1 viés, por ramo), atuando
sobre cerca de 25 frames depois das quatro reduções temporais do encoder. O
resultado vale para **esta** atenção, não para "atenção" em geral. A referência
certa é Okabe et al. (2018), *Attentive Statistics Pooling*, que é o que o código
implementa; Vaswani et al. (2017) descreve outro mecanismo (autoatenção
multi-cabeça) e não deve ser citado como a base deste incremento.

**A linhagem v1 → v4 é a evidência do ciclo incremental** (TC1 §4.7):

| versão | o que mudou | modelo |
|---|---|---|
| v1 | CNN rasa, LFCC (20 filtros) com deltas, pooling médio, sem augmentation | `baseline_lfcc_cnn` |
| v2 | mesma arquitetura + augmentation, recorte aleatório, pooling de estatísticas, pesos `sqrt`, limiar calibrado no dev | `baseline_lfcc_cnn_v2` |
| v3a | `n_filter` 20 → 70 (ablação de resolução; também desliga AMP e liga *grad clip*) | `baseline_lfcc_cnn_v3a` |
| v3 | v3a + encoder mais largo (canais até 128) + mais paciência | `baseline_lfcc_cnn_v3` |
| v4 | encoder LCNN (MFM) + pooling que preserva frequência | `baseline_lcnn_v4` |
| v4 | + ramo de espectrograma (**incremento 2**) | `fusion_lcnn_v4` |
| v4 | + atenção no pooling (**incremento 3**) | `attention_lcnn_v4` |

Dois ajustes para o texto do TC1: (1) os **deltas e delta-deltas do LFCC estão
no baseline em todas as versões** — o incremento 2 acrescenta só o
espectrograma; (2) a linha de base do contraste controlado é o
`baseline_lcnn_v4` (LCNN, um ramo), **não** a "CNN convencional" descrita no
TC1.

### 3-0. Arquitetura (o que está no código)

Seção que o TC1 ainda não tem. Valores de `configs/*_v4.yaml` e `src/`.

- **Pré-processamento:** 16 kHz, janela fixa de **4 s** (`fix_length`, completa
  por repetição), remoção de silêncio (`trim_silence`, `top_db: 30`) e
  normalização por pico.
- **Features**, ambas do **mesmo `|STFT|²`** (`n_fft` 512, janela 400 = 25 ms,
  passo 160 = 10 ms):
  - **LFCC:** 20 coeficientes, 70 filtros lineares, com delta e delta-delta →
    **60 × 401**.
  - **Espectrograma log-mel:** 80 bandas → 80 × 401.
- **Encoder:** LCNN com *Max-Feature-Map* (Lavrentyeva et al., 2019), um por
  representação.
- **Pooling:** estatísticas (média e desvio) preservando 4 faixas de
  frequência (`freq_bins: 4`); na atenção, média e desvio ponderados.
  Detalhe que a banca pode perguntar: no ramo LFCC as 60 linhas viram 3 depois
  das reduções, e o `AdaptiveAvgPool2d` com 4 faixas as estica para 4 faixas
  que se sobrepõem. Não invalida nada, mas é bom saber responder.
- **Fusão tardia:** os vetores de cada ramo são concatenados **depois do
  pooling**; uma cabeça com dropout 0,3 decide.
- **Treino:** Adam (lr 1e-3, weight decay 1e-4), batch 32, até 50 épocas,
  pesos de classe `sqrt`, `ReduceLROnPlateau` (fator 0,5, paciência 3), early
  stopping (paciência 12) e seleção de checkpoint pelo **EER do dev**, *grad
  clip* 5,0, semente 42.
- **Data augmentation** (prob. 0,5 cada): ruído gaussiano branco a SNR de
  **10–30 dB**, ganho de **±6 dB**, deslocamento de até 10% da janela, e recorte
  aleatório. Ligada do v2 em diante (o v1 treina sem ela). Ver a ressalva na
  Seção 5.
- **Hardware:** GTX 1650 (≈3 h por treino no 2019 LA).

### 3-A. Como funciona a fusão (duas coisas diferentes com o mesmo nome)

**Fusão de características** (`fusion_lcnn_v4`): dois encoders LCNN, um para
LFCC e outro para espectrograma log-mel, treinados **juntos**. As saídas são
concatenadas *depois do pooling* e uma cabeça só decide. O modelo aprende a
combinar as duas representações durante o treino.

**Fusão de scores** (`scripts/score_fusion.py`): dois modelos treinados
**separadamente**, cada um produz seu score, e os scores são combinados depois.
Não há treino conjunto.

As duas coexistem e não competem: o melhor resultado do projeto é uma fusão de
scores *entre* um modelo com fusão de características e um sem.

Detalhe compartilhado: LFCC e espectrograma partem do **mesmo `|STFT|²`**
(`src/features/stft.py`), calculado uma vez. Os dois ramos custam bem menos que
dois pipelines independentes.

---

## 4. Fusão de scores — o melhor resultado do projeto

| combinação | EER eval | |
|---|---|---|
| **baseline_v2 + fusion_v4** | **13,13%** | melhor |
| baseline_v2 + attention_v4 | 13,47% | |
| baseline_v2 + baseline_v3 | 15,84% | ¹ |
| fusion_v4 + attention_v4 | 19,88% | **controle** |

¹ Não descrever este par como "só resolução": o v3 muda também os canais do
encoder (≈241 mil parâmetros contra ≈23 mil do v2). O par que isolaria a
resolução é v2 + v3a, que não foi fundido.

De 18,99% para **13,13%**: ganho de 5,86 pp sobre o melhor modelo isolado.

**O controle é a parte importante.** Combinar `fusion_v4` com `attention_v4` —
dois modelos parecidos, mesmo encoder, mesmo pooling family — rende só −0,30 pp.
Combinar modelos **diferentes** (uma CNN rasa com LFCC e uma LCNN com dois ramos)
rende −5,86 pp. Isso mostra que o ganho vem da **diversidade entre os modelos**,
não do simples ato de somar scores. Sem esse controle, a afirmação não se
sustentaria.

**Regras de combinação e o que é aplicável ao vivo:**

| regra | EER (v2+fusion_v4) | streamável? |
|---|---|---|
| `rank` | 13,13% | **não** |
| `mean` | 14,03% | sim |

A regra `rank` atribui postos, e para isso precisa do **conjunto inteiro** de
scores. Num fluxo ao vivo existe uma janela por vez, então ela não existe. Os
0,90 pp de diferença são o preço da viabilidade em tempo real — e o monitor usa
`mean` por isso.

---

## 5. Robustez ao canal — onde o ranking se inverte

`scripts/robustness_eval.py`, eval completo, 71.237 áudios por condição.

| condição | baseline_v2 | fusion_v4 | vence |
|---|---|---|---|
| limpo | **18,99%** | 20,18% | v2 |
| opus 25 kbps | **20,04%** | 22,08% | v2 |
| banda estreita (8 kHz) | 35,95% | **25,53%** | **v4 por 10,4 pp** |
| ruído 5 dB SNR | 42,41% | **27,42%** | **v4 por 15,0 pp** |
| **degradação máxima** | **+23,42 pp** | **+7,24 pp** | |

**Este é o resultado mais forte do TC1.** O modelo que vence no benchmark limpo
é o que desaba no canal degradado. A ordem se inverte. Escolher modelo pelo EER
limpo — que é o que a literatura reporta — leva à escolha errada para qualquer
aplicação real.

O que custa o quê:

**Ressalva obrigatória: parte da robustez a ruído é "dentro da distribuição".**
O treino usa ruído gaussiano branco a 10–30 dB e ganho de ±6 dB, com a **mesma
função** `add_noise` (`src/preprocess/augment.py`) que o teste de robustez
aplica. As condições de 20 dB, 10 dB e ±6 dB já foram vistas no treino. As
condições **realmente inéditas** são **ruído a 5 dB, Opus e banda estreita** —
e são exatamente as duas linhas em que o ranking se inverte (5 dB e banda
estreita). A inversão continua justa, porque os dois modelos tiveram a mesma
augmentation, mas isso precisa estar declarado no texto.

**O codec testado é Opus, não MP3.** O `infer.py` aceita MP3, mas nenhuma
avaliação foi feita com esse formato. O texto não deve afirmar robustez a MP3.

- **O codec Opus custa pouco**: +1,2 a +2,8 pp até 15 kbps, abaixo do que o
  Teams usa na prática.
- **Perder a banda alta custa muito**, e o ruído acústico do interlocutor custa
  mais ainda.

**O ponto de operação não transfere.** O limiar gravado no checkpoint foi
calibrado em áudio limpo. Sob o mesmo Opus a 15 kbps, o recall do v2 **sobe**
(0,72 → 0,86) e o do v4 **cai** (0,55 → 0,39) — mesma perturbação, direções
contrárias. Por isso o monitor exibe **score**, não veredito.

### 5.1 Régua externa (Seção 5.9 do TC1)

**Baselines oficiais — confirmados na fonte primária.** Todisco et al. (2019),
Tabela 1, cenário LA, eval, resultados agrupados sobre todos os ataques:

| sistema | min t-DCF | EER | posição entre 50 |
|---|---|---|---|
| B02 (LFCC-GMM) | 0,2116 | **8,09%** | 28ª |
| B01 (CQCC-GMM) | 0,2366 | **9,57%** | 33ª |
| melhor sistema (T05) | 0,0069 | 0,22% | 1ª |

27 das 48 equipes superaram o B02. Os números "21,13% / 15,80%" que apareceram
em buscas anteriores **não** são do eval de 2019 e foram descartados.

**Por que essa régua não é equivalente.** Os baselines oficiais processam o
áudio **com** o silêncio. Este projeto **remove** o silêncio (`trim_silence:
true`, `top_db: 30`). Müller et al. (2021) mostram que isso muda o problema:

- No ASVspoof 2019, o bonafide tem silêncio inicial e final bem mais longo. Uma
  rede densa que recebe **um único número**, a duração do silêncio inicial,
  chega a **15,12% de EER** no eval (85% de acurácia).
- O RawNet2, baseline oficial do ASVspoof 2021, passa de **3,61%** com silêncio
  para **15,50% ± 5,2** com o silêncio removido no treino: cinco vezes pior.
- Um LSTM treinado com silêncio e avaliado com o silêncio removido do eval vai
  de 7,35% para **35,32%**.

**A régua equivalente existe, e está no mesmo artigo.** A Tabela 2 de Müller et
al. treina modelos no ASVspoof 2019 **com o silêncio removido** e avalia no
eval, o mesmo protocolo deste projeto:

| modelo, silêncio removido | EER eval 2019 |
|---|---|
| ResNet (CQT), Müller et al. | 27,23% ± 3,2 |
| CNN (CQT), Müller et al. | 26,27% ± 3,5 |
| LSTM (CQT), Müller et al. | 27,28% ± 1,4 |
| RawNet2, Müller et al. | 15,50% ± 5,2 |
| **baseline_lfcc_cnn_v2** (este) | **18,99%** |
| **fusion_lcnn_v4** (este) | **20,18%** |
| **v2 + fusion_v4, fusão de scores** (este) | **13,13%** |

Sob o protocolo equivalente, **os sete modelos deste projeto (18,99% a 21,56%)
ficam abaixo dos três modelos CQT de Müller (26–27%)**, e a fusão de scores
(13,13%) fica abaixo da média do RawNet2 (15,50%). Os modelos isolados ficam
acima do RawNet2, mas dentro do desvio que o próprio Müller mede para ele
(15,50 ± 5,2 → até 20,7%).

Diferenças que ficam registradas, para não comparar mais do que é comparável:
limiar do trim (Müller usa `top_db` 40, este projeto 30 — aqui o corte é um
pouco mais agressivo); features (CQT e forma de onda contra LFCC e log-mel);
duração (Müller usa o áudio inteiro, este projeto janela fixa de 4 s); e
repetições (Müller reporta média ± desvio de várias execuções, este projeto uma
execução por modelo).

**Como escrever:** citar B01/B02 como os baselines oficiais, dizer
explicitamente que eles usam o silêncio, citar Müller et al. para o efeito do
silêncio, e usar a Tabela 2 de Müller como a comparação de mesmo protocolo. Nem
esconder o 8,09%, nem comparar com ele como se fosse equivalente.

### 5.2 Uma limitação que a régua externa expõe: uma execução por modelo

Müller et al. reportam desvios de **1,4 a 5,2 pp entre execuções do mesmo
modelo**. Este projeto treinou cada modelo **uma vez** (semente 42). Não há como
saber a variância daqui sem retreinar, mas se ela for da mesma ordem:

| efeito medido | tamanho | acima de 1,4–5,2 pp? |
|---|---|---|
| diferença por ataque entre v2 e v4 (Seção 7) | até 30 pp | **sim** |
| inversão de ranking sob canal (Seção 5) | 10,4 e 15,0 pp | **sim** |
| ganho da fusão de scores (Seção 4) | 5,86 pp | **sim** |
| incremento da fusão de características (Seção 3) | −1,38 pp | **não** |
| incremento da atenção (Seção 3) | +0,95 pp | **não** |

Os resultados grandes sobrevivem. **Os dois incrementos do TC1 (−1,38 e +0,95
pp) ficam dentro da faixa de variação entre execuções que Müller observa**, e
não podem ser atribuídos ao método com uma execução só. Retreinar com mais
sementes resolveria (≈3 h por treino, medido), mas o caminho barato é declarar
a limitação no texto e apoiar a conclusão sobre a fusão na Seção 7, onde o
efeito é por ataque e muito maior.

---

## 6. O modelo está mesmo aprendendo? (verificação de atalhos)

Um detector pode acertar pelo motivo errado: se os áudios spoof forem
sistematicamente mais curtos ou mais baixos, o modelo aprende duração e energia
em vez de artefato de síntese.

### 6.1 A base tem atalhos? (`scripts/check_shortcut.py`)

Mede o EER que se obteria **usando só** uma variável trivial, e compara com um
teste de permutação (500 embaralhamentos, limiar no percentil 5).

| partição | variável | EER trivial | p5 | veredito |
|---|---|---|---|---|
| train | duração após o trim | **43,80%** | 47,20% | **significativo** |
| eval | duração, RMS, duração pós-trim | RMS: **32,02%** | — | **os três significativos** |

**Sim, a base tem atalhos.** Isso é fato sobre o ASVspoof, não sobre o modelo.

> Nota de método: a primeira versão deste script usava um limiar fixo de 40% e
> deu *passou* num efeito real de 43,80%. Limiar arbitrário não serve; o teste
> de permutação deriva o limiar dos próprios dados.

### 6.2 O modelo usa os atalhos? (`scripts/check_score_confound.py`)

Correlação de Spearman **dentro de cada classe** entre o score do modelo e a
variável trivial. Dentro da classe, a variável não pode ajudar a decidir — se
ainda assim correlaciona, o modelo está olhando para ela.

| variável | ρ | variância explicada |
|---|---|---|
| RMS | −0,140 | **2,0%** |
| duração | −0,099 | **1,0%** |

O piso de ruído é derivado do tamanho da amostra (`2/√(n−1)`), não de um limiar
escolhido a dedo.

**Conclusão:** a dependência existe e é estatisticamente real, mas explica 1–2%
da variância do score. Os outros 98% vêm de outra coisa. **O modelo aprende o
artefato, não o atalho** — com a ressalva quantificada.

> Nota de método: a primeira rodada usou 220 áudios bonafide e deu "desprezível"
> por 4 milésimos. Isso não era resultado negativo, era **indeterminado**. Com
> amostra maior o efeito virou significativo. Amostra pequena não prova ausência.

---

## 7. Desempenho por ataque — o resultado que o EER global esconde

`scripts/per_attack_eval.py`, eval completo, 4.914 áudios por ataque contra os
mesmos 7.355 bonafide.

| ataque | baseline_v2 | fusion_v4 | diferença | vence |
|---|---|---|---|---|
| A13 | 36,50% | **6,54%** | −29,96 pp | fusion_v4 |
| A16 | 22,75% | **0,03%** | −22,72 pp | fusion_v4 |
| A07 | 22,43% | **0,02%** | −22,41 pp | fusion_v4 |
| A17 | 11,27% | **0,41%** | −10,86 pp | fusion_v4 |
| A18 | 15,01% | **4,63%** | −10,38 pp | fusion_v4 |
| A19 | 6,98% | **0,00%** | −6,98 pp | fusion_v4 |
| A08 | 3,88% | **0,03%** | −3,85 pp | fusion_v4 |
| A09 | 2,32% | **0,12%** | −2,20 pp | fusion_v4 |
| A14 | **12,01%** | 17,74% | +5,73 pp | v2 |
| A12 | **36,18%** | 47,99% | +11,81 pp | v2 |
| A15 | **15,02%** | 29,37% | +14,35 pp | v2 |
| A10 | **30,63%** | 45,54% | +14,91 pp | v2 |
| A11 | **3,85%** | 33,74% | +29,89 pp | v2 |
| **EER global** | **18,99%** | 20,18% | | |

### 7.1 Os dois modelos erram em ataques diferentes

É o achado central desta seção. Os EERs globais são quase iguais (18,99% contra
20,18%) e **os perfis são quase opostos**: `fusion_v4` vence em 8 dos 13
ataques, `baseline_v2` nos outros 5, com diferenças de até 30 pp nos dois
sentidos.

O par A13 / A11 resume tudo:

- **A13** é o pior ataque do `baseline_v2` (36,50%) e o `fusion_v4` quase o
  resolve (6,54%).
- **A11** é quase resolvido pelo `baseline_v2` (3,85%) e derruba o `fusion_v4`
  (33,74%).

Mesma base, mesmos ataques, dois modelos com EER global equivalente — e o erro
distribuído de formas incompatíveis.

**Isto explica a Seção 4.** A fusão de scores rende 5,86 pp (18,99% → 13,13%)
porque os modelos são complementares, e agora isso está *medido*, não inferido.
O grupo de controle da Seção 4 já apontava nessa direção; aqui se vê o
mecanismo. A correlação de Spearman entre os dois perfis é ρ = +0,344
(p = 0,250) — com 13 ataques não dá para afirmar independência, mas também não
há evidência de que errem juntos.

**O teto da fusão.** Tomando o melhor dos dois modelos em cada ataque — um
oráculo que sabe qual ataque está enfrentando — o EER médio seria **8,42%**. É
um limite superior inatingível na prática, porque essa informação não existe em
operação, mas mostra que a fusão medida (13,13%) captura boa parte da margem
disponível. Vale registrar que **8,42% atenderia o RNF03**, e que o que separa
o sistema disso não é capacidade de discriminação, e sim não saber qual ataque
está vendo.

### 7.2 O EER global penaliza o fusion_v4 por outra coisa

| | média dos EERs por ataque | EER global | custo do agrupamento |
|---|---|---|---|
| baseline_v2 | 16,83% | 18,99% | **+2,16 pp** |
| fusion_v4 | **14,32%** | 20,18% | **+5,86 pp** |

Na média por ataque o `fusion_v4` é o **melhor** dos dois (14,32% contra
16,83%) — o inverso do ranking global. A diferença está no custo de agrupar:
o EER global usa **um limiar só** para os 13 ataques, e o `fusion_v4` distribui
os scores de cada ataque em faixas muito diferentes (desvio de 18,46 pp entre
ataques, contra 11,98 pp do v2). Um limiar único o serve mal.

Ou seja: o que o EER global mede no `fusion_v4` não é falta de capacidade de
discriminar, e sim **inconsistência da escala de score entre ataques**. Isso
reposiciona o incremento de fusão de características — ele não é pior; ele é
pior *sob um ponto de operação único*, que é exatamente a fragilidade já
documentada na Seção 5 ("o ponto de operação não transfere").

> Ressalva de método: cada EER por ataque usa o seu próprio limiar ótimo. A
> média deles não é uma métrica operacional — é diagnóstico. Nenhum sistema
> real escolhe limiar por ataque, porque não sabe qual ataque está enfrentando.

### 7.3 A10 e A12 são os únicos que resistem aos dois

| | A10 | A12 |
|---|---|---|
| baseline_v2 | 30,63% | 36,18% |
| fusion_v4 | 45,54% | 47,99% |

São os **únicos dois ataques em que ambos os modelos passam de 30%**. Todos os
outros casos difíceis são difíceis para *um* dos modelos: A13 derruba o v2 mas
não o v4; A11 derruba o v4 mas não o v2. `fusion_v4` no A12 dá 47,99%, que é
indistinguível do acaso.

### 7.3.1 O que separa os difíceis: geração autorregressiva

Cruzando os EERs medidos com o gerador de forma de onda de cada ataque:

| ataque | v2 | fusion_v4 | gerador de forma de onda |
|---|---|---|---|
| A12 | 36,18% | 47,99% | **WaveNet (autorregressivo)** |
| A10 | 30,63% | 45,54% | **WaveRNN (autorregressivo)** |
| A15 | 15,02% | 29,37% | **WaveNet (autorregressivo)** |
| A13 | 36,50% | 6,54% | filtragem + concatenação |
| A11 | 3,85% | 33,74% | Griffin-Lim |
| A14 | 12,01% | 17,74% | vocoder clássico (STRAIGHT) |
| A16 | 22,75% | 0,03% | concatenação |
| A07 | 22,43% | 0,02% | vocoder clássico (WORLD) |
| A18 | 15,01% | 4,63% | vocoder clássico (vocoder MFCC) |
| A17 | 11,27% | 0,41% | filtragem de forma de onda |
| A19 | 6,98% | 0,00% | filtragem espectral |
| A08 | 3,88% | 0,03% | **neural**, source-filter |
| A09 | 2,32% | 0,12% | vocoder clássico (Vocaine) |

> Os nomes dos vocoders seguem a tabela de ataques de Wang et al. (2020), a
> descrição da base ASVspoof 2019. **Conferir na fonte antes de publicar.** A
> cópia de agosto deste resumo rotulava A09, A14 e A18 como "WORLD" e A08 como
> "WaveNet" — os quatro rótulos estavam errados.

| grupo | v2 | fusion_v4 |
|---|---|---|
| autorregressivos (A10, A12, A15) | **27,28%** | **40,97%** |
| todos os outros (10 ataques) | 13,70% | **6,33%** |

Para o `fusion_v4` a diferença é de **6,5x**. Teste de permutação exato (os 286
trios possíveis entre os 13 ataques): **p = 0,0070** para o `fusion_v4` — o trio
autorregressivo é o 2º mais difícil entre 286 — e **p = 0,0490** para o `v2`.
Significativo nos dois, com folga bem maior no `fusion_v4`.

> **Correção de uma hipótese anterior deste documento.** A versão anterior desta
> seção atribuía a dificuldade a "vocoder neural", em oposição aos vocoders
> paramétricos clássicos do treino. **Isso está errado, e o contraexemplo é
> limpo: o A08 usa gerador neural** (*neural source-filter*) **e é praticamente
> resolvido pelos dois modelos** — 3,88% e 0,03%.
>
> O eixo que separa não é "neural", é **autorregressivo**. WaveNet e WaveRNN
> geram amostra a amostra, condicionando cada amostra nas anteriores, e não
> preservam a estrutura fonte-filtro. O *neural source-filter* do A08 é neural
> mas mantém essa estrutura — e com ela sobrevivem os artefatos que o detector,
> treinado majoritariamente em vocoders clássicos, aprendeu a procurar.

**Limite desta explicação.** O gerador é necessário mas não suficiente: **A12 e
A15 usam o mesmo WaveNet** e diferem em 21,16 pp no `baseline_v2` (36,18% contra
15,02%). Alguma outra coisa no pipeline — modelo acústico, dados de treino do
ataque — também pesa. E o grupo tem só 3 ataques; a separação é grande e
mecanicamente plausível, mas a amostra é pequena.

### 7.3.2 A11 é o caso mais instrutivo de complementaridade

| | A11 (Griffin-Lim) |
|---|---|
| baseline_v2 | **3,85%** |
| fusion_v4 | 33,74% |

O Griffin-Lim reconstrói a fase iterativamente a partir da magnitude, e deixa
um artefato bem característico. O `baseline_v2`, só com LFCC, quase o resolve;
o `fusion_v4`, que acrescenta um ramo de espectrograma log-mel, **piora 30 pp**.

Acrescentar uma representação não é gratuito: o ramo extra pode diluir a
evidência em que o ramo original se apoiava. Isto é o lado negativo da fusão de
características, medido — e é a contrapartida honesta do ganho de −1,38 pp
relatado na Seção 3.

### 7.4 O que escrever a partir disto

- O EER global de 18,99% é uma média entre ataques quase resolvidos (A09 com
  2,32%) e ataques em que o modelo está perto do acaso (A13 com 36,50%).
  Reportar só o agregado esconde os dois extremos.
- O `fusion_v4` tem perfil **bimodal**: seis ataques abaixo de 0,5% e cinco
  acima de 17%. O `baseline_v2` é mais uniforme. São modelos com
  comportamentos qualitativamente distintos, não versões melhores e piores do
  mesmo.
- A complementaridade medida aqui é a justificativa mecanicista da fusão de
  scores, e liga a Seção 4 à Seção 7.
- A dificuldade se concentra na **geração autorregressiva** (WaveNet, WaveRNN),
  não em "vocoder neural" — o A08 é neural e é resolvido. Ver 7.3.1.
- A fusão de características tem um custo medido: no A11 ela **piora 30 pp**.
  O ganho agregado de −1,38 pp é um saldo, não um ganho uniforme.

## 8. Requisitos da APS — o que foi atendido e o que não foi

| requisito | exigência | medido | veredito |
|---|---|---|---|
| RNF01 | ≤ 30 s por análise (arquivo de até 60 s) | **2,745 s** (CPU) | **atendido, 11x de folga** |
| RNF02 | F1 ≥ 0,85 | **0,69 a 0,83** (melhor: v2, 0,8304) | **NÃO atendido** |
| RNF03 | EER ≤ 10% | **13,13%** (melhor) | **NÃO atendido** |

Numeração: RNF02/RNF03 são da APS; no TC1 o requisito de F1 é o **RNF04**.

> **Correção.** A versão anterior desta tabela dava o RNF02 como "medido
> 0,9456, atendido". **0,9456 não vem de nenhum modelo**: é o F1 do
> classificador trivial "tudo spoof" (precisão 0,897, recall 1 →
> 2·0,897/1,897). Os F1 dos modelos, no limiar calibrado no dev
> (`calibrate_threshold: true`), ficam entre 0,69 e 0,83 — v2 com 0,8304 e
> fusion_v4 com 0,7103. Nenhum modelo atinge 0,85.

### 8.1 O RNF02 não foi atendido — e também é mal especificado

As duas afirmações coexistem. O requisito **não foi atendido** por nenhum
modelo real. E, ao mesmo tempo, com 89,7% de spoof no eval, um classificador
que responde **"spoof" para tudo** obtém F1 = **0,9456** — acima do exigido,
sem olhar para o áudio. Ou seja: o requisito é satisfeito pelo detector inútil
e não é satisfeito pelos detectores reais.

Esse achado vale mais que o cumprimento do requisito. É um resultado de
engenharia de requisitos: **F1 sobre classe majoritária não mede capacidade de
detecção**. (Convenção a declarar no texto, TC1 §4.11: a classe positiva é o
**spoof**, `pos_label=1` em `src/metrics.py`.) A métrica correta para a tarefa é o EER, que é independente de
limiar e não pode ser enganado dessa forma. Recomendação para a APS: substituir
o RNF02 por um alvo de EER, ou exigir F1 **macro**.

### 8.2 O RNF03 não foi atendido, e isso precisa estar no texto

O melhor resultado do projeto é 13,13% contra os 10% exigidos. Declarar
atendimento seria falso. O caminho para fechar a lacuna existe e está medido:
fusão de scores já trouxe 5,86 pp; o que falta provavelmente vem de encoder
pré-treinado em fala (wav2vec/WavLM), que é escopo além deste trabalho.

### 8.3 Latência medida (`scripts/bench_latencia.py`)

A medição usa a **janela deslizante**, não o `infer.py`. O `infer.py` corta o
sinal em `audio.duration`: num envio de 60 s analisaria só os primeiros 4 s —
6,7% do arquivo. Para perícia isso é inaceitável, então a aplicação janela o
arquivo inteiro e agrega.

---

## 9. A aplicação ao vivo

### 9.1 A janela é de 4 segundos, e não é parâmetro livre

O modelo classifica trechos de `audio.duration` = 4,0 s porque foi assim que ele
foi treinado — `fix_length` força esse comprimento em toda amostra. Mudar a
janela exigiria retreinar e invalidaria todas as métricas acima.

**Consequência dura:** um trecho sintético **mais curto que 4 s** nunca ocupa uma
janela inteira, e o modelo sempre o vê misturado com áudio real. Passo menor não
resolve — é limite de resolução, não de amostragem.

O mesmo limite tem uma consequência operacional: um **arquivo** mais curto que a
janela também não fecha nenhuma janela. O `AnalisadorContinuo.finalizar()` emite
o trecho final nesse caso, completado por repetição como no treino, com o peso
reduzido na proporção (um enunciado de 2,6 s pesa 0,65). Sem isso o monitor
devolvia zero leituras para a maioria dos áudios do ASVspoof, que são mais
curtos que 4 s.

### 9.2 Janelas sobrepostas não são observações independentes

Janela de 4 s com passo de 2 s: janelas vizinhas compartilham metade do áudio.
`n` janelas cobrem `(n−1)·passo + janela` segundos, o que equivale a
`cobertura / janela` janelas independentes.

Uma média de 5 janelas cobre 12 s — **3 janelas independentes**, não 5. O resumo
reporta os dois números; dizer "média de 5" sugeriria mais solidez do que existe.

### 9.3 Ponderação das janelas

O `preprocess_waveform` remove o silêncio e completa por **repetição**:

| fala na janela | após o trim | o modelo vê |
|---|---|---|
| 100% | 4,00 s | sinal íntegro |
| 50% | 2,08 s | o mesmo trecho 2x |
| 25% | 1,09 s | o mesmo trecho 4x |
| **10%** | **0,48 s** | **o mesmo trecho 8x** |

A 10% de fala o modelo recebe meio segundo em loop — entrada que **não existe no
treino**. Por isso o peso de cada janela na média é a **própria fração de fala**:
é a grandeza que causa a repetição, sem constante de ajuste no meio.

**O canal é tratado de forma binária, não contínua, e isso é deliberado.** A
degradação por banda estreita foi medida (+5,35 pp no `fusion_v4`, +16,96 pp no v2), e a resposta medida é
*excluir*, não atenuar. Atribuir peso intermediário exigiria uma curva
EER × qualidade que ninguém mediu. O portão de canal tem três estados
(`larga` / `estreita` / `indeterminado`); `indeterminado` entra na média — na
dúvida o sistema continua medindo em vez de se calar sem evidência.

### 9.4 Custo de rodar ao vivo (medido)

CPU, 4 threads, sem GPU. Caminho completo: janela → features → rede → agregação.

| | ms/janela | x tempo real | carga |
|---|---|---|---|
| 1 modelo (`fusion_v4`) | 80,4 | 25,7x | 3,9% de 1 núcleo |
| 2 modelos (fusão ao vivo) | 96,9 | **21,3x** | **4,7% de 1 núcleo** |

Com janela de 4 s e passo de 2 s são **0,5 janelas/s** a processar. A mesma CPU
entrega 71 janelas/s em inferência pura: folga de **143x**. Memória: 778 MB
residentes, dos quais ~500 MB são o próprio PyTorch (os pesos somam 1,5 MB —
23.778 + 348.866 parâmetros).

Contra o treino, no mesmo hardware, lote de 32:

| | ms/amostra |
|---|---|
| treino (forward + backward + Adam) | 58,2 |
| inferência (forward puro) | **14,0** |

Razão de **4,16x**, estrutural: o backward recalcula gradiente camada a camada e
o Adam mantém dois momentos por parâmetro. Mas o que separa de fato os dois
regimes é o volume — o treino atravessa 25.380 amostras por época; o ao vivo,
0,5 por segundo. **Rodar ao vivo não se parece com treinar.**

---

## 10. Camada 2 — medir o canal real

### 10.1 A distinção entre as duas camadas

**Camada 1** (feita): `robustness_eval.py` simula o canal em software — codec
Opus e limitação de banda — sobre os 71.237 áudios do eval. Mede codec e banda.

**Camada 2** (executável, ainda não executada): tocar os mesmos áudios dentro de
uma chamada **real** e capturar o retorno. Mede também supressão de ruído,
cancelamento de eco e ganho automático, que não são simuláveis de forma honesta.

**A camada 1 é um limite inferior da degradação.** Os números da Seção 5 são o
mínimo que o canal custa, não o total.

### 10.2 Como a camada 2 é executada (`scripts/canal_real.py`)

```bash
python scripts/canal_real.py preparar --config configs/fusion_v4.yaml --n-por-classe 40
python monitor.py --config ... --checkpoint ... --gravar chamada.wav
python scripts/canal_real.py alinhar --pasta outputs/canal_real --gravacao chamada.wav
```

O problema técnico é o alinhamento: a gravação chega como um bloco de minutos e
sem saber onde cada áudio começa não há rótulo, e sem rótulo não há EER.

**Marcar com um bipe não funciona** — e a razão é instrutiva. A supressão de
ruído é treinada para remover o que *não* é fala, e um seno puro é o exemplo
canônico disso. O marcador sumiria exatamente no cenário que se quer medir.

A solução é **correlação cruzada do envelope de energia** contra a referência
tocada. Codec, supressão e AGC mudam espectro e amplitude, não *quando* a fala
acontece; normalizar antes de correlacionar remove o efeito do ganho. O ajuste é
em duas etapas — atraso global da chamada, depois refino por trecho — e o refino
é sequencial para acompanhar a deriva de relógio entre as placas, que é
cumulativa.

Correlação medida (pior caso de cada cenário):

| cenário | correlação |
|---|---|
| canal limpo | 0,798 |
| opus + banda estreita 8 kHz | **0,798** |
| ruído puro (microfone errado) | 0,118 |

O codec praticamente não toca no envelope — que é o ponto do método. Limiar em
0,5, com folga de 1,6x para baixo e 4,2x para cima. Abaixo do limiar o trecho é
**descartado**: um recorte mal alinhado carrega o rótulo do vizinho e produziria
um EER que *parece* resultado.

O procedimento inclui um **controle**: repetir tudo sem chamada nenhuma. Se o
controle já divergir do eval limpo, a diferença é do procedimento, não do Teams.

**Não executada** — decisão registrada em 12.3: instrumentação validada, execução como trabalho futuro.

### 10.3 Bases públicas que já trazem canal

| base | o que traz | uso |
|---|---|---|
| **ASVspoof 2021 LA** | ataques de 2019 por VoIP e PSTN reais, 6 codecs | avaliação; **sem partição de treino** por regra |
| **ASVspoof 2021 DF** | codecs de mídia, áudio recomprimido | avaliação |
| **CFAD** | 12 tipos de falsificação, versões *clean* / *noisy* / *codec*, splits disjuntos | **avaliação e treino** |
| **In-the-Wild** | 37,9 h achadas na internet (17,2 h falsas), 58 figuras públicas | avaliação, caso mais difícil |
| **ASVspoof 5** | fala *crowdsourced* em condições não-estúdio, 32 algoritmos de ataque, ataques adversariais | **avaliação e treino** — ver 10.7 |

**ASVspoof 2021 LA** é a mais direta: por regra do desafio não há partição de
treino — os sistemas são treinados na 2019 LA, que é a base deste projeto. Os
checkpoints avaliam nela sem nenhuma mudança. Ressalva: é codec + transmissão,
**não** inclui o processamento de um cliente de conferência.

**Por que LA e não DF.** A trilha DF traz codecs de mídia (armazenamento) e
inclui todo o áudio dos dois *Voice Conversion Challenge* (2018 e 2020) além do
ASVspoof 2019 — cerca de 600 mil enunciados. Isso mistura três mudanças de uma
vez: corpus de origem novo, vocoders novos e compressão. Não há como atribuir a
diferença a nenhuma delas.

A LA tem a estrutura oposta, e é ela que serve aqui:

- **Os ataques são os mesmos A07–A19** do eval de 2019. A tabela por ataque da
  Seção 7 transfere linha a linha.
- **Existe uma condição de referência sem codec e sem transmissão**, equivalente
  ao cenário de 2019. É o controle pareado dentro da própria base.
- **Uma das condições é OPUS real, sobre rede real.** A Seção 5 mede Opus
  *simulado*. Comparar as duas quantifica o quanto a simulação subestima — que é
  exatamente a afirmação de "limite inferior" feita em 10.1, hoje sem número.

DF responde a outra pergunta — generalização a fontes e vocoders novos — e é uma
segunda etapa legítima. Mas responde com três variáveis mudando juntas, enquanto
LA responde com uma.

**Detalhe operacional que trava a importação.** Os rótulos vêm no
`trial_metadata.txt` do *eval-package*, com oito campos e em ordem diferente do
protocolo de 2019:

```
2019:  LA_0079 LA_E_1234567 -     A07    spoof
2021:  LA_0009 LA_E_9332881 alaw  ita_tx A07  spoof notrim eval
       locutor arquivo      codec canal  ataque chave trim  fase
```

O `parse_protocol_with_systems` lê o ataque em `parts[3]` e a chave em
`parts[4]` — no arquivo de 2021 isso daria `ita_tx` e `A07`. Como `A07` não é
chave válida, **todas as linhas seriam descartadas em silêncio** e o protocolo
sairia vazio. O `scripts/importar_asvspoof2021.py` converte, listando antes as
condições presentes:

```bash
python scripts/importar_asvspoof2021.py --metadata keys/LA/CM/trial_metadata.txt --listar
python scripts/importar_asvspoof2021.py --metadata ... --codec opus --amostra 10000 \
    --config-base configs/fusion_v4.yaml --audio-dir <flac do 2021>
```

O script **gera o config** de avaliação em vez de pedir que ele seja copiado à
mão. Copiar `fusion_v4.yaml` e trocar só os caminhos manteria o
`experiment.name`, e como os artefatos do `evaluate.py` são nomeados por ele, a
avaliação do 2021 gravaria **por cima** das métricas e dos scores do eval de
2019 — e o cache de features do eval seria apagado e recriado com o áudio novo.
O config derivado muda o nome e desliga o cache.

**CFAD** permite *treinar* com canal, porque as versões ruidosa e com codec têm
partição de treino. É em mandarim, o que confunde idioma com canal — mas o
desenho da base resolve isso: como as três versões partem do mesmo material, o
idioma é constante e o **delta** entre `clean` → `noisy` e `clean` → `codec`
isola o canal. Esse delta é diretamente comparável ao delta da Seção 5
(+23,42 pp no v2 contra +7,24 pp no fusion_v4). Se a inversão de ranking se
repetir em outro idioma, com outros ataques e outro grupo gerador, ela deixa de
ser peculiaridade da simulação e vira resultado.

> **Atenção — o ruído e o codec do CFAD são simulados, não capturados.** A
> própria descrição da base diz que ruído de fundo e codecs "são simulados".
> Portanto o CFAD é **camada 1 feita por outro grupo**, não camada 2. O valor
> dele é a independência (outro idioma, outros geradores, outra equipe), não o
> realismo do canal. Nenhuma base pública substitui a gravação da Seção 10.2.

Um dos 12 tipos do CFAD é *partially fake* (trecho falso dentro de áudio real).
O modelo deste projeto decide por enunciado inteiro, então esse tipo mede outra
tarefa — separar, não jogar na média.

### 10.4 Taxonomia do realismo do canal

Consolidando: nem toda "base com ruído" mede a mesma coisa. Esta tabela decide o
que cada fonte pode sustentar no texto.

| nível | o que mede | fontes disponíveis |
|---|---|---|
| **0 — limpo** | benchmark, sem canal | ASVspoof 2019 LA eval |
| **1 — canal simulado** | codec, banda, ruído aditivo, música de fundo | `robustness_eval.py` (este projeto), CFAD *noisy*/*codec*, ADD 2022 track LF |
| **1,5 — transmissão real** | codec real + rede (VoIP, PSTN) | **ASVspoof 2021 LA** |
| **2 — cliente de conferência** | supressão de ruído, cancelamento de eco, AGC | **só gravando** — `scripts/canal_real.py` |

Os níveis 0 e 1 estão medidos. O 1,5 está disponível publicamente e não exige
gravação. O 2 não existe em base pública conhecida e é o que o procedimento da
Seção 10.2 produz.

### 10.5 O que NÃO entra: falsificação parcial

Uma família inteira de bases próximas mede **outra tarefa**: Half-Truth (HAD),
ADD 2022 track PF e ADD 2023 track 1.2 tratam de *partially fake* — trechos
sintéticos curtos inseridos dentro de uma gravação autêntica, às vezes só uma ou
duas regiões por frase.

O modelo deste projeto emite **um score por enunciado**. Detectar falsificação
parcial exige localizar *onde* está o trecho falso: rótulo por quadro,
arquitetura com saída temporal e métrica de localização. Não é uma versão mais
difícil do mesmo problema — é outro problema.

Isso está registrado aqui como **trabalho futuro identificado**, não como
lacuna: reconhecer a distinção e delimitá-la é resultado de revisão da área.
A Seção 9.1 já mostra o limite correlato dentro deste trabalho — um trecho
sintético mais curto que a janela de 4 s nunca ocupa uma janela inteira.

### 10.6 Independência entre as bases candidatas

Bases do mesmo grupo podem compartilhar o áudio autêntico, e nesse caso somá-las
**não** produz confirmação independente.

Verificado: o áudio real do CFAD vem de **AISHELL-1, AISHELL-3, THCHS-30 e dois
corpora MAGICDATA**. O ADD 2022 é construído sobre **AISHELL-3**. Os dois
compartilham fonte bonafide, além de compartilharem idioma e equipe (Yi, Tao et
al., CAS).

Consequência prática: depois do CFAD, o ADD 2022 LF acrescenta **um** eixo que
nada mais cobre — **música de fundo** — e pouco além disso. Vale como um ponto
extra de robustez, não como segunda confirmação da Seção 5. A confirmação
independente vem do ASVspoof 2021 LA, que é outro idioma, outro grupo e outra
fonte de áudio autêntico.

### 10.7 ASVspoof 5 — a adição de maior valor, e o que ela custa

Construído sobre o **MLS English**, com ~2.000 locutores em condições acústicas
diversas (as edições anteriores eram estúdio). Sete partições disjuntas por
locutor, mais de 20 ataques *crowdsourced* e **7 ataques adversariais**, que
aparecem pela primeira vez na série.

| partição | locutores | áudios | bonafide |
|---|---|---|---|
| train | 400 | 182.357 | 18.797 |
| dev | 785 | 140.950 | 31.334 |
| eval (track 1) | 737 | **680.774** | 138.688 |

**Por que é melhor que o CFAD para este trabalho:** é em **inglês**, como a base
atual. Não há confundidor de idioma — a diferença de desempenho é atribuível a
condição de aquisição e a ataque, não a língua. É a comparação limpa que o CFAD
só consegue por subtração.

**O eixo novo que ele traz** não é transmissão, é **aquisição**: microfone
doméstico, sala qualquer, locutor qualquer. Isso é ortogonal ao canal da Seção 5
(codec, banda, ruído somado depois) e ortogonal à camada 2 (processamento do
cliente de conferência). São três degradações distintas.

**Efeito colateral útil para a APS:** com 20,4% de bonafide no eval (contra 10,3%
no 2019 LA), o classificador trivial "tudo spoof" obtém F1 = **0,8866** — ainda
acima do RNF02, mas com margem bem menor. Isso reforça a Seção 8.1: o problema é
a métrica, não o desequilíbrio específico de uma base.

**O custo, calculado com a taxa medida deste projeto** (113 amostras/s
consumidas pela GPU, registrado em `configs/fusion_v4.yaml`):

| | amostras/época | min/época | treino completo (50 épocas) |
|---|---|---|---|
| ASVspoof 2019 LA | 25.380 | 3,7 | **3,1 h** |
| ASVspoof 5 | 182.357 | 26,9 | **22,4 h** |

São **7,2x mais dados por época** e o eval é **9,6x** maior. Numa máquina que já
travou durante o treino atual (Seção 11), treinar do zero no ASVspoof 5 é um
compromisso sério, não um experimento extra.

**Recomendação:** usar o ASVspoof 5 como **avaliação cruzada** — rodar os
checkpoints já treinados no eval track 1, sem retreinar. Isso é generalização
entre bases, resultado legítimo e citável, e custa uma avaliação em vez de 22
horas por config. Treinar nele fica como trabalho futuro explícito.

### 10.8 Bases avaliadas e descartadas, com o motivo

Registrar o descarte vale tanto quanto registrar a adoção: mostra que a seleção
foi feita, e o motivo é técnico em cada caso.

**LRLspoof** (INTERSPEECH 2026, 66 idiomas, 2.732 h, MIT, 452 GB) — **descartada
por incompatibilidade de métrica.** A base é **spoof-only**: não há áudio
bonafide. Sem a classe bonafide não existe taxa de falso aceite, e portanto
**não existe EER** — a métrica central deste trabalho. A base reporta SRR a um
limiar fixo calibrado externamente (*threshold transfer*).

Há uma segunda razão, e ela é do próprio resultado deste projeto: a Seção 5
mediu que **o ponto de operação não transfere** — sob a mesma perturbação, o
recall de um modelo sobe e o de outro cai. A metodologia da LRLspoof assume
justamente que esse transporte é válido. Não é motivo para desqualificar a base,
mas é motivo para não construir um capítulo sobre ela.

**PlaybackSpoof** (165 GB, licença "other", sem artigo localizado) — **descartada
por ser outra classe de ataque.** É detecção de *replay* (ataque de
apresentação): o áudio é fala humana genuína, reproduzida por alto-falante e
recapturada. É a trilha **PA** do ASVspoof, não a **LA**.

O conflito é mais profundo que o escopo. Este trabalho passa a Seção 5 inteira
ensinando o modelo a **ignorar** artefato de canal e de captura, para que a
decisão dependa do artefato de síntese. Detecção de replay exige o oposto:
**atender** ao artefato de reprodução e recaptura, porque é ele que denuncia o
ataque. Os dois objetivos puxam em direções contrárias — um modelo ajustado para
um está estruturalmente em desvantagem no outro.

---

## 11. Limitações e pendências

**Baselines oficiais — resolvido.** Confirmados em Todisco et al. (2019),
Tabela 1: B02 8,09%, B01 9,57%. A comparação e a ressalva do silêncio estão em
5.1; a limitação de uma execução por modelo, em 5.2.

**O áudio ao vivo é o mix.** O loopback entrega a soma de todos os
participantes. Não há atribuição por pessoa; o resultado é sobre o *trecho*, não
sobre quem falou.

**A camada 2 ainda não foi executada.** O código existe e está testado ponta a
ponta com canal simulado, mas nenhuma chamada real foi medida.

**A degradação medida é limite inferior.** Ver 10.1.

**O RNF02 e o RNF03 não foram atendidos.** Ver 8.1 e 8.2.

**Atalhos existem na base e o modelo depende deles em 1–2%.** Ver Seção 6.

**Uma execução por modelo.** Os dois incrementos do TC1 (−1,38 e +0,95 pp)
ficam dentro da variância entre execuções que Müller et al. observam (1,4 a
5,2 pp). Ver 5.2. A defesa da fusão se apoia no EER por ataque (Seção 7).

**Robustez a ruído parcialmente dentro da distribuição.** 20 dB, 10 dB e ±6 dB
foram vistos no treino; inéditos são 5 dB, Opus e banda estreita. Ver Seção 5.

**Atenção mínima.** O resultado negativo vale para uma projeção linear de 258
parâmetros, não para atenção em geral. Ver Seção 3.

**Canal real não avaliado.** ASVspoof 2021 LA: só o importador. Camada 2: só a
instrumentação.

**t-DCF não calculado.** O `save_score_file` já grava os scores no formato
oficial; com os scores ASV fornecidos pelos organizadores do ASVspoof 2019, o
script oficial dá o min t-DCF, comparável direto com a Tabela 1 de Todisco et
al. (2019). Custo: uma rodada rápida, sem retreino.

**MP3 não avaliado.** O teste de robustez usou Opus.

---

## 12. Plano final do TC2 (escopo fechado)

Escrito sob restrição de tempo. A regra é: **o valor marginal de mais uma base é
menor que o valor marginal de escrever as seções que já têm dado.**

### 12.1 O que já está medido e só precisa ser escrito

Sete modelos comparados, fusão de scores com grupo de controle, robustez ao
canal com inversão de ranking, verificação de atalhos em duas camadas, latência
do RNF01, o achado do RNF02 trivial, custo de operação ao vivo, resolução
temporal e ponderação das janelas. Nada disso precisa de execução nova.

### 12.2 Fazer (barato, alto valor)

1. ~~Regenerar as tabelas por ataque~~ — **feito**, ver Seção 7.
2. **ASVspoof 2021 LA, com subamostragem.** Uma comparação pareada: condição de
   referência (sem codec) contra Opus real, nos dois modelos principais.

   **Régua para o 2021 LA**, de Müller et al. (2021), Tabela 4 e Tabela 1:

   | sistema | LA 2021 |
   |---|---|
   | RawNet2 treinado com silêncio | 10,38% |
   | **RawNet2 treinado sem silêncio** | **27,39%** |
   | só a duração do silêncio inicial (1 número) | 19,93% |

   A linha que compara com este projeto é a do meio: mesmo protocolo (silêncio
   removido). Ressalva: são números da fase de progresso da CodaLab, sobre um
   subconjunto do eval, e não sobre o eval completo.
3. ~~Obter os baselines oficiais do ASVspoof 2019~~ — **feito**, ver 5.1. Com
   eles veio a régua de mesmo protocolo (Müller et al., Tabela 2) e a
   limitação de 5.2.

**Não rode o eval de 2021 inteiro.** São 181.566 áudios, e não é preciso.
Simulado no regime deste projeto (EER ≈ 20%, 10,3% de bonafide), a dispersão do
EER estimado por tamanho de amostra:

| áudios por condição | bonafide | IC 95% do EER |
|---|---|---|
| 2.000 | 206 | ± 2,59 pp |
| 5.000 | 515 | ± 1,81 pp |
| **10.000** | **1.030** | **± 1,28 pp** |
| 25.000 | 2.575 | ± 0,85 pp |
| 71.237 (eval de 2019) | 7.337 | ± 0,49 pp |

Os efeitos a distinguir são a inversão de ranking (10,4 pp) e o ganho da fusão
de scores (5,86 pp). **10.000 por condição resolve os dois com folga** — e custa
menos que uma avaliação completa de 2019. O ganho da fusão de características
(1,38 pp) fica abaixo da resolução dessa amostra, mas ele já está medido no eval
completo de 2019 e não precisa ser refeito aqui.

### 12.3 Cortar

**A execução da camada 2.** O motivo é que o ASVspoof 2021 LA **absorve a maior
parte do que ela provaria**: transmissão real, codec real, com controle pareado
nos mesmos ataques. O que sobra de exclusivo para a camada 2 é apenas o
processamento do cliente de conferência (supressão de ruído, cancelamento de
eco, AGC) — incremento mais estreito do que parecia quando o 2021 LA ainda não
estava na mesa.

**O trabalho da camada 2 não se perde**, e não deve ser omitido do texto. Ele
entra como **instrumentação validada e trabalho futuro**: o método de
alinhamento por correlação de envelope, a razão medida de um marcador por bipe
não funcionar, as margens de correlação (0,798 no canal degradado contra 0,118
em ruído) e o procedimento com grupo de controle. Construir e validar a
instrumentação é contribuição; executá-la fica declarado como próximo passo.

**Também cortados:** ASVspoof 5 (só se sobrar tempo, e apenas avaliação
cruzada), CFAD, ADD, e qualquer retreino.

### 12.4 O que o TC2 conclui

- A fusão de características ajuda (−1,38 pp); a atenção não (Seção 3).
- A fusão de scores entre modelos **diversos** ajuda muito mais (−5,86 pp), e o
  grupo de controle mostra que o ganho vem da diversidade (Seção 4).
- **O ranking dos modelos se inverte sob degradação de canal simulado**
  (Seção 5) — o resultado central. **Ainda não há confirmação em canal real:**
  o ASVspoof 2021 LA tem só o importador pronto, nenhuma avaliação rodada, e a
  camada 2 não foi executada.
- O EER agregado esconde os ataques mais fortes: A10 e A12 ficam entre 30% e
  48% nos dois modelos (Seção 7.3).
- O modelo aprende artefato, não atalho, com dependência residual de 1–2%
  (Seção 6).
- Nenhum modelo atinge o RNF02 (F1 ≥ 0,85), que ainda assim é satisfeito por um
  classificador trivial; o RNF03 também não é atendido (Seção 8). Ambos
  documentados com evidência.

---

## 13. O que o texto do TC1 precisa mudar

### 13.1 Resultados esperados vs. obtidos

| o TC1 prometeu | obtido | |
|---|---|---|
| F1 > 0,90 | 0,69 a 0,83 | não |
| EER < 8% | 13,13% (fusão de scores), 18,99% (melhor isolado) | não |
| ganho a cada incremento | fusão −1,38 pp; atenção +0,95 pp | só na fusão |

A meta de EER < 8% veio do B02 (8,09%), que processa o áudio **com** o
silêncio. A comparação justa é a Tabela 2 de Müller et al. (2021), mesmo
protocolo sem silêncio (Seção 5.1): os sete modelos ficam abaixo dos CNN/ResNet
com CQT (26–27%) e a fusão de scores fica abaixo da média do RawNet2 (15,50%).

### 13.2 Por seção

- **Resumo/Abstract:** reescrever com os resultados — fusão de características
  −1,38 pp; atenção não melhora; fusão de scores entre modelos diferentes
  13,13%; ranking se inverte sob canal (+23,42 pp no v2 contra +7,24 pp no
  fusion_v4).
- **Arquitetura:** seção nova, a partir da Seção 3-0 deste resumo.
- **§4.7 (ciclo incremental):** usar a linhagem v1 → v4 da Seção 3; deltas já
  no baseline; linha de base = `baseline_lcnn_v4`.
- **§4.11 (métricas):** classe positiva = spoof; limiar calibrado no dev;
  acrescentar t-DCF (Seção 11).
- **Seção 5:** trocar os resultados esperados pelas tabelas das Seções 3, 4, 5
  e 7 deste resumo.
- **Conclusão:** quatro achados — (1) o dev não prevê o eval; (2) a fusão de
  scores ganha pela diversidade entre os modelos; (3) o EER agregado esconde os
  ataques mais fortes (A10/A12); (4) o ranking se inverte no canal degradado.
- **Limitações:** tudo o que está na Seção 11.

### 13.3 Promessas do texto sem entrega no repositório

Decisão da equipe: cortar do texto ou fazer.

- **Vozes dos colaboradores** (TC1 4.9, 5.1, 5.5, 5.6, 7.1 e conclusão). O
  repositório só tem a pasta prevista no README, sem script nem resultado. A
  7.1 afirma que os termos de consentimento foram assinados — se a coleta não
  aconteceu, a afirmação sai.
- **API (FastAPI) e front-end (React)**, Marcos 5 a 7. O README diz que só
  `ml/` está implementado. Confirmar se existem em outro repositório.
- **MP3.** Declarar que o teste de robustez usou Opus.

### 13.4 Pendência: fundir com a cópia de agosto

A cópia de agosto (no Projeto) está desatualizada, mas tem tabelas que esta
versão não tem: EER por ataque dos seis modelos, robustez nas 11 condições e
dev vs. eval com r = 0,951. Essas tabelas precisam ser trazidas para cá —
conferindo cada número contra a execução, e corrigindo os rótulos de ataque
(Seção 7.3.1).

---

## Comandos que produzem cada número

```bash
# Seção 2  — integridade da base
python scripts/check_data.py --config configs/fusion_v4.yaml --deep

# Seção 3  — EER por modelo
python evaluate.py --config configs/<cfg>.yaml --checkpoint checkpoints/<ckpt>.pt
python scripts/make_report.py

# Seção 4  — fusão de scores
python scripts/score_fusion.py --scores a.npz b.npz --rule rank

# Seção 5  — robustez ao canal
python scripts/robustness_eval.py --config ... --checkpoint ...

# Seção 6  — atalhos e confundidores
python scripts/check_shortcut.py --config ... --particao train
python scripts/check_score_confound.py --config ... --scores eval.npz --n 10000

# Seção 7  — por ataque
python scripts/per_attack_eval.py --config ... --checkpoint ...

# Seção 8  — latência (RNF01)
python scripts/bench_latencia.py --config ... --checkpoint ... --device cpu

# Seção 10 — camada 2
python scripts/canal_real.py preparar --config ... --n-por-classe 40
python scripts/canal_real.py alinhar --pasta outputs/canal_real --gravacao chamada.wav
```
