# Resumo consolidado para a escrita do TCC

Documento de referência com **tudo o que foi medido** no projeto. Serve tanto ao
TC1 (artigo científico) quanto à APS (engenharia de software).

> **Regra deste documento:** só entra número que saiu de execução registrada.
> Onde o número depende de uma rodada que precisa ser refeita, o lugar está
> marcado com `[REGENERAR]` e o comando que produz o valor. Nada é estimado.
>
> Versionado em `ml/docs/` de propósito: a versão anterior morava em
> `ml/outputs/`, que é gitignored, e se perdeu quando o ambiente foi reciclado.

Última atualização: 29/09/2026 · 591 testes automatizados passando (8 avisos).

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
| eval | 71.237 | **A07–A19 (11 inéditos + A16/A19, que repetem os algoritmos de A04/A06)** |
| **total** | **121.461** | |

O eval tem 7.355 bonafide e 63.882 spoof — **4.914 por ataque**, e **89,7% de
spoof**. Esse desequilíbrio volta a importar na Seção 8.

O ponto metodológico central: **onze dos treze ataques do eval não aparecem no
treino**; A16 e A19 reutilizam os algoritmos de A04 e A06, com outros dados
(Wang et al., 2020).
Todo número de eval neste documento é generalização para ataque não visto, não
desempenho em distribuição conhecida. É por isso que os EERs são de dezenas de
pontos percentuais e não de unidades.

A integridade da base é verificada por `scripts/check_data.py --deep`, que
decodifica os arquivos em vez de ler só o cabeçalho — um FLAC truncado reporta a
duração original e passaria despercebido até derrubar o treino horas depois.

### 2.1 Pré-processamento e features contra o baseline oficial LFCC-LCNN

Conferido no código dos baselines, não em descrição de artigo:

- **ASVspoof 2021 LA, baseline B03 (LFCC-LCNN):** repositório
  `asvspoof-challenge/2021`, `LA/Baseline-LFCC-LCNN/project/baseline_LA/model.py`
  e `sandbox/util_frontend.py` (classe `LFCC`).
- **Variante treinada no 2019 LA:** Wang & Yamagishi (2021), repositório
  `nii-yamagishilab/project-NN-Pytorch-scripts`,
  `project/03-asvspoof-mega/lfcc-lcnn-lstmsum-p2s/01/model.py`.

| | baseline 2021 LA | variante 2019 LA | este projeto (v2 / fusion_v4) |
|---|---|---|---|
| CMVN | **não** | **não** | **não** |
| normalização das features | nenhuma — `BatchNorm2d(affine=False)` após cada MFM, dentro da rede | idem | média e desvio **por áudio**, um único valor para a matriz inteira |
| amplitude | sem normalização | sem normalização | normalização por pico |
| silêncio | mantido (o VAD está comentado no código) | mantido | **removido** (`top_db` 30) |
| pré-ênfase | 0,97 | 0,97 | não |
| coeficientes | 20 LFCC, c0 trocado pela log-energia, + Δ + ΔΔ = 60 | idem | 20 LFCC + Δ + ΔΔ = 60 |
| filtros lineares | 20 | 20 | 20 (v2) / 70 (fusion_v4) |
| janela / passo | 20 ms / 10 ms | 20 ms / 10 ms | 25 ms / 10 ms |
| FFT | 1024 | 512 | 512 |
| **faixa de frequência** | **0–4 kHz** (`lfcc_max_freq = 0.5`) | 0–8 kHz | 0–8 kHz |

**CMVN não faz parte do baseline.** A classe do modelo tem um
`normalize_input`, mas ele nunca é chamado — o comentário no código diz *"not
relevant to this code"*. O CMVN vem da verificação de locutor, em que a
coloração estacionária do espectro (microfone, canal) é incômodo a remover. Na
detecção de spoofing, parte dessa coloração pode ser o próprio artefato do
vocoder; o baseline deixa o LFCC cru e normaliza por lote dentro da rede, o que
ajusta a escala sem apagar as diferenças de espectro entre áudios. (A
justificativa é leitura deste projeto; o código não a explicita.)

**A normalização por instância deste projeto não é CMVN.** CMVN normaliza cada
coeficiente ao longo do tempo; aqui é uma média e um desvio para o espectrograma
inteiro, o que tira nível e escala globais e não remove coloração de canal. O
efeito é o esperado: ganho de ±6 dB custa no máximo 0,6 pp (Seção 5).

**Onde o projeto diverge, e por quê:**

- **Silêncio removido** — deliberado, pelo atalho documentado por Müller et al.
  (2021); é o que torna comparável a régua de mesmo protocolo da 5.1.
- **Normalização por pico e por instância** — o RMS sozinho separa as classes
  com 32% de EER no eval (Seção 6); normalizar a amplitude impede o modelo de
  aprender esse atalho.
- **Pré-ênfase e c0 como log-energia** — não adotados; não foram testados
  isoladamente, e o efeito deles neste pipeline não é conhecido.

**A faixa de 0–4 kHz do baseline de 2021 é o dado mais relevante da tabela.**
Os organizadores limitaram o LFCC à metade inferior do espectro no baseline do
LA de 2021 — a edição que é, por construção, telefonia e VoIP. O comentário no
código diz *"only uses [0, 0.5 * Nyquist_freq range for LFCC]"*. É coerente com
o que este projeto mediu: banda estreita custa **+16,96 pp** ao v2 (Seção 5) e
o canal real, +11,28 pp (5.3). Um modelo que decide com a faixa de 4–8 kHz
perde essa informação no primeiro trecho de telefonia. **Se o objetivo for o
canal real, a mudança que segue o raciocínio do ASVspoof não é CMVN — é
limitar o LFCC a 0–4 kHz.** Exige retreino e está fora do escopo (12.3);
registrada como trabalho futuro na Seção 11.

---

## 3. Resultados dos modelos isolados

EER no eval completo (71.237 áudios). Menor é melhor.

| modelo | EER eval | precisão | recall | F1 |
|---|---|---|---|---|
| **baseline_lfcc_cnn_v2** | **18,99%** | 0,9838 | 0,7184 | **0,8304** |
| fusion_lcnn_v4 | 20,18% | 0,9999 | 0,5508 | 0,7103 |
| baseline_lfcc_cnn | 20,78% | 0,9814 | 0,7004 | 0,8174 |
| baseline_lfcc_cnn_v3 | 21,10% | 0,9987 | 0,5749 | 0,7297 |
| attention_lcnn_v4 | 21,13% | 0,9998 | 0,5374 | 0,6990 |
| baseline_lfcc_cnn_v3a | 21,23% | 0,9986 | 0,6196 | 0,7647 |
| baseline_lcnn_v4 | 21,56% | 0,9985 | 0,5284 | 0,6911 |

Precisão, recall e F1 com classe positiva = spoof, no limiar calibrado no dev.
Fonte: `scripts/make_report.py` (`outputs/report/comparativo_eval.md`), rodado
em 25/09/2026.

**O limiar do dev é conservador demais no eval.** Precisão de 0,98 a 0,9999 e
recall de 0,53 a 0,72: quase nenhum falso alarme, mas de um quarto a metade do
spoof passa. No dev os ataques são os do treino e o spoof pontua alto; nos
ataques inéditos, boa parte dos scores cai abaixo do corte. É por isso que o
fusion_v4 tem EER perto do v2 e F1 bem menor: o EER independe do limiar, o F1
não. É mais uma face de "o ponto de operação não transfere" (Seção 5).

### 3.1 O dev não prevê o eval — medido

EER de dev é o do checkpoint escolhido (menor EER de dev; a época bate com a
coluna "Melhor" do `make_report`). Fonte: `outputs/*_history.json`.

| modelo | EER dev | EER eval | eval / dev |
|---|---|---|---|
| fusion_lcnn_v4 | **0,03%** | 20,18% | 766x |
| attention_lcnn_v4 | 0,04% | 21,13% | 531x |
| baseline_lfcc_cnn_v3 | 0,48% | 21,10% | 44x |
| baseline_lcnn_v4 | 0,48% | 21,56% | 45x |
| baseline_lfcc_cnn_v3a | 0,71% | 21,23% | 30x |
| baseline_lfcc_cnn_v2 | 9,35% | **18,99%** | 2x |
| baseline_lfcc_cnn | 9,74% | 20,78% | 2x |

- Cinco modelos praticamente **resolvem o dev** (0,03% a 0,71%) e todos caem
  para 20–22% no eval. Os dois que ficam em ~9,5% no dev são os **dois melhores
  no eval** (v2) e o terceiro (v1).
- Correlação entre EER de dev e EER de eval nos sete modelos: **Spearman
  ρ = −0,11** (p = 0,84, permutação exata sobre as 5.040 ordens) e **Pearson
  r = −0,61**. Não há relação positiva; o sinal, se algum, é invertido.
- **O contraste controlado é v2 → v3a** (mesma arquitetura, só `n_filter`
  20 → 70, mais AMP/grad clip): dev **9,35% → 0,71%**, eval **18,99% →
  21,23%**. Mais resolução espectral fecha o dev e piora o eval.
- O mecanismo aparece por ataque (Seção 7): a alta resolução resolve A07, A16 e
  A19 — A16 e A19 são os **mesmos algoritmos** de A04 e A06, do treino — e
  perde em A12, A13 e A18. O dev só tem A01–A06; ele mede exatamente o que a
  alta resolução aprende, e nada do que ela perde.
- Consequência de método: **selecionar o checkpoint e calibrar o limiar pelo
  dev** (como aqui e como é o padrão da área) escolhe por um critério que não
  informa sobre ataques inéditos. Ver também a precisão ≈ 1 e o recall baixo
  acima.

> **O "r = 0,951" da cópia de agosto não deve ir para o texto.** Não está em
> nenhum commit nem em nenhuma saída regenerada, e a relação dev × eval medida
> acima é nula ou negativa. Se ele media outra coisa, não há como saber.

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

Dois ajustes em relação ao texto do TC1, para o relatório do TC2: (1) os **deltas e delta-deltas do LFCC estão
no baseline em todas as versões** — o incremento 2 acrescenta só o
espectrograma; (2) a linha de base do contraste controlado é o
`baseline_lcnn_v4` (LCNN, um ramo), **não** a "CNN convencional" descrita no
TC1.

### 3-0. Arquitetura (o que está no código)

Seção que o texto do TC1 não tem e o do TC2 precisa ter. Valores de `configs/*_v4.yaml` e `src/`.

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

**A diversidade é medida, não só inferida.** Correlação de Spearman entre os
perfis de EER por ataque (13 ataques) de cada par, contra o ganho da fusão:

| par | ρ dos perfis por ataque | EER fundido |
|---|---|---|
| baseline_v2 + fusion_v4 | **+0,34** | **13,13%** |
| baseline_v2 + attention_v4 | +0,39 | 13,47% |
| baseline_v2 + baseline_v3 | +0,64 | 15,84% |
| fusion_v4 + attention_v4 (controle) | **+0,97** | 19,88% |

Quanto menos os perfis se parecem, maior o ganho — a ordem é a mesma nos quatro
pares. Com quatro pontos isso é ilustração, não teste; mas explica o controle:
fusion_v4 e attention_v4 erram praticamente nos mesmos ataques (ρ = 0,97).
(Spearman com posto médio nos empates.)

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

## 5. Robustez ao canal — simulado × real

`scripts/robustness_eval.py`, eval completo, 71.237 áudios por condição.

> **Conferido — a saturação não afeta esta seção.** Estas tabelas foram
> calculadas pela probabilidade, antes da correção da 5.3. A reavaliação numa
> amostra estratificada de 10.002 áudios (`robustness_eval.py --amostra 10000`,
> rodada em 29/09/2026, que mostra o EER pelos dois caminhos) deu **nenhum
> bonafide saturado (0 de 1.032) em nenhuma das 11 condições, nos dois
> modelos**, e o EER pela probabilidade **idêntico** ao EER pelos log-odds em
> todas as 22 linhas. A amostra reproduz o eval completo:
>
> | condição | v2 completo | v2 amostra | v4 completo | v4 amostra |
> |---|---|---|---|---|
> | limpo | 18,99% | 19,02% | 20,18% | 20,25% |
> | ruído 20 dB | 27,75% | 27,59% | 21,27% | 21,18% |
> | ruído 10 dB | 34,99% | **33,23%** | 23,95% | 24,26% |
> | ruído 5 dB | 42,41% | 41,51% | 27,42% | 27,89% |
> | ganho −6 dB | 19,44% | 19,26% | 20,36% | 20,25% |
> | ganho +6 dB | 19,57% | 19,77% | 20,08% | 20,24% |
> | Opus 30 kbps | 19,75% | 19,96% | 21,38% | 21,71% |
> | Opus 25 kbps | 20,04% | 20,32% | 22,08% | 22,40% |
> | Opus 15 kbps | 20,12% | 20,74% | 23,02% | 22,95% |
> | banda estreita | 35,95% | 36,22% | 25,53% | 25,67% |
> | banda estreita + Opus | 32,85% | 32,65% | 25,04% | 25,31% |
>
> 21 das 22 diferenças ficam dentro de ±1,28 pp; a exceção (v2 com ruído a
> 10 dB, −1,76 pp) é o esperado por acaso em 22 comparações a 95% (≈ 1,1). A
> amostra usa o mesmo ruído do eval completo — o `Subset` preserva o índice
> original, e o ruído é derivado de (semente, época, índice) —, então a
> diferença é só de amostragem. **O vencedor é o mesmo nas 11 condições** nas
> duas medidas.

| condição | baseline_v2 | Δ | fusion_v4 | Δ | vence | no treino? |
|---|---|---|---|---|---|---|
| limpo | **18,99%** | — | 20,18% | — | v2 | — |
| ruído 20 dB | 27,75% | +8,76 | **21,27%** | +1,09 | **v4 por 6,5 pp** | **sim** |
| ruído 10 dB | 34,99% | +16,00 | **23,95%** | +3,77 | **v4 por 11,0 pp** | **sim** |
| ruído 5 dB | 42,41% | +23,42 | **27,42%** | +7,24 | **v4 por 15,0 pp** | não |
| ganho −6 dB | **19,44%** | +0,45 | 20,36% | +0,18 | v2 | sim |
| ganho +6 dB | **19,57%** | +0,58 | 20,08% | −0,10 | v2 | sim |
| Opus 30 kbps | **19,75%** | +0,76 | 21,38% | +1,20 | v2 | não |
| Opus 25 kbps | **20,04%** | +1,05 | 22,08% | +1,90 | v2 | não |
| Opus 15 kbps | **20,12%** | +1,13 | 23,02% | +2,84 | v2 | não |
| banda estreita (8 kHz) | 35,95% | +16,96 | **25,53%** | +5,35 | **v4 por 10,4 pp** | não |
| banda estreita + Opus | 32,85% | +13,86 | **25,04%** | +4,86 | **v4 por 7,8 pp** | não |
| **pior degradação** | | **+23,42** | | **+7,24** | | |

Fonte: `outputs/report/robustez_eval.md` (25/09/2026). Os outros cinco modelos
não têm avaliação de robustez.

O fusion_v4 vence nas **cinco** condições de ruído e banda estreita; o v2 vence
nas seis restantes (limpo, ganho e Opus), sempre por menos de 3 pp.

**Sob degradação simulada, a ordem se inverte.** O modelo que vence no
benchmark limpo é o que desaba com banda estreita e ruído.

**Ressalva obrigatória: parte da robustez a ruído é "dentro da distribuição".**
O treino usa ruído gaussiano branco a 10–30 dB e ganho de ±6 dB, com a **mesma
função** `add_noise` (`src/preprocess/augment.py`) que o teste de robustez
aplica. As condições de 20 dB, 10 dB e ±6 dB já foram vistas no treino; as
**inéditas** são ruído a 5 dB, Opus e banda estreita. Isso precisa estar
declarado no texto.

> **Correção de uma afirmação anterior desta seção.** Ela dizia que a inversão
> aparecia só nas condições inéditas. A tabela completa desmente: **a inversão
> já aparece a 20 dB e a 10 dB, que estão na distribuição de treino** — o v2
> piora 8,76 pp com um ruído que viu no treino, o fusion_v4 1,09 pp. Isso
> **fortalece** o resultado em vez de enfraquecer: a diferença de robustez a
> ruído não é questão de ter visto ou não a perturbação, porque os dois viram a
> mesma. É da representação/arquitetura.

> **Atenção — a Seção 5.3 muda a leitura desta tabela.** No canal de transmissão
> **real** (ASVspoof 2021 LA, Opus sobre redes VoIP de verdade), a ordem volta a
> ser a do áudio limpo: `baseline_v2` 29,43%, `fusion_v4` 32,62%. A robustez do
> `fusion_v4` à banda estreita e ao ruído *simulados* não virou vantagem no
> canal real. A conclusão que se sustenta: **a degradação simulada não prediz
> qual modelo resiste ao canal real.**

O que custa o quê, **na simulação**:


**O codec testado é Opus, não MP3.** O `infer.py` aceita MP3, mas nenhuma
avaliação foi feita com esse formato. O texto não deve afirmar robustez a MP3.

- **Ganho de ±6 dB não custa nada** (≤ 0,6 pp) — esperado, a normalização por
  pico o desfaz.
- **O codec Opus simulado custa pouco**: +0,8 a +1,1 pp no v2 e +1,2 a +2,8 pp
  no fusion_v4, até 15 kbps. É o único eixo em que o fusion_v4 é o mais sensível
  dos dois. A Seção 5.3 mostra que o Opus **real** custa de 7 a 11 vezes mais.
- **Perder a banda alta custa muito**, e o ruído custa mais ainda — no v2. No
  fusion_v4 o pior caso inteiro (ruído a 5 dB) custa +7,24 pp.
- Opus sobre banda estreita dá **menos** EER que banda estreita sozinha nos dois
  modelos (32,85% contra 35,95% no v2). Não há explicação medida para isso;
  não interpretar sem investigar.

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
| incremento da fusão, por ataque (7.0) | −37,65 a +25,78 pp | **sim** |
| dev × eval (3.1) | dev 0,03–9,74%, eval 18,99–21,56% | **sim** |
| inversão de ranking sob canal **simulado** (Seção 5) | 6,5 a 15,0 pp | **sim** — mas não se repete no canal real (5.3) |
| ordem no canal real, v2 × v4 (5.3) | 3,19 pp | **não** — acima só do IC de amostragem (±1,28 pp), não da variância entre execuções |
| fusão de scores no canal real (5.3) | 4,50 pp | **não** — dentro da faixa, mas repete o sentido e a ordem de grandeza do 5,86 pp de 2019 |
| ganho da fusão de scores (Seção 4) | 5,86 pp | **sim** |
| incremento da fusão de características (Seção 3) | −1,38 pp | **não** |
| incremento da atenção (Seção 3) | +0,95 pp | **não** |

Os resultados grandes sobrevivem. **Os dois incrementos do TC1 (−1,38 e +0,95
pp) ficam dentro da faixa de variação entre execuções que Müller observa**, e
não podem ser atribuídos ao método com uma execução só. Retreinar com mais
sementes resolveria (≈3 h por treino, medido), mas o caminho barato é declarar
a limitação no texto e apoiar a conclusão sobre a fusão na Seção 7, onde o
efeito é por ataque e muito maior.

### 5.3 Canal real: ASVspoof 2021 LA

Os mesmos ataques A07–A19 do eval de 2019, agora **transmitidos por redes reais**.
Fase `eval` (o conjunto oficial: 14.816 bonafide, 133.360 spoof), amostra
estratificada de 10.000 por condição, mesma semente nas duas condições e nos dois
modelos — os dois modelos veem exatamente os mesmos áudios.

| | referência (sem codec) | **Opus real** | variação |
|---|---|---|---|
| baseline_v2 | 18,15% | **29,43%** | +11,28 pp |
| fusion_v4 | 19,51% | **32,62%** | +13,11 pp |
| fusão `rank` (v2 + v4) | — | **24,94%** | −4,50 pp sobre o v2 |

**Controles que validam a medida:**

- A condição de referência reproduz o eval de 2019 nos dois modelos (18,15%
  contra 18,99%; 19,51% contra 20,18%), dentro do IC 95% de ±1,28 pp da amostra.
  A mudança no Opus é do canal, não de diferença entre as bases.
- Os `.flac` do 2021 não decodificam no libsndfile (38 de 50 testados, "unknown
  error in flac decoder", versão 1.2.2) e foram convertidos para WAV uma vez
  (`scripts/converter_para_wav.py`). A conversão é exata: o baseline_v2 na
  referência deu **métricas idênticas** pelos dois caminhos — FFmpeg direto e
  WAV convertido (EER 18,15%, acurácia 0,7485, precisão 0,9872, recall 0,7299).

**Achado 1 — a simulação subestima o canal real em uma ordem de grandeza.**

| | Opus simulado (25 kbps, Seção 5) | Opus real | razão |
|---|---|---|---|
| baseline_v2 | +1,05 pp | +11,28 pp | ~11x |
| fusion_v4 | +1,90 pp | +13,11 pp | ~7x |

A Seção 10.1 chamava a camada 1 de "limite inferior" sem número. Agora tem: o
custo real é de 7 a 11 vezes o simulado. Ressalva: o "Opus real" é codec **e
transmissão** (três redes: `ita_tx`, `loc_tx`, `sin_tx`, com taxa e perdas
desconhecidas), enquanto a simulação é só o codec numa taxa fixa. A diferença
inclui tudo o que a rede faz além do codec — que é justamente o que se queria
medir.

> **Correção de medida — a primeira execução deu 49,06% para o `fusion_v4`, e
> era artefato.** O score salvo era o softmax em float32, que arredonda para
> exatamente 1,0 quando a margem entre os logits passa de ~17 (medido). Se
> bonafide e spoof empatam em 1,0, o EER mede o arredondamento, não o modelo — um
> teste do projeto mostra classes perfeitamente separadas pelo logit dando EER ≥
> 40% pelo softmax. `scripts/checar_saturacao.py` mediu:
>
> | arquivo de scores | bonafide em 1,0 | spoof em 1,0 | valores distintos |
> |---|---|---|---|
> | todos os de 2019 (eval completo) | **0** | até ~26 mil | — |
> | 2021 referência (v2 e v4) | **0** | — | — |
> | 2021 Opus, baseline_v2 | **0** | — | — |
> | 2021 Opus, fusion_v4 | **975 de 1.002 (97%)** | 8.926 de 8.999 (99%) | 52 |
>
> Só o último estava contaminado. Spoof em 1,0 com bonafide abaixo é acerto e
> não muda o EER, então **todos os EERs de 2019, as duas referências de 2021 e o
> 29,43% do baseline_v2 no Opus valiam**. O EER passou a ser calculado sobre os
> log-odds (logit[1] − logit[0]), que não saturam; o `fusion_v4` no Opus foi
> reavaliado e deu **32,62%**. Vale registrar como aconteceu: o sintoma que
> denunciou o erro foi a análise por ataque — quatro ataques com EER idêntico
> (48,65%) e tamanhos de amostra diferentes, o que só ocorre com scores
> empatados.

**Achado 2 — a robustez simulada não virou vantagem no canal real.** Sob banda
estreita e ruído simulados, o `fusion_v4` era o robusto (Seção 5). Sob
transmissão real, o `baseline_v2` vence por **3,19 pp** (29,43% contra 32,62%),
diferença maior que o IC 95% de ±1,28 pp de cada medida. A ordem no canal real é
a do áudio limpo e do Opus simulado, não a da banda estreita e do ruído. A
simulação não previu qual modelo resiste — isso não invalida a Seção 5, invalida
a extrapolação dela.

**Achado 2b — a diversidade entre modelos se confirma no canal real.** A fusão
`rank` dos dois dá **24,94%**, −4,50 pp sobre o melhor isolado — o mesmo
fenômeno da Seção 4 (−5,86 pp no eval de 2019), agora sob transmissão real. Por
ataque, os menores EERs do `fusion_v4` (A17 14,78%, A19 17,26%, A09 17,88%)
são onde a fusão mais ganha (fundidos, A09 cai para 6,74%) — mas note que ali o
v2 isolado já é tão bom ou melhor (5,34% no A09; ver 5.3.1). Os dois falham juntos
em A10 e A12 (42,98% e 37,53% fundidos) — os mesmos dois que já resistiam aos
dois modelos em 2019 (Seção 7.3).

**Achado 2c — a regra streamável não captura esse ganho.** A regra `mean` (a do
monitor) dá **29,41%**, igual ao `baseline_v2` sozinho (29,43%). O motivo é a
calibração: 97% dos bonafide recebem probabilidade 1,0 do `fusion_v4` no Opus,
então a média das probabilidades é, na prática, o v2 deslocado. A regra `max`
repete o 49,06% pelo mesmo motivo — ela opera sobre probabilidades. O ganho da
fusão existe na **ordenação** dos dois modelos, e a `rank` precisa do conjunto
inteiro; uma fusão ao vivo que o aproveite exige calibrar cada modelo **no canal
de destino**, o que fica como trabalho futuro.

**Achado 3 — o limiar calibrado colapsa.** No Opus real, os dois modelos
classificam **tudo** como spoof no limiar do checkpoint: recall 1,0000 e precisão
0,8998, que é exatamente a proporção de spoof da amostra. É o classificador
trivial da Seção 8.1, produzido por um canal real e não por construção.

**Consequência para o monitor ao vivo.** A escolha do `fusion_v4` para chamadas
se apoiava na robustez simulada, e o canal real a contradiz: o `baseline_v2` é
melhor sozinho (29,43% contra 32,62%), e a fusão pela média — a única
streamável — empata com ele (29,41%) custando o dobro de processamento. **O
monitor passa a recomendar o `baseline_v2` sozinho.**

### 5.3.1 O canal real por ataque

`per_attack_eval.py` nos quatro configs do 2021 (29/09/2026). Cada ataque tem
~690 spoof na amostra, então o IC 95% por ataque é de ±1,5 pp (EER ≈ 5%) a
±3,4 pp (EER ≈ 40%): **diferenças abaixo de ~5 pp por ataque não se leem.**
EER (%):

| ataque | v2 2019 | v2 ref. 2021 | v2 Opus real | v4 2019 | v4 ref. 2021 | v4 Opus real |
|---|---|---|---|---|---|---|
| A07 | 22,43 | 21,73 | 41,22 | 0,02 | 0,00 | 37,13 |
| A08 | 3,88 | 3,39 | 9,27 | 0,03 | 0,00 | 21,45 |
| A09 | 2,32 | 1,31 | **5,34** | 0,12 | 0,17 | 17,88 |
| A10 | 30,63 | 31,07 | 43,61 | 45,54 | 43,53 | 49,50 |
| A11 | 3,85 | 3,77 | **19,86** | 33,74 | 30,86 | 48,74 |
| A12 | 36,18 | 33,13 | 40,55 | 47,99 | 46,20 | 41,80 |
| A13 | 36,50 | 34,23 | 46,79 | 6,54 | 6,15 | **22,37** |
| A14 | 12,01 | 10,94 | **21,27** | 17,74 | 16,61 | 41,86 |
| A15 | 15,02 | 15,35 | **16,93** | 29,37 | 29,00 | 42,68 |
| A16 | 22,75 | 22,99 | 43,48 | 0,03 | 0,00 | **34,92** |
| A17 | 11,27 | 9,98 | 15,19 | 0,41 | 1,06 | 14,78 |
| A18 | 15,01 | 15,79 | 27,86 | 4,63 | 5,07 | **18,13** |
| A19 | 6,98 | 6,83 | 16,44 | 0,00 | 0,00 | 17,26 |
| **global** | 18,99 | 18,15 | **29,43** | 20,18 | 19,51 | 32,62 |
| média por ataque | 16,83 | 16,19 | **26,75** | 14,32 | 13,74 | 31,42 |

**1. A referência de 2021 reproduz 2019 ataque por ataque.** Correlação de
postos entre os perfis: ρ = 0,99 nos dois modelos; nenhum ataque difere mais
de 3,05 pp, dentro do IC. Valida a importação do 2021 e a Seção 7 ao mesmo
tempo: o perfil por ataque é propriedade do modelo, não ruído da amostra.

**2. O Opus real apaga exatamente o que dava vantagem ao `fusion_v4`.** Os
ataques que ele zerava — A07, A08, A16 e A19, 0,00% na referência — vão a
17–37%. O perfil do v2 sobrevive ao canal (ρ referência × Opus = 0,92, custo
espalhado de +2 a +20 pp); o do `fusion_v4` se desfaz (ρ = 0,67, custo de até
+37 pp). Na média por ataque o `fusion_v4` passa de **melhor** (13,74% contra
16,19%) a **pior** (31,42% contra 26,75%). É o mecanismo da Seção 5.3: a
vantagem do `fusion_v4` era detecção quase perfeita de alguns ataques, e é essa
precisão que o canal real destrói. Hipótese compatível, não medida: os ataques
que ele zerava dependem de detalhe espectral fino (A07/A16/A19 são justamente
os que a resolução de 70 filtros resolve, Seção 7.5), e o Opus atenua a banda
alta (−4,4 dB entre 6 e 8 kHz, medido no `channel.py`).

**3. A vantagem nos ataques por filtragem sobrevive, em parte.** Onde o
`fusion_v4` continua vencendo no Opus real estão A13 (22,37% contra 46,79%) e
A18 (18,13% contra 27,86%) — dois dos três ataques por filtragem da hipótese da
7.5; o terceiro, A17, empata dentro do IC (14,78% contra 15,19%). Também segue
com ele o A16 (34,92% contra 43,48%), e o A07 por margem próxima do IC (4,1 pp). Reforça a hipótese da 7.5 sem
prová-la (post hoc, poucos ataques).

**4. A diversidade encolhe no canal real, e a fusão rende menos.** ρ entre os
perfis de v2 e `fusion_v4`: 0,34 em 2019, 0,36 na referência, **0,48** no Opus
real. A fusão rende −5,86 pp em 2019 e −4,50 pp no Opus. Mesmo sentido da
relação da Seção 4 (perfis mais parecidos, ganho menor) — um quinto ponto,
ainda ilustrativo. O oráculo (melhor dos dois por ataque) vai de 8,21% na
referência a 23,12% no Opus.

**5. A10 e A12 continuam os piores** para os dois modelos nas três condições.

Comandos desta seção:

```bash
python evaluate.py --config outputs/asvspoof2021/opus_eval_n10000__fusion_lcnn_v4.yaml --checkpoint checkpoints/fusion_lcnn_v4.pt
python scripts/per_attack_eval.py --config outputs/asvspoof2021/opus_eval_n10000__fusion_lcnn_v4.yaml --checkpoint checkpoints/fusion_lcnn_v4.pt
# os quatro configs (none/opus × v2/v4), como na 5.3.1:
# for cfg in outputs/asvspoof2021/*__*.yaml: per_attack_eval.py --config cfg --checkpoint <do modelo>
python scripts/score_fusion.py --name fusao_2021_opus \
    --model outputs/asvspoof2021/opus_eval_n10000__baseline_lfcc_cnn_v2.yaml checkpoints/baseline_lfcc_cnn_v2.pt \
    --model outputs/asvspoof2021/opus_eval_n10000__fusion_lcnn_v4.yaml checkpoints/fusion_lcnn_v4.pt
```

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

### 7.0 Os sete modelos (matriz completa)

Fonte: `outputs/report/por_ataque_eval.md` (25/09/2026). EER (%).

| ataque | v1 | v2 | v3a | v3 | lcnn_v4 | fusion_v4 | attention_v4 | média |
|---|---|---|---|---|---|---|---|---|
| A07 | 22,28 | 22,43 | 0,24 | 0,13 | 0,27 | 0,02 | 0,01 | 6,48 |
| A08 | 2,68 | 3,88 | 0,05 | 0,00 | 0,59 | 0,03 | 0,05 | 1,04 |
| A09 | 1,19 | 2,32 | 0,19 | 0,19 | 0,29 | 0,12 | 0,08 | 0,63 |
| A10 | 31,24 | 30,63 | 32,62 | 35,88 | 37,50 | 45,54 | 44,61 | **36,86** |
| A11 | 4,60 | 3,85 | 4,42 | 11,69 | 7,96 | 33,74 | 32,03 | 14,04 |
| A12 | 43,50 | 36,18 | 56,47 | 50,05 | 52,28 | 47,99 | 48,11 | **47,80** |
| A13 | 34,59 | 36,50 | 59,58 | 48,31 | 44,19 | 6,54 | 5,51 | 33,60 |
| A14 | 10,71 | 12,01 | 0,96 | 2,08 | 12,14 | 17,74 | 28,49 | 12,02 |
| A15 | 19,23 | 15,02 | 8,75 | 12,05 | 9,54 | 29,37 | 32,33 | 18,04 |
| A16 | 22,08 | 22,75 | 0,70 | 0,30 | 0,75 | 0,03 | 0,09 | 6,67 |
| A17 | 11,20 | 11,27 | 10,28 | 8,22 | 10,50 | 0,41 | 0,39 | 7,47 |
| A18 | 21,48 | 15,01 | 30,14 | 32,41 | 34,11 | 4,63 | 5,56 | 20,48 |
| A19 | 8,99 | 6,98 | 0,13 | 0,10 | 0,37 | 0,00 | 0,03 | 2,37 |
| **global** | 20,78 | **18,99** | 21,23 | 21,10 | 21,56 | 20,18 | 21,13 | |
| **média por ataque** | 17,98 | 16,83 | 15,73 | 15,49 | 16,19 | **14,32** | 15,18 | |

**Três famílias de perfil.** Correlação de Spearman entre os perfis por ataque:
v1 × v2 = 0,97; v3 × v3a = 0,97; v3a × lcnn_v4 = 0,91; v3 × lcnn_v4 = 0,90;
fusion_v4 × attention_v4 = 0,97; entre famílias, 0,34 a 0,85. As famílias são:

1. **LFCC com 20 filtros** (v1, v2): falham nos ataques parecidos com o treino
   (A07, A16, A19 — A16 e A19 usam os algoritmos de A04 e A06) e acertam A11.
2. **LFCC com 70 filtros, um ramo** (v3a, v3, lcnn_v4): resolvem A07/A16/A19,
   falham em A12, A13 e A18.
3. **LFCC + espectrograma** (fusion_v4, attention_v4): resolvem também A13, A17
   e A18, e falham em A10, A11, A14 e A15.

**Os dois incrementos, ataque a ataque** (contraste controlado, em pp):

| ataque | lcnn_v4 → fusion_v4 | fusion_v4 → attention_v4 |
|---|---|---|
| A13 | **−37,65** | −1,03 |
| A18 | **−29,48** | +0,93 |
| A17 | **−10,09** | −0,02 |
| A12 | −4,29 | +0,12 |
| A11 | **+25,78** | −1,71 |
| A15 | **+19,83** | +2,96 |
| A10 | +8,04 | −0,93 |
| A14 | +5,60 | **+10,75** |
| demais (5) | −0,72 a −0,17 | −0,04 a +0,06 |
| **global** | **−1,38** | **+0,95** |

- **A fusão de características não é um efeito de 1,38 pp.** É uma troca de
  até 37,65 pp num sentido e 25,78 pp no outro, cujo saldo é −1,38. Esses
  efeitos por ataque são muito maiores que a variância entre execuções
  (1,4–5,2 pp, Seção 5.2): **é aqui que a defesa do incremento 2 se apoia.**
- **A atenção quase não muda nada.** Onze dos treze ataques mudam menos de
  3 pp; a exceção é A14 (+10,75). Perfil com ρ = 0,97 em relação ao fusion_v4.
  Coerente com 258 parâmetros a mais (Seção 3).

**O oráculo entre os sete** (melhor modelo em cada ataque) daria **7,00%** de
EER médio por ataque — mesmo limite teórico da 7.1, com mais modelos.

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

### 7.3 A10 e A12 resistem aos sete modelos

| | A10 | A12 |
|---|---|---|
| menor EER entre os sete | 30,63% (v2) | 36,18% (v2) |
| maior EER entre os sete | 45,54% (fusion_v4) | 56,47% (v3a) |
| média dos sete | **36,86%** | **47,80%** |

São os **únicos dois ataques em que todos os sete modelos passam de 30%**. O
terceiro mais difícil, A13 (média 33,60%), é resolvido pela família com
espectrograma (5,51% e 6,54%). Nenhuma configuração testada — resolução,
encoder, segundo ramo, atenção — resolve A10 ou A12. A12 na média dos sete dá
47,80%: acaso.

### 7.3.1 O que separa os difíceis: geração autorregressiva

Cruzando os EERs medidos com o gerador de forma de onda de cada ataque:

| ataque | v2 | fusion_v4 | gerador de forma de onda |
|---|---|---|---|
| A12 | 36,18% | 47,99% | **WaveNet (autorregressivo)** |
| A10 | 30,63% | 45,54% | **WaveRNN (autorregressivo)** |
| A15 | 15,02% | 29,37% | **WaveNet (autorregressivo)** |
| A13 | 36,50% | 6,54% | filtragem de forma de onda (VC sobre voz de TTS por seleção de unidades) |
| A11 | 3,85% | 33,74% | Griffin-Lim |
| A14 | 12,01% | 17,74% | vocoder clássico (STRAIGHT) |
| A16 | 22,75% | 0,03% | concatenação |
| A07 | 22,43% | 0,02% | vocoder clássico (WORLD + pós-filtro GAN) |
| A18 | 15,01% | 4,63% | vocoder MFCC (modifica o resíduo da fala de origem) |
| A17 | 11,27% | 0,41% | filtragem de forma de onda |
| A19 | 6,98% | 0,00% | filtragem espectral |
| A08 | 3,88% | 0,03% | **neural**, source-filter |
| A09 | 2,32% | 0,12% | vocoder clássico (Vocaine) |

> **Conferido na fonte em 25/09/2026** — Wang et al. (2020), Tabela 1
> ("Summary of LA spoofing systems"), arXiv 1911.01601v4, p. 6:
>
> | ataque | entrada | modelo acústico / conversão | gerador de forma de onda |
> |---|---|---|---|
> | A07 | texto | RNN | WORLD + pós-processamento GAN |
> | A08 | texto | RNN autorregressivo | neural source-filter |
> | A09 | texto | RNN | Vocaine |
> | A10 | texto | Tacotron 2 + d-vector | WaveRNN |
> | A11 | texto | Tacotron 2 + d-vector (igual ao A10) | Griffin-Lim |
> | A12 | texto | RNN | WaveNet (AR) |
> | A13 | fala de TTS (seleção de unidades) | *moment matching* (VC) | filtragem de forma de onda |
> | A14 | fala de TTS | RNN sobre features de ASR (VC) | STRAIGHT |
> | A15 | fala de TTS | RNN sobre features de ASR (VC) | WaveNet (AR) |
> | A16 | texto | CART | concatenação (= A04) |
> | A17 | fala humana | VAE (VC) | filtragem de forma de onda |
> | A18 | fala humana | i-vector/PLDA linear (VC) | **vocoder MFCC** |
> | A19 | fala humana | GMM-UBM (VC) | filtragem espectral + OLA (= A06) |
>
> O conflito do A18 está **resolvido: vocoder MFCC** (a descrição "vocoder
> glotal DNN" veio de um resumo de busca e estava errada). A13 não é
> "concatenação": a *fonte* é uma voz de TTS por seleção de unidades, e o
> gerador é filtragem de forma de onda. A10, A12 e A15 continuam os únicos com
> gerador autorregressivo. O A08 tem modelo acústico autorregressivo e gerador
> não autorregressivo — o eixo é o gerador. A cópia de agosto deste resumo
> rotulava A09, A14 e A18 como "WORLD" e A08 como "WaveNet"; os quatro estavam
> errados.

| grupo | v2 | fusion_v4 |
|---|---|---|
| autorregressivos (A10, A12, A15) | **27,28%** | **40,97%** |
| todos os outros (10 ataques) | 13,70% | **6,33%** |

Para o `fusion_v4` a diferença é de **6,5x**. Teste de permutação exato (os 286
trios possíveis entre os 13 ataques), agora nos sete modelos:

| modelo | trio AR | outros 10 | posição do trio entre 286 | p |
|---|---|---|---|---|
| attention_v4 | 41,68% | 7,22% | 1º | **0,0035** |
| fusion_v4 | 40,97% | 6,33% | 2º | **0,0070** |
| v1 | 31,32% | 13,98% | 9º | 0,0315 |
| lcnn_v4 | 33,11% | 11,12% | 11º | 0,0385 |
| v2 | 27,28% | 13,70% | 14º | 0,0490 |
| v3 | 32,66% | 10,34% | 14º | 0,0490 |
| v3a | 32,61% | 10,67% | 19º | 0,0664 |

**Como escrever:** forte na família com espectrograma (p < 0,01); nos outros
cinco, no limite de 0,05 ou acima (v3a). São sete testes sobre os mesmos 13
ataques e não independentes entre si, então **não** somar como "7 confirmações".
O que se sustenta: A10 e A12 difíceis para todos (7.3); A15 difícil só para
parte dos modelos (8,75% no v3a contra 32,33% no attention_v4).

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
um artefato bem característico. Todos os modelos só com LFCC ficam entre 3,85% e
11,69%; os dois com ramo de espectrograma ficam em 32–34%. No contraste
controlado (lcnn_v4 → fusion_v4, só o segundo ramo muda), o A11 **piora
25,78 pp** (7,96% → 33,74%). A diferença de 30 pp entre v2 e fusion_v4 mistura
esse efeito com resolução e encoder; para o texto, usar os 25,78 pp.

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
- A fusão de características tem um custo medido: no A11 ela **piora 25,78 pp**
  (contraste controlado) e ganha 37,65 pp no A13. O ganho agregado de −1,38 pp
  é um saldo, não um ganho uniforme — e é essa troca por ataque, não o saldo,
  que supera a variância entre execuções.
- A atenção quase não muda o perfil (ρ = 0,97 com o fusion_v4).
- Os modelos se agrupam em três famílias de perfil (7.0), e a fusão de scores
  rende mais quanto mais distantes as famílias (Seção 4).

### 7.5 Os baselines oficiais por ataque confirmam três achados

Wang et al. (2020), Tabela 8, dão o EER por ataque de B01 (CQCC-GMM) e B02
(LFCC-GMM) no eval. **Ressalva: os baselines usam o áudio com silêncio**; os
valores absolutos não são comparáveis aos deste projeto, a ordem entre ataques
é o que interessa.

| ataque | B01 CQCC | B02 LFCC-20 | v2 LFCC-20 | v3a LFCC-70 | lcnn_v4 | fusion_v4 |
|---|---|---|---|---|---|---|
| A07 | 0,00 | **12,86** | **22,43** | 0,24 | 0,27 | 0,02 |
| A08 | 0,04 | 0,37 | 3,88 | 0,05 | 0,59 | 0,03 |
| A09 | 0,14 | 0,00 | 2,32 | 0,19 | 0,29 | 0,12 |
| A10 | 15,16 | 18,97 | 30,63 | 32,62 | 37,50 | 45,54 |
| A11 | 0,08 | 0,12 | 3,85 | 4,42 | 7,96 | 33,74 |
| A12 | 4,74 | 4,92 | 36,18 | 56,47 | 52,28 | 47,99 |
| A13 | **26,15** | 9,57 | 36,50 | 59,58 | 44,19 | **6,54** |
| A14 | 10,85 | 1,22 | 12,01 | 0,96 | 12,14 | 17,74 |
| A15 | 1,26 | 2,22 | 15,02 | 8,75 | 9,54 | 29,37 |
| A16 | 0,00 | **6,31** | **22,75** | 0,70 | 0,75 | 0,03 |
| A17 | **19,62** | 7,71 | 11,27 | 10,28 | 10,50 | **0,41** |
| A18 | 3,81 | 3,58 | 15,01 | 30,14 | 34,11 | 4,63 |
| A19 | 0,04 | **13,94** | **6,98** | 0,13 | 0,37 | 0,00 |
| global | 9,57 | 8,09 | 18,99 | 21,23 | 21,56 | 20,18 |

**1. A falha do LFCC com 20 filtros se repete com outro classificador.** O B02
usa LFCC com 20 filtros lineares — a mesma resolução do v1/v2 — e um GMM em vez
de CNN. Falha **nos mesmos três ataques**: A07, A16 e A19 (12,86%, 6,31% e
13,94%), que o B01 (CQCC) resolve (≤ 0,04%). O v1/v2 falha nos mesmos (7–23%),
e o v3a, com 70 filtros, os resolve (≤ 0,70%). Correlação de postos dos perfis:
v1/v2 × B02 = 0,62–0,64; v3a/lcnn_v4 × B01 = 0,78–0,81. **Mesma representação,
classificadores diferentes, mesmas falhas: a causa é a resolução espectral, não
o classificador.** Isso sustenta a explicação da Seção 3.1 com evidência
externa. Wang et al. registram o mesmo: o B02 "já é fraco em A04 e A06", e por
isso não se beneficia de A16/A19 serem algoritmos do treino.

**2. O gerador de forma de onda pesa mais que o modelo acústico — conclusão do
próprio artigo.** A10 e A11 têm o **mesmo** modelo acústico e diferem só no
gerador (WaveRNN × Griffin-Lim); Wang et al. usam esse par para concluir que o
gerador domina. É o mesmo eixo da 7.3.1, agora com citação: "the acoustic model
seemed to have less of an impact than the waveform generation method".

**3. O dev também não previa o eval nos baselines oficiais.** B01: 0,43% no
dev, 9,57% no eval (Wang et al., Tabelas 7 e 8), atribuído pelos autores aos
ataques novos. A Seção 3.1 não é peculiaridade deste projeto.

**Hipótese nova (post hoc, 3 ataques): o ramo de espectrograma captura os
métodos de filtragem.** A13 e A17 geram a onda **filtrando uma fala existente**,
e o A18 modifica o resíduo da fala de origem. Wang et al. apontam a filtragem
(A13, A17) como o que os baselines mais erram. Média desses três ataques:
lcnn_v4 **29,60%**, fusion_v4 **3,86%**, attention_v4 3,82%. É onde está quase
todo o ganho do incremento 2 (Tabela da 7.0: −37,65, −10,09 e −29,48 pp). Foi
formulada **depois** de ver os dados, então no texto entra como hipótese, não
como resultado.

**Divergência a registrar, sem interpretar além do que se mede:** o A12
(WaveNet) é fácil para os baselines (4,7–4,9%) e está entre os mais difíceis
aqui (36–56%). Os baselines usam o silêncio; é possível que parte da facilidade
do A12 venha dele (Müller et al., 2021), mas isso não foi medido.

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

A degradação simulada também mostra o F1 apontando na direção errada. Na
amostra de 10.002 áudios (29/09/2026), o `baseline_v2` tem F1 **0,8281** no
áudio limpo e **0,9056** com Opus a 15 kbps — acima do RNF02 — enquanto o EER
**piora** de 19,02% para 20,74%. O codec desloca os scores para o lado spoof,
o recall sobe (0,7149 → 0,8563) e o F1 melhora com um modelo que discrimina
pior. O `fusion_v4` sofre o deslocamento oposto: recall 0,5547 → 0,3929.

O canal real produziu esse caso sem que ninguém o construísse. No ASVspoof 2021
LA com Opus (Seção 5.3), o `fusion_v4` classifica **todos** os áudios como spoof
no limiar calibrado (recall 1,0000, precisão 0,8998 = a proporção de spoof da
amostra) e obtém F1 = **0,9473**, acima do RNF02. Um ponto de operação que não
distingue nada atende o requisito.

Esse achado vale mais que o cumprimento do requisito. É um resultado de
engenharia de requisitos: **F1 sobre classe majoritária não mede capacidade de
detecção**. (Convenção a declarar no texto do TC2, §4.11: a classe positiva é o
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

### 9.5 Sensibilidade a deslocamentos de poucos milissegundos (medida)

Achado lateral do teste ao vivo. Na sessão "limpo" (Seção 10.2) os 40 áudios da
playlist são recortados da própria referência — o mesmo sinal, sem canal
nenhum. Mesmo assim os scores diferiram dos do eval de 2019 para os mesmos
arquivos: mediana ~0,01, **máximo 0,17** (0,501 → 0,669), dois dos 40 trocando
de lado no limiar, e o EER da amostra indo de 22,50% para 27,50%.

A causa era o recorte: o alinhamento por envelope tem resolução de 10 ms e
deslocava cada áudio em **até 5 ms** (medido: 79 amostras com a gravação
idêntica à referência). Cinco milissegundos são meio passo do STFT (hop de
10 ms): todos os quadros das features mudam de fase, e o modelo responde. O
alinhamento ganhou um ajuste fino pela forma de onda, com erro medido de zero
amostras no caminho limpo, com atraso arbitrário e sob Opus + banda estreita.

**O que isso diz do modelo, e por que importa ao vivo:** o score de um mesmo
áudio depende de onde o enquadramento começa, em escala de milissegundos. No
monitor as janelas caem em posições arbitrárias da fala, então esse jitter faz
parte de toda leitura ao vivo — mais um motivo para agregar janelas (9.2, 9.3)
em vez de decidir por uma só. Ressalva: 40 áudios, um modelo; é indício, não
medida da sensibilidade. Medir direito seria avaliar o eval com deslocamentos
de 0 a 10 ms e comparar os EERs.

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
python scripts/canal_real.py preparar --config configs/baseline_v2.yaml --n-por-classe 20
python monitor.py --config ... --checkpoint ... --gravar outputs/canal_real/chamada.wav
python scripts/canal_real.py alinhar --pasta outputs/canal_real --gravacao outputs/canal_real/chamada.wav --sessao chamada
python scripts/comparar_sessoes.py outputs/canal_real/*/config_canal_real.yaml
```

Passo a passo completo, com as sessões limpo / controle / chamada e o que
anotar: [`docs/ROTEIRO_TESTE_AO_VIVO.md`](ROTEIRO_TESTE_AO_VIVO.md).

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

**Execução completa não feita** — decisão registrada em 12.3. Uma **execução
reduzida** (40 áudios, três sessões) está prevista como teste funcional do
monitor ao vivo; o número que ela dá é indicativo, não uma medida de EER.

### 10.2.1 Execução reduzida — resultados até aqui (parcial)

Playlist de 40 áudios do eval de 2019 (20 bonafide, 20 spoof, os 13 ataques),
`baseline_v2`. Com 20 bonafide, cada bonafide mal ordenado move o EER em
~2,5 pp: os números são indicativos. O que importa é a **direção** dos scores,
áudio por áudio (`comparar_sessoes.py`).

| sessão | o que muda | EER (mesmos áudios) | acertos no limiar |
|---|---|---|---|
| eval de 2019 | — | 22,50% | 30/40 |
| **limpo** (recortes da própria playlist) | nada | **22,50%** — scores idênticos | 30/40 |
| perdas simuladas (1 × 10 ms por áudio) | um clique | 17,50% | 29/40 |
| reamostragem simulada (16 → 48 → 16 kHz) | some 7,6–8 kHz | 30,00% | 25/40 |
| reamostragem + ruído branco a −31 dB do pico | topo some e piso sobe | 42,50% | 28/40 |
| controle com aprimoramentos do driver ligados | caminho real de saída + efeitos Realtek | 48,53% (37 áudios) | 16/37 |
| **controle sem aprimoramentos** (alto-falante → loopback, sem chamada) | caminho real de saída | **35,00%** | 25/40 |

**Validação do procedimento.** A sessão "limpo" reproduz o eval de 2019 amostra
a amostra (depois do ajuste fino do alinhamento, Seção 9.5). Diferenças nas
outras sessões são do caminho do som, não do recorte.

**O controle leva todo áudio a ~1,0**, bonafide inclusive (0,006 → 0,999). Não é
perda de amostra: o controle final teve zero descontinuidades (buffer do WASAPI
de 1 s), e perdas simuladas quase não mexem no score. `comparar_espectro.py`
mostra o que o caminho fez, nos mesmos áudios, após normalizar por pico:

- **7,8–8 kHz: −47,8 dB.** É a ida e volta de taxa (16 kHz do arquivo, 48 kHz
  do dispositivo, 16 kHz do modelo). O filtro antialiasing de qualquer
  conversão para 16 kHz apaga essa faixa: medido em ruído branco, −30 dB em
  7,7–7,9 kHz e −103 dB acima de 7,9 kHz, em qualquer qualidade do soxr.
- **Piso das pausas: +31,5 dB** (−62,5 → −30,9 dB do pico), com o espectro
  subindo mais onde a fala tem pouca energia (0–250 Hz e 4–7,5 kHz, +10 dB) e
  menos onde ela é forte (250 Hz–2 kHz, +2 dB) — assinatura de compressão de
  dinâmica que puxa as partes baixas para cima. Suspeito: os "aprimoramentos
  de áudio" do driver (Realtek, "Efeitos Padrão do Dispositivo").

**Só a faixa do topo já desloca todos os scores para cima** (bonafide 0,306 →
0,984; 0,066 → 0,376), inutiliza o limiar e custa ~7,5 pp de EER. **O modelo
usa a faixa de 7,6–8 kHz.** Isso tem uma consequência estrutural para o
monitor: a captura do sistema é a 48 kHz e a conversão para 16 kHz apaga essa
faixa **sempre**, em qualquer chamada. É o mesmo raciocínio que levou os
organizadores do ASVspoof 2021 LA a limitar o LFCC do baseline a 0–4 kHz
(Seção 2.1): um detector para canal real não pode depender do topo da banda.

**O piso simulado destrói a ordenação, mas empurra os scores para o lado
oposto.** Com ruído branco no nível medido, o EER vai a 42,50% — quase acaso,
perto do controle —, só que os scores **descem** (spoof 0,999 → 0,766;
0,972 → 0,183), enquanto no controle todos **subiram** a ~1,0. Ruído de fundo
é o que uma gravação humana tem; somá-lo aproxima tudo de "bonafide". Conclusões:

- **O modelo é muito sensível ao conteúdo das pausas.** Mexer só no piso leva
  o EER perto do acaso — coerente com a dependência de silêncio documentada
  por Müller et al. (2021) no ASVspoof 2019 LA.
- **O piso do controle não é ruído somado.** O que o subiu preservou a "cor"
  do áudio original: compatível com compressão de dinâmica ou equalização,
  que amplificam o próprio conteúdo baixo — os "aprimoramentos de áudio" do
  driver continuam o suspeito principal.

**Controle sem os aprimoramentos: o caminho real ficou reproduzido em
software.** Com os aprimoramentos de áudio desativados, o espectro do
controle é igual ao do limpo até 7,5 kHz (+0,1 dB em todas as faixas), o piso
das pausas volta ao original (−62,3 → −62,2 dB) e só a faixa do topo some
(7,8–8 kHz: −51,6 dB). Os scores batem, áudio por áudio, com a simulação de
reamostragem (diferenças de ~0,01); o alinhamento recuperou 40/40, todos
ajustados à amostra. Duas conclusões:

- **Os "aprimoramentos de áudio" do driver eram o segundo efeito** — a
  compressão que subiu o piso em 31,5 dB e levou tudo a ~1,0. Desligá-los é
  pré-requisito do monitor; vai para o texto e para o roteiro.
- **O que sobra é estrutural: a conversão de taxa.** Toda captura ao vivo passa
  por 48 kHz e volta a 16 kHz; a faixa de 7,6–8 kHz nunca chega ao modelo, e
  o `baseline_v2` depende dela: todos os scores sobem (bonafide 0,066 → 0,369;
  0,306 → 0,983), o limiar deixa de separar e o EER da amostra vai de 22,50% a
  30–35%.

**Medido com IC** (`robustness_eval.py --amostra 10000 --so clean captura_48k`,
10.002 áudios estratificados, IC 95% ±1,28 pp; a condição `captura_48k` é a
mesma ida e volta 16 → 48 → 16 kHz, com o mesmo `soxr` do monitor):

| modelo | EER limpo | EER captura 48 kHz | custo | bonafide acima do limiar |
|---|---|---|---|---|
| baseline_v2 | 19,02% | **24,70%** | +5,68 pp | **10% → 71%** |
| fusion_v4 | 20,25% | **24,98%** | +4,73 pp | **100%** — os 1.032 em probabilidade 1,0 |

(Bonafide acima do limiar: derivado de precisão e recall no limiar do
checkpoint. v2 limpo: precisão 0,9837, recall 0,7149; captura: 0,9209 e
0,9523. fusion_v4 na captura: recall 1,0000, precisão 0,8968 = a proporção de
spoof da amostra.)

**A ordenação perde ~5 pp; o ponto de operação colapsa.** Os dois modelos
empatam na captura (24,70% contra 24,98%, dentro do IC). O limiar, não: no v2,
7 em cada 10 humanos passam a ser marcados como sintéticos; no fusion_v4
**todo** bonafide satura em probabilidade 1,0 — o EER pela probabilidade dá
50,00% e o modelo vira o classificador trivial "tudo é spoof", o mesmo
comportamento do Opus real do 2021 LA (5.3). O EER de 24,98% só é visível
porque é calculado pelos log-odds (correção da 5.3).

**Consequências:**

- **Para o monitor** (que mostra probabilidade): o fusion_v4 é inutilizável ao
  vivo *com o soxr* (com o FIR deixa de saturar: 10.2.8), e o v2 funciona com o
  limiar deslocado. A recomendação do v2 (5.3)
  ganha uma segunda justificativa, independente do canal da chamada.
- **A captura custa mais que o Opus simulado** (v2: +5,68 pp contra +1,30 pp
  do Opus a 25 kbps na mesma amostra) — e está presente em toda chamada.
- **Mitigação sem retreino — implementada** (`scripts/calibrar_captura.py`):
  o limiar passa a ser o ponto de EER no `dev` (A01–A06) passado pela mesma
  ida e volta, e vai para uma cópia do checkpoint com os mesmos pesos. O eval
  não é tocado na calibração; o efeito se mede nele depois. Para o fusion_v4 o
  script recusa: com todos os humanos em probabilidade 1,0, não existe limiar
  em probabilidade.

**Resultado da recalibração (baseline_v2).** No dev com captura (10.002 áudios,
EER 15,90%), o limiar foi de 0,6539 para **0,9829**: os humanos acima dele
caíram de 81,0% para 15,9%. No **eval** (A07–A19, não vistos na calibração),
derivado de precisão e recall:

| áudio | limiar | humanos marcados como sintéticos | sintéticos que passam |
|---|---|---|---|
| captura | original 0,6539 | 71% | 5% |
| captura | **recalibrado 0,9829** | **17%** | **31%** |
| limpo (16 kHz nativo) | original 0,6539 | 10% | 29% |
| limpo (16 kHz nativo) | recalibrado 0,9829 | 0,1% | 69% |

- **Na captura, o ponto de operação volta a ser usável**: de 7 em cada 10
  humanos acusados para menos de 2 em 10. O EER não muda (24,70%): só o corte
  se move, a ordenação é a mesma.
- **O equilíbrio do dev não transfere inteiro para o eval** (17% contra 31%, e
  não 16% contra 16%): os ataques do eval são inéditos e mais difíceis, e um
  limiar calibrado nos ataques do treino deixa passar mais deles. É a mesma
  limitação de qualquer calibração fora do domínio de teste — registrada, não
  corrigida.
- **Um limiar por caminho do áudio.** Em áudio nativo de 16 kHz com a banda
  inteira, o limiar recalibrado deixa passar 69% dos sintéticos; lá vale o
  original. O que decide é se o áudio passou por uma taxa acima de 16 kHz e
  foi convertido — ver 10.2.2.

### 10.2.2 Qual limiar vale para qual áudio

A faixa de 7,6–8 kHz não some só na captura ao vivo. Some em **qualquer
conversão para 16 kHz**: o `librosa.load(sr=16000)`, que o projeto usa para
abrir arquivos, reamostra com o mesmo tipo de filtro (`soxr_hq`). Então:

| origem do áudio | chega ao modelo com 7,6–8 kHz? | limiar |
|---|---|---|
| arquivo gravado a 16 kHz com a banda inteira (ex.: ASVspoof) | sim | original |
| captura ao vivo (loopback a 48 kHz) | não | recalibrado |
| arquivo gravado a 44,1/48 kHz: celular, microfone, WhatsApp, MP3, vídeo | não | recalibrado |
| arquivo a 8 kHz (telefonia) | não — e nada acima de 4 kHz | nenhum dos dois foi medido |

Ou seja: o limiar original só vale para o formato da base de treino. Quase todo
áudio do mundo real nasce a 44,1 ou 48 kHz e cai no caso do recalibrado.

**Escolha automática — implementada** (`src/limiares.py`). O `monitor.py` e o
`infer.py` aplicam a tabela acima sozinhos: captura ao vivo, arquivo acima de
16 kHz ou de taxa desconhecida → recalibrado; nativo de 16 kHz → original;
abaixo de 16 kHz → original, com aviso de condição não medida. A cópia
recalibrada é achada ao lado do checkpoint original e só é aceita com os mesmos
pesos (impressão digital do `state_dict`); sem ela, o sistema avisa em vez de
inventar um limiar. **Limite da regra:** ela lê a taxa do arquivo, não o
conteúdo — um arquivo de 16 kHz que já foi convertido antes também perdeu o
topo da banda e recebe o limiar original. Isso
reforça a conclusão da 10.2.1: um detector para uso real não deveria depender
do topo da banda (Seção 2.1).

- **Com retreino:** aumentação com a ida e volta, ou LFCC limitado abaixo de
  7,5 kHz — o raciocínio do baseline de 2021 (Seção 2.1). Trabalho futuro.

Depois disso, a sessão "chamada" (Meet/Teams) mede o que a chamada soma ao
caminho de captura.

### 10.2.3 Teste funcional com áudio real (qualitativo)

Monitor ao vivo, `baseline_v2`, limiar escolhido automaticamente (0,9829,
captura), aprimoramentos de áudio desligados, 1 perda de amostra por sessão.
Dois áudios tocados do navegador, sem rótulo por janela:

| áudio | score médio | mediano | janelas acima do limiar | fração acima de 4 kHz (p90, por janela) |
|---|---|---|---|---|
| vídeo narrado por IA (YouTube) | 0,490 | 0,420 | 0 de 19 | 11–94% |
| audiobook narrado por humano | **0,807** | **0,857** | **7 de 32** | **0–5% na maior parte** |
| voz do Google Tradutor (TTS neural) | 0,469 | 0,371 | 0 de 45 | 1–83%, variando |

**A ordem se inverteu:** a voz humana recebeu scores mais altos que a
sintética, e só ela disparou o limiar. A última coluna explica: o audiobook
chega praticamente sem nada acima de 4 kHz — banda estreita, provavelmente por
compressão ou pela gravação — enquanto o vídeo de IA chega em banda larga. Sem
a parte alta do espectro, os scores deste modelo sobem (medido: banda estreita
+16,96 pp, Seção 5; captura +5,68 pp e limiar colapsado, 10.2.1).

**O score está respondendo à largura de banda do áudio, não à síntese.** Dois
exemplos não são medida; o valor deste teste é mostrar, com áudio real, a
dependência que o controle mediu com IC. Soma-se a isso o limite de
generalização conhecido: o modelo viu os ataques de 2019 (A01–A06 no treino,
A07–A19 no eval), e uma voz de IA comercial atual é de outra geração.

**Não é defeito do programa — verificado em três elos.** (1) O caminho
offline reproduz o eval de 2019 amostra a amostra (sessão "limpo", 10.2.1).
(2) O monitor dá o mesmo score do caminho offline para o mesmo áudio
(`tests/test_monitor_consistencia.py`, diferença < 1e-6). (3) O caminho de
captura foi reproduzido em software, áudio a áudio (controle sem
aprimoramentos, 10.2.1) — e nele os ataques de 2019 continuam indo a ~1,0
(A09, A11, A08: 0,998–1,000). O que falha é o modelo diante de vozes que não
são de 2019.

**Conclusão para o texto:** o sistema funciona de ponta a ponta — captura,
janelas, limiar por tipo de áudio, agregação —, mas este modelo não serve para
uso real. Os dois caminhos de correção estão identificados: treino com banda
limitada ou LFCC restrito abaixo de 7,5 kHz (o raciocínio do baseline de 2021,
Seção 2.1) e dados de ataques atuais (ASVspoof 5, Seção 10.7).

### 10.2.4 Teste pela interface web e o efeito do player

Playlists da demonstração (`scripts/montar_demo.py`: 3 bonafide e 3 spoof do
eval de 2019, 4 s de silêncio entre eles), `baseline_v2`, aprimoramentos
desligados, limiar escolhido automaticamente:

| playlist | envio de arquivo | ao vivo, Reprodutor do Windows | ao vivo, VLC |
|---|---|---|---|
| reais (bonafide) | 0,03 — sem indício | **0,33 — sem indício** (0 janelas acima) | 0,98 — 7 de 9 janelas acima |
| falsos (spoof) | 1,00 — indício | **1,00 — indício** (8 de 8) | 1,00 — indício |

Pelo Reprodutor do Windows o sistema separa os dois grupos ao vivo. Pelo VLC,
**o mesmo arquivo de vozes reais** vai de 0,33 a 0,98: o player altera o áudio
(reamostragem ou filtros próprios) o bastante para o modelo. Mais um exemplo da
dependência do caminho do som — e uma instrução prática: na demonstração, tocar
pelo Reprodutor do Windows ou pelo `canal_real.py tocar`.

### 10.2.5 A perda na captura era do conversor de taxa — corrigida sem retreino

A captura, o `tocar` e a leitura de arquivos acima de 16 kHz passaram a usar um
FIR de 2047 coeficientes com corte em 7,9 kHz (`src/preprocess/reamostragem.py`)
no lugar do `soxr`. Em ruído branco, na ida e volta 16 → 48 → 16 kHz,
7,6–7,8 kHz vai de −8,0 dB para 0,0 dB e 7,8–7,95 kHz de −32,7 dB para −2,8 dB.
Custo ao vivo: ~2 ms por bloco de 100 ms.

Medido no eval (`robustness_eval.py --amostra 10000`, mesmos 10.002 áudios,
IC ±1,28 pp), `baseline_v2`:

| condição | EER | precisão | recall | humanos acima do limiar original |
|---|---|---|---|---|
| limpo | 19,02% | 0,9837 | 0,7149 | 10% |
| captura com soxr | 24,70% | 0,9209 | 0,9523 | 71% |
| **captura com FIR** | **19,41%** | 0,9810 | 0,7271 | **12%** |

**A captura deixou de custar**: +0,39 pp, dentro do IC, contra +5,68 pp antes.
O ponto de operação volta junto: recalibrado no dev passado pelo FIR, o limiar
fica em **0,686** (EER no dev 10,33%), contra 0,654 do original e 0,983 do
caminho com soxr. Ou seja: a degradação que a Seção 10.2.1 atribuía "à captura"
era quase toda do filtro do conversor, que apagava a faixa de 7,6–8 kHz.

O que continua: o filtro não recupera o que outro programa já cortou antes da
captura — o player (VLC, 10.2.4), o codec da chamada (Opus em banda larga corta
perto de 8 kHz) ou um arquivo convertido antes. Para esses casos, a correção
continua sendo treinar sem depender do topo da banda (Seção 2.1).

### 10.2.6 Demonstração com a própria voz: controle pareado (resultado negativo)

5 frases gravadas pelo autor (Audacity, WASAPI, mono, 48 kHz, português) e
refeitas por *copy-synthesis* com Griffin-Lim (o vocoder do ataque A11) e WORLD
(A02/A03/A05/A07) — `gerar_sintetico.py`. Original e falso têm o mesmo locutor,
microfone, sala e texto; só o vocoder muda. Scores (`avaliar_arquivos.py`,
`baseline_v2`, limiar 0,686):

| frase | original (humano) | Griffin-Lim | WORLD |
|---|---|---|---|
| 1 | **0,984** | 0,564 | 0,866 |
| 2 | **0,980** | 0,648 | 0,962 |
| 3 | **0,989** | 0,850 | 0,936 |
| 4 | **0,992** | 0,848 | 0,976 |
| 5 | **0,811** | 0,448 | 0,585 |

**Versões sintéticas acima do original da mesma frase: 0 de 10.** O modelo
marcou a voz humana como sintética em todas as frases, e as versões refeitas
pelo vocoder ficaram *abaixo* dela. Com o canal controlado, a conclusão fica
isolada: a decisão responde às condições de gravação (microfone, sala, língua,
processamento do driver), não aos artefatos do vocoder — e o vocoder, ao
reconstruir a voz, suaviza justamente o que levava o score para cima. Mesmo
padrão do audiobook (10.2.3).

**Regravação sem os aprimoramentos do microfone** (2 frases em inglês, 3 em
português):

| frase | original | Griffin-Lim | WORLD |
|---|---|---|---|
| 1 (inglês) | 0,901 | 0,446 | 0,443 |
| 2 (inglês) | 0,472 | 0,264 | 0,413 |
| 1 (português) | 0,898 | 0,720 | 0,718 |
| 2 (português) | 0,593 | 0,315 | 0,453 |
| 3 (português) | 0,688 | 0,707 | 0,898 |

- **Os aprimoramentos do microfone empurravam a voz humana para cima:** média
  dos originais de 0,95 (5 de 5 acima do limiar) para 0,71 (3 de 5). É o mesmo
  efeito dos aprimoramentos da saída (10.2.1).
- **Língua: sem efeito claro** (inglês 0,69, português 0,73; 2 e 3 frases).
- **As versões do vocoder seguem abaixo do original: 18 de 20** nas duas
  rodadas. Isso não é do driver. O modelo acerta o ataque A11 (Tacotron2 +
  Griffin-Lim) no eval, mas não o Griffin-Lim aplicado a outra voz: o que ele
  aprendeu do A11 não é o artefato genérico do vocoder, e sim marcas do sistema
  e da base do ASVspoof (modelo acústico, locutores, condições de gravação).

**Consequência:** para a demonstração, as playlists do ASVspoof (10.2.4)
funcionam ao vivo; a voz própria entra como evidência da limitação de
generalização, com controle pareado. A correção é de dados: treino com
gravações em condições variadas (microfones comuns, salas, outras línguas).

### 10.2.7 Reprodução pelo ar: alto-falante do celular → microfone (qualitativo)

As playlists da demonstração (10.2.4: 3 bonafide e 3 spoof, A09/A11/A19)
tocadas pelo alto-falante de um celular e captadas pelo microfone do
computador (aba Ao vivo, fonte "microfone", `baseline_v2`, limiar 0,69),
30/09/2026:

| rodada | reais (bonafide) | falsos (spoof) | janelas acima do limiar (falsos) |
|---|---|---|---|
| equalização do celular ligada, 1 | 0,08 | 0,35 | 4 de 18 |
| equalização do celular ligada, 2 | 0,07 | 0,30 | 2 de 14 |
| equalização desligada | **0,19** | **0,15** | 0 |

Referência, mesmos arquivos pelo som do computador: reais 0,33, falsos 1,00.

- **Pelo ar, nada passa do limiar:** os falsos caem de 1,00 para 0,15–0,35. É o
  cenário *Physical Access* da ASVspoof (repetição por alto-falante), e o
  modelo foi treinado só em *Logical Access*.
- **A separação não se sustenta.** Com a equalização (reforço de graves) do
  celular, os falsos ficaram acima dos reais nas duas rodadas; sem ela, a ordem
  se inverteu (0,15 contra 0,19). Uma configuração do alto-falante basta para
  mudar o resultado.
- **Coerente com a sensibilidade às pausas (10.2.1):** ruído somado às pausas
  derrubou os scores dos spoofs na simulação (0,999 → 0,766; 0,972 → 0,183). A
  gravação pelo ar enche as pausas de ruído e reverberação da sala, o que o
  modelo associa a voz humana; o alto-falante pequeno ainda altera graves e
  agudos.
- **Não calibrar um limiar com isso.** Três pares não sustentam um limiar;
  calibrar para o ar exigiria tocar uma amostra rotulada pelo mesmo caminho e
  medir o EER, como na 10.2.1.

**Para a apresentação:** a demonstração principal vai pelo som do computador
(10.2.4); o microfone serve para a voz ao vivo e para mostrar a limitação.

### 10.2.8 fusion_v4 com o FIR: a saturação era do conversor

`robustness_eval.py --amostra 10000 --so clean captura_48k_fir`, mesmos 10.002
áudios do eval, limiar do checkpoint 0,7173, 02/10/2026:

| condição | EER | precisão | recall | humanos em probabilidade 1,0 |
|---|---|---|---|---|
| limpo | 20,25% | 0,9996 | 0,5547 | 0 de 1.032 |
| captura com soxr (10.2.1) | 24,98% | 0,8968 | 1,0000 | **1.032 de 1.032** |
| **captura com FIR** | **20,16%** | 0,9992 | 0,5621 | **0 de 1.032** |

- **A captura deixou de custar** (−0,09 pp, dentro do IC) e **a saturação
  sumiu**. Ela vinha do soxr apagar 7,6–8 kHz, não do modelo. O fusion_v4 passa
  a ser usável ao vivo, e com isso a fusão dos dois modelos também pode ser
  testada ao vivo (antes, a média das probabilidades virava a do v2 sozinho).
- **Ponto de operação conservador:** no limiar 0,7173, derivado de precisão e
  recall, ~0,4% dos humanos ficam acima (cerca de 4 de 1.032) e ~44% dos
  sintéticos passam. É o perfil já visto no eval completo (recall 0,55).
- **O Opus real continua saturando** (97% dos humanos em 1,0, Seção 5.3): lá a
  causa é o codec e a rede, não o conversor.

**Recalibração (`calibrar_captura.py`):** no dev com captura FIR, EER 0,13%
(log-odds e probabilidade), nenhum humano em 1,0; limiar 0,7173 → **0,9984**.
O dev só tem os ataques do treino (A01–A06), e com as classes quase separadas
o ponto de EER fica no extremo: é o mesmo 0,03% de validação que não previu o
eval (Seção 5.2). Um limiar mais alto deixa passar mais sintéticos inéditos.

Medido no eval (mesma amostra, captura FIR), derivado de precisão e recall:

| limiar | humanos acima | sintéticos que passam | recall | precisão |
|---|---|---|---|---|
| original 0,7173 | ~0,4% (cerca de 4 de 1.032) | 44% | 0,5621 | 0,9992 |
| recalibrado 0,9984 | 0% | **50%** | 0,5017 | 1,0000 |

Confirmado: o limiar do dev troca ~4 alarmes falsos por ~540 sintéticos a mais
passando. No fusion_v4 a recalibração pelo EER do dev não serve, porque o dev
quase separável não informa sobre ataques inéditos. Decisão: usar o limiar
original nele (renomear a cópia `_captura` para o sistema não a achar). O
limiar do caminho da demonstração sai do Incremento 4, medido com vozes reais.

### 10.2.9 Fusão ao vivo na captura FIR: o EER melhora, o limiar não acompanha

`robustness_eval.py --amostra 10000 --so captura_48k_fir` nos dois modelos, no
eval e no dev (mesmos 10.002 áudios em cada partição), e `fusao_ao_vivo.py`
sobre os scores salvos. baseline_v2 com a cópia `_captura` (limiar 0,686, o do
ao vivo); fusion_v4 com o limiar original 0,7173 (10.2.8). 02/10/2026.

| | EER | limiar | humanos acima | sintéticos que passam |
|---|---|---|---|---|
| baseline_v2 | 19,41% | 0,686 | 10,7% | 28,8% |
| fusion_v4 | 20,16% | 0,7173 | 0,4% | 43,8% |
| fusão pela média (a do monitor) | **14,2%** | | | |
| fusão por postos (referência) | 13,3% | | | |

- **O ganho da fusão volta com o FIR:** −5,2 pp sobre o melhor isolado. É o
  padrão da Seção 4 no eval limpo (média 14,03%, postos 13,13%; aqui também
  0,9 pp entre as duas regras). No Opus real a média empatava com o v2
  (29,41% contra 29,43%, 5.3) porque o fusion_v4 saturava; sem saturação, a
  regra do monitor recupera a diversidade entre os modelos.
- **Critério combinado antes (o ponto de EER da fusão no dev) falhou.** EER da
  fusão no dev 0,8%, limiar 0,5363. No eval: 0,4% dos humanos acima e 42,2%
  dos sintéticos passando. Os outros dois limiares testados dão o mesmo
  formato: 0,686 (o do v2) 0,3% e 48,5%; 0,7016 (média dos dois) 0,3% e
  49,1%. Nos três, passam mais sintéticos que com o v2 sozinho (28,8%).

**Segundo critério, combinado em 02/10/2026 antes de calcular o número no
eval.** Do dev para o eval se mantém a fração de humanos acima do limiar, não a
de sintéticos que passam (os ataques do dev são os do treino). O limiar da
fusão passa a ser escolhido no dev com **o mesmo alarme falso que o v2 tem no
dev no limiar 0,686**. Regra de decisão: a fusão volta a ser a recomendação do
monitor se, nesse limiar, deixar passar **pelo menos 2 pp menos sintéticos que
o v2** no eval (28,8%, ou seja, 26,8% ou menos); senão fica o v2 sozinho (5.3)
e a fusão entra só como resultado de EER. Este critério veio depois de ver o
primeiro falhar e é declarado assim no texto.

**Resultado** (`fusao_ao_vivo.py` com `--dev`, saída em
`outputs/fusao_ao_vivo_captura.json`):

| limiar da fusão | humanos acima, dev → eval | sintéticos que passam (eval) |
|---|---|---|
| 0,686 (o do v2) | 0,3% → 0,3% | 48,5% |
| 0,7016 (média dos dois limiares) | 0,2% → 0,3% | 49,1% |
| 0,5363 (ponto de EER da fusão no dev) | 0,7% → 0,4% | 42,2% |
| **0,3473 (alarme falso do v2 no dev, 10,3%)** | **10,3% → 10,6%** | **16,5%** |
| v2 sozinho, 0,686 | 10,3% → 10,7% | 28,8% |

- **Regra cumprida.** Com o mesmo alarme falso do v2 (10,6% contra 10,7% no
  eval), a fusão deixa passar 16,5% dos sintéticos contra 28,8%: **−12,4 pp**,
  bem além dos 2 pp combinados. Na captura com FIR a fusão volta a ser a
  recomendação. O alarme falso no eval tem IC 95% de cerca de ±1,9 pp (1.032
  humanos).
- **O lado dos humanos se mantém do dev para o eval; o dos sintéticos, não.**
  É o que torna o critério do alarme falso transferível e o do EER não.
- **Por que os limiares acima de 0,5 falham.** Acima de 0,5 a média só marca
  sintético quando os dois modelos concordam: um score abaixo de 2·limiar − 1
  basta para o sintético passar. Em 0,686, 47,8% dos sintéticos passam
  qualquer que seja o outro modelo. Só um modelo pega 38,3% dos sintéticos
  (26,6% só o v2, 11,7% só o fusion_v4), justamente a diversidade que dá o
  ganho no EER; em 0,686 passam 31,4 pp deles, em 0,3473 passam 0,2%.
- **Por que o dev escolheu 0,5363.** No dev o fusion_v4 sozinho tem recall
  1,0000 (10.2.8): não existe ali o caso "o v2 pega, o fusion_v4 não", e o
  ponto de EER da fusão fica decidido só pela cauda dos humanos.
- **Validade.** Captura FIR (16 → 48 → 16 kHz) na amostra de 10.002 áudios. Não
  vale para o Opus real, onde o fusion_v4 satura (5.3), nem para arquivos
  nativos de 16 kHz, onde o limiar da fusão não foi medido.
- **O monitor ainda não usa esse limiar.** Com dois `--checkpoint` ele aplica o
  limiar do primeiro (0,686 ou 0,7173), o ponto de 48 a 49% de sintéticos
  passando; a interface web roda um modelo só. Até ligar o limiar da fusão, a
  recomendação prática do monitor continua o v2 sozinho.

### 10.2.10 Entrada Realtek do notebook: exclusivo x compartilhado (teste de equipamento)

`scripts/testar_microfone.py`, "Grupo de microfones (Realtek(R) Audio)", a mesma
frase gravada em WASAPI exclusivo e compartilhado, 48 kHz, 02/10/2026.

**O microfone medido é o do fone de ouvido, não o embutido.** As duas rodadas
foram feitas com um fone com microfone conectado, e com ele o driver Realtek
troca a fonte dessa entrada para o microfone do fone sem mudar o nome nem
mostrar outro dispositivo (informado pelo usuário em 14/10/2026). O microfone
embutido do notebook, junto da webcam, que é o da apresentação, continua sem
teste.

| rodada | | exclusivo | compartilhado |
|---|---|---|---|
| 1 (sem ruído de fundo, 16 bits) | pico | −40,4 dBFS | −21,7 dBFS |
| | fala (p95 do RMS) | −53,2 dBFS | −36,1 dBFS |
| 2 (ruído branco de fundo, 24 bits) | pico | −37,4 dBFS | −18,4 dBFS |
| | fala (p95 do RMS) | −48,6 dBFS | −32,5 dBFS |
| | ruído no começo / no fim | −59,3 / −65,1 dBFS | −48,7 / −49,3 dBFS |

- **O compartilhado aplica ganho fixo de +16 a +17 dB** nas duas rodadas (fala
  e ruído sobem juntos): volume ou reforço do microfone no Windows, que o
  exclusivo não aplica. É da entrada, não do microfone, e deve se repetir no
  embutido; falta medir.
- **Nenhum indício de supressão de ruído na rodada 2**, nos dois modos: o
  espectro do ruído não muda de formato do começo para o fim nem de um modo
  para o outro (critério: uma faixa até 8 kHz 10 dB fora do ganho geral). A
  queda de 5,8 dB do ruído no exclusivo fica abaixo do limite de 6 dB, e o
  começo tem um evento a −54 dBFS entre 0,5 e 1 s que puxa a medida.
- **Retirado:** o "−17 dB em 250–500 Hz" da primeira análise veio da rodada 1,
  feita sem o ruído de fundo (confirmado pelo usuário); ali o "ruído" era o do
  próprio microfone e da sala, e o resultado não se repetiu com ruído branco.
- **Consequência:** a coleta usa o exclusivo, como previsto no plano. O nível
  do exclusivo é baixo (pico de −37 a −40 dBFS, no microfone do fone); o modelo
  normaliza pelo pico, e a gravação vai em 24 bits.
- **Risco para a coleta e a banca:** um fone com microfone conectado ao
  notebook troca o microfone em silêncio, e o monitor continua mostrando
  "Grupo de microfones". Nenhum fone com microfone ligado ao notebook durante
  a gravação e a demonstração.
- Na rodada 2 as checagens de fala e de piso reprovam por construção: o ruído
  branco ficou a 20 dB da fala.

Dispositivos de entrada no WASAPI, consultados no driver (sem gravar):

| dispositivo | taxas aceitas a 1 canal |
|---|---|
| Grupo de microfones (Realtek(R) Audio) — com o fone conectado, o microfone do fone | 48 kHz, exclusivo e compartilhado |
| Interface de áudio digital (2- USB Digital Audio) | só 96 kHz, exclusivo e compartilhado |
| Microfone (DroidCam Audio) | 44,1 kHz nos dois modos; 48 kHz só no exclusivo |

A fonte de microfone passou a aceitar 96 kHz (razão 6 para 16 kHz, o mesmo
FIR: a 96 kHz os 2047 coeficientes dão −0,1 dB em 7,8 kHz e −103 dB em
8,1 kHz).

### 10.2.11 Oclusão da banda alta: quanto os modelos dependem de 7,6–8 kHz

Item do plano do Incremento 4 ("oclusão da banda alta, só com avaliação"). O
plano tira a dependência de 7,6–8 kHz do treino (LFCC de 0 a 4 kHz ou descarte
aleatório da banda alta); este teste diz se vale a pena e quanto. Indício
anterior: com o soxr, que atenua 7,6–7,8 kHz e quase apaga o resto até 8 kHz,
a captura custou +5,68 pp no baseline_v2 (19,02% → 24,70%, 10.2.1) e +4,73 pp
no fusion_v4 (20,25% → 24,98%, 10.2.8). O soxr, porém, mexe também na fase e
no restante da banda; a oclusão só zera o espectro acima do corte.

Condições novas do `robustness_eval.py`: `oclusao_7600`, `oclusao_7000`,
`oclusao_6000` e `oclusao_4000` (zera tudo a partir do corte, fase zero), na
amostra de 10.002 áudios do eval 2019, com os checkpoints originais.

**Critério, definido em 22/10/2026, antes da medição.** Se tirar 7,6–8 kHz
(`oclusao_7600`) subir o EER em **2 pp ou mais** em algum dos dois modelos
(acima do IC 95% de cerca de ±1,3 pp de cada medida), a dependência pesa: os
modelos do Incremento 4 recebem a correção, e um treino curto compara as duas
opções do plano. Abaixo de 2 pp nos dois, a correção não entra e isso é
registrado. Os cortes em 7, 6 e 4 kHz só descrevem até onde a dependência vai;
o de 4 kHz não mede o custo de um LFCC de 0 a 4 kHz, que exigiria treinar.

**Resultado** (22/10/2026, 10.002 áudios do eval, EER pelos log-odds):

| corte (zera acima de) | baseline_v2 | fusion_v4 |
|---|---|---|
| nenhum (`clean`) | 19,02% | 20,25% |
| **7,6 kHz** | **25,29% (+6,27 pp)** | **31,12% (+10,87 pp)** |
| 7 kHz | 34,20% (+15,18 pp) | 27,73% (+7,48 pp) |
| 6 kHz | 39,33% (+20,31 pp) | 24,37% (+4,12 pp) |
| 4 kHz | 38,76% (+19,74 pp) | 25,38% (+5,13 pp) |

- **Critério atingido nos dois modelos** (+6,27 e +10,87 pp, contra 2 pp): a
  correção entra no treino do Incremento 4, e um treino curto compara as duas
  opções do plano.
- **No fusion_v4, tirar só 7,6–8 kHz satura os humanos:** 1.032 de 1.032 em
  probabilidade 1,0 (EER pela probabilidade 50,00%). A saturação da captura com
  soxr (10.2.8) vinha disso, a faixa sumir, e não do conversor em si.
- **A curva do fusion_v4 não é monótona:** cortar em 6 ou 4 kHz custa menos
  que cortar em 7,6 kHz. O que ele aprendeu parece ser a borda do espectro
  perto de 7,6 kHz, não a falta de banda alta em geral. O baseline_v2 piora sem
  parar até 6 kHz.
- **O limiar colapsa:** baseline_v2 em 7 kHz e fusion_v4 em 7,6 e 7 kHz ficam
  com recall 1,0 e precisão 0,8968, o classificador trivial (8.1).
- **Hipótese, não medida:** no Opus real, 97% dos humanos também foram a 1,0
  (5.3); se aquele canal corta o topo da banda, o mecanismo pode ser o mesmo.

**Treino curto das duas correções** (desenho e critério definidos em
22/10/2026, antes de treinar). Três treinos do baseline_v2 na 2019 LA, iguais
exceto pela correção (`tests/test_configs_banda.py` confere), 10 épocas e a
mesma semente:

- `banda_controle`: sem correção;
- `banda_lfcc4k`: LFCC de 0 a 4 kHz (`features.lfcc.f_max: 4000`);
- `banda_descarte`: metade das amostras de treino perde o espectro acima de um
  corte sorteado entre 4 e 7,9 kHz, no áudio bruto (`audio.degradacao`).

Cada um avaliado nos mesmos 10.002 áudios em `clean`, `oclusao_7600` e
`oclusao_4000`. Uma correção funciona se o EER em `oclusao_7600` ficar pelo
menos 2 pp abaixo do controle e o EER `clean` no máximo 2 pp acima dele. Se as
duas funcionarem, fica a de menor EER em `oclusao_7600`; se a diferença entre
elas for menor que 2 pp, a de menor EER `clean`. Se nenhuma funcionar, o
resultado é registrado e a escolha volta para o grupo. No fusion_v4, a opção
do LFCC também precisaria limitar o ramo de espectrograma, que este teste não
cobre.

**10 épocas, e quando refazer com o treino completo** (combinado em
22/10/2026, antes de treinar). O baseline_v2 original rodou 34 épocas e teve a
melhor na 27 (EER do dev 9,35%, contra 12,17% na época 10): com 10 épocas os
modelos ainda estão aprendendo, e o controle não deve reproduzir os 19,02% do
original; ele é a régua só das outras duas versões de 10 épocas (cerca de 30 a
40 min cada na GTX 1650). O resultado de 10 épocas vale se a decisão pelo
critério sair com folga: só uma correção funciona, ou as duas funcionam e a
vencedora fica pelo menos 2 pp à frente em `oclusao_7600`. Se nenhuma
funcionar (o descarte pode só aprender mais devagar) ou se as duas ficarem a
menos de 2 pp uma da outra, os três treinos são refeitos completos (até 50
épocas com parada antecipada, como o original), em configs com outro nome, e
o mesmo critério decide.

**Resultado com 10 épocas** (22/10/2026, mesmos 10.002 áudios; entre
parênteses, a diferença para o controle):

| treino | melhor época (dev) | limpo | corte em 7,6 kHz | corte em 4 kHz |
|---|---|---|---|---|
| `banda_controle` | 7 (12,52%) | 24,43% | 29,18% | 40,51% |
| `banda_lfcc4k` | 10 (14,37%) | 25,19% (+0,76) | **25,22% (−3,96)** | 25,30% |
| `banda_descarte` | 10 (14,25%) | 25,11% (+0,68) | **26,38% (−2,80)** | 33,89% |

- **As duas correções funcionam:** no corte em 7,6 kHz caem mais de 2 pp em
  relação ao controle, e no limpo nenhuma sobe mais de 2 pp.
- **Mas empatam:** o LFCC de 0 a 4 kHz fica 1,16 pp à frente no corte, menos
  que os 2 pp de folga. Pela regra combinada antes, os três treinos são refeitos
  completos (`banda_*_completo.yaml`, até 50 épocas).
- **O LFCC de 0 a 4 kHz não depende de nada acima de 4 kHz** (25,19%, 25,22% e
  25,30% nas três condições), o que vale também para telefonia e codecs de banda
  estreita. O descarte reduz a dependência de 7,6–8 kHz (o corte passa a custar
  1,27 pp, contra 4,75 pp no controle), mas o corte em 4 kHz ainda custa 8,78 pp.
- **Os três estão subtreinados:** o controle fica em 24,43% no limpo, contra
  19,02% do baseline_v2 completo, e as duas correções ainda melhoravam no dev
  na época 10.

**Resultado com treino completo** (23/10/2026, até 50 épocas com parada
antecipada, mesmos 10.002 áudios; entre parênteses, a diferença para o
controle):

| treino | épocas (melhor, dev) | tempo | limpo | corte em 7,6 kHz | corte em 4 kHz |
|---|---|---|---|---|---|
| `banda_controle_completo` | 49 (42, 10,08%) | 1h28m | 18,14% | 24,33% | 38,55% |
| `banda_lfcc4k_completo` | 50 (50, 8,78%) | 1h29m | 23,34% (+5,20) | 23,34% (−0,99) | 23,46% (−15,09) |
| `banda_descarte_completo` | 50 (48, 9,74%) | 2h57m | 20,54% (+2,40) | 21,34% (−2,99) | 28,50% (−10,05) |

- **O controle reproduz o baseline_v2 original** dentro do IC (18,14% contra
  19,02% no limpo, 24,33% contra 25,29% no corte, 38,55% contra 38,76% em
  4 kHz): o teste vale.
- **Pelo critério, nenhuma correção passou.** O LFCC de 0 a 4 kHz falha nas
  duas condições: custa 5,20 pp no limpo e só tira 0,99 pp no corte (a faixa de
  4 a 8 kHz carrega informação que o modelo completo usa). O descarte tira
  2,99 pp no corte, mas custa 2,40 pp no limpo, 0,40 pp acima da tolerância de
  2 pp, dentro do IC de cada medida. Pela regra combinada antes, o resultado é
  registrado assim e a escolha volta para o grupo (plano, decisões pendentes).
- **A validação de novo não prevê a avaliação:** o LFCC de 0 a 4 kHz tem o
  melhor dev (8,78%) e o pior eval limpo (23,34%); ele chegou às 50 épocas
  ainda melhorando no dev.
- **O descarte dobra o tempo de treino:** 2h57m contra 1h28m, com o
  carregamento de dados passando de 49 s para 169 s por época.
- **O motivo original da correção mudou em parte:** o plano dizia que a faixa
  some em qualquer áudio gravado a 44,1 ou 48 kHz, mas isso era do soxr; com o
  FIR a captura a preserva (10.2.5). Ela ainda some nos codecs e na telefonia
  (eval da ASVspoof 5, chamadas reais).

**Decisão do grupo** (23/10/2026): adotar o descarte aleatório nos modelos do
Incremento 4, sabendo que ele ficou 0,40 pp acima da tolerância no limpo. O
texto registra isso. Depois do treino, o descarte da 2019 tinha o custo da FFT
em comprimento ruim (96.001 amostras: 42 ms contra 3 ms); a oclusão passou a
completar com zeros até um comprimento rápido, o que vaza ~1e-6 da energia
acima do corte e não muda as janelas de 4 s da avaliação.

### 10.2.12 Teste com a voz própria: real contra clone (critério combinado antes)

Ensaio da meta do plano com um locutor fora do treino (um integrante do grupo),
antes da coleta com o orientador. Preparado e registrado em 24/10/2026, antes de
qualquer gravação; os números entram aqui depois, como saírem.

- **Modelos:** fusão pela média dos log-odds de `fusion_lcnn_v4_2019_degr` e
  `fusion_lcnn_v4_asv5` (10.9), no caminho da interface web: janelas de 4 s com
  passo de 2 s, média ponderada pela fração de fala, também nos log-odds. A
  regra registrada primeiro era a média das probabilidades; mudou no mesmo dia,
  antes de qualquer gravação, pela calibração abaixo.
- **Limiar:** escolhido antes, só com os humanos dos devs, na captura FIR
  (`captura_48k_fir`): 10% de alarme falso nos humanos da amostra de 10 mil do
  dev da 2019 e do dev do recorte da ASVspoof 5, juntos
  (`scripts/limiar_fusao.py`). Nenhum áudio do teste entra na escolha. Score
  igual ou acima do limiar é sintético.
- **Material:** 10 frases (`docs/etica/frases_teste_voz_propria.txt`) lidas no
  microfone embutido do notebook, sem fone conectado, em WASAPI exclusivo, na
  taxa do dispositivo e em 24 bits, com os aprimoramentos do Windows desligados
  (`scripts/gravar_frases.py`). O clone sai de uma gravação de cerca de 1 minuto
  de outro texto (`docs/etica/texto_referencia_clonagem.txt`), para a
  ferramenta não ver as frases do teste. Uma ferramenta de clonagem; quais
  entram no treino segue pendente no plano.
- **Clonagem (24/10/2026):** XTTS v2 (Coqui), no próprio computador, na GTX 1650
  (`scripts/clonar_xtts.py`, ambiente conda separado, semente fixa por frase),
  escolhido porque a demonstração ao vivo não pode depender de internet nem
  mandar a voz para fora. Referência: o trecho com fala da gravação de 90 s (1,8
  a 73 s, 71,2 s), em 16 bits e com ganho até o pico de −3 dBFS (o original tinha
  13 MB e pico de −26 dBFS). Cada frase levou de 8,7 a 13,4 s para gerar, 1,04 a
  1,28 vez a duração da fala; o modelo levou 133,5 s para carregar na primeira
  vez, com o download. Os clones saem a 24 kHz, com o pico normalizado, e
  reproduzem o ruído da referência: 18 a 24 dB abaixo da fala, contra 21 a 25 dB
  nas gravações reais.
- **Caminhos:** arquivo, as gravações reais contra os clones como a ferramenta
  os entregou; microfone, as mesmas gravações reais contra os clones tocados
  pelo alto-falante do notebook e gravados pelo mesmo microfone
  (`gravar_frases.py pelo-ar`). Placar pelo `scripts/teste_voz_propria.py`.
- **Critério (a meta do plano):** em cada caminho, pelo menos 8 de 10 reais
  julgados humanos e 8 de 10 clones julgados sintéticos. Arquivo sem janela
  útil conta como erro. Regravar só durante a gravação, pelas checagens
  (saturação, fala, banda, piso de ruído), sem ver score; depois disso toda
  gravação conta.
- **Ruído do microfone embutido (decidido em 24/10/2026, antes de gravar):** em
  dois ensaios de nível, com as 10 frases lidas na distância normal de fala, o
  ruído ficou 16 a 20 dB abaixo da fala com o carregador ligado e 17 a 21 dB sem
  ele. Sem o carregador sumiu o zumbido de 240 Hz, mas não o ronco grave (85% da
  energia do ruído abaixo de 500 Hz); um passa-altas de 150 Hz só levaria a 22 a
  24 dB. Chegar aos 35 dB da checagem exigiria falar a uns 10 cm do microfone,
  diferente da banca. Regra do teste: piso de ruído de pelo menos 15 dB
  (`--piso-minimo 15`); no caminho microfone, o real e o clone passam pelo
  mesmo ruído. A checagem de fala passou a contar só o que fica pelo menos 10 dB
  acima do ruído (com o corte de 30 dB do pico, a gravação inteira contava como
  fala), e a gravação passou de 10 para 12 s: as frases levaram cerca de 7,5 s e
  terminavam coladas no fim. Na gravação da referência, a tecla e a mão no
  notebook chegaram ao microfone como pancadas graves (~150 ms, metade da
  energia abaixo de 150 Hz) e a checagem acusava fala nas bordas sem ninguém
  falar; a detecção de fala passou a usar um passa-altas de 150 Hz e a ignorar
  trechos de menos de 0,25 s. Os ensaios de nível e a referência passaram só
  pelas checagens, sem o modelo.
- **Só descritivo:** o score de cada modelo, a comparação por frase (clone acima
  do real da mesma frase) e as checagens.
- **Leitura combinada antes:** o caminho arquivo sozinho pode ser acertado por
  um detector de microfone (o real passa pelo microfone, o clone não); o
  caminho microfone põe o mesmo canal nas duas classes e é o que vale para a
  demonstração. Se a meta falhar, o resultado entra como saiu, com a limitação
  em 5.7.
- **Limitações:** um locutor, que não é o orientador; uma ferramenta; 10
  frases; uma sala. A gravação e o clone ficam em `data/vozes/teste_proprio/`
  (fora do Git), com termo assinado, e o clone é apagado depois da banca.

**Calibração nos devs** (24/10/2026; 1.026 humanos no dev da 2019 e 2.223 no da
ASVspoof 5, 3.249 juntos). Os dois devs escolheram os checkpoints (parada
antecipada), então os números deles são otimistas; o lado dos sintéticos é só
referência, porque o limiar sai dos humanos. No dev da 2019 os ataques são os do
treino; no da ASVspoof 5 (A09 a A16), nenhum dos dois modelos os viu.

| regra | limiar | dev | EER da fusão | humanos acima | sintéticos passam |
|---|---|---|---|---|---|
| média das probabilidades | 0,4351 | 2019 | 2,32% | 6,3% | 1,0% |
| | | ASVspoof 5 | 21,48% | 11,7% | 33,0% |
| média dos log-odds | −4,437 (score 0,0117) | 2019 | 4,67% | 7,8% | 3,1% |
| | | ASVspoof 5 | 11,87% | 11,0% | 12,4% |

Cada modelo sozinho: `fusion_lcnn_v4_2019_degr` 2,09% no dev da 2019 e 37,80% no
da ASVspoof 5; `fusion_lcnn_v4_asv5` 40,35% e 20,59%.

- **Na média das probabilidades a fusão não ganha do melhor modelo em nenhum
  dev,** porque o modelo do recorte dá probabilidades minúsculas: o limiar dele
  é 0,00017, e os humanos ficam perto de 0. Ele só pesa quando tem quase
  certeza; no dev da ASVspoof 5, 258 dos 259 humanos acima do limiar foram
  marcados só pelo modelo da 2019.
- **Nos log-odds ele pesa,** e a fusão fica bem abaixo do melhor modelo no dev
  da ASVspoof 5 (11,87% contra 20,59%), a 2,58 pp dele no da 2019. Com o mesmo
  alvo de humanos, os sintéticos que passam no dev da ASVspoof 5 caem de 33,0%
  para 12,4%. Decidido em 24/10/2026: a média dos log-odds, no teste e no
  monitor (`--limiar-fusao`).
- **Na tela,** o limiar de 0,0117 deixava a decisão no primeiro 1% da barra. O
  monitor e a interface passaram a mostrar sigmoide(logit(score) −
  logit(limiar)), com o limiar sempre em 0,50; a ordem e a decisão não mudam.

**Resultado (24/10/2026; `outputs/teste_voz_propria.json`).** Limiar −4,437 nos
log-odds (score 0,0117). Nenhum clone foi gerado de novo.

| caminho | reais julgados humanos | clones julgados sintéticos | meta |
|---|---|---|---|
| arquivo (real × clone do XTTS) | 10 de 10 | **0 de 10** | não atingida |
| microfone (real × clone tocado pelo alto-falante) | 10 de 10 | **2 de 10** | não atingida |

**A meta não foi atingida em nenhum caminho.** O lado humano passou nos dois; os
clones passaram quase todos como voz humana.

Descritivo (não é o critério), em log-odds de cada modelo; "pares" é a fração dos
100 pares (clone, real) em que o clone fica acima:

| modelo | real | clone | clone pelo ar | pares clone | pares pelo ar |
|---|---|---|---|---|---|
| `fusion_lcnn_v4_2019_degr` | +1,19 a +6,54 | −7,65 a −1,22 | +1,59 a +5,85 | 0% | 65% |
| `fusion_lcnn_v4_asv5` | −18,41 a −11,66 | −10,30 a −5,14 | −17,82 a −11,25 | 100% | 66% |
| fusão (média) | −8,61 a −4,88 | −6,99 a −5,00 | −7,00 a −3,10 | 45% | 62% |

- **O modelo da 2019 inverte no caminho arquivo:** a voz real gravada no
  microfone fica do lado sintético (log-odds positivo em 10 de 10) e o clone
  limpo do XTTS do lado humano, sem exceção (0% dos pares). É a mesma resposta às
  condições de gravação da Seção 10.2.6.
- **O modelo do recorte separa o caminho arquivo:** todos os clones acima de
  todas as gravações reais (100% dos pares); no limiar próprio dele (−8,67, do
  dev da ASVspoof 5), 8 de 10 clones e 0 de 10 reais acima. Na média, o modelo da
  2019 puxa os clones para o lado humano. Resultado exploratório, achado depois
  de ver os dados: vale como hipótese para outro ensaio com critério combinado
  antes, não como resultado do teste.
- **Pelo ar, nenhum dos dois separa** (65% e 66% dos pares, com 10 de cada): o
  caminho alto-falante → microfone apaga o que o modelo do recorte usava no
  arquivo, e o da 2019 vê o clone tocado como vê a voz real. Repete a Seção 10.2.7
  com um clonador atual.
- **Limitação do caminho microfone:** os clones tocaram mais baixo que a voz
  (pico de −37,2 a −30,8 dBFS, contra −28,8 a −21,4 dBFS nas gravações reais), e 9
  das 10 gravações pelo ar não passaram na checagem de fala (menos de 4 s acima
  do ruído); pela regra combinada, contam como foram aceitas na gravação. Não dá
  para separar o efeito do nível do efeito do canal.
- **Consequência:** com os modelos atuais, a demonstração planejada (clone pelo
  microfone) não funciona. As correções que o plano prevê atacam exatamente o
  que falhou: clones de ferramentas atuais no treino, nas duas classes e também
  regravados pelo alto-falante, e o XLS-R. Se o XTTS entra no treino ou fica de
  fora segue para o grupo.

### 10.2.13 Ensaio 2: o caminho arquivo com o modelo do recorte (em preparação)

**Decisão do grupo (24/10/2026):** confirmar, com dados novos, a hipótese que o
descritivo do 10.2.12 levantou: o `fusion_lcnn_v4_asv5` sozinho separa a voz
real gravada do clone do XTTS como arquivo. A demonstração passaria a ter o
clone analisado como arquivo e o clone pelo ar como limitação medida (a mudança
da meta ainda passa pelo orientador). Reserva, se o ensaio 2 falhar: o XLS-R.

**Diagnóstico antes do ensaio (exploratório, nos dados do ensaio 1):**
- **Taxa de amostragem não explica a separação.** O clone sai a 24 kHz e a
  gravação a 48 kHz, e cada um chegava aos 16 kHz por uma conversão diferente.
  Levando tudo antes a 48 kHz (`teste_voz_propria.py --taxa-comum 48000`), os
  log-odds do modelo mudaram no máximo 0,015, e todos os clones seguem acima de
  todas as gravações reais.
- **O limiar é o problema.** Pela regra da fusão (10% de alarme falso nos
  humanos dos devs, captura FIR), o limiar do modelo sozinho fica em −6,566
  (`limiar_fusao.py escolher --so fusion_lcnn_v4_asv5`), e só 2 de 10 clones
  passariam dele; no limiar do treino do modelo (−8,67, ponto de EER no dev do
  recorte), 8 de 10. As gravações reais ficam entre −18,41 e −11,66, bem abaixo
  dos humanos dos devs: o microfone da sala parece mais humano ao modelo que a
  base.

**Critério do ensaio 2 (fixado em 24/10/2026, antes de gravar):**
- **Modelo e limiar:** `fusion_lcnn_v4_asv5` sozinho, no limiar do treino dele,
  −8,671 nos log-odds (`outputs/limiar_fusao/limiar_asv5_treino.json`, de
  `limiar_fusao.py treino`); média das janelas ponderada pela fala, nos
  log-odds. Escolha do grupo, com risco declarado: foi nesse limiar que o
  padrão apareceu nos dados do ensaio 1. Não há vozes reais para calibrar no
  microfone; recalibrar por outra via (como a degradação de codecs) fica para
  depois, e só vale com dados novos.
- **Material:** 10 frases novas (`docs/etica/frases_ensaio2.txt`), lidas pelo
  mesmo locutor do ensaio 1, no mesmo microfone e sala, com as mesmas regras de
  gravação (`--piso-minimo 15`, 12 s). Clones do XTTS v2 com a mesma referência
  e a mesma configuração (`clonar_xtts.py`, sementes 1 a 10). Clone que não
  disser a frase inteira e certa é gerado de novo com outra semente, antes de
  qualquer score, e fica registrado.
- **Caminho:** só o arquivo (gravação real × clone como o XTTS entregou), com os
  dois levados a 48 kHz antes da conversão para 16 kHz (`--taxa-comum 48000`).
  O caminho pelo ar não é medido.
- **Meta:** pelo menos 8 de 10 reais julgados humanos e 8 de 10 clones julgados
  sintéticos. Arquivo sem janela útil conta como erro.
- **Limitações declaradas antes:** um locutor, cuja voz e scores já foram vistos
  no ensaio 1, e que é o mesmo da referência do clone; uma ferramenta; uma sala.
  Mesmo que passe, o resultado é otimista para uma voz desconhecida, como a da
  banca.

**O fim dos clones do XTTS (antes do placar do ensaio 2):** todo clone termina
com o som caindo de uma vez para o silêncio digital, cerca de 0,57 s antes do fim
do arquivo, seguido de 417 ms de zeros exatos (10.000 amostras a 24 kHz, que a
biblioteca acrescenta depois de cada frase). Pela energia, a queda acontece no
meio da fala, cortando a última sílaba, em 3 dos 10 clones do ensaio 1 e em 7 dos
10 do ensaio 2; nos outros, corta só o ruído de fundo copiado da referência.
Exploratório, nos clones do ensaio 1: sem essa cauda, os log-odds do modelo do
recorte mudam no máximo 0,72 (nos dois sentidos) e todos seguem acima das
gravações reais. A pista do fim abrupto não explica a separação. No ensaio 2
vale a regra combinada: clone com a frase cortada é gerado de novo com outra
semente, antes do placar.

**Clones do ensaio 2, antes do placar:** a primeira geração (sementes 1 a 10)
foi refeita inteira, ao ouvir, com as sementes 101 a 110. Três ainda travavam
palavras e terminavam antes da hora (f04, f05 e f10); gerados de novo com as
mesmas sementes, saíram idênticos (o XTTS é determinístico pela semente), e com
as sementes 204, 205 e 210 ficaram certos. Conjunto final: sementes 101, 102,
103, 204, 205, 106, 107, 108, 109 e 210 (`data/vozes/ensaio2/clone/clonagem.json`).
Gravações reais: as 10 frases passaram nas checagens.

**Resultado do ensaio 2 (24/10/2026; `outputs/ensaio2.json`):**

| caminho | reais julgados humanos | clones julgados sintéticos | meta |
|---|---|---|---|
| arquivo (real × clone do XTTS, taxa comum de 48 kHz) | 10 de 10 | **9 de 10** | **atingida** |
| microfone | — | — | não medido |

- **Primeira vez que a meta é atingida num caminho, com critério combinado
  antes.** O placar rodou no ambiente do XTTS e de novo no do projeto: os mesmos
  log-odds (diferença 0).
- **Descritivo:** reais de −14,59 a −12,35; clones de −11,62 a −4,53; todos os
  clones acima de todas as gravações reais (100 de 100 pares). O clone que
  passou como humano (f08, −11,62) ficou a 0,73 da gravação real mais alta, e
  dois dos que foram pegos (f01, −8,50; f03, −8,26) ficaram a menos de 0,5 do
  limiar: a ordem é estável entre os dois ensaios, o ponto de corte é frágil.
- **Ensaio 1 e ensaio 2 juntos, no mesmo modelo e limiar:** reais de −18,41 a
  −12,35 (20 de 20 humanos); clones de −11,62 a −4,53, com 17 de 20 acima do
  limiar. Os dados do ensaio 1 não contam para o critério, porque o limiar foi
  escolhido depois deles.
- **O que o resultado não mostra:** é um locutor só, que é também a voz da
  referência e cujos scores já tinham sido vistos; uma ferramenta; uma sala; o
  clone analisado como arquivo, não tocado. Para a banca, falta pelo menos uma
  voz que o sistema nunca viu.

### 10.2.14 Ensaio 3: uma voz que o sistema nunca ouviu (em preparação)

O ensaio 2 passou com a voz do L01, já vista no ensaio 1. O ensaio 3 repete o
critério com o locutor L02, voluntário de fora do grupo, para ver se o resultado
vale para uma voz nova, como a da banca.

**Critério (fixado em 25/10/2026, antes de qualquer gravação do L02):**
- **Modelo e limiar:** os do ensaio 2, sem nenhuma mudança: `fusion_lcnn_v4_asv5`
  sozinho, −8,671 nos log-odds (`outputs/limiar_fusao/limiar_asv5_treino.json`),
  média das janelas ponderada pela fala, nos log-odds.
- **Material:** o L02 lê o texto de referência (`docs/etica/texto_referencia_clonagem.txt`,
  90 s) e as 10 frases de `docs/etica/frases_ensaio3.txt`, novas para todos,
  no mesmo microfone e sala, sem fone e sem carregador, com as regras dos
  ensaios (`--piso-minimo 15`, 12 s, regravar só pelas checagens). A referência
  é preparada por `scripts/preparar_referencia.py`, que reproduz o preparo
  feito à mão na referência do L01 (trecho com fala, pico a −3 dBFS, 16 bits).
- **Clones:** XTTS v2 local, mesma configuração (sementes 1 a 10). Clone que
  travar ou não disser a frase inteira é gerado de novo com outra semente,
  antes do placar, e fica registrado.
- **Caminho:** só o arquivo, com `--taxa-comum 48000`.
- **Meta:** pelo menos 8 de 10 reais julgados humanos e 8 de 10 clones julgados
  sintéticos. Arquivo sem janela útil conta como erro.
- **Sem espiar:** nenhum score da voz do L02 antes do placar, nem pela aba Ao
  vivo nem pela Enviar arquivo.
- **Leitura combinada antes:** se passar, há um resultado com uma voz que o
  sistema nunca ouviu, ainda com uma ferramenta, uma sala e um locutor novo só.
  Se falhar, entra como saiu, e o ensaio 2 fica valendo só para a voz do L01; a
  reserva continua o XLS-R.
- **Ética:** termo assinado pelo L02 antes de gravar, com o item de clonagem;
  código no lugar do nome; gravações e clones em `data/vozes/L02/`, fora do Git,
  e os clones apagados depois da banca.

**Material do L02 e mudanças antes do placar (25/10/2026, sem nenhum score visto):**
- **Referência:** 90 s, fala de 4,1 a 87,9 s (texto lido inteiro, mais devagar
  que o L01: ~84 s contra ~70 s); preparada pelo `preparar_referencia.py` em
  84,9 s.
- **Frases reais:** as 10 passaram nas checagens (ruído de 15 a 22 dB abaixo da
  fala). Em 12 s, o L02 ficou sem fôlego na f06 e quase na f10, pela avaliação
  dele; as duas foram regravadas com 15 s (`--segundos 15`) e passaram (9,6 s de
  fala cada). As outras 8 ficaram com 12 s.
- **Clones sem o ponto final:** na primeira geração, 3 de 10 clones (f04, f06,
  f09) terminaram falando "ponto". Os 10 foram apagados e gerados de novo com a
  frase sem o ponto final, que passou a ser o padrão do `clonar_xtts.py`
  (`--com-ponto-final` repete os ensaios 1 e 2). Diferença em relação ao ensaio
  2, que mandou a frase com o ponto.
- **Conjunto final dos clones:** sementes 1 a 10, sem o ponto final, nenhum
  refeito; os 10 foram ouvidos e dizem a frase inteira e certa. Pela energia,
  f01, f04 e f10 terminam com som de fala no ar, mas ao ouvir não falta palavra.
  Ruído herdado da referência: 17 a 26 dB abaixo da fala.

**Defeitos do XTTS v2 vistos até aqui** (para o texto e para a demonstração):
1. **Fim abrupto:** todo clone cai de uma vez para o silêncio digital e termina
   com 417 ms de zeros; a última sílaba sai cortada em parte deles (3 de 10 no
   ensaio 1, 7 de 10 no ensaio 2). Não explica a separação (10.2.13).
2. **Palavras travadas e fim antes da hora:** 3 de 10 no ensaio 2 (f04, f05,
   f10). Dependem da semente: com a mesma semente o áudio sai idêntico, com
   outra o defeito some.
3. **Pontuação lida em voz alta:** o ponto final virou a palavra "ponto" em 3 de
   10 clones do L02. Contorno: mandar a frase sem o ponto final.
4. **Ruído herdado:** o clone reproduz o ruído de fundo da referência (18 a 24
   dB abaixo da fala, contra 21 a 25 dB nas gravações reais do L01).
5. **Começo sem silêncio:** o clone começa a falar logo no início do arquivo
   (0,09 a 0,34 s em 9 dos 10 clones do L02), e a checagem de fala, feita para
   gravações de microfone, acusa "fala encosta no início". Por isso os clones
   aparecem com "checagens: fala" no placar, só como informação. O f10 do L02 é
   a exceção: cerca de 2 s de ruído antes da fala, e a checagem passa.

Todos aparecem ao ouvir o clone, sem olhar score; a regra dos ensaios é refazer
com outra semente antes do placar. Na demonstração, ouvir o clone antes de
mostrá-lo e refazer (`r` no modo interativo) se precisar.

**Resultado do ensaio 3 (25/10/2026; `outputs/ensaio3_L02.json`):**

| caminho | reais julgados humanos | clones julgados sintéticos | meta |
|---|---|---|---|
| arquivo (real × clone do XTTS, taxa comum de 48 kHz) | 10 de 10 | **10 de 10** | **atingida** |
| microfone | — | — | não medido |

- **A meta passa com uma voz que o sistema nunca tinha ouvido.** Todos os clones
  acima de todas as gravações reais (100 de 100 pares).
- **Descritivo, em log-odds:** reais de −15,27 a −10,58; clones de −7,94 a −0,97;
  limiar em −8,671. A gravação real mais perto do limiar (f07) ficou 1,91 abaixo
  dele, e o clone mais perto (f05), 0,73 acima. Entre as duas classes sobrou um
  intervalo de 2,64, maior que o do L01 no ensaio 2 (0,73).
- **Ensaios 2 e 3 juntos** (mesmo modelo, limiar e caminho, critério combinado
  antes nos dois): reais 20 de 20 humanos, clones 19 de 20 sintéticos.
- **O silêncio das gravações não explica a separação** (exploratório, depois
  do placar): as gravações reais têm 2 a 4 s de ruído antes da fala, e os clones
  começam a falar logo. Cortando as reais para o trecho com fala (0,5 s de
  folga), os log-odds mudam de −1,24 a +2,74, nos dois sentidos, e as 20 seguem
  abaixo do limiar (L01 de −14,88 a −11,61; L02 de −14,23 a −9,28). A f01 do L02
  chega a 0,61 do limiar: a margem do lado humano também é estreita.
- **O que o resultado não mostra:** dois locutores, uma ferramenta (o XTTS v2,
  que no ensaio 3 recebeu as frases sem o ponto final), uma sala, um microfone;
  o clone analisado como arquivo, não tocado. A voz da banca, de outra pessoa
  ou gravada em outro lugar, segue sem medida; o clone pelo ar segue sem
  separação (10.2.12).

### 10.2.15 Ensaio 4: uma segunda ferramenta de clonagem (em preparação)

Os ensaios 2 e 3 mediram clones do XTTS v2, que ao ouvido soam genéricos. O
ensaio 4 pergunta se o detector pega clones de uma ferramenta mais recente, ou
se o resultado vale só para o XTTS. É a segunda ferramenta que o plano prevê.

**Critério (fixado em 25/10/2026, antes de gerar qualquer clone com ela):**
- **Ferramenta:** Chatterbox Multilingual (Resemble AI, licença MIT, 2025), no
  próprio computador, num ambiente conda separado (`clonar_xtts.py --ferramenta
  chatterbox`), com os padrões dela. O pacote do pip (`chatterbox-tts` 0.1.7)
  traz o modelo multilíngue v2 (`t3_mtl23ls_v2`), e não a v3 do repositório;
  corrigido em 25/10/2026, antes de gerar qualquer clone. Ele usa só os
  primeiros 10 s da referência (o XTTS usa 30 s). Todo áudio que ela gera leva
  uma marca d'água inaudível (Perth); fica registrado.
- **Material:** as mesmas referências preparadas e as mesmas gravações reais dos
  ensaios 2 (L01, `frases_ensaio2.txt`) e 3 (L02, `frases_ensaio3.txt`). Clones
  novos, sementes 1 a 10, frase sem o ponto final, em `clone_chatterbox/` ao
  lado de `clone/`. Clone que travar ou não disser a frase inteira é refeito com
  outra semente antes do placar.
- **Detector e caminho:** os dos ensaios 2 e 3, sem mudança: `fusion_lcnn_v4_asv5`,
  −8,671 nos log-odds, caminho arquivo com `--taxa-comum 48000`
  (`teste_voz_propria.py --clones clone_chatterbox`).
- **Meta:** para cada locutor, pelo menos 8 de 10 clones julgados sintéticos. O
  lado real já foi medido (10 de 10 em cada) e só se repete no placar.
- **Leitura combinada antes:** se passar nos dois locutores, o resultado não é só
  do XTTS. Se falhar, o detector reconhece o XTTS mas não uma ferramenta mais
  nova: entra como limitação, no texto e na demonstração, que usa o XTTS. O
  realismo de cada ferramenta, julgado de ouvido, é só descritivo.
- **Limitações declaradas antes:** as gravações reais e os scores delas já
  foram vistos; dois locutores, uma sala, um microfone.

**Clones do ensaio 4, antes do placar (25/10/2026):** `chatterbox-tts` 0.1.7,
t3 v2, sementes 1 a 10 nos dois locutores, frase sem o ponto final, nenhum
refeito; os 20 foram ouvidos e dizem a frase inteira. Saem a 24 kHz, com 4,9 a
7,0 s de fala, perto do ritmo das leituras reais (o XTTS fala mais devagar: 6,9 a
11,0 s nas frases do L01). Em 6 dos 10 clones do L01, o Chatterbox avisou
repetição de um som e encerrou a frase à força; em dois deles (f01, f06) o áudio
termina com energia de fala, mas ao ouvir não falta palavra. Na GTX 1650 o modelo
ocupa quase os 4 GB: cada frase levou de 25 a 108 s, e uma, 366 s, com a memória
transbordando para a do sistema. Inviável ao vivo nessa placa; a demonstração
segue com o XTTS.

**Resultado do ensaio 4 (26/10/2026; `outputs/ensaio4_L01.json` e `ensaio4_L02.json`):**

| locutor | reais julgados humanos | clones do Chatterbox julgados sintéticos | pares clone acima do real |
|---|---|---|---|
| L01 | 10 de 10 | **3 de 10** | 96 de 100 |
| L02 | 10 de 10 | **9 de 10** | 100 de 100 |

- **Meta não atingida:** o critério pedia pelo menos 8 de 10 em cada locutor, e o
  L01 ficou em 3. O detector não pega com folga os clones de uma ferramenta mais
  nova; o resultado dos ensaios 2 e 3 vale para o XTTS.
- **Descritivo, em log-odds:** clones do Chatterbox do L01 de −12,61 a −5,86 (o
  limiar é −8,671), do L02 de −9,55 a −6,11; os do XTTS do L02 iam de −7,94 a
  −0,97. O Chatterbox fica mais perto do lado humano, e mais ainda na voz do L01.
- **A ordem quase não erra:** em 196 dos 200 pares (clone, gravação real do
  mesmo locutor), o clone fica acima. Juntando os ensaios 2 a 4, o modelo do
  recorte põe o clone acima da voz real do mesmo locutor em 396 de 400 pares; o
  que falha é o ponto de corte, que não se ajusta à voz de cada pessoa.
  Observação depois dos dados, não critério.

### 10.2.16 Captura do microfone: uma emenda suspeita a cada 1,1 s (sem perda de amostras)

**Achado (25/10/2026):** ouvidas no VLC e no reprodutor do Windows, as gravações
reais "travavam". Não há buraco de silêncio nem salto grande no sinal, mas nas
emendas entre os blocos de 100 ms que o gravador pedia ao driver o erro de
previsão da amostra é anormal num instante regular: a cada 1,1 s da captura,
11% a 20% dessas emendas ficam acima de 5 vezes o normal, contra menos de 1% nas
outras emendas. Aparece em todas as gravações reais e referências dos ensaios 1 a
3 (L01 e L02, dias diferentes).

**Causa ainda não explicada.** A primeira hipótese era o encaixe dos blocos
fixos de 4.800 amostras com pacotes de 528 do driver (que se repetiria a cada
52.800 amostras, os 1,1 s); não se confirmou: o driver entrega pacotes de 480
amostras, que cabem exatamente em 4.800. **Mudança:** a fonte do microfone
passa a receber os pacotes no tamanho do driver (`blocksize=0`); o FIR guarda a
fase entre pacotes de qualquer tamanho (testado); o gravador mostra o pacote,
avisa perdas e aceita `--bloco-ms 100` para repetir a captura antiga. Na
primeira gravação assim (L01, uma frase), os pontos a cada 1,1 s não se
destacam (mediana 1,5 vez o normal, contra 1,7 em pontos ao acaso; antes, 2,3).
Mas ao ouvido ela soou "em câmera lenta", e a voz saiu 7% mais grave que a
média do L01 no ensaio 2, o que cabe na variação de uma frase para outra.

**Teste com um tom de 1.000 Hz** (26/10/2026), tocado pelo alto-falante e gravado
nas duas capturas, 31 a 32 dB acima do ruído: a frequência gravada é 1.000,00 Hz
nas duas (a velocidade está certa; não há câmera lenta), e a fase do tom é
contínua nas duas, sem nenhum tranco nos instantes de 1,1 s (nenhuma amostra
perdida ou repetida). Ao ouvido, as duas soam iguais. **Conclusão:** a captura
não perde amostras; o "travado" ouvido não se confirmou como falha de captura. A
emenda medida na fala segue como um artefato pequeno, de causa não explicada,
sem efeito visível no tom. A captura fica com os pacotes do driver, que não
mostrou nenhum custo.

**O "travado" está na gravação, e parece vir do microfone** (26/10/2026). Tocado
pelo Python, fora dos reprodutores, o áudio também soa travado. O espectrograma
da fala mostra muitos estalos curtos de banda larga (linhas verticais até 12 kHz),
também no meio de vogais. Contando os estalos de 1 ms acima de 7 kHz: 116 na
referência do L01 (1,3 por segundo) e 61 na do L02; nenhum no tom e quase nenhum
nos clones. Nas gravações antigas, eles aparecem em pares separados por 110 ms
exatos (11 pacotes de 10 ms do driver), 20 de 22 deles nas pausas do L02: um
estalo a cada 110 ms em dez seguidos daria os 1,1 s, e só um em dez cairia numa
emenda de 100 ms, o que explica a primeira medida. Não há degrau de ganho nas
pausas. **Hipótese:** processamento dentro do microfone embutido ("Grupo de
microfones", provavelmente uma matriz com supressão de ruído ou formação de
feixe no driver ou no hardware, que o modo exclusivo não desliga), trocando de
estado entre fala e pausa. É o risco "o microfone faz processamento no hardware"
do plano; o teste do microfone embutido tinha ficado para depois. No painel da
Realtek (Realtek Audio Console), os efeitos do microfone já estavam desligados
("Desabilitar Tudo"), com reforço de +10 dB e formato compartilhado de 48 kHz e
16 bits; os estalos não vêm dessas opções. Falta gravar a mesma frase com outro
microfone para separar o microfone embutido de uma interferência elétrica. Os resultados dos ensaios ficam como
saíram: as gravações reais, com esses estalos, foram julgadas humanas, e a banca
usa o mesmo microfone.

**Efeito nos ensaios, declarado:** a classe real dos ensaios 1 a 3 e as
referências dos clones têm essa emenda áspera. Os resultados ficam como saíram;
o artefato é uma imperfeição pequena da gravação real (o tom mostrou que não
há perda de amostras), que mesmo assim foi julgada humana em 30 de 30 arquivos. O ao vivo usa a mesma fonte, então os testes ao vivo pelo
microfone (10.2.10 e os da aba Ao vivo) também tinham o defeito.

**Causa achada em 10.2.34 (26/10/2026):** não é o microfone. No modo exclusivo,
a captura entrega cerca de 1,47 vez mais pacotes que o tempo real: ela insere
cópias exatas de trechos recentes (110 ms com os blocos de 100 ms, 20 ms com os
pacotes do driver). O tom de 1.000 Hz não via isso, porque 10 ms são 10 ciclos
exatos: a cópia sai com a fase contínua. A "câmera lenta" ouvida era isso.

### 10.2.17 Imperfeições dos sintéticos, parte 1: onde o detector vê o sintético (exploratório)

**Por quê (26/10/2026):** no ensaio 4, o Chatterbox passou pelo detector (L01: 3
de 10 clones acima do limiar). Mudar o limiar foge da proposta do trabalho, e o
XLS-R pré-treinado seria uma caixa-preta (proposta no plano). O grupo passou a
estudar as imperfeições dos próprios áudios sintéticos: onde o detector vê o
sintético, que marcas os geradores deixam e se elas valem para ferramentas que
o treino não viu. **Exploratório:** sem critério combinado antes e sem mudar
nenhum placar. Serve para escolher hipóteses, que depois serão testadas com
critério em dados que não serviram para escolhê-las.

**Material das partes 1 a 4:** os arquivos dos ensaios 2 a 4. L01: 10 gravações
reais, 10 clones do XTTS e 10 do Chatterbox; L02: o mesmo. Reais pelo microfone
embutido, com os estalos de 10.2.16; clones a 24 kHz. Modelo e caminho do
ensaio 2: `fusion_lcnn_v4_asv5`, limiar −8,671 nos log-odds, `--taxa-comum 48000`.

**Método (`scripts/estudo_faixas.py`, `outputs/estudo_faixas.json`):** já a 16
kHz, retira uma faixa por vez (Butterworth de ordem 8, ida e volta) e mede o
log-odds médio do arquivo (janelas ponderadas pela fala, como no placar). Com o
áudio inteiro, a média do log-odds e os clones acima do limiar repetem os
placares: L01 reais −13,27 (0/10), XTTS −7,30 (9/10), Chatterbox −9,85 (3/10);
L02 reais −12,86 (0/10), XTTS −5,85 (10/10), Chatterbox −7,51 (9/10).

Variação do log-odds ao tirar a faixa, nos clones menos nos reais do mesmo
locutor (negativo: a faixa pesa a favor de "sintético" nos clones):

| faixa retirada | XTTS L01 | XTTS L02 | Chatterbox L01 | Chatterbox L02 |
|---|---|---|---|---|
| 0–500 Hz | −2,32 | +1,07 | −3,84 | −0,56 |
| 500–1000 Hz | −1,31 | −1,22 | +0,98 | +0,56 |
| 1–2 kHz | −2,22 | −0,90 | +1,13 | −0,91 |
| 2–3 kHz | −1,42 | −1,19 | −0,60 | −0,20 |
| 3–4 kHz | −1,75 | −1,94 | −1,30 | −1,05 |
| 4–5 kHz | −1,00 | −1,68 | −0,30 | −0,33 |
| 5–6 kHz | −0,72 | −0,83 | +0,00 | −1,24 |
| 6–7 kHz | −0,61 | −0,37 | −0,02 | −0,55 |
| 7–8 kHz | −0,25 | −0,31 | −0,16 | −0,10 |

**Leitura:**
- Tirar qualquer faixa sobe o log-odds de todos, reais inclusive. Sem 0–500 Hz,
  os reais do L01 sobem 7,19 em média e os 10 passam do limiar. O modelo lê
  "espectro incompleto" como sintético; a retirada sozinha não localiza a pista.
- Em relação aos reais, as faixas de 2 a 5 kHz pesam contra os clones nos
  quatro pares; 3–4 kHz é a maior e a mais estável (−1,05 a −1,94). Acima de 6
  kHz o peso é pequeno (−0,02 a −0,61). Abaixo de 2 kHz o sinal muda de um par
  para outro.
- Nenhuma faixa sozinha esconde o XTTS: sem qualquer uma delas, 8 a 10 de 10
  clones seguem acima do limiar. A pista está espalhada pela banda média.

### 10.2.18 Parte 2: o relógio do gerador na modulação (exploratório)

**Hipótese:** os geradores montam o som em passos fixos (tokens, quadros do
vocoder). Um pico estreito no espectro de modulação (os ritmos em que a energia
de cada faixa sobe e desce) marcaria o relógio do gerador.

**Método (`scripts/estudo_relogio.py`, `outputs/estudo_relogio.json` e `.png`):**
log-energia de 8 faixas a cada 2 ms, nos trechos de fala de pelo menos 1 s, a 16
kHz; espectro de modulação até 250 Hz, com 1 Hz de resolução; pico = 3 dB acima
do contorno. O pente regular é medido pela autocorrelação do excesso sobre o
contorno, de 12 a 160 Hz.

**Resultado:** nenhum pico em nenhum grupo. As gravações reais têm um pente
regular: espaçamento de 9 Hz, autocorrelação +0,91 (L01) e +0,92 (L02),
ondulação de 1,66 e 1,46 dB. Os clones não: autocorrelação de +0,13 a +0,22,
ondulação de 0,12 a 0,49 dB. O espaçamento é o dos estalos do microfone a cada
110 ms (10.2.16; 1/0,110 s = 9,09 Hz). **O relógio achado é o da captura, não o
do gerador.** Alerta para as partes seguintes: as gravações reais dos ensaios
têm uma marca do microfone, e uma característica nova precisa ser conferida em
áudio real de outra origem, senão o detector aprende "tem estalo = humano".
(10.2.34: a marca é da captura, cópias de 110 ms; sem elas, o pente some.)

### 10.2.19 Parte 3: linhas fixas no espectro (exploratório)

**Hipótese:** a altura da voz muda o tempo todo, então no espectro médio os
harmônicos se espalham; um tom de frequência fixa (resto do vocoder, ruído do
microfone) continua uma linha estreita. Vocoders neurais deixam tons ao
reamostrar (Pons et al., ICASSP 2021, "Upsampling artifacts in neural audio
synthesis").

**Método (`scripts/estudo_linhas.py`, `outputs/estudo_linhas.json`):** espectro de
Welch com 1 Hz de resolução, na taxa de cada arquivo, menos o contorno (mediana
em 41 Hz). As linhas são escolhidas num grupo (mediana dos arquivos, acima de
300 Hz) e contadas em todos; a linha está presente no arquivo se passa 3 dB do
contorno a até 2 Hz. A réplica é escolher num locutor e conferir no outro.

**Linhas que se repetem no outro locutor** (arquivos com a linha):

| linha | escolhida em | real L01 | real L02 | XTTS L01 | XTTS L02 | Chatterbox L01 | Chatterbox L02 |
|---|---|---|---|---|---|---|---|
| 6.078 Hz | XTTS L01 | 0/10 | 3/10 | 9/10 | 9/10 | 1/10 | 0/10 |
| 8.139 Hz | XTTS L02 | 3/10 | 4/10 | 10/10 | 9/10 | 0/10 | 1/10 |
| 7.014 Hz | XTTS L02 | 1/10 | 2/10 | 10/10 | 8/10 | 0/10 | 3/10 |
| 7.850 Hz | Chatterbox L01 | 1/10 | 1/10 | 0/10 | 0/10 | 10/10 | 10/10 |
| 7.450 Hz | Chatterbox L02 | 2/10 | 0/10 | 1/10 | 1/10 | 9/10 | 10/10 |
| 6.550 Hz | Chatterbox L02 | 0/10 | 1/10 | 2/10 | 3/10 | 8/10 | 10/10 |
| 527 Hz | real L01 | 9/10 | 9/10 | 2/10 | 1/10 | 1/10 | 0/10 |
| 655 Hz | real L01 | 9/10 | 9/10 | 1/10 | 0/10 | 1/10 | 1/10 |

**Leitura:**
- Cada gerador deixa as próprias linhas, as mesmas nos dois locutores, e o
  microfone também deixa as dele. As frequências mudam de uma ferramenta para
  outra: é uma **assinatura de cada gerador**, não uma marca universal.
- Nem toda linha escolhida é específica: 7.611 Hz, escolhida no XTTS do L01,
  aparece em 7 de 10 e 4 de 10 gravações reais.
- As linhas do Chatterbox caem, na maioria, em múltiplos de 50 Hz (6.550, 7.450,
  7.850), que é a taxa de quadros do vocoder dele. Daí a parte 4.
- **Por que o detector não usa as linhas.** Pelo caminho do modelo (16 kHz), a
  linha chega inteira na resolução de 1 Hz:
  - 6.078 Hz do XTTS: +5,6 dB (L01) e +5,1 dB (L02);
  - 7.850 Hz do Chatterbox: +5,2 e +5,6 dB;
  - controle a ±100 Hz da linha: +0,1 a +0,7 dB.

  Na janela de 25 ms do detector (400 amostras, FFT de 512, como nos
  `fusion_v4*.yaml`), a mesma linha fica entre −0,1 e +1,0 dB, perto do controle
  (0,0 a +0,2). Depois ainda passa pelos 70 filtros do LFCC e pelos 80 do mel,
  mais largos que um ponto da FFT. **A marca chega ao modelo, mas a resolução
  dele é larga demais para vê-la.**

### 10.2.20 Parte 4: o pente na taxa de quadros do vocoder (exploratório)

**Hipótese:** o vocoder monta o som quadro a quadro, e uma marca repetida a cada
quadro vira um pente de linhas nos múltiplos da taxa de quadros. Ela surgiu
olhando as linhas da parte 3, nos mesmos arquivos; não é uma confirmação
independente. Taxas tiradas do código instalado:
- **XTTS v2:** HiFi-GAN com fatores 8, 8, 2, 2, ou seja, 256 amostras a 24 kHz e
  93,75 quadros/s (`TTS/tts/layers/xtts/hifigan_decoder.py`).
- **Chatterbox:** HiFT com fatores 8, 5, 3 e iSTFT de passo 4, ou seja, 480
  amostras a 24 kHz e 50 quadros/s (`chatterbox/models/s3gen/s3gen.py`).
- **Microfone, previsto antes de medir:** os estalos a cada 110 ms (10.2.16) dão
  9,0909 Hz.

**Método (`scripts/estudo_pente.py`, `outputs/estudo_pente.json`):** a força do
pente, por arquivo, é o excesso médio (1 Hz) nos múltiplos da base menos o dos
pontos do meio. É medida em 4–11,5 kHz no nativo e em 4–7,6 kHz no caminho do
modelo. A separação é medida pela AUC. Uma varredura de 40 a 200 Hz mede o pente
mais forte sem saber o gerador.

**Resultado no nativo** (força mediana em dB; entre colchetes, a AUC contra os
20 reais):

| grupo | base 93,75 Hz | base 50 Hz | base 9,0909 Hz | varredura |
|---|---|---|---|---|
| reais L01 | +0,10 | −0,51 | +2,03 | +2,18 |
| reais L02 | −0,25 | −0,77 | +1,86 | +1,89 |
| XTTS L01 | +0,85 [0,96] | +0,09 [1,00] | −0,02 [0,00] | +1,36 [0,16] |
| XTTS L02 | +0,03 [0,57] | +0,09 [1,00] | −0,00 [0,00] | +1,05 [0,07] |
| Chatterbox L01 | −0,40 [0,17] | +0,22 [0,99] | +0,00 [0,00] | +1,15 [0,07] |
| Chatterbox L02 | −0,27 [0,30] | +0,25 [1,00] | +0,00 [0,00] | +1,06 [0,04] |

**Leitura:**
- **O pente previsto do microfone está em todas as gravações reais e em nenhum
  clone.** O microfone sozinho separa os reais dos clones (AUC 0,00, ou 1,00 no
  outro sentido). É o atalho que um detector treinado nessas gravações
  aprenderia. Na varredura, os 100 Hz dos reais (20 de 20 no nativo) são 11 ×
  9,09 Hz.
- **A AUC de 50 Hz contra os reais é espúria.** Ela separa o Chatterbox, mas
  também o XTTS (1,00), porque o pente do microfone põe metade dos múltiplos de
  50 Hz num vale (reais em −0,51 e −0,77).
- **Comparando um gerador com o outro, sem o microfone:**
  - base de 50 Hz, Chatterbox contra XTTS: AUC 0,58 (L01) e 0,75 (L02);
  - base de 93,75 Hz, XTTS contra Chatterbox: 1,00 e 0,72;
  - no caminho do modelo: 0,77 e 0,72; 0,83 e 0,61.

  O relógio do vocoder existe, mas é fraco (de +0,03 a +0,85 dB nas medianas,
  cada gerador na própria base) e
  pouco confiável por arquivo.
- **A varredura que não sabe o gerador falha:** o pente mais forte é o dos
  reais (AUC dos clones de 0,04 a 0,16).

**Síntese das partes 1 a 4:**
1. O detector vê o sintético espalhado pela banda média (2–5 kHz) e lê espectro
   incompleto como sintético.
2. Os geradores deixam marcas finas (linhas fixas próprias de cada ferramenta;
   um pente fraco na taxa de quadros). A resolução de 25 ms do detector é larga
   demais para vê-las.
3. A captura também deixa marcas finas (pente de 9,09 Hz, linhas de 527 e 655
   Hz). As gravações reais dos ensaios não servem sozinhas de referência humana
   para características desse tipo.
4. Nada até aqui é universal: as linhas mudam de gerador para gerador, e há
   relógio dos dois lados.

**Próximo:** a parte 5 repete as medidas nas bases ASVspoof. Lá os humanos têm
outra origem, há muitos geradores e há ataques que o estudo não viu.

### 10.2.21 Parte 5: linhas e pente nas bases ASVspoof (exploratório)

**Por quê:** nas partes 3 e 4, os humanos vinham todos do mesmo microfone, e os
geradores eram só dois. Aqui os humanos são os das bases e há 22 ataques.

**Método (`scripts/estudo_bases.py`; `outputs/estudo_bases_<rótulo>.json`):**
- **Amostra:** 100 arquivos por classe (semente 42), cortados nos primeiros 3 s,
  para que todas as classes tenham o mesmo número de segmentos no espectro (2 Hz
  de resolução, 11 segmentos).
- **Bases:** a 2019 LA (treino, A01–A06) e o recorte da ASVspoof 5 (treino,
  A01–A08; dev, A09–A16). O eval fica para um teste com critério.
- **Linhas da classe:** escolhidas na metade A dos arquivos e contadas na metade B
  e na metade B dos humanos.
- **Duas medidas por arquivo, sem saber o gerador:** a fração da banda de 2 a 7,6
  kHz com linha (6 dB acima do contorno) e o pente varrido (parte 4, 4–7,6 kHz).
  A separação é a AUC contra os humanos da mesma base.

**Resultado:**
- **2019 LA:** nenhum ataque tem linha fixa. Nos ataques, a mediana mais alta é
  +2,2 dB (A02), em 7.998 Hz, a borda dos 8 kHz, que aparece igual nos humanos
  (11/50). As duas
  medidas não separam: AUC de 0,34 a 0,60 (linhas) e de 0,34 a 0,68 (pente). Os
  geradores da 2019 (WORLD, WaveNet, concatenação) não têm camadas de
  reamostragem aprendidas, de onde vêm os tons segundo Pons et al.
- **ASVspoof 5:** marcas em parte dos ataques.

  | ataque | marca | arquivos com a marca | humanos |
  |---|---|---|---|
  | A14 (dev) | pente de 62,5 Hz; as 5 linhas mais fortes em múltiplos de 62,5 Hz (4.000, 7.562, 7.812, 7.875 e 7.938 Hz, +4,4 a +5,9 dB) | pente: base de 62,5 Hz em 55/100 e de 187,5 Hz em 40/100; linhas: 32 a 37/50 | linhas: 0 a 7/50 |
  | A13 (dev) | linhas em 7.732, 5.512 e 2.756 Hz (+6,7, +2,5, +2,3 dB) | 45, 33 e 26/50 | 9, 5 e 6/50 |
  | A07 (treino) | linha em 7.000 Hz (+5,2 dB); pente de 93,75 Hz | linha: 40/50; pente: 9/100 | linha: 8/50 |
  | A01–A06 (treino) | a mesma linha de 7.000 Hz, mais fraca (+2,4 a +3,3 dB) | 20 a 26/50 | 8/50 |
  | A08–A12, A15, A16 | nenhuma linha acima de +2,3 dB | — | — |

  Os 62,5 Hz são compatíveis com um vocoder a 16 kHz e passo de 256 amostras;
  5.512 e 2.756 Hz, com reamostragem a 22,05 kHz (22.050/4 e 22.050/8). A
  documentação desses ataques não foi conferida.
- **As medidas que não sabem o gerador só pegam o A14:** pente com AUC 0,97 e
  linhas com 0,79. Nos outros 14 ataques da ASVspoof 5, AUC de 0,33 a 0,51
  (linhas) e de 0,34 a 0,59 (pente).
- **Repetição com 500 arquivos por classe** (`--n 500`, semente 42;
  `outputs/estudo_bases_*_n500.json`). Confirma o quadro:
  - A14: pente com AUC 0,96, base de 62,5 Hz em 260 de 500 arquivos;
  - A13: linha de 7.732 Hz em 209/250, contra 35/250 nos humanos;
  - A07: linha de 7.000 Hz em 182/250, contra 26/250;
  - A01–A06 da ASVspoof 5: a mesma linha em 114 a 132/250;
  - demais ataques: AUC de 0,35 a 0,65 nas duas medidas, contando a 2019.

  Aparece também um relógio fraco de 50 Hz no A16: base de 50 Hz em 33/500, e
  linhas em múltiplos de 50 (6.650, 7.050, 7.300 e 7.700 Hz, +1,4 a +1,6 dB, em
  75 a 85/250, contra 25 a 39/250 nos humanos).

**Leitura:** as marcas finas são reais e explicáveis, mas não universais.
- Aparecem nos geradores com reamostragem aprendida: o XTTS, o Chatterbox e uma
  parte dos ataques da ASVspoof 5.
- A frequência muda de um gerador para outro.
- Na 2019 LA não aparecem.
- Uma medida simples por arquivo só separa o gerador de marca mais forte (A14).

Para o detector, isso sugere um ramo de resolução fina, de convolução ao longo
da frequência, que procure "linha estreita" em qualquer lugar e não numa
frequência decorada. Há um risco: microfones e ambientes também têm linhas
(527 e 655 Hz no microfone embutido, parte 3). Uma hipótese a testar é que a
linha do gerador acompanha a fala, e a do ambiente continua nas pausas.

### 10.2.22 Parte 6: o que o XLS-R enxerga (exploratório; regra fixada antes)

**Por quê (26/10/2026):** os detectores com XLS-R são os que mais generalizam na
literatura, e o grupo quer entender o que eles captam para trazer ao modelo
próprio. O XLS-R entra como **instrumento de medida, não como detector**: os
pesos pré-treinados da Meta (`facebook/wav2vec2-xls-r-300m`, baixados com o aval
do grupo) ficam congelados e não entram no sistema.

**Método (`scripts/estudo_ssl.py`; vetores em `outputs/ssl/`):**
- **Representações:** cada arquivo é preparado como no treino (16 kHz pelo FIR,
  sem silêncio nas pontas, pico em 1) e passa em janelas de 4 s com passo de 2 s.
  A média no tempo de cada representação vira um vetor: a convolução de entrada
  e as 25 saídas do transformer.
- **Controle:** a média e o desvio do LFCC do nosso detector (120 valores).
- **Sonda:** uma regressão logística por representação, com o C escolhido por
  validação cruzada só no treino. Ela é treinada no treino do recorte da
  ASVspoof 5 (2.000 humanos e 250 de cada ataque, A01–A08).
- **Medida:** no dev do recorte (400 humanos e 150 de cada ataque A09–A16, que a
  sonda não viu), na 2019 LA de treino (400 humanos e 100 de cada ataque) e nos
  clones dos ensaios (XTTS e Chatterbox, L01 e L02). O limiar de cada sonda é o
  ponto de EER no dev.

**Regra fixada antes dos resultados:** a melhor representação é a de maior AUC
média nos ataques do dev da ASVspoof 5. Os ensaios e a 2019 LA só medem.

**Explicar:** o score da melhor representação é comparado com medidas simples:
- energia por faixa, inclinação do espectro, planura da banda alta;
- pico sobre a média, HNR;
- linhas e pente (partes 4 e 5);
- silêncio retirado e duração.

A mesma sonda treinada só nessas medidas mostra o quanto elas sozinhas
explicam. Depois, a retirada de faixas (como na parte 1) mostra onde a melhor
camada vê o sintético.

**Extração (26/10/2026, 02:24–02:53):** 6.660 arquivos, nenhum pulado
(`outputs/ssl/extracao.log`). Para comparar, o nosso detector (`fusion_lcnn_v4_asv5`)
pontuou os mesmos arquivos pelo caminho da avaliação: 4 s desde o início e
limiar no ponto de EER do mesmo dev (`estudo_ssl.py detector`,
`outputs/ssl/detector.json`). O EER dele nesse dev foi de 20,75%, perto dos
20,56% do dev inteiro (10.7).

**Resultado das sondas (`outputs/ssl/sondas.json`).** AUC dev = média das AUCs
dos 8 ataques do dev; nos ensaios e na 2019, a AUC de todos os falsos contra
todos os humanos.

| representação | EER dev | AUC dev | AUC ensaios | AUC 2019 LA |
|---|---|---|---|---|
| nosso detector (`fusion_lcnn_v4_asv5`) | 20,75% | 0,830 | 0,983 | 0,657 |
| sonda no LFCC (controle) | 19,54% | 0,848 | 0,999 | 0,632 |
| XLS-R, convolução de entrada | 42,25% | 0,597 | 0,877 | 0,610 |
| camada 0 | 44,21% | 0,581 | 0,974 | 0,518 |
| camada 2 | 34,71% | 0,701 | 0,998 | 0,609 |
| camada 3 | 21,50% | 0,813 | 1,000 | 0,748 |
| camada 4 | 12,96% | 0,943 | 1,000 | 0,885 |
| camada 5 | 2,75% | 0,997 | 1,000 | 0,955 |
| camada 6 | 0,21% | 1,000 | 1,000 | 0,993 |
| **camada 7 (a melhor pela regra)** | **0,21%** | **1,000** | **0,950** | **0,998** |
| camada 8 | 0,50% | 0,999 | 0,770 | 0,996 |
| camada 11 | 0,46% | 1,000 | 0,264 | 0,992 |
| camada 18 | 0,50% | 1,000 | 0,136 | 0,984 |
| camada 24 | 0,50% | 1,000 | 0,785 | 0,992 |

**Por ataque, no dev (AUC contra os humanos):**

| ataque | nosso detector | sonda no LFCC | camada 7 |
|---|---|---|---|
| A09 | 0,995 | 0,885 | 1,000 |
| A10 | 0,994 | 0,891 | 1,000 |
| A11 | 0,999 | 0,989 | 1,000 |
| **A12** | **0,113** | **0,138** | **1,000** |
| A13 | 0,730 | 0,954 | 1,000 |
| A14 | 0,999 | 0,999 | 1,000 |
| A15 | 0,912 | 0,940 | 1,000 |
| A16 | 0,898 | 0,990 | 1,000 |

Pelo limiar escolhido no dev, a camada 7 acerta:
- **dev:** os 150 arquivos de cada ataque (A11: 148) e 399 de 400 humanos;
- **2019 LA, outra base:** 400 de 400 humanos e de 69 (A06) a 100 de 100 em cada
  ataque;
- **ensaios:**
  - L01: reais 10/10, XTTS 10/10, Chatterbox 9/10;
  - L02: reais 8/10, XTTS 10/10, Chatterbox 4/10.

**Leitura das sondas:**
1. **A informação que generaliza não está no espectro local.** A convolução de
   entrada do XLS-R, que vê 25 ms de cada vez, separa menos que o LFCC. A
   separação aparece da camada 3 à 6 do transformer, as que relacionam os quadros
   no tempo. Nas camadas 6 e 7, o EER nos ataques não vistos cai para 0,21%,
   contra 20,75% do nosso detector, com o mesmo tipo de classificador do controle
   no LFCC.
2. **A camada 7 pega o A12 (AUC 1,000),** que o nosso detector e o LFCC
   invertem (0,113 e 0,138). Nas medidas simples, o A12 é quase igual aos humanos
   (|d| ≤ 0,33 em todas; `outputs/ssl/explicar.json`).
3. **As camadas profundas invertem nos ensaios** (camadas 10 a 18: AUC de 0,136 a
   0,521). Elas aprendem o canal das bases e acham as gravações do nosso
   microfone mais "falsas" que os clones. As do meio (5 a 7) não sofrem disso.
4. **Ressalva sobre os ensaios:** ali, até a sonda no LFCC dá AUC 0,999, porque
   real e clone diferem também no canal (microfone contra áudio limpo). Os
   ensaios sozinhos não provam generalização; o dev e a 2019 provam mais.

**Explicar (`outputs/ssl/explicar.json`):** as medidas simples explicam pouco do
score da camada 7: R² de 0,25 no dev e na 2019. A sonda treinada só com elas
separa pouco: AUC 0,666 no dev e 0,690 na 2019. Nos ensaios ela inverte (AUC
0,000), porque o silêncio e a duração mudam com o canal (os clones do XTTS
terminam com 417 ms de zeros; o microfone tem ruído de fundo). O que a camada 7
usa não é energia por faixa, inclinação, planura, HNR, linhas, pente nem
silêncio.

**Retirada de faixas na camada 7 (`outputs/ssl/faixas.json`;** 30 arquivos por
classe do dev e os 60 dos ensaios). Variação do score ao tirar a faixa, nos
falsos menos nos humanos do mesmo conjunto:

| faixa | A09 | A12 | A13 | A14 | A16 | XTTS L01 | XTTS L02 | Chatterbox L01 | Chatterbox L02 |
|---|---|---|---|---|---|---|---|---|---|
| 0–500 Hz | −3,65 | −1,63 | −1,51 | −3,01 | −3,61 | −1,86 | −2,38 | −2,34 | −1,41 |
| 1–2 kHz | −1,15 | −0,78 | −1,05 | −2,01 | −1,70 | −0,31 | −0,24 | −0,13 | +0,10 |
| 3–4 kHz | −0,17 | −0,06 | +0,01 | −0,46 | −0,27 | +0,21 | +0,13 | +0,37 | +0,26 |
| 7–8 kHz | +0,19 | +0,03 | +0,00 | +0,01 | −0,02 | +0,10 | −0,04 | +0,20 | −0,10 |

O comportamento é o oposto do nosso detector:
- **Efeito de tirar faixas:** no nosso detector, faltar faixa deixa tudo "mais
  falso", até os humanos (parte 1). Na camada 7, faltar faixa deixa os falsos
  "mais humanos", e nos humanos o score quase não muda (de −0,38 a +0,82 no dev).
- **Onde está a pista:** a da camada 7 está em **0–500 Hz**, onde ficam a altura
  da voz (F0) e os primeiros harmônicos, e depois em 1–2 kHz; a do nosso detector
  estava em 2–5 kHz.

Hipótese, junto com a leitura das sondas: o XLS-R olha como a altura e a
vozeação evoluem no tempo, algo que o espectro médio não mostra.

**Teste do tempo (`estudo_ssl.py tempo`, `outputs/ssl/tempo.json`; 30 arquivos
por classe do dev e da 2019, os 60 dos ensaios).** Os quadros de 20 ms da
convolução são reordenados antes do transformer, e a mesma sonda da camada 7
mede o resultado. Como a sonda é linear, o score do arquivo é a média dos scores
dos quadros, o que permite decompô-lo por tipo de quadro:
- pausa: energia abaixo do percentil 20 da janela;
- fala: acima do percentil 60;
- transição: os 20% de maior variação de energia.

| conjunto | original | embaralhado | embaralhado em 100 ms | invertido | quadros de pausa | de fala | de transição |
|---|---|---|---|---|---|---|---|
| dev ASVspoof 5 | 1,000 | 0,389 | 0,797 | 0,804 | 0,982 | 1,000 | 0,999 |
| 2019 LA | 0,997 | 0,542 | 0,883 | 0,835 | 0,951 | 0,999 | 0,987 |
| ensaios | 0,950 | 0,855 | 1,000 | 0,999 | 0,924 | 0,831 | 0,781 |

**Leitura:**
1. **Nas bases, a pista da camada 7 está na ordem dos quadros no tempo.**
   Embaralhar todos os quadros derruba a AUC para 0,389 e 0,542. Embaralhar só
   dentro de blocos de 100 ms ou inverter o tempo também derruba (0,797 a 0,883).
   Conta como cada trecho de 20 ms se encadeia com os vizinhos e com a frase toda.
2. **Nos ensaios, a ordem quase não importa** (0,855 a 1,000). Isso confirma que
   ali a separação vem do canal, não dessa pista.
3. **Nas bases, a evidência está sobretudo nos quadros de fala** (1,000 e
   0,999); um pouco menos nas pausas (0,982 e 0,951).

Ressalva: embaralhar também tira a entrada do que o XLS-R viu no
pré-treino, para as duas classes. A queda da AUC mostra que a ordem natural é
necessária para separar, mas a sonda não foi retreinada com os quadros
embaralhados.

**Síntese da parte 6:**
- O XLS-R separa ataques que nunca viu pela **dinâmica da voz no tempo**,
  concentrada na banda baixa (altura e vozeação em 0–500 Hz, formantes em 1–2
  kHz), e não pelo espectro de cada trecho.
- O nosso detector faz o contrário: olha a "cor" do espectro em 2–5 kHz, em
  pedaços de 25 ms, com pouca relação entre eles no tempo. Por isso cai em
  geradores que imitam bem o espectro médio, como o A12.

### 10.2.23 Parte 7: a linha fixa acompanha a fala? (critério combinado antes)

**Hipótese (escrita na parte 5, antes do teste):** a linha do gerador faz parte
do som sintetizado e some nas pausas; a do ambiente é constante e aparece mais
nas pausas. Se valesse, separaria "tom de máquina que fala" de "zumbido da sala".

**Método (`scripts/estudo_acompanha.py`; `outputs/estudo_acompanha_*.json`):**
espectrograma com janela de 0,25 s e passo de 50 ms. O excesso da linha é a
potência nela sobre a mediana dos vizinhos a 16–48 Hz. Fala: quadros acima do
percentil 60 de energia; pausa: abaixo do percentil 20. Diferença = excesso
mediano na fala − nas pausas.

**Critério e linhas, fixados no commit antes de rodar:** a linha "acompanha a
fala" se a diferença é positiva em pelo menos 70% dos arquivos e "fica nas
pausas" se é negativa em pelo menos 70%. As linhas são as das partes 3 e 5.

**Resultado** (excesso mediano em dB; arquivos com diferença positiva):

| grupo | linha | fala | pausa | diferença | dif. > 0 | veredito |
|---|---|---|---|---|---|---|
| real L01 | 527 Hz | +3,23 | +2,66 | +0,58 | 5/10 | indefinida |
| real L01 | 655 Hz | +3,12 | +4,90 | −1,00 | 2/10 | fica nas pausas |
| real L02 | 527 Hz | +3,03 | +3,92 | −0,93 | 3/10 | fica nas pausas |
| real L02 | 655 Hz | +0,16 | +3,03 | −3,19 | 1/10 | fica nas pausas |
| XTTS L01 | 6.078 Hz | +3,97 | +3,92 | +0,12 | 6/10 | indefinida |
| XTTS L01 | 7.014 Hz | +3,14 | +4,12 | −1,20 | 2/10 | fica nas pausas |
| XTTS L01 | 8.139 Hz | +4,77 | +2,73 | +2,22 | 9/10 | acompanha a fala |
| XTTS L02 | 6.078 Hz | +4,66 | +2,73 | +2,10 | 9/10 | acompanha a fala |
| XTTS L02 | 7.014 Hz | +3,82 | +3,64 | +0,30 | 6/10 | indefinida |
| XTTS L02 | 8.139 Hz | +2,88 | +1,15 | +1,77 | 9/10 | acompanha a fala |
| Chatterbox L01 | 6.550 Hz | +4,56 | +5,58 | −0,62 | 3/10 | fica nas pausas |
| Chatterbox L01 | 7.450 Hz | +2,86 | +5,92 | −2,92 | 0/10 | fica nas pausas |
| Chatterbox L01 | 7.850 Hz | +2,32 | +3,56 | −1,32 | 4/10 | indefinida |
| Chatterbox L02 | 6.550 Hz | +3,63 | +5,82 | −2,16 | 0/10 | fica nas pausas |
| Chatterbox L02 | 7.450 Hz | +5,25 | +8,34 | −3,14 | 0/10 | fica nas pausas |
| Chatterbox L02 | 7.850 Hz | +2,04 | +4,50 | −2,32 | 1/10 | fica nas pausas |
| A14 (dev, 100) | 4.000 Hz | +9,02 | +12,41 | −3,99 | 12/100 | fica nas pausas |
| A14 | 7.562 Hz | +10,45 | +11,24 | −0,83 | 37/100 | indefinida |
| A14 | 7.875 Hz | +11,66 | +10,36 | +0,95 | 67/100 | indefinida |
| A13 (dev, 100) | 2.756 Hz | +4,41 | +6,52 | −2,57 | 16/100 | fica nas pausas |
| A13 | 5.512 Hz | +6,63 | +9,44 | −2,83 | 18/100 | fica nas pausas |
| A13 | 7.732 Hz | +11,16 | +14,39 | −3,19 | 9/100 | fica nas pausas |
| A07 (treino, 100) | 7.000 Hz | +4,20 | +9,46 | −5,41 | 7/100 | fica nas pausas |

**Hipótese rejeitada.** Das 19 linhas de gerador, só 3 acompanham a fala, todas
do XTTS; 11 ficam nas pausas, como as do microfone. O tom do gerador se comporta
como um zumbido constante do vocoder, que não depende do que está sendo dito: a
pausa de um áudio sintético não é silêncio, é o vocoder ligado. A regra "linha
que acompanha a fala" não separa o gerador da sala.

**Consequência para o detector:** nas 11 linhas que ficam nas pausas, o excesso
nas pausas vai de +4,1 a +14,4 dB. O pré-processamento do treino corta o
silêncio das pontas, e o analisador ao vivo pesa as janelas pela fração de fala:
as duas escolhas afastam o modelo justamente das partes onde a marca do gerador
é mais forte. Testar o efeito exige outro treino e fica como proposta.

### 10.2.24 Parte 8: a dinâmica da voz com medidas nossas (critério combinado antes)

**Pergunta:** a pista temporal do XLS-R (parte 6) aparece em medidas de dinâmica
que a gente calcula e entende, sem o XLS-R?

**Método (`scripts/estudo_dinamica.py`; `outputs/dinamica/`).** São 38 medidas
por arquivo:
- **contorno da altura (pYIN, 10 ms):** desvio, variação quadro a quadro,
  microvariação em 50 ms, suavidade, declínio;
- **vozeação:** fração, trocas por segundo, duração dos trechos, incerteza;
- **trajetórias de energia em 0–0,5, 0,5–1, 1–2 e 2–4 kHz:** fluxo, autocorrelação
  a 10 e a 50 ms, modulação de 4 a 16 Hz;
- **centroide da banda baixa;**
- **acoplamento altura-energia;**
- **H1−H2 no tempo.**

Os arquivos, o classificador e as réguas são os da parte 6. Extração: 6.660
arquivos, nenhum pulado.

**Critério (no commit antes de rodar):** ganho de pelo menos 0,05 de AUC sobre a
sonda no LFCC, no dev da ASVspoof 5 e na 2019 LA.

| representação | EER dev | AUC dev | AUC ensaios | AUC 2019 LA | A12 (dev) |
|---|---|---|---|---|---|
| LFCC (controle) | 19,54% | 0,848 | 0,999 | 0,632 | 0,14 |
| dinâmica | 39,25% | 0,630 | 0,521 | 0,589 | 0,27 |
| LFCC + dinâmica | 20,00% | 0,837 | 0,998 | 0,655 | 0,13 |

**Não passou:**
- dinâmica: −0,218 no dev e −0,043 na 2019;
- LFCC + dinâmica: −0,012 e +0,023.

As medidas que mais pesam são as autocorrelações e os fluxos de energia em 0,5–4
kHz, não as da altura da voz.

**Leitura:** a pista do XLS-R não se reduz a resumos globais da altura, da
vozeação ou da energia por faixa. Ela parece estar no encadeamento fino entre os
quadros, que um modelo precisa aprender. Fica em aberto se estatísticas
temporais dos quadros locais bastam ou se é preciso um modelo de sequência
(parte 9).

### 10.2.25 Parte 9: estatísticas temporais dos quadros locais (critério combinado antes)

**Pergunta:** trocar o pooling de média e desvio do nosso modelo por
estatísticas temporais bastaria? Ou é preciso um modelo de sequência?

**Método (`scripts/estudo_estatisticas_tempo.py`; `outputs/estat_tempo/`).** Para
cada dimensão dos quadros locais, calcula média, desvio, variação quadro a
quadro e autocorrelação. Dois tipos de quadro:
- convolução do XLS-R: 512 dimensões, 20 ms, atrasos de 20 a 160 ms;
- LFCC do detector: 20 coeficientes, 10 ms, atrasos de 10 a 160 ms.

Os arquivos, a sonda e as réguas são os da parte 6. Conferência: a média da
convolução é igual à da parte 6.

**Critério (no commit antes de rodar):** ganho de pelo menos 0,05 de AUC sobre a
média e o desvio das mesmas características, no dev e na 2019.

| representação | dimensões | EER dev | AUC dev | AUC ensaios | AUC 2019 LA | A12 |
|---|---|---|---|---|---|---|
| convolução do XLS-R, média | 512 | 42,25% | 0,597 | 0,877 | 0,610 | 0,339 |
| convolução, média e desvio | 1.024 | 33,50% | 0,709 | 0,931 | 0,634 | 0,230 |
| convolução, com estatísticas temporais | 3.584 | 36,04% | 0,658 | 0,995 | 0,597 | 0,059 |
| LFCC com deltas, média e desvio (o controle) | 120 | 19,54% | 0,848 | 0,999 | 0,632 | 0,138 |
| LFCC estático, média e desvio | 40 | 25,00% | 0,839 | 0,875 | 0,601 | 0,482 |
| LFCC estático, com estatísticas temporais | 160 | 19,79% | 0,831 | 0,999 | 0,741 | 0,100 |

**Não passou:**
- convolução: −0,051 no dev e −0,037 na 2019;
- LFCC: −0,007 no dev e +0,140 na 2019.

As estatísticas temporais ajudam o LFCC na 2019, mas não no dev, e nenhuma
representação pega o A12.

**Leitura das partes 8 e 9 juntas:**
- O que as camadas 3 a 7 do XLS-R fazem não se reduz a resumos da dinâmica:
  nem medidas feitas à mão (parte 8), nem autocorrelações dos quadros locais.
- Precisa ser aprendido por um modelo de contexto. Trocar o pooling do nosso
  modelo não bastaria.
- Essas camadas foram aprendidas sem nenhum áudio falso, só prevendo trechos
  escondidos de fala humana. A parte 10 testa se esse princípio explica a
  separação.

### 10.2.26 Parte 10: a surpresa do modelo de fala natural (critério combinado antes)

**Pergunta:** o XLS-R nunca viu um áudio falso. No pré-treino, ele esconde
trechos de fala humana e tenta reconhecer o verdadeiro entre distratores. A fala
sintética torna essa tarefa diferente?

**Método (`scripts/estudo_surpresa.py`; `outputs/surpresa/`):**
- **máscaras** como no pré-treino: início em 7,5% dos quadros, 10 quadros cada,
  6 repetições com sementes fixas por janela;
- **medidas, por quadro escondido:**
  - perda contrastiva contra 100 distratores do mesmo áudio;
  - acerto (o verdadeiro em 1º);
  - similaridade com o verdadeiro;
- **amostra:** 80 arquivos por classe da parte 6, ou seja, 1.340 no total;
- **conferência:** a perda calculada aqui é igual à do próprio modelo (maior
  diferença 2,4·10⁻⁷).

**Critério (no commit antes de rodar):** |AUC − 0,5| ≥ 0,2 na perda, no dev e na
2019, no mesmo sentido.

| conjunto | AUC da perda | perda mediana dos humanos | por classe (AUC) |
|---|---|---|---|
| dev ASVspoof 5 | 0,886 | 1,75 | A09 0,71; A10 0,73; A11 0,86; **A12 0,99**; A13 0,91; A14 1,00; A15 0,91; A16 0,97 |
| 2019 LA | 0,766 | 2,03 | A01 0,76; A02 0,68; A03 0,50; A04 0,87; A05 0,97; A06 0,81 |
| ensaios | **0,000** | **3,18** | os quatro grupos de clones 0,00 (medianas de 1,78 a 2,16) |

**Passou:** dev 0,886 e 2019 0,766, no mesmo sentido. A fala sintética
surpreende mais o modelo de fala natural; as outras duas medidas também mostram
isso (acerto: AUC 0,126 e 0,243; similaridade: 0,040 e 0,182). É uma pista **de
classe única**: vem de um modelo que só viu fala humana. Ela pega o A12, que o
nosso detector inverte.

**Mas nos ensaios ela se inverte por completo.** As gravações do nosso microfone
(ruído, estalos, sala) surpreendem mais (mediana 3,18) que os clones (1,78 a
2,16) e que os humanos das bases (1,75 a 2,03). Para um modelo de fala natural, o
canal ruim parece mais "estranho" que um clone limpo.

**Combinação com o nosso detector (`estudo_surpresa.py combinar`,
`outputs/surpresa/combinar.json`).** Pesos iguais, fixados antes; cada score vira
z pelos humanos do dev, e os dois são somados.

| conjunto | detector | surpresa | soma |
|---|---|---|---|
| dev ASVspoof 5 | 0,835 | 0,886 | 0,983 (A12 0,09 → 0,89; A13 0,76 → 1,00) |
| 2019 LA (teste) | 0,682 | 0,766 | 0,849 (A04 0,48 → 0,79; A05 0,36 → 0,83; A06 0,39 → 0,68) |
| ensaios | 0,983 | 0,000 | 0,168 |

**Leitura:**
1. As duas pistas se completam. O detector pega o que a surpresa pega menos
   (A09–A11), e a surpresa pega o que o detector inverte (A12, A04–A06).
2. O princípio "um modelo de fala natural estranha a fala sintética" explica
   parte do que o XLS-R faz e não depende de conhecer os geradores.
3. Para o uso real (microfone, chamada), o canal decide. Um modelo de
   naturalidade só serve se tiver aprendido que o canal varia, por exemplo com as
   degradações do nosso treino (RawBoost, codecs, captura).

Proposta para o modelo próprio: anotada no plano como decisão pendente.

### 10.2.27 Parte 10b: o canal na surpresa (critério combinado antes)

**Pergunta:** nos ensaios, a surpresa se inverte. Isso acontece porque a medida é
frágil a qualquer canal, ou porque real e clone passaram por canais diferentes
(microfone contra arquivo limpo)?

**Método (`estudo_surpresa.py canal`, `outputs/surpresa/canal.json`).** O mesmo
canal é aplicado às duas classes, em 30 arquivos por classe do dev e da 2019 (480
no total), com as mesmas máscaras em todas as condições:
- ruído branco a 20 dB;
- Opus;
- **o ruído do nosso microfone a 20 dB:** só as pausas das 20 gravações reais
  dos ensaios (88,2 s, quadros até 6 dB acima do percentil 20 de energia, em
  trechos de 100 ms ou mais), com os estalos inclusos.

**Critério (no commit antes de rodar):** o problema é o descasamento se a AUC da
perda no dev cair no máximo 0,10 em relação ao limpo em pelo menos 2 das 3
condições.

| condição (nas duas classes) | AUC dev | perda mediana, humanos / falsos (dev) | AUC 2019 |
|---|---|---|---|
| limpo | 0,883 | 1,74 / 2,59 | 0,767 |
| ruído branco a 20 dB | 0,817 | 2,28 / 2,89 | 0,752 |
| Opus | 0,852 | 1,78 / 2,58 | 0,779 |
| ruído do nosso microfone a 20 dB | 0,830 | 1,98 / 2,80 | 0,799 |

**Passou nas três:** quedas de 0,066, 0,032 e 0,053 no dev; na 2019, a AUC fica
entre 0,752 e 0,799.

**Leitura:** com o mesmo canal nas duas classes, a ordem entre real e falso se
mantém, mesmo com o ruído e os estalos do nosso microfone. A inversão dos ensaios
vem do descasamento de canal, não da medida em si. Mas o canal sobe a surpresa
de todos, e por isso um limiar absoluto não serviria.

Num ataque real, o clone também passa pelo canal (telefone, alto-falante,
microfone). Esse é o caso "pelo ar", em que o nosso detector não separa (10.2.12).
Ele deve ser o primeiro teste se a proposta do modelo de naturalidade seguir.
Ressalva para esse teste: no ensaio 1, os clones pelo ar tocaram ~8 dB mais
baixo, e menos sinal sobre o mesmo ruído também sobe a surpresa; o nível precisa
ser igualado antes de gravar.

### 10.2.28 A marca d'água Perth do Chatterbox (exploratório)

**Por quê (26/10/2026):** o Chatterbox põe em todo áudio gerado a marca Perth
(Resemble AI), verificável com o pacote `perth`. Duas perguntas:
1. Ela está nos nossos clones?
2. As linhas fixas atribuídas ao vocoder dele (6.550, 7.450 e 7.850 Hz, 10.2.19 e
   10.2.20) poderiam ser a marca, e não o vocoder?

**Método (`scripts/estudo_perth.py`, no ambiente `chatterbox`;
`outputs/estudo_perth.json`).** A marca é detectada em todos os grupos. Depois,
ela é aplicada aos reais e aos clones do XTTS (que não a têm), e as linhas são
contadas antes e depois, com a regra da parte 3. Os áudios marcados ficam numa
pasta temporária, apagada no fim.

| grupo | com a marca | 6.550 Hz | 7.450 Hz | 7.850 Hz |
|---|---|---|---|---|
| real L01 / L02 | 0/10 / 0/10 | 0/10 / 1/10 | 2/10 / 0/10 | 1/10 / 1/10 |
| real L01 / L02, com a marca aplicada | 10/10 / 10/10 | 0/10 / 1/10 | 2/10 / 0/10 | 1/10 / 1/10 |
| XTTS L01 / L02 | 0/10 / 0/10 | 2/10 / 3/10 | 1/10 / 1/10 | 0/10 / 0/10 |
| XTTS L01 / L02, com a marca aplicada | 10/10 / 10/10 | 2/10 / 3/10 | 1/10 / 1/10 | 0/10 / 0/10 |
| Chatterbox L01 / L02 | **10/10 / 10/10** | 8/10 / 10/10 | 9/10 / 10/10 | 10/10 / 10/10 |

**Leitura:**
1. **A marca está em todos os 20 clones do Chatterbox e em nenhum dos 40
   outros áudios.** Os clones que escapam do nosso detector (ensaio 4) carregam
   um selo de origem verificável. Mas o Chatterbox é de código aberto (MIT), e
   quem quiser enganar pode desligar a marca. Ela só pega quem não se esforça e
   não serve como detector geral.
2. **As linhas fixas não são a marca.** Aplicada aos outros áudios, a marca não
   cria nenhuma das três linhas (as contagens não mudam). A atribuição ao
   vocoder (10.2.19, 10.2.20) se mantém.

### 10.2.29 Piloto de destilação: um aluno pequeno imitando a camada 7 (critério combinado antes)

**Por quê (26/10/2026):** a camada 7 do XLS-R separa ataques que ele nunca viu
(10.2.22), mas o grupo não quer o XLS-R como detector, e as medidas feitas à mão
não chegam lá (10.2.24 e 10.2.25). Na destilação, o modelo grande entra como
professor, só no treino. O conhecimento das 436 mil horas chega embutido nas
respostas dele sobre os nossos áudios, e o detector final é um aluno pequeno,
nosso, que roda sem ele. Decidido pelo grupo em 26/10/2026 como piloto; o treino
completo ainda passa pelo orientador.

**Método (`scripts/destilacao.py`; dados em `outputs/destilacao/`):**
- **Professor:** a camada 7 do XLS-R 300M congelado, quadro a quadro (20 ms),
  nas mesmas janelas de 4 s e nos mesmos arquivos da parte 6:
  - treino: 4.000 arquivos do recorte da ASVspoof 5;
  - dev: 1.600;
  - 2019 LA: 1.000;
  - ensaios: 60.
- **Aluno:** LFCC com deltas e log-mel (140 por quadro de 10 ms, normalizados na
  janela); uma convolução de passo 2 (para 20 ms); um transformer de 4 camadas
  (dimensão 256); uma saída de 1.024 dimensões.
- **Perda:** L1 mais −log σ(cosseno) por quadro, como no DistilHuBERT, sobre os
  alvos padronizados.
- **Rótulos:** não são usados no treino do aluno.
- **Escolha do checkpoint:** pela perda de destilação no dev, sem rótulo.
- **Avaliação:** uma sonda linear treinada nas respostas do aluno (média no
  tempo), com o mesmo treino e as mesmas réguas da parte 6. A sonda do professor
  aplicada ao aluno sai só como informação.

**Critério (no commit antes de rodar):** o piloto passa se a sonda no aluno
superar a sonda no LFCC (0,848 no dev, 0,632 na 2019) em pelo menos 0,05 de AUC
no dev (média dos ataques) e na 2019 LA.

**Régua extra, pedida pelo grupo (26/10/2026): o eval da ASVspoof 5.**
- **Amostra:** a já usada em 10.9 (`data/ASVspoof5_recorte/eval`), com 10 mil
  arquivos: 2.036 humanos e cerca de 500 de cada ataque A17–A32, com o codec de
  cada um em `data/recortes/asvspoof5_eval_amostra10000_codec_s42.tsv`.
- **Uso:** só medida, depois de tudo escolhido. A camada (a 7), os limiares
  (ponto de EER do dev) e o checkpoint do aluno (perda no dev) vêm do dev.
- **Leituras:** o XLS-R camada por camada, o LFCC, o nosso detector e o aluno,
  por ataque e por codec. É a busca de um padrão que sobreviva aos ataques e
  codecs que nenhum dos modelos viu.

**Treino (26/10/2026, rodado pela sessão a pedido do usuário, que estava longe
da máquina).** A primeira tentativa, em precisão mista, deu NaN desde o primeiro
lote. A convolução de passo 2 em fp16 gera NaN nesta GTX 1650; o treino passou
para fp32 (commit `8faee3a`).
- **Aluno:** 3,78 milhões de parâmetros, 30 épocas de cerca de 85 s.
- **Perda no dev:** 1,2615 na primeira época e 1,0354 na melhor (época 29,
  escolhida pela perda, sem rótulo).
- **Arquivos:** `checkpoints/aluno_xlsr_l7.pt` e `.json`.

**Resultado (`outputs/destilacao/avaliacao.json`):**

| representação | EER dev | AUC dev | AUC 2019 LA | AUC ensaios |
|---|---|---|---|---|
| sonda no LFCC (a régua do critério) | 19,54% | 0,848 | 0,632 | 0,999 |
| nosso detector, mesmos arquivos (10.2.22) | 20,75% | 0,830 | 0,657 | 0,983 |
| **aluno, sonda nova** | 18,46% | **0,872** | **0,800** | 0,955 |
| aluno, sonda do professor (informação) | 16,75% | 0,911 | 0,808 | 0,944 |
| professor, camada 7 (10.2.22) | 0,21% | 1,000 | 0,998 | 0,950 |

**Por ataque (aluno, sonda nova):**
- dev: A09 0,999; A10 0,996; A11 0,987; **A12 0,361**; A13 0,727; A14 0,998;
  A15 0,912; A16 0,996;
- 2019: A01 0,99; A02 0,99; A03 0,98; A04 0,47; A05 0,88; A06 0,50;
- ensaios, certos pelo limiar do dev:
  - L01: reais 10/10, XTTS 0/10, Chatterbox 2/10;
  - L02: reais 10/10, XTTS 10/10, Chatterbox 8/10.

**Critério: não passou.** O ganho sobre o LFCC foi de +0,024 no dev, abaixo de
0,05, e de +0,168 na 2019.

**Leitura:**
- **O aluno pegou parte do que o professor sabe.** Na 2019, uma base que ele
  nunca viu, a AUC sobe de 0,632 para 0,800, e ele supera o nosso detector nas
  duas réguas. O treino foi de só 4.523 janelas.
- **O que falta é o principal:** o A12 (0,36 contra 1,00 do professor) e o A04 e
  o A06 da 2019, perto do acaso. São os geradores que imitam o espectro médio, e
  o aluno ainda não aprendeu o contexto que os denuncia.
- **A perda ainda caía no fim** (1,0373 na época 20, 1,0354 na 29). Mais dados
  de destilação e um aluno maior são os próximos passos naturais; cada um seria
  um novo piloto, com critério próprio.

**Expectativa para o eval da ASVspoof 5, registrada antes de ver os números
(26/10/2026):** o aluno acima do nosso detector no geral e abaixo dele em parte
dos codecs de banda estreita. A destilação usou só áudio limpo, sem as
degradações de canal do treino. Chegar perto do professor é improvável.

**Resultado no eval da ASVspoof 5 (26/10/2026).**
- **Amostra:** 9.979 arquivos (2.036 humanos; A17–A32).
- **Medida:** só medida. A camada 7, os limiares (ponto de EER do dev) e o
  checkpoint do aluno vieram do dev.
- **Arquivos:** `outputs/ssl/sondas_com_eval.json`,
  `outputs/ssl/detector_eval.json`, `outputs/ssl/codecs_eval.json` e
  `outputs/destilacao/avaliacao_com_eval.json`.
- **Entradas do aluno:** preparadas sem rodar o professor
  (`preparar --sem-professor`, 20.371 janelas).

| ataque | nosso detector | sonda no LFCC | aluno (sonda nova) | professor (camada 7) |
|---|---|---|---|---|
| **geral** | **0,659** | **0,724** | **0,828** | **0,971** |
| A17 | 0,97 | 0,88 | 0,91 | 0,97 |
| A18 | 0,47 | 0,75 | 0,83 | 0,94 |
| A19 | 0,28 | 0,31 | 0,60 | 0,99 |
| A20 | 0,28 | 0,51 | 0,61 | 0,99 |
| A21 | 0,99 | 0,78 | 0,97 | 1,00 |
| A22 | 0,96 | 0,84 | 0,92 | 0,99 |
| A23 | 0,86 | 0,80 | 0,93 | 0,98 |
| A24 | 0,96 | 0,93 | 0,89 | 0,97 |
| A25 | 0,89 | 0,68 | 0,81 | 0,99 |
| A26 | 0,78 | 0,79 | 0,87 | 0,99 |
| A27 | 0,21 | 0,69 | 0,81 | 0,98 |
| A28 | 0,85 | 0,65 | 0,78 | 0,91 |
| A29 | 1,00 | 0,92 | 0,98 | 0,98 |
| A30 | 0,18 | 0,64 | 0,76 | 0,93 |
| A31 | 0,50 | 0,73 | 0,75 | 0,94 |
| A32 | 0,28 | 0,62 | 0,81 | 0,99 |

Humanos abaixo do limiar do dev: detector 1.430, aluno 1.483 e professor 1.726,
de 2.036. A sonda do professor aplicada ao aluno dá AUC geral de 0,861.

**Por codec** (AUC dos falsos contra os humanos do mesmo codec):

| codec | detector | LFCC | aluno | professor |
|---|---|---|---|---|
| sem codec | 0,681 | 0,845 | 0,898 | 0,999 |
| Opus banda larga | 0,690 | 0,730 | 0,870 | 0,994 |
| AMR banda larga | 0,650 | 0,718 | 0,818 | 0,988 |
| Speex banda larga | 0,657 | 0,721 | 0,805 | 0,994 |
| EnCodec (neural) | 0,635 | 0,673 | 0,831 | 0,901 |
| MP3 | 0,692 | 0,752 | 0,897 | 0,999 |
| M4A | 0,656 | 0,712 | 0,832 | 0,996 |
| MP3 + EnCodec | 0,623 | 0,706 | 0,824 | 0,881 |
| Opus banda estreita | 0,632 | 0,729 | 0,824 | 0,958 |
| AMR banda estreita | 0,636 | 0,658 | 0,750 | 0,988 |
| Speex banda estreita | 0,608 | 0,668 | 0,755 | 0,984 |
| aparelhos (Bluetooth, cabo) | 0,645 | 0,707 | 0,822 | 0,998 |

**Leitura:**
1. **No teste mais difícil, o aluno supera o nosso detector com folga:** 0,828
   contra 0,659. Os sete ataques em que o detector se inverte (A18, A19, A20,
   A27, A30, A31 e A32, de 0,18 a 0,50) ficam de 0,60 a 0,83 no aluno: nenhum
   abaixo do acaso.
2. **O aluno ganha do detector em todos os 12 codecs.** A expectativa
   registrada acertou no geral e errou nos codecs de banda estreita, para o lado
   pessimista. O pior codec do aluno (AMR banda estreita, 0,750) ainda fica acima
   do melhor do detector (MP3, 0,692).
3. **O detector ainda é melhor em alguns ataques** (A17, A21, A22, A24, A25, A28 e
   A29). As duas pistas se completam, como na parte 10, e a soma é um próximo
   passo natural.
4. **O professor mostra o teto:** 0,971 no geral. Só o EnCodec o derruba um pouco
   (0,901; MP3 + EnCodec, 0,881).
5. **Ressalva:** o critério do piloto (dev e 2019) continua não atingido. O eval
   é uma régua extra, pedida pelo grupo, que não muda esse veredito.

### 10.2.30 O aluno somado ao detector (critério combinado antes)

**Pergunta, do grupo (26/10/2026):** o detector ainda ganha do aluno em alguns
ataques do eval (A17, A21, A24, A25, A28). Somados, os dois ajudam mesmo?

**Método (`scripts/combinar_detector_aluno.py`; `outputs/destilacao/combinacao.json`):**
- **Scores, arquivo a arquivo:**
  - do detector (`fusion_lcnn_v4_asv5`, caminho da avaliação);
  - do aluno (sonda nova).
- **Conjuntos:** o dev, a 2019, os ensaios e o eval da ASVspoof 5, os mesmos
  arquivos de 10.2.29.
- **Principal, fixado antes:** pesos iguais, cada score em z pelos humanos do dev.
- **Secundário, exploratório:** uma regressão logística de dois pesos ajustada no
  dev, medida só fora dele.
- **Limiares:** ponto de EER do dev de cada score.

**Critério (no commit antes de rodar):** a soma de pesos iguais ajuda se superar
o melhor dos dois sozinhos em pelo menos 0,02 de AUC no eval da ASVspoof 5 e na
2019 LA, sem perder mais de 0,02 nos ensaios.

**Resultado (26/10/2026; 1.600, 1.000, 60 e 9.979 arquivos com os dois scores):**

| conjunto | detector | aluno | soma (pesos iguais) | logística (ajustada no dev) |
|---|---|---|---|---|
| dev ASVspoof 5 | 0,830 | 0,872 | 0,852 | 0,877 (dentro da amostra) |
| 2019 LA | 0,657 | 0,800 | 0,763 | 0,805 |
| ensaios | 0,983 | 0,955 | 0,995 | 0,939 |
| eval ASVspoof 5 | 0,659 | 0,828 | 0,781 | 0,828 |

**Não passou:** a soma ficou −0,048 no eval e −0,038 na 2019 em relação ao melhor
sozinho, o aluno. Nos ensaios ganhou +0,012.

**Leitura:**
- **O detector traz mais erro do que pista.** Nos ataques em que ele se inverte,
  a soma puxa o aluno para baixo: A19 0,60 → 0,43, A20 0,61 → 0,44, A27 0,81 →
  0,58, A30 0,76 → 0,49, A32 0,81 → 0,61.
- **Onde o detector era melhor, a soma só empata com ele:** A17 0,96, A21 0,99,
  A24 0,95, A25 0,89, A28 0,85.
- **Por codec, a soma fica abaixo do aluno em todos os 12.**
- **A logística ajustada no dev confirma:** os pesos são −0,09 para o detector e
  0,80 para o aluno, ou seja, ela praticamente descarta o detector e repete o
  aluno.
- **Nos ensaios, a soma separa melhor no geral (AUC 0,995), mas pelo limiar do
  dev deixa passar mais clones do L01** (XTTS 3/10, Chatterbox 2/10) que o
  detector sozinho (7/10 e 6/10). Esses ensaios usam o caminho da avaliação
  (4 s desde o início), não o do placar.
- **Conclusão:** nas réguas gerais, o aluno sozinho é o melhor; somar o
  detector atual não compensa. Ele só agrega no canal dos nossos ensaios, e até
  isso pede cuidado com o limiar.

### 10.2.31 Piloto 2 de destilação: mais dados e o canal (critério combinado antes)

**Por quê (26/10/2026):** no piloto 1, a perda ainda caía no fim do treino, e o
aluno, que só viu áudio limpo, deixa passar quase todos os clones do L01 pelo
microfone (10.2.30). O piloto 2 ataca as duas coisas.

**Método (`scripts/destilacao.py`; dados em `outputs/destilacao/asv5_train2/`).**
O que muda em relação ao piloto 1:
- **Dados:** 20 mil arquivos do treino do recorte, 6.000 humanos e 1.750 de cada
  ataque A01–A08 (semente 42). Eles contêm os 4 mil do piloto 1.
- **Canal:** cada janela ganha 2 cópias degradadas das entradas do aluno, além
  da limpa.
  - A degradação é a do `configs/fusion_v4_asv5.yaml` (RawBoost, captura FIR,
    codecs e corte da banda alta), mais ruído de 10 a 30 dB com chance de 0,5.
  - Em cada época, o aluno vê uma das 3 versões, sorteada, e tenta dar a resposta
    que o professor deu sobre a janela **limpa**.
- **Alvos comprimidos:** PCA de 1.024 para 256 dimensões, ajustada nos quadros do
  treino do piloto 1 (87,9% da variância).
  - A PCA não tira a separação: a sonda do professor nas representações
    projetadas dá 1,000 no dev e 0,987 na 2019, contra 1,000 e 0,998 sem PCA.
  - Com 128, 384 e 512 dimensões: 1,000/0,992, 0,999/0,993 e 0,998/0,992.
- **Mesmos do piloto 1:** a arquitetura do aluno, a perda e a escolha do
  checkpoint pela perda no dev limpo, sem rótulo.
- **Treino:** 20 épocas.

**Critério (no commit antes de rodar):** o mesmo do piloto 1. A sonda no aluno 2
precisa superar a sonda no LFCC (0,848 no dev, 0,632 na 2019) em pelo menos 0,05
de AUC no dev e na 2019. A comparação com o aluno 1 no eval da ASVspoof 5 e nos
ensaios é informação.

**Execução (26/10/2026, pela sessão):**
- **Preparo:** 20.000 arquivos e 22.924 janelas, das 13h05 às 14h24.
- **Treino:** 20 épocas de cerca de 155 s. A perda no dev foi de 1,139 na época
  1 a 0,978 na época 20, ainda caindo.
- **Aluno:** 3,59 milhões de parâmetros; `checkpoints/aluno2_xlsr_l7.pt`.
- **Avaliação:** `outputs/destilacao/avaliacao_piloto2.json`.

| representação | EER dev | AUC dev | AUC 2019 LA | AUC eval ASVspoof 5 | AUC ensaios |
|---|---|---|---|---|---|
| sonda no LFCC | 19,54% | 0,848 | 0,632 | 0,724 | 0,999 |
| nosso detector | 20,75% | 0,830 | 0,657 | 0,659 | 0,983 |
| aluno 1 (10.2.29) | 18,46% | 0,872 | 0,800 | 0,828 | 0,955 |
| **aluno 2, sonda nova** | **10,75%** | **0,961** | **0,819** | **0,879** | **0,506** |
| aluno 2, sonda do professor | 7,54% | 0,977 | 0,854 | 0,926 | 0,371 |
| professor, camada 7 | 0,21% | 1,000 | 0,998 | 0,971 | 0,950 |

**Critério: passou.** Ganho sobre o LFCC de +0,112 no dev e de +0,187 na 2019.

**Por ataque e por codec (aluno 2, sonda nova):**
- **dev:** A09 0,99; A10 0,99; A11 1,00; **A12 0,92** (aluno 1: 0,36); A13
  0,90; A14 0,99; A15 0,92; A16 0,98;
- **2019:** A01 0,98; A02 0,99; A03 0,97; A04 0,65; A05 0,80; A06 0,52;
  humanos abaixo do limiar: 324/400 (aluno 1: 222);
- **eval, por codec:**
  - sem codec 0,91;
  - Opus banda larga 0,90; AMR banda larga 0,88; Speex banda larga 0,84;
  - EnCodec 0,90; MP3 0,89; M4A 0,91; MP3 + EnCodec 0,87;
  - Opus banda estreita 0,84; AMR banda estreita 0,82; Speex banda estreita 0,79;
  - aparelhos 0,88.

**Mas, nos ensaios, o aluno 2 não separa (AUC 0,506).** Pelo limiar do dev:

| locutor | reais certos | XTTS pegos | Chatterbox pegos |
|---|---|---|---|
| L01 | 4/10 | 1/10 | 2/10 |
| L02 | 9/10 | 7/10 | 2/10 |

Os scores de reais e clones ficam no mesmo lugar: as medianas vão de −3,04 a
−1,81, todas do lado humano.

**Leitura:**
1. **Mais dados e o canal no treino resolvem a generalização entre bases,
   ataques e codecs.** O aluno 2 passa no critério, pega o A12 e fica acima do
   aluno 1 em todos os codecs do eval.
2. **O aluno 2 aprendeu a ignorar o canal, e nos ensaios isso tira a separação.**
   Ali, real e clone diferem sobretudo no canal (microfone contra arquivo limpo).
   Por isso o LFCC dava 0,999 e embaralhar os quadros não mudava a camada 7
   (10.2.22). Separado do canal, o que sobra não distingue os nossos clones do
   XTTS e do Chatterbox da nossa voz gravada.
3. **O teste justo é real e clone pelo mesmo canal** (o clone pelo ar, ou o canal
   igualado). Até lá, o resultado dos ensaios não mede a pista do sintético.

### 10.2.32 Ensaios com o canal igualado (exploratório, sem critério)

**Pergunta:** com o ruído do nosso microfone também nos clones, os alunos e o
detector separam a voz real dos clones dos ensaios?

**Método (`scripts/ensaio_canal_igualado.py`;
`outputs/destilacao/ensaio_canal_igualado.json`):**
- **Clones:** os 40 dos ensaios 2 a 4, lidos a 16 kHz pelo FIR, mais o ruído
  das pausas das 20 gravações reais (como em 10.2.27). SNRs de 15, 20 e 25 dB, a
  faixa de ruído medida nas gravações reais (10.2.12).
- **Reais:** as mesmas 20 gravações, sem mudança.
- **Medidas:** aluno 1, aluno 2 e o nosso detector, com os limiares do dev.

Não é o clone pelo ar: faltam o alto-falante e a sala. Os áudios com ruído ficam
numa pasta temporária, apagada no fim.

**Resultado (26/10/2026).** AUC e, pelo limiar do dev, quantos de 10 ficam do
lado certo:

| SNR do ruído nos clones | detector | aluno 1 | aluno 2 |
|---|---|---|---|
| 15 dB | 0,954 | 0,796 | 0,345 |
| 20 dB | 0,970 | 0,877 | 0,374 |
| 25 dB | 0,975 | 0,910 | 0,422 |
| sem ruído (10.2.31) | 0,983 | 0,955 | 0,506 |

| a 20 dB, do lado certo | detector | aluno 1 | aluno 2 |
|---|---|---|---|
| reais L01 / L02 | 10 / 10 | 10 / 10 | 4 / 9 |
| XTTS L01 / L02 | 7 / 6 | 0 / 10 | 0 / 3 |
| Chatterbox L01 / L02 | 3 / 4 | 0 / 5 | 0 / 1 |

**Leitura:**
1. **O ruído não explica a queda do aluno 2.** Com o ruído do microfone nos
   clones, ele fica invertido (0,345 a 0,422) e continua acusando 6 das 10
   vozes reais do L01.
2. **O que o aluno 2 estranha são as nossas gravações reais.** Isso bate com a
   surpresa da parte 10: para o modelo de fala natural, a voz gravada pelo
   microfone embutido (mediana 3,18) é mais estranha que os clones (1,78 a 2,16).
   O microfone tem processamento interno e estalos a cada 110 ms (10.2.16), e o
   ruído das pausas não reproduz isso. O aluno 2, que imita melhor esse modelo,
   herdou a estranheza. (10.2.34: o que ele estranha são as cópias da captura;
   sem elas, aceita 10 de 10 vozes reais do L01.)
3. **O detector atual segue separando os ensaios (0,954 a 0,975)**, mesmo com o
   ruído nos clones. Ele foi treinado com a captura simulada (10.7) e não depende
   só do ruído.
4. **O próximo teste decisivo é gravar a voz real com outro microfone,** já
   pendente em 10.2.16. Se o aluno 2 aceitar como real a voz gravada sem o
   processamento do microfone embutido, o problema é o aparelho, não a pista.

### 10.2.33 Outro microfone: o headset com fio (exploratório)

**O que foi feito (26/10/2026):** o L01 gravou 5 das 10 frases do ensaio 2 com
um headset com fio, ligado na entrada do notebook. O gravador e o driver são os
mesmos; o microfone muda sozinho. Ao ouvido, o L01 notou que as gravações do
headset também "travam".

**Medidas do caminho de captura (`scripts/estudo_estalos.py`,
`outputs/estudo_estalos.json`):**

| grupo | arquivos | pente de 9,09 Hz (dB) | fundo nas pausas (dBFS) | saltos > 15 dB em 10 ms (por s) |
|---|---|---|---|---|
| embutido L01 | 10 | +2,03 | −66,8 | 0,38 |
| embutido L02 | 10 | +1,86 | −71,3 | 0,09 |
| **headset L01** | 5 | **−0,00** | **−86,7** | **0,63** |
| clones XTTS L01 | 10 | −0,02 | −150,0 (silêncio digital) | 0,86 |
| clones Chatterbox L01 | 10 | +0,00 | −77,4 | 0,60 |

No espectrograma, a fala do headset tem bordas retas e pausas quase escuras.
Não há perda de amostras nem pacote repetido nas gravações nativas.

**Os detectores com a voz real do headset**
(`scripts/ensaio_canal_igualado.py --sem-ruido`, mesmos clones do L01; arquivos
`outputs/destilacao/microfone_*_L01.json`). AUC e, pelo limiar do dev, quantos
reais ficam do lado certo:

| L01 real pelo | detector | aluno 1 | aluno 2 |
|---|---|---|---|
| microfone embutido | 0,975 (10/10) | 0,940 (10/10) | 0,310 (4/10) |
| **headset** | **0,000 (0/5)** | **0,200 (4/5)** | **0,170 (1/5)** |

**Leitura:**
1. **Os estalos a cada 110 ms vêm do caminho do microfone embutido.** O pente
   some no headset, com o mesmo driver e o mesmo gravador. Isso responde, em
   parte, a dúvida de 10.2.16: não é o driver nem o gravador.
2. **O caminho do headset aplica uma redução de ruído com portão.** O fundo nas
   pausas fica 20 dB mais baixo que no embutido, e há mais saltos bruscos de
   nível. É provavelmente o "travado" que se ouve: o ruído some e volta de uma
   vez a cada palavra.
3. **Com esse processamento, a voz real parece sintética para todos os
   detectores.** Até o detector atual, que separa o embutido (0,975), põe as 5
   frases do headset acima de todos os clones (AUC 0,000). Pausas limpas demais e
   bordas retas lembram o áudio gerado.
4. **Para a segurança na prática, isso é um alerta:** redução de ruído (de
   chamadas, aplicativos e drivers) pode fazer voz real ser acusada de falsa. Os
   próximos testes precisam do caminho sem esse processamento: desligar a redução
   de ruído da entrada do headset (Windows e Realtek) e regravar, ou usar um
   gravador de celular sem redução de ruído.

**Corrigido em 10.2.34:** a captura do headset também tinha pacotes repetidos
(29%); a frase "não há perda de amostras nem pacote repetido" estava errada. As
leituras 1 a 3 mudam: o pente sumiu porque o gravador trocou os blocos de 100 ms
pelos pacotes do driver (as cópias passaram a ser de 20 ms), não por causa do
microfone; e não há portão de ruído, e sim nível muito baixo no modo exclusivo
(fala em −57 dBFS, pausas no arredondamento de 16 bits). Sem as cópias, o
detector dá 0,660 para o headset (antes 0,000).

### 10.2.34 Defeito na captura: o modo exclusivo insere cópias (diagnóstico)

**Por quê (26/10/2026):** o L01 ouviu as gravações do headset "travando", como
as do microfone embutido. O pedido foi procurar defeito no próprio gravador.

**Método (`scripts/diagnostico_captura.py`; `outputs/diagnostico_captura.json`).**
Em cada gravação, os trechos de 10 ms (480 amostras a 48 kHz) idênticos bit a
bit a um trecho anterior. Num sinal de microfone, isso não acontece por acaso;
silêncio digital não conta. Depois, capturas de 4 s do ruído da sala (nenhum
som tocado, nada salvo) em cada configuração do `sounddevice` (PortAudio
V19.7.0-devel), no "Grupo de microfones (Realtek(R) Audio)".

**Todas as gravações reais têm cópias; nenhum clone tem:**

| pasta | trechos de 10 ms repetidos | distância da cópia |
|---|---|---|
| ensaio 1, reais do L01 | 3.958/12.000 (33,0%) | 11 trechos (110 ms) |
| ensaio 2, reais do L01 | 3.814/12.000 (31,8%) | 110 ms |
| reais do L02 | 3.619/12.600 (28,7%) | 110 ms |
| referências da clonagem, L01 / L02 | 32,4% / 27,9% | 110 ms |
| clones pelo ar do ensaio 1 | 3.665/11.318 (32,4%) | 110 ms |
| headset do L01 | 1.758/6.000 (29,3%) | 2 trechos (20 ms) |
| clones do XTTS (L01 e L02, ensaios 1 a 3) | 0 | — |
| clones do Chatterbox, L01 / L02 | 98/5.776 / 19/5.916 | 1 a 5 trechos, só no fim do arquivo |
| celular do L01 (gravador do aparelho) | 0/10.307 | — |

Nos clones do Chatterbox, as repetições são o enchimento do fim do arquivo: as
amostras valem no máximo 1 degrau de 16 bits. Não é o mesmo defeito.

**A causa é o modo exclusivo.** Ruído da sala, 4 s por configuração:

| configuração | trechos repetidos | amostras por segundo |
|---|---|---|
| exclusivo, 1 canal, pacotes do driver (o gravador) | 194/592 (32,8%), a 20 ms | ~71 mil |
| exclusivo, blocos de 100 ms (o gravador antigo) | 187/580 (32,2%), a 110 ms | — |
| exclusivo, 2 canais | 192/590 (32,5%) | — |
| exclusivo, latência de 20 a 60 ms | 29,1% a 30,8% das janelas de 64 amostras | 69 a 72 mil |
| exclusivo, latência baixa (11,3 e 14,7 ms) | nenhum trecho inteiro, mas 22,1% a 23,5% das janelas de 64 amostras | 63 a 65 mil |
| compartilhado, 1 ou 2 canais | 0 (nenhuma janela de 64 amostras) | 47.966 e 48.013 |

No exclusivo, com latência de 20 ms ou mais, o PortAudio chama a captura em
rajadas: a mediana entre duas chamadas é 0,01 a 0,05 ms, e as pausas têm o
tamanho da latência. Cada chamada traz 10 ms. Ele lê o buffer do driver mais vezes do que o driver o enche, e a leitura a
mais devolve um trecho antigo. Com a latência baixa, a cópia só deixa de cair
alinhada nos trechos de 10 ms.

**As cópias são inseridas, nada se perde.** Na fala:
- na emenda de um trecho novo com a cópia, o salto da amostra é 14 vezes (L01),
  4 vezes (L02) e 19 vezes (headset) o do meio do trecho;
- na saída da cópia, ele é normal (0,95 a 1,02), e os trechos novos seguem em
  ordem.

Apagar as cópias devolve o sinal contínuo (`diagnostico_captura.py reparar`;
cópias em `data/vozes/reparadas/`, fora do Git). As frases do ensaio 2 vão de
12,00 s para 8,19 s em média: a voz gravada ficava 1,47 vez mais longa, a
"câmera lenta" de 10.2.16. O tom de 1.000 Hz não via isso, porque 10 ms são 10
ciclos exatos.

**Outros defeitos do exclusivo neste notebook.** Ruído da sala, nos dois modos:
- o compartilhado fica +40,0 dB acima do exclusivo, igual em todas as faixas
  (39,2 a 40,1 dB, de 50 Hz a 23 kHz). É o ganho do Windows, que o exclusivo não
  aplica, sem mudar a forma do espectro;
- o exclusivo entrega 16 bits (pedido em int32, os valores vêm em múltiplos de
  2¹⁶). O compartilhado tem resolução abaixo de 1/100 do degrau de 16 bits.

**O headset não tem portão de ruído.** A fala dele ficou em −57,2 dBFS (pico
−40,8). Nas pausas, as amostras valem 0 a 3 degraus de 16 bits (23% são zero):
é o arredondamento, que o pico normalizado do modelo amplia. A leitura 2 de
10.2.33 estava errada.

**Medidas do caminho, sem as cópias** (`scripts/estudo_estalos.py`,
`outputs/estudo_estalos_reparadas.json`):

| grupo | pente de 9,09 Hz (dB) | fundo nas pausas (dBFS) | saltos > 15 dB (por s) |
|---|---|---|---|
| embutido L01, com cópias | +2,03 | −66,8 | 0,38 |
| embutido L01, sem cópias | −0,00 | −66,7 | 0,10 |
| embutido L02, sem cópias | −0,03 | −71,2 | 0,03 |
| headset L01, sem cópias | −0,01 | −86,8 | 0,61 |
| celular L01 | −0,02 | −56,2 | 0,45 |

O pente de 9,09 Hz (10.2.18, 10.2.20) era o eco das cópias de 110 ms e some sem
elas.

**Os detectores com a voz real sem o defeito** (`ensaio_canal_igualado.py
--sem-ruido`, os 20 clones do L01 do XTTS e do Chatterbox;
`outputs/destilacao/microfone_*_L01.json`). AUC e, pelo limiar do dev, quantos
reais ficam do lado certo:

| L01 real | detector | aluno 1 | aluno 2 |
|---|---|---|---|
| embutido, com cópias (10.2.33) | 0,975 (10/10) | 0,940 (10/10) | 0,310 (4/10) |
| embutido, sem cópias | 0,935 (10/10) | 0,880 (10/10) | 0,655 (10/10) |
| headset, com cópias (10.2.33) | 0,000 (0/5) | 0,200 (4/5) | 0,170 (1/5) |
| headset, sem cópias | 0,660 (4/5) | 0,410 (4/5) | 0,720 (5/5) |
| celular (10 frases do ensaio 2) | 0,870 (9/10) | 0,810 (10/10) | 0,545 (9/10) |

Os clones pegos são os mesmos nos três casos: detector 7/10 (XTTS) e 6/10
(Chatterbox), aluno 1 0/10 e 2/10, aluno 2 1/10 e 2/10.

**Placar dos ensaios 2 a 4 com os reais sem as cópias** (mesmo protocolo:
`fusion_lcnn_v4_asv5`, limiar −8,671, `--taxa-comum 48000`;
`outputs/teste_voz_propria_reparado_*.json`): reais 10/10 nos dois locutores;
clones pegos L01 XTTS 9/10, L01 Chatterbox 3/10, L02 XTTS 10/10, L02 Chatterbox
9/10, os mesmos de antes. A média dos reais sobe pouco: L01 −13,27 → −13,03, L02
−12,86 → −12,68 nos log-odds.

**A surpresa do XLS-R (10.2.26) sem as cópias** (`surpresa_arquivo`, mesmas
máscaras; `outputs/surpresa/reparadas.json`). Perda mediana e AUC dos 40 clones
dos ensaios contra cada grupo real (abaixo de 0,5: o real surpreende mais, a
inversão de 10.2.26):

| voz real | perda mediana | AUC dos clones |
|---|---|---|
| L01 embutido, com / sem cópias | 3,41 / 2,55 | 0,000 / 0,020 |
| L02 embutido, com / sem cópias | 3,18 / 2,12 | 0,000 / 0,285 |
| L01 headset, sem cópias | 2,34 | 0,115 |
| L01 celular | 2,12 | 0,343 |

Clones: XTTS L01 2,12, Chatterbox L01 2,03, XTTS L02 1,85, Chatterbox L02 1,85.
As cópias explicam parte da inversão, não toda: até o celular surpreende mais
que os clones. O resto fica com o canal (sala e microfone contra arquivo
limpo), como em 10.2.27.

**Leitura:**
1. **O defeito era do nosso caminho de captura (PortAudio em WASAPI exclusivo
   com este driver), não do microfone.** Ele está em todas as gravações reais,
   nas referências da clonagem, nos clones pelo ar e no ao vivo pelo microfone
   (mesma fonte). Os clones do XTTS e do Chatterbox foram feitos a partir de
   referências com cópias.
2. **Os placares dos ensaios se mantêm.** Com os reais recuperados, nenhum
   veredito muda. Os resultados ficam como saíram, com o defeito declarado.
3. **O "estranhamento" do aluno 2 era o defeito.** Sem as cópias, ele aceita as
   10 vozes reais do L01 e deixa de inverter (0,310 → 0,655). Mas segue fraco
   contra estes clones: pelo limiar do dev, pega 3 dos 20.
4. **O headset parecia sintético pelas cópias de 20 ms e pelo nível baixo.** Sem
   as cópias, o detector sobe de 0,000 para 0,660; o resto fica com o nível de
   −57 dBFS, perto do arredondamento de 16 bits.
5. **O modo compartilhado não tem nenhum dos dois defeitos** neste notebook
   (taxa certa, sem cópias, resolução maior e o ganho do Windows). O plano pede
   o exclusivo; a troca é decisão do grupo.

**Decisão e checagem (26/10/2026).** O grupo trocou o padrão para o
compartilhado: gravador (`gravar_frases.py`), monitor e aba Ao vivo. O exclusivo
ficou como `--exclusivo`, só para diagnóstico. As checagens depois de gravar
ganharam a de cópias (`checar_copias` em `src/capture/checagens.py`): falha com
qualquer trecho de 10 ms idêntico a um anterior. Trechos de até 1 degrau de 16
bits não contam, para o enchimento do Chatterbox não acusar. Conferida em todas
as pastas de `data/vozes/`:

| pastas | arquivos acusados | trechos repetidos |
|---|---|---|
| reais do embutido: ensaio 1, ensaio 2, L02 | 10/10 em cada | 3.958, 3.814, 3.619 |
| referências da clonagem, L01 / L02 | 2/2 / 2/2 | 5.225 / 4.879 |
| clones pelo ar do ensaio 1 | 10/10 | 3.665 |
| headset do L01 | 5/5 | 1.758 |
| clones do XTTS (4 pastas) e do Chatterbox (2) | 0 | 0 |
| celular do L01 | 0/10 | 0 |
| reais recuperados (L01, L02, headset) | 0 | 0 |

**Nível no compartilhado.** No exclusivo, a fala do embutido tinha pico mediano
de −25,1 dBFS no L01 (máximo −23,1) e −29,4 no L02 (−26,5); o headset, −41,1
(−35,6). Com os +40 dB do compartilhado medidos acima, a fala do embutido
passaria de 0 dBFS. O volume do microfone no Windows precisa baixar antes de
gravar, conferido com uma frase de teste (protocolo: pico entre −12 e −3 dBFS).

### 10.2.35 Ensaio 5: os dois caminhos com a captura corrigida (critério combinado antes)

**Por quê (26/10/2026):** todo o material de voz real dos ensaios 1 a 4, as
referências da clonagem e os clones pelo ar saíram com as cópias do modo
exclusivo (10.2.34). O caminho pelo microfone nunca foi medido sem elas. Com o
modo compartilhado e o nível acertado (frase de teste: pico −9,9 e −8,2 dBFS,
0 de 1.200 trechos repetidos, ruído 23 a 25 dB abaixo da fala), o L01 regrava
tudo.

**Critério (fixado em 26/10/2026, antes de gravar):**
- **Modelo e limiar:** os dos ensaios 2 a 4, sem mudança: `fusion_lcnn_v4_asv5`
  sozinho, −8,671 nos log-odds (`outputs/limiar_fusao/limiar_asv5_treino.json`),
  média das janelas ponderada pela fala, `--taxa-comum 48000`.
- **Material:** L01, microfone embutido em modo compartilhado, com o nível da
  frase de teste, sem fone e sem carregador, mesma sala. Referência nova (texto
  de referência, 90 s, preparada pelo `preparar_referencia.py`). Dez frases
  novas para todos (`docs/etica/frases_ensaio5.txt`, 21 e 22 palavras: com a
  captura certa, as de 17 a 19 palavras davam menos de 4 s de fala em 4 das 10
  frases recuperadas do L01). Regras de gravação dos ensaios (`--piso-minimo
  15`, 12 s, regravar só pelas checagens).
- **Clones:** XTTS v2 e Chatterbox Multilingual locais, da referência nova,
  sementes 1 a 10, sem o ponto final. Clone que travar, cortar palavra ou não
  disser a frase inteira e certa é gerado de novo com outra semente antes do
  placar, e fica registrado.
- **Clone pelo ar:** cada clone tocado pelo alto-falante do notebook
  (aprimoramentos da saída desligados) e gravado pelo mesmo microfone. O volume
  da saída é acertado no primeiro clone para o pico gravado ficar a até 3 dB do
  pico típico das frases reais, e fica o mesmo para os 20 (no ensaio 1, os
  clones tocaram ~8 dB mais baixo).
- **Caminhos e meta:** arquivo (real × clone) e microfone (real × clone pelo
  ar), para cada ferramenta. Pelo menos 8 de 10 reais julgados humanos e 8 de
  10 clones julgados sintéticos em cada caminho. Arquivo sem janela útil conta
  como erro.
- **Sem espiar:** nenhum score deste material antes do placar.
- **Leitura combinada antes:**
  - Pelo ar, se passar: a falha do ensaio 1 (2 de 10) vinha em boa parte da
    captura. Se falhar: o caminho pelo ar segue como limitação medida, como na
    proposta da demonstração.
  - Pelo arquivo, a comparação é com os ensaios 2 e 4 do L01 (XTTS 9 de 10,
    Chatterbox 3 de 10). A referência nova e a gravação real nova mudam juntas,
    e o ensaio não separa as duas.
- **Limitações declaradas antes:** um locutor, cuja voz e scores já foram
  vistos; uma sala; o limiar escolhido com o risco declarado em 10.2.13.

**Durante a geração (26/10/2026, nenhum score visto):** o L01 ouviu os clones do
Chatterbox com sotaque de Portugal. É o mesmo modelo do ensaio 4
(`chatterbox-tts` 0.1.7, multilíngue t3 v2, idioma `pt`), que não tem escolha de
variante; a Resemble publica ajustes separados para o português do Brasil e o
de Portugal. Pelo critério, sotaque não é motivo para gerar de novo: os clones
ficam, e o sotaque entra como limitação (um ouvinte brasileiro estranharia o
clone; o detector não sabe de sotaque).

**Primeira tentativa do clone pelo ar, descartada (26/10/2026, nenhum score
visto):** os clones do XTTS tocaram no fone de ouvido, não no alto-falante. O
pico gravado ficou entre −24,9 e −49,6 dBFS, contra −6,9 a −13,6 dBFS das frases
reais, e as checagens de fala e de ruído falharam. Os arquivos salvos saíram da
pasta do ensaio antes de refazer; a gravação recomeça pelo alto-falante, sem
fone. O ruído nas pausas das frases reais (−44,5 dBFS) é o da
frase de teste (−42,4): o mesmo microfone.

**Segunda tentativa, descartada, e um ajuste no critério (26/10/2026, nenhum
score visto):** pelo alto-falante, o pico mediano dos clones do XTTS ficou em
−14,4 dBFS, contra −9,75 das frases reais: 4,7 dB abaixo, fora dos 3 dB do
critério. A fala ficou só 7,3 dB acima do ruído no quadro mediano (11,4 nas
reais), e a checagem achou menos de 4 s de fala em 9 dos 10. O ruído durante o
clone é o da sala (−54,8 dBFS antes de tocar, −55,0 no quadro p10): é nível, não
ruído do alto-falante. Ajuste antes de regravar: os arquivos do XTTS têm pico em
0 dBFS e os do Chatterbox de −2,8 a −12,4, e um volume só não serviria para os
20. O `gravar_frases.py pelo-ar` passa a tocar cada clone com o pico em −1 dBFS;
o resto do critério não muda.

**Por onde tocar (26/10/2026, nenhum score visto).** O Windows já estava em
100%: o alto-falante do notebook não chega a 3 dB da voz real (no ensaio 1, no
exclusivo, também ficou ~8 dB abaixo). A troca depende só do equipamento, não
de resultado, e fica combinada assim:
- **Com uma caixinha de som externa** (preferida): ela é a saída do Windows,
  fica na frente do notebook, a ~25 cm do microfone, onde fica a boca de quem
  fala. O volume é acertado no primeiro clone, e o resto do critério vale como
  está. É também o cenário do ataque: alguém tocando o clone perto do microfone.
- **Sem caixinha:** alto-falante do notebook a 100%, e a diferença de nível é
  declarada. A checagem de fala dos clones pelo ar fica só descritiva. As outras
  checagens e a regra "arquivo sem janela útil conta como erro" continuam.

Sem caixinha: alto-falante do notebook a 100%, tampa aberta. Pico mediano dos
clones pelo ar: XTTS −15,4 dBFS, Chatterbox −15,8, contra −9,8 das frases reais.
A checagem de fala falhou em todos os 20 (descritiva); o f09 do Chatterbox falhou
no piso de ruído (14 dB) e foi regravado (16 dB) antes do placar.

**Placar (26/10/2026;** `outputs/ensaio5_L01_xtts.json` e
`outputs/ensaio5_L01_chatterbox.json`**):**

| caminho | reais julgados humanos | clones pegos | meta |
|---|---|---|---|
| arquivo, XTTS | 10/10 | 7/10 | não atingida |
| arquivo, Chatterbox | 10/10 | 0/10 | não atingida |
| microfone (pelo ar), XTTS | 10/10 | 0/10 | não atingida |
| microfone (pelo ar), Chatterbox | 10/10 | 0/10 | não atingida |

Descritivo, nos log-odds (limiar −8,671):

| grupo | média (mín. a máx.) | AUC contra os reais | acima do real da mesma frase |
|---|---|---|---|
| reais | −13,84 (−15,75 a −12,55) | — | — |
| XTTS, arquivo | −7,84 (−10,27 a −5,16) | 1,000 | 10/10 |
| XTTS, pelo ar | −13,21 (−14,74 a −11,56) | 0,690 | 8/10 |
| Chatterbox, arquivo | −13,36 (−16,31 a −10,77) | 0,560 | 6/10 |
| Chatterbox, pelo ar | −16,75 (−20,08 a −15,30) | 0,010 | 0/10 |

Contra os ensaios com a referência antiga (com cópias), mesmo limiar:

| clones do L01 | média dos clones | pegos |
|---|---|---|
| XTTS, ensaio 2 → ensaio 5 | −7,30 → −7,84 | 9/10 → 7/10 |
| Chatterbox, ensaio 4 → ensaio 5 | −9,85 → −13,36 | 3/10 → 0/10 |

Os reais do L01 foram de −13,27 (ensaio 2, com cópias) para −13,84.

**Leitura:**
1. **A meta não foi atingida em nenhum caminho nem ferramenta.** As vozes reais
   ficam todas do lado certo; o que falha são os clones.
2. **Pelo ar, a limitação se confirma sem o defeito da captura.** Nenhum clone é
   pego. O do XTTS ainda fica um pouco acima do real (AUC 0,690); o do Chatterbox
   fica abaixo de todas as vozes reais (0,010): pelo alto-falante, ele parece
   mais humano que a voz real. Os clones chegaram 5,6 dB abaixo da voz, o que
   está declarado.
3. **Pelo arquivo, o XTTS mantém a ordem e perde o ponto de corte.** Todos os
   clones ficam acima de todas as vozes reais (AUC 1,000), mas 3 caem abaixo do
   limiar. É o mesmo padrão dos ensaios 2 a 4.
4. **O Chatterbox com a referência nova não se separa nem na ordem** (AUC
   0,560): a média dos clones caiu 3,5 nos log-odds e ficou no nível da voz real.
5. **A queda dos clones vem da referência ou das frases, não das gravações
   reais.** Com o limiar fixo, o veredito de um clone só depende dele. Este ensaio
   não separa referência e frases. Uma hipótese a testar: os clones herdavam
   da referência com cópias algo que o detector pegava. Se for isso, os acertos
   dos ensaios 2 a 4, todos com referências com cópias, eram em parte o defeito.
   Eles ficam como saíram, com essa ressalva.

### 10.2.36 A referência com cópias explica a queda do ensaio 5? (exploratório, leitura combinada antes)

**Pergunta (26/10/2026, antes de qualquer score):** a queda dos clones pegos
entre os ensaios 2 e 4 e o ensaio 5 (10.2.35) vem da referência ou das frases?

**Desenho:** as mesmas 10 frases do conjunto 5, sementes 1 a 10, sem o ponto
final, nas duas ferramentas, clonadas de três referências do L01:
- **antiga:** a dos ensaios 1 a 4, com as cópias da captura
  (`teste_proprio/referencia/referencia_envio.wav`);
- **antiga reparada:** a mesma gravação sem as cópias (`diagnostico_captura.py
  reparar` e `preparar_referencia.py`);
- **nova:** a do ensaio 5 (clones já feitos).

Mesmo detector, limiar e caminho do ensaio 5 (`--taxa-comum 48000`). Os clones
não são ouvidos nem refeitos (a geração é automática, durante o jantar do L01);
fica declarado.

**Leitura combinada antes,** pela média dos log-odds dos clones e por quantos
passam do limiar −8,671:
- antiga acima das outras duas, e reparada perto da nova: as cópias da
  referência deixavam os clones mais fáceis de pegar;
- antiga e reparada perto, as duas acima da nova: é a sessão de gravação (sala,
  nível, 16 bits), não as cópias;
- as três perto: a queda veio das frases.

Junto, só descritivo: o aluno 1, o aluno 2 e a sonda da camada 7 do XLS-R
(10.2.22) no material do ensaio 5 e nestes clones.

**Referências (26/10/2026).** A antiga tem 2.310 de 7.120 trechos de 10 ms
repetidos; a reparada (48,4 s) e a nova (55,2 s), nenhum. Placar
(`outputs/referencia_*.json` e os do ensaio 5):

| clones do conjunto 5 | XTTS: média dos log-odds (pegos) | Chatterbox: média (pegos) |
|---|---|---|
| referência antiga, com cópias | −8,57 (6/10) | −12,53 (0/10) |
| antiga reparada | −9,59 (4/10) | −13,30 (0/10) |
| nova (ensaio 5) | −7,84 (7/10) | −13,36 (0/10) |

**Leitura, pela regra combinada:** as três referências ficam perto, então a
queda veio das frases. No XTTS não há padrão (a nova é a mais alta). No
Chatterbox, a referência com cópias sobe 0,8, contra a queda de 3,5 entre os
ensaios 4 e 5: as cópias explicam pouco. "Frases" inclui o que mudou junto com
elas: as sementes, o ponto final e, no ensaio 2, a escuta com regeração. As
cópias da referência não explicam os acertos dos ensaios 2 a 4. Descritivo: com
a referência antiga, os clones do XTTS falam mais devagar (f09: 13,1 s contra
9,5 s com a reparada).

**Os candidatos no mesmo material (descritivo;** `outputs/destilacao/ensaio5.json`,
`outputs/ssl/sondas_ensaio5.json`**).** AUC contra as 10 vozes reais e, pelo
limiar do dev de cada um, quantos clones são pegos. Todos aceitam as 10 vozes
reais. "Detector (interface)" é o caminho do placar (janelas, 48 kHz);
"detector (avaliação)" é o do `evaluate.py`.

| clones | detector (interface) | detector (avaliação) | aluno 1 | aluno 2 | sonda XLS-R, camada 7 |
|---|---|---|---|---|---|
| XTTS, arquivo | 1,000 | 1,00 (8/10) | 0,97 (8/10) | 0,78 (6/10) | 1,00 (9/10) |
| Chatterbox, arquivo | 0,560 | 0,92 (0/10) | 0,63 (0/10) | 0,40 (1/10) | 1,00 (5/10) |
| XTTS, pelo ar | 0,690 | 0,79 (1/10) | 0,32 (0/10) | 0,36 (1/10) | 0,98 (0/10) |
| Chatterbox, pelo ar | 0,010 | 0,48 (0/10) | 0,21 (0/10) | 0,35 (1/10) | 0,88 (0/10) |
| XTTS, ref. antiga / reparada | — | 1,00 / 0,99 | 0,78 / 0,58 | 0,47 / 0,36 | 1,00 (9/10) / 1,00 (8/10) |
| Chatterbox, ref. antiga / reparada | — | 0,84 / 0,81 | 0,67 / 0,63 | 0,46 / 0,71 | 1,00 (7/10) / 1,00 (7/10) |

Sonda, todas as camadas no conjunto: camada 7 0,983, camada 8 1,000; camadas 10 a
18 de 0,799 a 0,935; LFCC 0,776 (a sonda do LFCC acusa as 10 vozes reais).

**Leitura (descritiva):**
1. **A sonda da camada 7 põe todos os clones acima de todas as vozes reais pelo
   arquivo, nas duas ferramentas e nas três referências (AUC 1,00).** Ela foi
   treinada só no recorte da ASVspoof 5, sem as nossas vozes, o nosso microfone
   ou estas ferramentas. É o Chatterbox que o detector não separa (0,560).
2. **Pelo ar, a ordem quase se mantém (0,98 e 0,88), mas o limiar do dev não
   pega nenhum.** O canal desloca os scores; o limiar teria de ser calibrado
   com o nosso canal nas duas classes (a coleta do plano).
3. **Os alunos não reproduzem isso** (AUC geral 0,599 e 0,486): a destilação
   não levou o que a camada 7 vê nestes clones.
4. **A inversão das camadas 10 a 18 nos ensaios (10.2.22) não aparece sem as
   cópias** (0,799 a 0,935).
5. **O detector separa o Chatterbox melhor no caminho da avaliação (0,92) que no
   da interface (0,560),** com os mesmos arquivos. Os caminhos diferem no trecho
   lido e na conversão de taxa; a investigar antes de qualquer conclusão sobre a
   demonstração.

### 10.2.37 Dois diagnósticos no ensaio 5: o trecho que cada caminho lê e o aluno fora do domínio (exploratório)

**1. Por que o detector separa o Chatterbox na avaliação (0,92) e não na
interface (0,560)** (`outputs/caminhos_ensaio5.json`). Os mesmos arquivos, o
`fusion_lcnn_v4_asv5`, AUC contra as 10 vozes reais:

| trecho lido | XTTS | Chatterbox | XTTS pelo ar | Chatterbox pelo ar |
|---|---|---|---|---|
| primeiros 4 s (avaliação; 16 kHz direto ou por 48 kHz, igual) | 1,000 | 0,920 | 0,790 | 0,480 |
| 4 s a partir de 2 s | 0,960 | 0,620 | 0,650 | 0,200 |
| 4 s a partir de 4 s | 1,000 | 0,275 | 0,550 | 0,110 |
| últimos 4 s | 0,870 | 0,200 | 0,350 | 0,030 |
| média das janelas (interface) | 1,000 | 0,560 | 0,690 | 0,010 |
| só a primeira janela da interface | 1,000 | 0,890 | 0,790 | 0,480 |
| interface, janelas pesadas pela fala de verdade (`trechos_de_fala`) | 1,000 | 0,680 | 0,680 | 0,056 |

- **A conversão de taxa não muda nada.** O que muda é o trecho.
- **O Chatterbox parece sintético no começo da frase e humano no fim,** também
  com recortes alinhados ao início da fala nas duas classes (AUC 0,880 no
  início, 0,560 no fim). Por janela, os clones vão de −12,15 a −14,14; as vozes
  reais ficam entre −15,06 e −14,07 nas janelas com fala.
- **A interface trata o ruído da sala como fala.** Com o ruído a menos de 30 dB
  da voz, o corte de silêncio não acha pausa, e as janelas finais das gravações
  reais (só ruído) entram com peso cheio. Elas parecem mais sintéticas (−11,23).
  Pesar pela fala de verdade, como a checagem faz, sobe o Chatterbox de 0,560
  para 0,680.
- **Não é um erro de conta:** os caminhos fazem perguntas diferentes (o começo
  da frase contra a frase inteira). Mudar a regra da interface é decisão de
  calibração, no dev, antes de um ensaio novo.

**2. O aluno imita o professor no nosso domínio?** (`outputs/destilacao/
fidelidade_ensaio5.json`). O professor (camada 7) respondeu sobre os 50 arquivos
do ensaio 5 (184 janelas); perda de destilação de cada aluno, a mesma do treino
(menor é melhor):

| janelas | aluno 1 | aluno 2 |
|---|---|---|
| dev da ASVspoof 5, humanos | 0,985 | 0,922 |
| dev da ASVspoof 5, falsos | 1,051 | 0,995 |
| clones como arquivo (XTTS / Chatterbox) | 0,996 / 1,020 | 0,926 / 0,976 |
| voz real do L01 pelo microfone | 1,094 | 1,064 |
| clones pelo ar (XTTS / Chatterbox) | 1,114 / 1,141 | 1,096 / 1,119 |

- **O aluno imita mal só o áudio que passou pelo nosso microfone.** Nos clones
  em arquivo, em português, a perda é a do dev; não é a língua.
- **A diferença é do tamanho do treino inteiro:** no aluno 2, a perda no dev foi
  de 1,139 na época 1 a 0,978 na 20. Na voz real do L01 (1,064), ele imita como
  no começo do treino.
- **A sonda do professor nas respostas do aluno** dá AUC 0,723 (aluno 1) e
  0,500 (aluno 2) no ensaio 5, contra 0,983 no professor.
- **Leitura:** mais épocas com os mesmos dados melhorariam a ASVspoof (a perda
  ainda caía), mas não o nosso canal. O aluno precisa ver, na destilação, áudio
  que passou por microfone e sala. Como a destilação não usa rótulos, qualquer
  fala gravada assim serve, desde que os locutores dos ensaios fiquem de fora.

### 10.2.38 Piloto 3 de destilação: o canal de sala e microfone (critério combinado antes)

**Por quê (26/10/2026):** os alunos imitam mal a camada 7 justamente no áudio
que passou por microfone e sala (10.2.37); mais épocas com os mesmos dados não
mudam isso. Tocar áudios pelo alto-falante à noite não é viável na casa do L01.
A destilação não usa rótulos: qualquer fala serve, gravada em canal real ou
passada por uma sala real medida.

**Dados de treino do aluno (todos sem rótulo; `scripts/destilacao.py`):**
1. **O conjunto do piloto 2:** 20 mil arquivos do recorte, 22.924 janelas, a
   limpa e 2 degradadas.
2. **Common Voice em português** (CC0, versão 27.0): 4.000 clipes de 1.749
   pessoas, cada um gravado pela própria pessoa, com o microfone que tinha
   (`scripts/montar_amostra_commonvoice.py`, seleção em
   `data/recortes/commonvoice_pt_4000_400_s42.tsv`).
3. **Salas simuladas:** os 4.000 arquivos do piloto 1 passados por uma posição
   sorteada do BUT ReverbDB (resposta ao impulso real e o silêncio gravado pelo
   mesmo microfone, a 10–30 dB; `src/preprocess/sala.py`).

Nos itens 2 e 3, o professor responde sobre o áudio como o aluno o ouve (no
piloto 2, sempre sobre o limpo). Todos os itens recebem as mesmas 2 versões
degradadas e a mesma PCA de 256 dimensões do piloto 2.

**Escolha do checkpoint, sem rótulo:** a média das perdas de destilação em três
devs:
- o dev da ASVspoof 5 (1.600 arquivos);
- o dev do Common Voice (400 clipes de outras 172 pessoas);
- o mesmo dev da ASVspoof passado por salas.

**Aluno e treino:** a mesma arquitetura do piloto 2 (4 camadas, dimensão 256),
do zero, semente 42, 20 épocas. Só os dados mudam.

**Avaliação:** as réguas do piloto 2 (dev, 2019 LA, eval da ASVspoof 5) e o
material do ensaio 5 (50 arquivos do L01, nunca usados no preparo nem na
escolha). Sonda nova treinada nas respostas do aluno, como antes.

**Critério (no commit antes de rodar):** o piloto 3 passa se cumprir as duas
condições:
- **No ensaio 5, fechar pelo menos metade da distância até o professor.**
  - **Arquivo:** AUC da sonda nova ≥ 0,80 com os clones do XTTS e do
    Chatterbox juntos contra a voz real. O aluno 2 dá 0,59 e o professor 1,00.
  - **Pelo ar:** ≥ 0,65 com os clones pelo ar juntos. O aluno 2 dá 0,355 e o
    professor 0,93.
- **Não piorar na ASVspoof.** AUC média no dev ≥ 0,941 e na 2019 LA ≥ 0,799,
  ou seja, no máximo 0,02 abaixo do aluno 2 (0,961 e 0,819).

**Descritivo:** a perda de imitação por grupo no ensaio 5, como em 10.2.37. A
expectativa é a voz real do L01 chegar a 0,05 dos humanos do dev.

**Limitações declaradas antes:**
- o ensaio 5 tem um locutor só e 50 arquivos;
- os microfones do BUT ficam a ~1,4 m de um alto-falante, mais longe que a voz
  do L01 do microfone;
- o Common Voice passa pelo processamento do navegador e pelo MP3;
- nenhum desses é o microfone do notebook da banca.

**Braço 3b, a "versão máxima" (combinado antes, 26/10/2026):** os mesmos dados
e devs, com um aluno maior (6 camadas, dimensão 384) e 30 épocas. Muda o tamanho
e o tempo de treino ao mesmo tempo, de propósito: é a busca do melhor aluno, não
uma medida isolada. O critério é o mesmo. Se os dois braços passarem, fica o de
menor perda média nos três devs, sem olhar o ensaio 5.

**Preparo (26 e 27/10/2026, pela sessão; nenhum treino ainda):**
- **Common Voice:** 4.000 clipes e 4.787 janelas de treino. No dev, 1 dos 400
  clipes tem o cabeçalho maior que o áudio (5,202 s contra 5,146 s) e foi pulado:
  ficam 399 clipes e 503 janelas.
- **BUT ReverbDB:** pacote `RIR-Only` (9,3 GB) baixado com aprovação do L01.
  Foram extraídas só as respostas ao impulso (2.325), 2 silêncios por posição
  (2.046) e as descrições (1.674). Há 1.674 posições em 9 salas.
- **Salas:** 4.000 arquivos e 4.523 janelas de treino; o dev passado por salas
  tem 1.600 arquivos e 1.992 janelas.
- **Degradação** das versões 1 e 2, a do piloto 2: RawBoost 50%, captura FIR
  30%, codec 30%, oclusão 50% e ruído 50%.
- **Réguas a mais na avaliação,** decididas antes do treino:
  - o dev da ASVspoof passado por salas, para os alunos 2, 3 e 3b;
  - a condição `sala` do `robustness_eval.py`, para o detector da demonstração
    e o `fusion_v4` da 2019 (10 mil áudios do eval da 2019).

**Treino do aluno 3 (27/10/2026, pela sessão com o aval do L01):** 3,59 milhões
de parâmetros; 32.234 janelas de treino, cada uma com 3 versões; 20 épocas em 77
minutos (`checkpoints/aluno3_xlsr_l7.pt`). A perda média nos três devs caiu de
1,095 na época 1 a 0,9365 na 20, ainda caindo. Só no dev da ASVspoof, a perda
foi 0,951; a do aluno 2 era 0,978.

**Resultado do piloto 3** (`outputs/destilacao/avaliacao_piloto3.json`; o aluno
2 nas mesmas réguas em `avaliacao_piloto2_reguas5.json`). Sonda nova, AUC:

| régua | aluno 2 | aluno 3 | critério |
|---|---|---|---|
| ensaio 5, arquivo (XTTS e Chatterbox) | 0,590 | 0,750 | ≥ 0,80: **não** |
| ensaio 5, pelo ar | 0,355 | 0,685 | ≥ 0,65: sim |
| dev da ASVspoof 5 (média dos ataques) | 0,961 | 0,917 | ≥ 0,941: **não** |
| 2019 LA | 0,819 | 0,778 | ≥ 0,799: **não** |
| eval da ASVspoof 5 (informação) | 0,879 | 0,857 | — |
| dev da ASVspoof 5 passado por salas (informação) | 0,629 | 0,685 | — |

**Critério: não passou.** No ensaio 5, por grupo: XTTS 1,00, Chatterbox 0,50,
XTTS pelo ar 0,72 e Chatterbox pelo ar 0,65 (aluno 2: 0,78, 0,40, 0,36 e 0,35).
As 10 vozes reais ficam do lado humano nos dois alunos.

**Descritivo:**
- **Imitação** (perda, como em 10.2.37):

  | grupo | aluno 2 | aluno 3 |
  |---|---|---|
  | humanos do dev da ASVspoof | 0,922 | 0,903 |
  | voz real do L01 | 1,064 | 0,982 |
  | clones pelo ar | 1,096 e 1,119 | 0,990 e 1,008 |

  A diferença entre a voz do L01 e os humanos do dev caiu de 0,142 para 0,079.
  A expectativa registrada era ficar dentro de 0,05.
- **A sonda do professor aplicada ao aluno 3** (informação, fora do critério):
  - dev 0,973, 2019 0,871, eval 0,930, salas 0,752;
  - ensaio 5 pelo arquivo 0,660;
  - ensaio 5 pelo ar 0,930, a mesma AUC do professor.
- **Por ataque no dev** (sonda nova): o A12 cai de 0,92 no aluno 2 para 0,66.
- A linha "Critério ... passa" que o `avaliar` imprime é a do piloto 1 (ganho
  sobre o LFCC), não a do piloto 3.

**Treino do aluno 3b (27/10/2026):** 11,43 milhões de parâmetros; 30 épocas de
~546 s (4 h 34 min). A perda média nos devs caiu de 1,051 na época 1 a 0,8600 na
29, a escolhida; a do aluno 3 terminou em 0,9365. Arquivo:
`checkpoints/aluno3b_xlsr_l7.pt`.

**Resultado dos três alunos e do professor** (sonda nova; professor: a sonda da
camada 7, 10.2.22 e 10.2.36). AUC:

| régua | aluno 2 | aluno 3 | **aluno 3b** | professor |
|---|---|---|---|---|
| dev da ASVspoof 5 (média dos ataques) | 0,961 | 0,917 | **0,989** | 1,000 |
| 2019 LA | 0,819 | 0,778 | **0,934** | 0,998 |
| eval da ASVspoof 5 | 0,879 | 0,857 | **0,964** | 0,971 |
| dev passado por salas | 0,629 | 0,685 | **0,782** | — |
| ensaio 5, arquivo | 0,590 | 0,750 | **0,785** | 1,00 |
| ensaio 5, pelo ar | 0,355 | 0,685 | **0,785** | 0,93 |

**Critério do 3b: não passou.** O ensaio 5 pelo arquivo ficou em 0,785, abaixo
de 0,80. As outras três condições passaram: pelo ar 0,785 (≥ 0,65), dev 0,989
(≥ 0,941) e 2019 0,934 (≥ 0,799). Nenhum braço passou; a regra de escolha entre
os dois não se aplica.

**Descritivo do 3b:**
- **Por ataque no dev:** A09 a A16 entre 0,96 e 1,00; o A12 dá 0,99.
- **Eval da ASVspoof 5 por codec:** de 0,94 (EnCodec, AMR e Speex de banda
  estreita) a 0,98.
- **Ensaio 5 por grupo:**
  - XTTS 0,79, Chatterbox 0,78, XTTS pelo ar 0,67, Chatterbox pelo ar 0,90;
  - as 10 vozes reais ficam do lado humano;
  - pelo limiar do dev, só 1 dos 40 clones é pego: o limiar não transfere.
- **Imitação:** humanos do dev 0,820, voz do L01 0,914, clones pelo ar 0,918 e
  0,933. A diferença entre o L01 e o dev ficou em 0,094.
- **Sonda do professor aplicada ao 3b:** dev 0,993, 2019 0,960, eval 0,970,
  ensaio 5 pelo ar 0,920 e pelo arquivo 0,595.

**Robustez a salas dos detectores** (`robustness_eval.py --amostra 10000 --so
clean sala`; os mesmos 10.002 áudios do eval da 2019; EER):

| detector | limpo | sala | humanos saturados em 1,0 na sala |
|---|---|---|---|
| `fusion_lcnn_v4_asv5` (o da demonstração) | 37,71% | 46,12% | 0 de 1.032 |
| `fusion_lcnn_v4` (2019) | 20,25% | 42,53% | 926 de 1.032 |

**Leitura:**
1. **Nenhum braço passou.** Os dois falham no ensaio 5 pelo arquivo (0,750 e
   0,785 contra 0,80); o aluno 3 também perde na ASVspoof, o 3b não.
2. **O aluno 3b é o melhor modelo nosso até agora na ASVspoof.** No eval da
   ASVspoof 5 chega a 0,964, contra 0,971 do professor, 0,879 do aluno 2 e
   0,659 do nosso detector (10.2.29). São 11,4 milhões de parâmetros, sem o
   XLS-R na hora de detectar. Tamanho e tempo de treino mudaram juntos;
   este braço não separa os dois.
3. **No nosso domínio, o ganho é grande, mas não basta.** No ensaio 5, o
   arquivo vai de 0,590 a 0,785 e o pelo ar de 0,355 a 0,785. O professor
   continua acima pelo arquivo (1,00). O limiar do dev não pega quase nenhum
   clone: falta calibrar com o nosso canal nas duas classes.
4. **As salas derrubam os dois detectores.** O da 2019 passa a chamar quase toda
   voz humana em sala de sintética: 926 de 1.032 vão a 1,0. O do recorte piora
   menos, de 37,71% para 46,12%.

### 10.2.39 Limiar calibrado pelo canal com outro locutor (critério combinado antes)

**Por quê (27/10/2026):** o limiar escolhido no dev não transfere para o nosso
microfone (no aluno 3b, pega 1 de 40 clones do ensaio 5; 10.2.38). A literatura
calibra o ponto de corte com dados do próprio canal: regressão logística, como
na ASVspoof 5, ou normalização por coorte, como na verificação de locutor. O
teste mais simples, sem gravar nada novo: o limiar sai só de vozes reais de
outra pessoa no mesmo microfone e é testado no L01.

**Desenho:**
- **Calibração:** as 10 gravações reais do L02 (frases do conjunto 3; mesmo
  notebook, microfone e sala), sem as cópias da captura
  (`data/vozes/reparadas/L02_embutido`). Nenhum clone entra.
- **Regra principal:** o limiar é o maior score entre os 10 reais do L02;
  acima dele, sintético. Nenhuma voz real do L02 é acusada.
- **Regra secundária (descritiva):** o segundo maior, ou seja, 1 de 10 reais do
  L02 acusado.
- **Modelos:**
  - o aluno 3b, com a mesma sonda nova do 10.2.38;
  - o professor (sonda da camada 7, treinada no treino do recorte);
  - o detector `fusion_lcnn_v4_asv5`, pelo caminho da interface.
- **Teste:** o material do ensaio 5 (L01). Reais; clones do XTTS e do
  Chatterbox pelo arquivo e pelo ar.
- **Meta, como nos ensaios:** pelo menos 8 de 10 reais julgados humanos e 8 de
  10 clones julgados sintéticos, em cada caminho e ferramenta.

**Leitura combinada antes:**
- **O aluno 3b ou o professor atinge a meta onde o limiar do dev falhava:** a
  calibração pelo canal, com outro locutor, resolve o ponto de corte naquele
  caminho.
- **Os reais do L01 passam do limiar do L02:** a variação entre pessoas pesa
  mais que o canal, e 10 gravações de uma pessoa não bastam.
- **Nem uma coisa nem outra:** a ordem é o limite, não o corte.

**Ressalvas declaradas antes:**
- Os scores do detector nos reais do L02 já foram vistos (10.2.34, média
  −12,68), e os do ensaio 5 também. Para ele, o resultado é só descritivo. Os
  testes limpos são o aluno 3b e o professor, cujos scores no L02 ainda não
  foram calculados.
- É um locutor só, com 10 arquivos.
- O L02 gravou no modo exclusivo (16 bits, sem o ganho do Windows), recuperado
  sem as cópias; o L01 gravou no compartilhado.

**Resultado (27/10/2026;** `outputs/calibracao_L02.json`**,**
`outputs/destilacao/calibracao_L02_aluno3b.json`**,** `outputs/ssl/L02_rep.npz`**).**
Pela regra principal (o maior score real do L02), acertos de 10 no ensaio 5:

| modelo | limiar | reais do L01 | XTTS, arquivo | Chatterbox, arquivo | XTTS, pelo ar | Chatterbox, pelo ar |
|---|---|---|---|---|---|---|
| **professor (camada 7)** | −3,269 | **10** | **10** | **10** | 7 | 1 |
| aluno 3b | −2,463 | 9 | 5 | 3 | 1 | 4 |
| detector (descritivo) | −9,976 | 10 | 9 | 0 | 0 | 0 |

Pela regra secundária (o segundo maior):
- **Professor:** reais 10, arquivo 10 e 10, pelo ar 8 (XTTS) e 3 (Chatterbox).
- **Aluno 3b:** reais 7, arquivo 7 e 8, pelo ar 5 e 9.
- **Detector:** reais 10, arquivo 10 e 1, pelo ar 0 e 0.

A sonda nova do aluno 3b deu os mesmos scores do 10.2.38 no ensaio 5
(conferido); a avaliação só imprimia mal um conjunto de uma classe só, e a
impressão foi corrigida.

**Leitura, pela regra combinada:**
1. **No professor, a calibração pelo canal com outro locutor resolve o ponto de
   corte pelo arquivo.** Reais 10 de 10, XTTS 10 de 10 e Chatterbox 10 de 10,
   onde o limiar do dev pegava 9 e 5 (10.2.36). É a primeira vez que o
   Chatterbox é separado no nosso caminho: a meta é atingida pelo arquivo nas
   duas ferramentas.
2. **Pelo ar, o professor melhora no XTTS (0 → 7 de 10; 8 pela regra
   secundária) e não no Chatterbox (1 de 10).** O caminho pelo ar segue como
   limitação.
3. **No aluno 3b, o limite é a ordem, não o corte.** Os reais do L01 (−3,34 a
   −2,41) e do L02 (−3,45 a −2,46) ficam no mesmo lugar, mas os clones não ficam
   acima deles.
4. **As vozes reais de pessoas diferentes caem no mesmo lugar no mesmo canal.**
   No professor, L02 de −5,15 a −3,27 e L01 de −4,99 a −3,88: 10 gravações de
   outra pessoa bastaram para pôr o corte. Com um locutor só, isso precisa de
   confirmação com mais vozes.

### 10.2.40 Régua em português: o BRSpeech-DF (descritivo, expectativa registrada antes)

**Por quê (27/10/2026):** todo o resultado em português até aqui vem de dois
locutores (L01 e L02) e duas ferramentas. O BRSpeech-DF (EMNLP 2025; CC BY 4.0
na página da base) é a primeira base pública de deepfake em português: leitores
de audiolivros do LibriVox (Brasil e Portugal) e clones deles feitos por cinco
ferramentas que clonam a partir de uma amostra de voz.

**Material (baixado com aprovação do L01, 11,7 GB da base):**
- **Teste:** 10 leitores, 1.320 vozes reais e 6.587 clones (~1.316 por
  ferramenta: XTTS, F5-TTS, Fish-Speech, YourTTS e ToucanTTS).
- **Validação:** outros 10 leitores, 1.161 reais e 5.795 clones.
- **Treino:** 4 de 37 blocos de reais (6 leitores, 8.020) e 6 de 274 de clones
  (8.106, todos do F5-TTS).
- As partes não repetem leitores.
- Amostra da régua: as 1.320 reais e 300 clones de cada ferramenta do teste,
  semente 42 (`scripts/montar_amostra_brspeech.py`, lista em
  `data/recortes/brspeech_df_test_300_s42.tsv`).

**Medidas:** AUC por ferramenta contra as reais, e quantos ficam do lado certo
pelo limiar do dev da ASVspoof 5. Modelos:
- a sonda da camada 7 do XLS-R (o professor);
- o aluno 3b;
- o `fusion_lcnn_v4_asv5`, pelo caminho da avaliação (os primeiros 4 s).

**Expectativa (antes de ver):**
- o professor acima de 0,95 de AUC em todas as ferramentas;
- o aluno 3b entre o detector e o professor;
- o detector abaixo de 0,80 em parte delas: nos autores da base, detectores
  treinados em inglês dão de 30% a 63% de EER.

O canal das vozes reais (audiolivros) é parecido com o da ASVspoof 5, também
construída sobre audiolivros (MLS em inglês). O que muda é a língua e as
ferramentas, não o microfone.

**Resultado (27/10/2026;** `outputs/ssl/sondas_brspeech.json`,
`outputs/ssl/detector_brspeech.json`, `outputs/destilacao/brspeech_aluno3b.json`**).**
Um clone do ToucanTTS ficou fora: 17 ms de fala depois do corte de silêncio.
Antes disso, duas falhas de encanamento foram corrigidas: o estudo do XLS-R só
lia `.wav` e quebrava com áudio curto demais. AUC contra as 1.320 vozes reais:

| ferramenta | detector | sonda no LFCC | professor (camada 7) | aluno 3b |
|---|---|---|---|---|
| XTTS | 0,86 | 0,88 | 0,60 | 0,57 |
| F5-TTS | 0,54 | 0,69 | 0,59 | 0,39 |
| Fish-Speech | 0,95 | 0,95 | 0,92 | 0,78 |
| ToucanTTS | 0,86 | 0,89 | 0,91 | 0,73 |
| YourTTS | 0,99 | 1,00 | 0,86 | 0,98 |
| **todas** | **0,84** | **0,880** | **0,776** | **0,690** |

Vozes reais do lado humano pelo limiar do dev da ASVspoof 5: detector 939, LFCC
971, professor 641 e aluno 3b 855, de 1.320.

Por camada do XLS-R, todas as ferramentas juntas: a 4 dá 0,942, a 3 0,918, a 5
0,885 e a 7 0,776; da 10 em diante, de 0,68 a 0,71. É descritivo: a camada 7
foi escolhida no dev da ASVspoof 5, antes desta régua.

**Leitura:**
1. **A expectativa não se confirmou.** O professor ficou em 0,776, não acima de
   0,95, e quase não separa o XTTS (0,60) e o F5-TTS (0,59). Aqui o detector
   (0,84) e até a sonda no LFCC (0,880) ficam acima dele.
2. **O contraste com o ensaio 5 é grande.** Lá, o professor deu 1,00 no XTTS
   pelo arquivo (10.2.36). Duas explicações possíveis, que esta régua não
   separa:
   - no ensaio 5, a voz real passou pelo nosso microfone e o clone não, e parte
     da separação seria o canal (pelo ar, com o mesmo canal, ele ainda deu 0,98
     e 0,88);
   - aqui, cada clone imita uma gravação de audiolivro, com o canal dela, e
     falta essa diferença.
3. **O F5-TTS é o mais difícil para todos** (0,39 a 0,69). É também a
   ferramenta dos blocos de treino que baixamos.
4. **O aluno 3b fica abaixo do professor e do detector** (0,690) e inverte o
   F5-TTS (0,39).
5. **Os limiares do dev não transferem para o português de audiolivro.** O
   professor acusa 679 das 1.320 vozes reais (51%).
6. **As camadas de baixo do XLS-R (3 e 4) separam melhor aqui** do que a 7.
   Uma escolha de camada que valha para as duas réguas teria de ser feita em
   dados que não sejam o teste.

### 10.2.41 Piloto 4 de destilação: um aluno com vários professores (critério combinado antes)

**Por quê (27/10/2026, pedido do L01):** cada régua mostrou um professor forte
num lugar diferente:
- a camada 7 do XLS-R na ASVspoof e nos clones do L01 pelo arquivo (10.2.36,
  10.2.39);
- a camada 4 no português de audiolivro (0,942, 10.2.40);
- o nosso detector nos clones do BRSpeech-DF (0,84);
- no F5-TTS todos erram, e só os rótulos o conhecem.

A destilação com vários professores é conhecida: o DistilHuBERT já imitava três
camadas, cada uma com a sua cabeça. Aqui o aluno aprende com todos ao mesmo
tempo (`scripts/destilacao_multi.py`).

**Aluno 4:**
- parte do aluno 3b (6 camadas, dimensão 384), com o mesmo corpo e a cabeça da
  camada 7;
- ganha três cabeças: a camada 4 do XLS-R (PCA de 256), o score do
  `fusion_lcnn_v4_asv5` em cada janela e a decisão real ou falso, treinada com
  os rótulos. A cabeça de decisão é o detector final.
- Perda: camada 7 + camada 4 + 0,5 × detector + 1,0 × rótulo, com pesos
  fixados antes.
- Cada janela ensina o que tiver; o alvo que falta sai daquela parte da perda.
- 12 épocas, taxa 3e-4, lotes de 16, semente 42.

**Dados de treino:**

| conjunto | camada 7 | camada 4 | detector | rótulo |
|---|---|---|---|---|
| ASVspoof 5 do piloto 2 (o treino do detector) | sim | — | — | sim |
| Common Voice (piloto 3) | sim | sim | sim | real |
| salas (piloto 3) | sim | sim | sim | sim |
| BRSpeech-DF, treino: 3.000 reais e 3.000 do F5-TTS | sim | sim | sim | sim |

No treino do BRSpeech-DF, um leitor tem 6.665 das 8.020 reais baixadas; o
sorteio de 3.000 mantém essa proporção, e isso fica declarado.

**Escolha do checkpoint:** a menor perda média em quatro devs:
- o dev da ASVspoof 5;
- o dev do Common Voice;
- o dev passado por salas;
- a validação do BRSpeech-DF: as 1.161 reais e 200 clones de cada ferramenta,
  de 10 leitores que não estão no teste.

Nenhuma régua de teste entra no treino nem na escolha.

**Critério (no commit antes de rodar):** o piloto 4 passa se a cabeça de
decisão cumprir todas as condições, com AUC:
- **BRSpeech-DF, teste:** ≥ 0,90 (o melhor modelo anterior, a sonda no LFCC,
  deu 0,880);
- **ensaio 5:** ≥ 0,90 pelo arquivo (XTTS e Chatterbox juntos) e ≥ 0,80 pelo ar;
- **eval da ASVspoof 5:** ≥ 0,95;
- **2019 LA:** ≥ 0,90.

**Descritivo:** a AUC por ferramenta; o dev passado por salas; e a meta do
ensaio 5 com o limiar das vozes reais do L02 (regra do 10.2.39).

**Execução (27/10/2026, pela sessão com o aval do L01):**
- **BRSpeech-DF:** treino com 3.000 reais e 3.000 do F5-TTS (12.383 janelas);
  validação com 2.161 arquivos (9.148 janelas).
- **PCA da camada 4:** 86,0% da variância em 256 componentes.
- **Camada 4 e detector** em 21.693 das 44.617 janelas de treino.
- **Treino:** 11,53 milhões de parâmetros, 12 épocas de ~930 s. A melhor perda
  média nos devs foi a da **época 1** (2,568); depois ela subiu, entre 2,85 e
  3,90.
  - **Por quê:** no treino, a perda da cabeça de decisão caiu de 0,137 para
    0,000, enquanto a perda no dev da ASVspoof (ataques A09–A16, que o treino não
    tem) foi de 3,38 a 4,58, e a do dev em salas de 2,45 a 3,96. A cabeça de
    decisão decorou os ataques do treino.
  - A imitação das camadas 7 e 4 seguiu caindo no treino, e o dev do Common
    Voice também (2,13 → 1,89).

**Resultado** (`outputs/destilacao/avaliacao_piloto4.json`; a cabeça de
decisão da época 1). AUC:

| régua | aluno 4 | critério | antes, o melhor |
|---|---|---|---|
| BRSpeech-DF, teste | **0,938** | ≥ 0,90: **sim** | 0,880 (sonda no LFCC) |
| ensaio 5, arquivo | 0,625 | ≥ 0,90: não | 0,785 (aluno 3b); professor 1,00 |
| ensaio 5, pelo ar | 0,440 | ≥ 0,80: não | 0,785 (aluno 3b) |
| eval da ASVspoof 5 | 0,826 | ≥ 0,95: não | 0,964 (aluno 3b) |
| 2019 LA | 0,753 | ≥ 0,90: não | 0,934 (aluno 3b) |
| dev passado por salas (informação) | 0,803 | — | 0,782 (aluno 3b) |

**Critério: não passou.** Só o BRSpeech-DF passou.

**Descritivo:**
- **BRSpeech-DF por ferramenta:** F5-TTS 1,00, YourTTS 0,98, ToucanTTS 0,97,
  XTTS 0,93 e Fish-Speech 0,82. Reais do lado humano: 810 de 1.320.
- **Ensaio 5:** XTTS 0,99 e Chatterbox 0,26 pelo arquivo; pelo ar, XTTS 0,71 e
  Chatterbox 0,17.
- **Com o limiar do L02** (10.2.39): reais 10, XTTS 9, Chatterbox 0, pelo ar 0
  e 0.
- **Dev da ASVspoof:** EER 19,25%; a AUC média cai de 0,989 (aluno 3b) para
  0,861.
- **Eval da ASVspoof 5:** A19 e A20 em 0,43 e 0,41.

**Leitura:**
1. **Os rótulos ensinam o que mostram, e só isso.** O F5-TTS, que todos os
   modelos erravam (0,39 a 0,69), vai a 1,00. O BRSpeech-DF inteiro sobe ao
   melhor resultado até agora (0,938), com leitores que o treino não viu.
2. **A cabeça de decisão generaliza pior que a sonda sobre a imitação.** Nos
   ataques que o treino não tem, a ASVspoof e a 2019 caem, e o Chatterbox do
   ensaio 5 fica abaixo da voz real (0,26). Uma época bastou para decorar o
   treino.
3. **Próximo passo natural, a decidir:** manter a imitação dos professores e
   tirar a cabeça de decisão (ou dar a ela peso bem menor), e pôr a sonda sobre
   a imitação, que generalizou melhor nos pilotos 2 a 3b. Não foi testado.

### 10.2.42 Régua do caminho pelo ar: o ReplayDF (descritivo, expectativa registrada antes)

**Por quê (27/10/2026):** o caminho pelo ar só foi medido com um locutor (o
L01), uma sala e um alto-falante (10.2.35, 10.2.39). O ReplayDF (Interspeech
2025; CC BY-NC 4.0, só pesquisa; 23 GB, baixado com aprovação do L01) traz real
e falso tocados por alto-falante e regravados, as duas classes pelo mesmo canal.

**Material:**
- 109 montagens, com 30 microfones e 33 alto-falantes;
- 6 línguas (en, de, fr, es, it, pl), sem português;
- falsos do Bark, do VITS e do XTTS v1.1 e v2.0; reais do M-AILABS.
- Amostra: 10 reais e 10 falsos por montagem, os falsos repartidos entre as
  ferramentas, semente 42. São 2.180 áudios (1.090 reais; Bark 327, VITS 327,
  XTTS v1.1 218, XTTS v2 218), em `scripts/montar_amostra_replaydf.py` e
  `data/recortes/replaydf_10_s42.tsv`.

**Medidas:** AUC por ferramenta contra os reais e acertos pelo limiar do dev da
ASVspoof 5. Modelos: a sonda da camada 7 (e as outras camadas), o detector
(caminho da avaliação), o aluno 3b e, se o piloto 4 terminar antes, o aluno 4.

**Expectativa (antes de ver):**
- todos abaixo do que dão sem regravação;
- o professor acima dos outros, na faixa de 0,80 a 0,90, como no ensaio 5 pelo
  ar (0,93);
- o detector abaixo de 0,75.

No artigo do ReplayDF, o melhor detector testado vai de 4,7% a 18,2% de EER
com a regravação.

**Resultado (27/10/2026;** `outputs/ssl/sondas_replaydf.json`,
`outputs/ssl/detector_replaydf.json`, `outputs/destilacao/replaydf_aluno3b.json`,
`outputs/destilacao/replaydf_aluno4.json`**).** Nenhum áudio pulado. AUC de
cada ferramenta contra os 1.090 reais regravados:

| ferramenta | detector | sonda no LFCC | professor (camada 7) | aluno 3b | aluno 4 |
|---|---|---|---|---|---|
| Bark | 0,87 | 0,73 | 0,76 | 0,68 | 0,77 |
| VITS | 0,71 | 0,57 | 0,76 | 0,64 | 0,68 |
| XTTS v1.1 | 0,58 | 0,62 | 0,86 | 0,54 | 0,68 |
| XTTS v2 | 0,74 | 0,69 | 0,85 | 0,62 | 0,76 |
| **todas** | **0,741** | **0,649** | **0,797** | **0,628** | **0,724** |

Reais do lado humano pelo limiar do dev: detector 839, LFCC 664, professor
1.071, aluno 3b 1.054 e aluno 4 963, de 1.090. O professor acusa só 199 dos
1.090 falsos.

Por camada do XLS-R, todas as ferramentas juntas:
- da 0 à 3: de 0,57 a 0,73;
- a 4: 0,785; a 7: 0,797;
- da 10 à 17: de 0,82 a 0,85 (a 17 dá 0,851);
- da 19 à 24: de 0,76 a 0,79.

É descritivo: a camada 7 foi escolhida antes, no dev.

**Leitura, contra a expectativa registrada:**
1. **O professor fica acima dos outros, como esperado, mas em 0,797,** logo
   abaixo da faixa prevista (0,80 a 0,90). Os dois XTTS são os mais separados
   por ele (0,85 e 0,86), como o XTTS do L01 pelo ar (0,98, 10.2.36).
2. **O detector fica em 0,741, abaixo de 0,75, como esperado.** O XTTS v1.1
   quase não se separa (0,58).
3. **O limiar do dev não serve pelo ar.** Com a regravação, os scores dos falsos
   vão para o lado humano: o professor deixa passar 891 de 1.090. É o mesmo
   problema do ensaio 5, que o limiar calibrado no canal resolveu em parte
   (10.2.39).
4. **Os alunos não herdam a robustez do professor à regravação** (3b 0,628). O
   aluno 4 sobe para 0,724.
5. **Cada canal favorece camadas diferentes do XLS-R:**
   - no dev da ASVspoof, quase todas dão perto de 1,00;
   - no português de audiolivro, as de baixo (a 4 dá 0,942; 10.2.40);
   - na regravação, as do meio para cima (de 10 a 17).

   Uma camada fixa perde em algum canal. Uma soma ponderada das camadas, que é
   o modelo `fusion_ssl` do plano, aprenderia a pesar cada uma.

### 10.2.43 Modelo final: XLS-R congelado com pesos por camada (critério combinado antes)

**Por quê (27/10/2026):** o grupo decidiu que o modelo final usa o XLS-R, para
reduzir o risco na pré-banca e na banca. Em vez do ajuste completo no Kaggle, o
primeiro passo é o XLS-R congelado. A melhor camada muda com o canal (10.2.40,
10.2.42), então uma cabeça pequena aprende quanto pesar cada uma das 25 camadas
(a soma ponderada do `fusion_ssl` do plano), com um ramo LFCC somado no fim
(`scripts/xlsr_congelado.py`).

**Cabeça:**
- pesos por camada (softmax das 25), sobre as médias no tempo de cada camada,
  padronizadas no treino;
- uma saída linear sobre a soma ponderada, mais uma saída linear sobre o LFCC
  (média e desvio, 120 valores), somadas;
- AdamW com taxa 3e-3 e decaimento 1e-2 (sem decaimento nos pesos das
  camadas), dropout 0,2, lotes de 256, até 60 épocas, parada após 8 sem
  melhora.

**Dados de treino:** cinco domínios, cada um com o mesmo peso total e as
classes equilibradas dentro dele.
- **ASVspoof 5:** os 4.000 arquivos do piloto 1.
- **BRSpeech-DF:** treino com 3.000 reais e 3.000 do F5-TTS.
- **Salas:** os 4.000 da ASVspoof passados por uma sala do BUT ReverbDB.
- **Common Voice:** 4.000 reais.
- **ReplayDF:** 20 reais e 20 falsos em cada uma de 55 montagens (2.200).

**Devs** (a escolha da época, pela perda média): o dev da ASVspoof 5, a
validação do BRSpeech-DF, o dev passado por salas, o dev do Common Voice e o
ReplayDF de outras 10 montagens (200 da amostra do 10.2.42).

**Réguas, fora do treino e da escolha:**
- o eval da ASVspoof 5 e a 2019 LA;
- o teste do BRSpeech-DF;
- o ReplayDF nas 44 montagens restantes: os 880 da amostra do 10.2.42, sem
  montagem em comum com o treino (`data/recortes/replaydf_montagens_s42.tsv`);
- o ensaio 5.

**Referências do professor (sonda da camada 7):**

| régua | AUC |
|---|---|
| eval da ASVspoof 5 | 0,971 |
| 2019 LA | 0,998 |
| BRSpeech-DF | 0,776 |
| ReplayDF, as 44 montagens | 0,780 |
| ensaio 5, arquivo | 1,00 |
| ensaio 5, pelo ar | 0,93 |

**Critério (no commit antes de rodar):** o modelo passa se cumprir tudo:
- **ganhos, onde o professor falha:** BRSpeech-DF ≥ 0,90 e ReplayDF (44
  montagens) ≥ 0,85;
- **sem perdas:** eval da ASVspoof 5 ≥ 0,95, 2019 LA ≥ 0,95, ensaio 5 pelo
  arquivo ≥ 0,95 e pelo ar ≥ 0,85.

**Descritivo:**
- a meta do ensaio 5 com o limiar das vozes reais do L02 (regra do 10.2.39);
- os pesos aprendidos das camadas;
- a AUC por ferramenta e por ataque.

**Execução (27 e 28/10/2026, pela sessão).**
- **Extração do XLS-R:** pulados 2 arquivos do BRSpeech-DF, com menos de 0,25 s
  depois do corte de silêncio, e o clipe do Common Voice de cabeçalho errado.
- **Treino da cabeça:** 14 épocas; parou depois de 8 sem melhora. A escolhida
  foi a 6 (perda média nos devs 0,179).
  - Por dev: ASVspoof 0,083, BRSpeech-DF 0,184, salas 0,439, Common Voice 0,021,
    ReplayDF 0,169.
  - Depois dela, a perda no dev em salas subiu de 0,439 a 0,662.
- **Camadas mais pesadas:** 6 (0,09), 5 (0,08), 4 (0,07), 7 (0,06), 10 e 9
  (0,05).
- **EER no dev da ASVspoof 5:** 3,04%.

**Resultado** (`outputs/xlsr_congelado.json`). AUC:

| régua | modelo final | critério | professor (camada 7) |
|---|---|---|---|
| eval da ASVspoof 5 | **0,975** | ≥ 0,95: sim | 0,971 |
| 2019 LA | **0,974** | ≥ 0,95: sim | 0,998 |
| BRSpeech-DF, teste | **0,921** | ≥ 0,90: sim | 0,776 |
| ReplayDF, as 44 montagens | **0,968** | ≥ 0,85: sim | 0,780 |
| ensaio 5, arquivo | **1,000** | ≥ 0,95: sim | 1,00 |
| ensaio 5, pelo ar | **0,975** | ≥ 0,85: sim | 0,93 |

**Critério: passou.**

**Descritivo:**
- **Por ferramenta:**
  - BRSpeech-DF: Fish-Speech 0,95, ToucanTTS 0,96, YourTTS 0,96, XTTS 0,87,
    F5-TTS 0,87;
  - ReplayDF: Bark 0,98, VITS 0,96, XTTS v1.1 0,97, XTTS v2 0,96;
  - eval da ASVspoof 5: todos os ataques entre 0,93 (A28) e 0,99;
  - 2019: A04 0,94 e A06 0,91, os outros 1,00.
- **Ensaio 5 por grupo:** XTTS e Chatterbox 1,00 pelo arquivo; pelo ar, XTTS
  1,00 e Chatterbox 0,95. Os clones das três referências (10.2.36) também ficam
  entre 0,94 e 1,00.
- **Meta do ensaio 5 com o limiar das vozes reais do L02** (regra do 10.2.39,
  limiar −0,517):
  - reais do L01, **6 de 10**;
  - XTTS 10 e Chatterbox 10 pelo arquivo; pelo ar, XTTS 10 e Chatterbox 9.
  - A meta não é atingida, só pelas vozes reais.
- **Onde ficam as vozes reais:**
  - L01: de −1,65 a −0,02;
  - L02: de −2,89 a −0,52;
  - clones pelo arquivo: o menor é 1,53; pelo ar, −0,95.
  - Pelo arquivo, um limiar entre −0,02 e 1,53 separaria tudo. Escolher esse
    limiar olhando o L01 seria calibrar no teste, o que não se faz.
- **Pelo limiar do dev** (−0,898): reais do L01 5 de 10; humanos do BRSpeech-DF
  531 de 1.320.

**Leitura:**
1. **O modelo final passa em todas as réguas, inclusive onde o professor
   falhava.**
   - No português de audiolivro, de 0,776 para 0,921.
   - Pelo ar, em montagens que o treino não viu, de 0,780 para 0,968.
   - Mantém a ASVspoof (0,975) e o ensaio 5 (1,000 e 0,975).
   - É o melhor modelo do projeto em todas as réguas ao mesmo tempo.
2. **A ordem está resolvida; falta o ponto de corte para a voz do L01.**
   - As vozes reais do L02 caem abaixo das do L01, e o limiar tirado delas acusa
     4 das 10 vozes reais do L01.
   - Uma diferença declarada antes pode pesar: o L02 gravou no modo exclusivo,
     recuperado; o L01, no compartilhado.
3. **Próximo passo:** calibrar com vozes reais de mais pessoas, gravadas no modo
   atual e no microfone da banca, com a regra combinada antes. Depois, integrar
   o modelo à interface e ao monitor, e um ensaio novo com o critério fixado
   antes.

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

**Formato e licença** (README do Zenodo, registro 14498691). Áudio em FLAC a
16 kHz: treino em `flac_T_aa.tar` a `flac_T_ae.tar`, dev em `flac_D_aa.tar` a
`flac_D_ac.tar`, eval em `flac_E_aa.tar` a `flac_E_aj.tar`; protocolos em
`ASVspoof5_protocols.tar.gz`. Os protocolos da trilha 1 têm 10 campos
separados por espaço, apesar do `.tsv` (o README diz "cinco colunas" e
lista dez): locutor, arquivo, gênero, codec, qualidade do codec, original do
codec, condição de adaptação do atacante, ataque, chave e um reservado.
Linhas: 182.357 (train), 140.950 (dev), 680.774 (eval), as mesmas da tabela.
Licença Open Data Commons Attribution (ODC-By); o áudio bonafide, CC BY 4.0.
Leitura em `src/data/asvspoof5.py`, que confere cada linha pelo conteúdo e
o total contra o README.

**Recorte** (`scripts/montar_recorte_asvspoof5.py`): lê os FLAC direto dos
`.tar`, sem extrair (extraídos, os 57,5 GB dobrariam), e confere o tamanho e
o MD5 de cada `.tar` contra o Zenodo. Seleção com semente fixa: todos os
bonafide e até N spoof por ataque. Cada áudio vira um FLAC de 16 bits com no
máximo 6 s, começando cerca de 0,5 s antes da primeira fala: a margem de
silêncio fica de propósito, para a degradação de canal do treino agir também
nas pausas (10.2.1). A seleção inteira é versionada em `data/recortes/`.

**Conferido nos protocolos reais** (21/10/2026): 10 campos por linha, nenhuma
ignorada, total igual ao do README nas três partições; o protocolo de trials
da trilha 2 é recusado. Os 8 `.tar` de treino e dev têm o tamanho do Zenodo.

| partição | enunciados | bonafide | locutores | ataques (spoof por ataque) | codec |
|---|---|---|---|---|---|
| train | 182.357 | 18.797 | 400 | A01–A08 (20.445 cada) | nenhum |
| dev | 140.950 | 31.334 | 785 | A09–A16 (13.702 cada) | nenhum |
| eval | 680.774 | 138.688 | 737 | A17–A32 (30.860 a 34.414) | C01–C11 em 509.172; 171.602 sem |

Recorte de treino com 5.000 spoof por ataque: 58.797 áudios (18.797 bonafide,
40.000 spoof).

**Recorte de treino montado** (21/10/2026; MD5 dos 5 `.tar` conferido): 58.797
FLAC de 16 kHz e 16 bits, todos legíveis, protocolo idêntico à seleção
versionada (`data/recortes/asvspoof5_train_5000_s42.txt`). 58.614 com 6 s
exatos, mediana 6,00 s, mínimo 3,58 s (só 1 abaixo de 4 s); 98,0 h de áudio em
6,07 GB (o plano estimava 11 GB contando int16 sem compressão).

**Recorte do dev montado** (22/10/2026; MD5 dos 3 `.tar` conferido): amostra
estratificada de 9.999 áudios (2.223 bonafide, 972 de cada ataque A09–A16, 712
locutores), 15,8 h em 1,12 GB, todos legíveis; seleção em
`data/recortes/asvspoof5_dev_amostra10000_s42.txt`. Treino e dev não têm
arquivo nem locutor em comum. 10 áudios do dev têm menos de 1 s (o mais curto,
0,09 s), e os 5 mais curtos são spoof (A11 e A13); no treino o mais curto tem
3,58 s. Duração curta só em spoof pode virar atalho: conferir no eval.

**Treino do fusion_v4 no recorte** (`configs/fusion_v4_asv5.yaml`, 23 a
24/10/2026, GTX 1650): 31 épocas, parada antecipada (paciência 12) depois da
melhor, a 19, com EER do dev de **20,56%**; 8h07m, cerca de 15 min por época
(45% esperando dados). A perda de treino cai de 0,225 a 0,018 enquanto o dev
fica entre 20,6% e 22% desde a época 3: o dev tem ataques que o treino não
tem (A09–A16), e a distância entre os dois é a generalização que o dev novo
foi escolhido para medir. O limiar calibrado no dev saiu **0,0002**: os
spoof inéditos recebem probabilidade quase zero, então qualquer limiar que
separe esse dev fica no extremo. Vale para EER; o limiar da demonstração sai
do Incremento 4, com vozes reais. Leitura combinada antes das réguas:
diferença entre modelos só conta acima de 2 pp nas amostras de 10 mil (IC de
cerca de ±1,3 pp por medida).

**Réguas da 2019 e da 2021 para o fusion_v4 do recorte** (24/10/2026, as
mesmas amostras dos outros modelos; EER):

| régua | fusion_v4 (2019) | fusion_v4 (recorte da ASVspoof 5) |
|---|---|---|
| eval 2019 completo | 20,18% | 37,76% |
| amostra de 10 mil, limpo | 20,25% | 37,71% |
| ruído 5 dB | 27,89% | 42,70% |
| Opus 25 kbps simulado | 22,40% | 38,17% |
| banda estreita | 25,67% | 37,42% |
| captura com soxr | 24,98% | 37,68% |
| corte em 7,6 kHz | 31,12% | 37,61% |
| 2021, referência sem codec | 19,51% | 36,77% |
| 2021, Opus real | 32,62% | 37,44% |

- **Pior em todas as réguas da 2019 e da 2021,** por 4,82 a 17,58 pp, acima
  dos 2 pp combinados: treinado na ASVspoof 5, o modelo reconhece mal os
  ataques da 2019 (A07–A19) mesmo no áudio limpo. É o cruzamento entre bases do
  plano, no sentido ASVspoof 5 → 2019, e o resultado é negativo.
- **Mas quase não muda com o canal:** o Opus real custa +0,67 pp sobre a
  referência (fusion_v4: +13,11), o corte em 7,6 kHz −0,10 pp (+10,87), a
  captura com soxr −0,03 pp (+4,73) e a banda estreita −0,29 pp (+5,42). O
  ruído a 5 dB ainda custa +4,99 pp (+7,64).
- **O resultado mistura duas mudanças:** a base de treino e a degradação nova.
  Separar as duas pede a outra célula do 2x2, o fusion_v4 treinado na 2019 com
  a mesma degradação.
- **A régua do próprio domínio, o eval da ASVspoof 5, ainda falta.** O dev do
  recorte, com ataques inéditos, ficou em 20,56%.

**Amostra do eval** (planejada em 23/10/2026; monta depois do treino do
fusion_v4, para não disputar CPU): baixando um `.tar` do eval por vez para o
SATA, tirando a amostra e apagando. Estratificada por (chave, ataque, codec),
com o áudio inteiro, sem corte. Pelo protocolo: 9.979 áudios (2.036
bonafide), os 16 ataques A17–A32 e 600 a 700 áudios por condição de codec
(132 a 143 bonafide cada). Com tão poucos bonafide por codec, o EER de um
codec isolado tem IC largo; comparar em grupos: codecs que o treino vê (C01,
C05, C08), não vistos (C02–C04, C06, C07, C09, C10) e aparelhos (C11). A
seleção vai também em `.tsv`, no formato da ASVspoof 5, que guarda o codec.

**Custo do RawBoost** (`src/preprocess/rawboost.py`, 22/10/2026, CPU, um
áudio de 6 s do recorte, média de 5): lnl 43 ms, isd 4 ms, ssi 10 ms,
lnl+isd 44 ms, lnl+isd+ssi 55 ms. Com a degradação ligada o cache do treino
se desliga, e esse custo entra em toda amostra de toda época, dividido pelos
workers do DataLoader.

**Codecs do aumento de dados** (`src/preprocess/codecs.py`, 22/10/2026): os
que o libsndfile já traz, em memória. Medidos num áudio de 6 s do recorte, a
16 kHz, por nível de compressão de 0 a 0,99: MP3 77 a 21 kbps (~20 ms; o nível
1,0 dá erro), Vorbis 77 a 30 kbps (~35 ms), Opus 260 a 9 kbps (40 a 115 ms;
0,9 a 0,99 dá 30 a 9 kbps, a faixa de voz). A 8 kHz: G.711 µ-law e A-law
64 kbps e GSM 06.10 13 kbps (1 a 11 ms). Nenhum desloca o sinal no tempo.
Do eval da ASVspoof 5 (`ASVspoof5.codec.config.csv`), o treino vê o tipo de
C1 e C8 (Opus) e C5 (MP3); AMR (C2, C9), Speex (C3, C10), Encodec (C4), AAC
(C6), MP3 com Encodec (C7) e os aparelhos de C11 ficam como não vistos. No
eval, reportar as duas partes separadas.

- **Treino e dev não têm codec.** No modelo treinado no recorte, a robustez a
  codec vem só do aumento de dados (RawBoost, codecs, ida e volta 16/48/16), e
  o eval mede codecs que o treino nunca viu.
- **Espelho:** o Hugging Face (`jungjee/asvspoof5`, público) tem os mesmos
  `.tar`, com tamanhos idênticos em bytes; lá o protocolo vem como `.tar` sem
  compressão.

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

### 10.9 Incremento 4: as duas células do fusion_v4 e a fusão entre elas

Três fusion_v4 nas mesmas réguas (24/10/2026): o original (2019, aumento antigo),
o da 2019 com a degradação nova (`configs/fusion_v4_2019_degr.yaml`: 50 épocas, o
teto, melhor na 47 com dev 1,96%, 4h39m) e o do recorte da ASVspoof 5 com a mesma
degradação (10.7). Os dois últimos só diferem na base de treino e de validação.
EER; amostras de 10 mil da 2019 e da 2021 e a amostra de 9.979 do eval da
ASVspoof 5, estratificada por codec.

| régua | original (2019) | 2019 + degradação | ASVspoof 5 + degradação | fusão média dos dois últimos |
|---|---|---|---|---|
| eval 2019 completo | 20,18% | 23,74% | 37,76% | — |
| 2019, 10 mil, limpo | 20,25% | 23,84% | 37,71% | 16,8% |
| 2019, 10 mil, corte em 7,6 kHz | 31,12% | 23,84% | 37,61% | 16,8% |
| 2019, 10 mil, ruído 5 dB | 27,89% | 27,14% | 42,70% | 24,2% |
| 2021, referência sem codec | 19,51% | 23,70% | 36,77% | 18,06% |
| 2021, Opus real | 32,62% | 24,05% | 37,44% | 18,16% |
| eval da ASVspoof 5 | 34,78% | 34,63% | 38,32% | 26,7% |

Eval da ASVspoof 5 por grupo de codec (bonafide do grupo entre parênteses):

| grupo | original (2019) | 2019 + degradação | ASVspoof 5 + degradação |
|---|---|---|---|
| sem codec (516) | 27,41% | 34,46% | 37,31% |
| codecs que o treino vê: C01, C05, C08 (414) | 42,58% | 33,68% | 37,18% |
| não vistos: C02–C04, C06, C07, C09, C10 (974) | 36,06% | 34,99% | 39,44% |
| aparelhos, C11 (132) | 35,52% | 32,56% | 40,15% |

- **A degradação nova (2019 + degradação contra o original) troca um pouco do
  limpo por canal.** Custa 3,56 pp no eval 2019 completo e 4,19 pp na referência
  da 2021, mas ganha 8,57 pp no Opus real (32,62% → 24,05%): o canal real deixa
  de custar (+0,35 pp sobre a referência, contra +13,11). No eval da ASVspoof 5
  empata no total, mas fica plano entre os grupos de codec (32,6% a 35,0%), onde
  o original vai de 27,41% sem codec a 42,58% nos codecs que o aumento simula.
- **A base (ASVspoof 5 contra 2019, mesma degradação) piora tudo,** inclusive o
  próprio eval da ASVspoof 5 (38,32% contra 34,63%). Treinar no recorte, como
  foi montado, não ajudou nem no domínio dele.
- **Mas os dois erram em ataques diferentes.** Por ataque no eval da ASVspoof 5,
  o do recorte acerta A17, A21, A22, A24 e A29 (2,82% a 10,65%), onde os da 2019
  ficam entre 36,19% e 55,73%; e erra A18, A19, A20, A27, A30 e A32 (52,67% a
  75,12%, invertidos), onde o da 2019 com degradação fica entre 4,80% e 27,20%.
- **Fusão pela média (exploratória, sem critério combinado antes).** A média das
  probabilidades dos dois, a regra do monitor, dá 16,8% na amostra da 2019 (contra
  23,84% do melhor sozinho), 18,16% no Opus real e 26,7% no eval da ASVspoof 5.
  Os 18,16% são o melhor resultado do projeto no canal real (antes: 24,94% pela
  fusão por postos do v2 e do fusion_v4, que não roda ao vivo, 5.3). Pelos postos:
  18,8%, 18,06% e 25,1%. O controle repete a Seção 4: os dois fusion_v4 da 2019,
  parecidos, fundidos dão só 33,1% no eval da ASVspoof 5 (melhor sozinho 34,63%).
- **Para virar a escolha da demonstração** falta combinar o critério antes,
  calibrar o limiar da fusão e medir com vozes reais (plano). Rodar dois modelos
  ao vivo já foi medido como viável (9.4).

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

**Robustez medida em dois modelos.** Só v2 e fusion_v4 têm as 11 condições.

**Atenção mínima.** O resultado negativo vale para uma projeção linear de 258
parâmetros, não para atenção em geral. Ver Seção 3.

**Canal real: só uma condição, numa amostra.** ASVspoof 2021 LA avaliado na
condição Opus contra a referência, 10.000 áudios por condição, dois modelos
(Seção 5.3). As outras condições do 2021 (a-law, G.722, PSTN etc.) não foram
rodadas. A camada 2 (cliente de conferência) continua só instrumentação.

**t-DCF — calculado (29/09/2026).** `scripts/tdcf.py`, eval completo de
2019, ASV fornecido pelos organizadores:

| sistema | EER CM | min t-DCF |
|---|---|---|
| baseline_lfcc_cnn_v2 | **18,99%** | 0,5075 |
| fusion_lcnn_v4 | 20,18% | **0,3818** |
| v2 + fusion_v4, fusão por postos | **13,13%** | **0,3129** |
| B02 (LFCC-GMM, **com** silêncio) | 8,09% | 0,2116 |
| B01 (CQCC-GMM, **com** silêncio) | 9,57% | 0,2366 |

Conferências que validam a medida: o EER CM bate com o `make_report` e com o
`score_fusion` nas três linhas; o ASV no seu limiar de EER dá Pfa = Pmiss =
2,46%, contra 2,48% de EER do ASV em Wang et al. (2020, Tab. 6); e as
contagens (5.370 target, 63.882 spoof) batem com o artigo.

**O t-DCF inverte o ranking do EER entre os dois modelos.** Pelo EER o v2 é
melhor (18,99% contra 20,18%); pelo min t-DCF o fusion_v4 é bem melhor (0,3818
contra 0,5075). Não é contradição: o EER olha um único ponto da curva (erros
iguais), e o t-DCF, com os custos do ASVspoof 2019 (Cfa_cm = 10), procura o
ponto em que **quase nenhum spoof passa**. É a região em que o fusion_v4 é
forte — ele zera seis ataques (Seção 7) e tem precisão de 0,9999 (Seção 3). É
a mesma leitura da 7.2: pela média por ataque o fusion_v4 também é o melhor.
**A métrica escolhida decide qual modelo "vence"**, e o texto deve dizer isso
em vez de eleger um.

A fusão por postos melhora nas duas métricas (0,3129), mas continua acima dos
baselines oficiais — que usam o silêncio (Seção 5.1). Não há régua de t-DCF
sob o mesmo protocolo: Müller et al. (2021) reportam só EER.

Para os outros cinco modelos, basta acrescentar os `.npz` em `--scores`
(opcional; os dois principais bastam para o texto).

**MP3 não avaliado.** O teste de robustez usou Opus.

**Front-end não testado contra o do baseline oficial.** O pipeline difere do
LFCC-LCNN oficial em pré-ênfase, c0 como energia, janela e faixa de frequência
(Seção 2.1). Nenhuma dessas diferenças foi isolada. A mais promissora para o
canal real é limitar o LFCC a 0–4 kHz, como no baseline do 2021 LA —
**trabalho futuro**, exige retreino.

---

## 12. Plano final do TC2 (escopo fechado)

Escrito sob restrição de tempo. A regra é: **o valor marginal de mais uma base é
menor que o valor marginal de escrever as seções que já têm dado.**

### 12.1 O que já está medido e só precisa ser escrito

Sete modelos comparados, fusão de scores com grupo de controle, robustez ao
canal simulado (com inversão de ranking) e real (sem ela), verificação de atalhos em duas camadas, latência
do RNF01, o achado do RNF02 trivial, custo de operação ao vivo, resolução
temporal e ponderação das janelas. Nada disso precisa de execução nova.

### 12.2 Fazer (barato, alto valor)

1. ~~Regenerar as tabelas por ataque~~ — **feito**, ver Seção 7.
2. ~~ASVspoof 2021 LA, com subamostragem~~ — **feito**, ver 5.3. Mudou a
   leitura da Seção 5.

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

- **O dev não prevê o eval** (3.1): cinco modelos ficam abaixo de 0,71% no dev
  e todos acima de 21% no eval; os dois piores no dev são os melhores no eval.
  ρ(dev, eval) = −0,11.
- A fusão de características ajuda (−1,38 pp) como saldo de trocas de até
  37,65 pp por ataque; a atenção quase não muda nada (Seção 3 e 7.0).
- A fusão de scores entre modelos **diversos** ajuda muito mais (−5,86 pp), e o
  grupo de controle mostra que o ganho vem da diversidade (Seção 4).
- **A degradação simulada não prediz o canal real** (Seções 5 e 5.3). Sob
  banda estreita e ruído simulados o `fusion_v4` é o robusto; sob transmissão
  real o `baseline_v2` vence por 3,19 pp (29,43% contra 32,62%). A simulação
  subestima o custo real do Opus em 7 a 11 vezes.
- A fusão de modelos diversos **se confirma no canal real** (−4,50 pp, 24,94%),
  mas só pela regra `rank`, que não é streamável; a média das probabilidades
  empata com o v2 sozinho (Seção 5.3).
- O EER agregado esconde os ataques mais fortes: A10 e A12 passam de 30% nos
  sete modelos (Seção 7.3) e, fundidos, continuam os piores no canal real
  (42,98% e 37,53%, Seção 5.3).
- O modelo aprende artefato, não atalho, com dependência residual de 1–2%
  (Seção 6).
- Nenhum modelo atinge o RNF02 (F1 ≥ 0,85), que ainda assim é satisfeito por um
  classificador trivial; o RNF03 também não é atendido (Seção 8). Ambos
  documentados com evidência.

---

## 13. O que o relatório do TC2 muda em relação ao texto do TC1

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
  13,13%; o ranking se inverte sob canal simulado (+23,42 pp no v2 contra
  +7,24 pp no fusion_v4), mas não no canal real (ASVspoof 2021 LA: 29,43%
  contra 32,62%).
- **Rascunho redigido** destas mudanças para o TC2, na numeração do texto do
  TC1: `ml/docs/TC2_REDACAO.md`.
- **Arquitetura:** seção nova, a partir da Seção 3-0 deste resumo.
- **§4.7 (ciclo incremental):** usar a linhagem v1 → v4 da Seção 3; deltas já
  no baseline; linha de base = `baseline_lcnn_v4`.
- **§4.11 (métricas):** classe positiva = spoof; limiar calibrado no dev;
  acrescentar t-DCF (Seção 11).
- **Seção 5:** trocar os resultados esperados pelas tabelas das Seções 3, 4, 5
  e 7 deste resumo.
- **Conclusão:** quatro achados — (1) o dev não prevê o eval; (2) a fusão de
  scores ganha pela diversidade entre os modelos; (3) o EER agregado esconde os
  ataques mais fortes (A10/A12); (4) a degradação simulada não prediz o canal
  real — o ranking se inverte na simulação e não se inverte no ASVspoof 2021 LA.
- **Limitações:** tudo o que está na Seção 11.

### 13.3 Promessas do texto sem entrega no repositório

- **Vozes dos colaboradores — a coleta NÃO aconteceu (confirmado, set/2026).**
  Sai do texto em 4.9, 5.1, 5.5, 5.6, 7.1, Marco 2 e conclusão, e do
  Resumo/Abstract. A frase da 7.1 sobre termos de consentimento assinados é
  **falsa** e precisa sair. O papel que a coleta teria (bonafide fora da
  ASVspoof) fica como trabalho futuro. Redação de substituição em
  `ml/docs/TC2_REDACAO.md`.
- **API e front-end — resolvido (30/09/2026).** A interface foi feita com
  FastAPI + Jinja2, numa aplicação só dentro de `ml/web/`, em vez de API
  separada + React (README). Descrita no TC2 na 4.8, na nova 4.13 e na 6.4.
- **MP3.** Declarar que o teste de robustez usou Opus. O RF01/RNF05 (aceitar
  MP3) é atendido na entrada do `infer.py`, mas não há avaliação de desempenho
  em MP3.

### 13.4 Cópia de agosto — substituída pela regeneração

A cópia de agosto não está no git (morava em `ml/outputs/`, gitignored). O que
ela tinha de exclusivo foi **regenerado** em 25/09/2026 com
`scripts/make_report.py` e o histórico de treino: EER por ataque dos sete
modelos (7.0), robustez nas 11 condições (Seção 5) e dev × eval (3.1). O
"r = 0,951" dela não foi reproduzido e não deve ser usado.

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
python scripts/canal_real.py preparar --config configs/baseline_v2.yaml --n-por-classe 20
python scripts/canal_real.py alinhar --pasta outputs/canal_real --gravacao ... --sessao <limpo|controle|chamada>
python scripts/comparar_sessoes.py outputs/canal_real/*/config_canal_real.yaml

# Seção 10.2.9 — fusão ao vivo na captura FIR (os dois modelos, eval e dev)
python scripts/robustness_eval.py --config configs/baseline_v2.yaml --checkpoint checkpoints/baseline_lfcc_cnn_v2_captura.pt --amostra 10000 --so captura_48k_fir [--partition dev]
python scripts/robustness_eval.py --config configs/fusion_v4.yaml --checkpoint checkpoints/fusion_lcnn_v4.pt --amostra 10000 --so captura_48k_fir [--partition dev]
python scripts/fusao_ao_vivo.py <v2 eval .npz> <v4 eval .npz> --dev <v2 dev .npz> <v4 dev .npz> --json outputs/fusao_ao_vivo_captura.json
```
