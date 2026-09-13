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

**Os dois incrementos do TC1, isolados** (comparando com `baseline_lcnn_v4`, que
é o mesmo encoder com um ramo só):

| incremento | efeito |
|---|---|
| fusão de características (LFCC + espectrograma) | **−1,38 pp** (21,56 → 20,18) |
| atenção no pooling | **+0,95 pp** (21,56 → 21,13, mas pior que a fusão) |

A conclusão honesta do TC1 é que **a fusão ajuda e a atenção não**. Reportar o
resultado negativo da atenção vale mais do que escondê-lo: ele foi medido com o
mesmo encoder, a mesma base e a mesma semente.

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
| baseline_v2 + baseline_v3 | 15,84% | |
| fusion_v4 + attention_v4 | 19,88% | **controle** |

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

- **O codec Opus custa pouco**: +1,2 a +2,8 pp até 15 kbps, abaixo do que o
  Teams usa na prática.
- **Perder a banda alta custa muito**, e o ruído acústico do interlocutor custa
  mais ainda.

**O ponto de operação não transfere.** O limiar gravado no checkpoint foi
calibrado em áudio limpo. Sob o mesmo Opus a 15 kbps, o recall do v2 **sobe**
(0,72 → 0,86) e o do v4 **cai** (0,55 → 0,39) — mesma perturbação, direções
contrárias. Por isso o monitor exibe **score**, não veredito.

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
| A14 | 12,01% | 17,74% | vocoder clássico |
| A16 | 22,75% | 0,03% | concatenação |
| A07 | 22,43% | 0,02% | vocoder clássico |
| A18 | 15,01% | 4,63% | vocoder clássico |
| A17 | 11,27% | 0,41% | filtragem de forma de onda |
| A19 | 6,98% | 0,00% | filtragem espectral |
| A08 | 3,88% | 0,03% | **neural**, source-filter |
| A09 | 2,32% | 0,12% | vocoder clássico |

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
| RNF02 | F1 ≥ 0,85 | 0,9456 | **atendido — mas ver abaixo** |
| RNF03 | EER ≤ 10% | **13,13%** (melhor) | **NÃO atendido** |

### 8.1 O RNF02 é um requisito mal especificado

Com 89,7% de spoof no eval, um classificador que responde **"spoof" para tudo**
obtém F1 = **0,9456** — acima do exigido, sem olhar para o áudio.

Esse achado vale mais que o cumprimento do requisito. É um resultado de
engenharia de requisitos: **F1 sobre classe majoritária não mede capacidade de
detecção**. A métrica correta para a tarefa é o EER, que é independente de
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
degradação por banda estreita foi medida (+5,35 pp), e a resposta medida é
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

`[REGENERAR]` — resultado da camada 2, quando executada.

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
python scripts/importar_asvspoof2021.py --metadata ... --codec opus --saida p_opus.txt
```

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

**Pendência externa — baselines oficiais do ASVspoof 2019 LA.** A Seção 5.9 do
TC1 precisa dos EERs oficiais de B01/B02 (pooled, eval) de Wang et al. (2020) /
Todisco et al. (2019) como régua externa. **Não foram obtidos**: o proxy do
ambiente bloqueia arxiv, isca-archive, asvspoof.org e datashare.ed.ac.uk, e as
buscas devolveram números contraditórios. **Não citar de memória** — puxar da
fonte primária.

**O áudio ao vivo é o mix.** O loopback entrega a soma de todos os
participantes. Não há atribuição por pessoa; o resultado é sobre o *trecho*, não
sobre quem falou.

**A camada 2 ainda não foi executada.** O código existe e está testado ponta a
ponta com canal simulado, mas nenhuma chamada real foi medida.

**A degradação medida é limite inferior.** Ver 10.1.

**O RNF03 não foi atendido.** Ver 8.2.

**Atalhos existem na base e o modelo depende deles em 1–2%.** Ver Seção 6.

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

1. **Regenerar as tabelas por ataque** (Seção 7). Duas execuções de minutos. É a
   seção que explica *por que* o modelo falha, e hoje está marcada
   `[REGENERAR]`.
2. **ASVspoof 2021 LA, com subamostragem.** Uma comparação pareada: condição de
   referência (sem codec) contra Opus real, nos dois modelos principais.
3. **Obter os baselines oficiais do ASVspoof 2019** (ver Seção 11). Sem eles a
   comparação com a literatura não tem régua.

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
- **O ranking dos modelos se inverte sob degradação de canal** (Seção 5) — o
  resultado central, agora com confirmação em canal real.
- O modelo aprende artefato, não atalho, com dependência residual de 1–2%
  (Seção 6).
- O RNF02 é satisfeito por um classificador trivial; o RNF03 não é atendido
  (Seção 8). Ambos documentados com evidência.

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
