# Redação das seções revisadas do TC1

Texto pronto para colar no documento do TC1, na numeração do PDF atual. Todo
número vem de `RESUMO_TCC.md`; a seção de origem aparece entre colchetes, em
comentário, depois de cada tabela ou parágrafo com dado.

Convenções:

- `[PENDENTE: ...]` marca um valor que ainda não existe ou que não está
  versionado. **Não entregar o texto com essas marcas.**
- As tabelas estão numeradas na ordem em que aparecem (Tabelas 1 a 7). As duas
  tabelas do cronograma (hoje Tabela 1 e Tabela 2) passam a ser a **Tabela 8 e a
  Tabela 9**.
- Citações no formato autor-data da ABNT (NBR 10520). As referências novas e as
  correções estão no fim do arquivo.
- Termos estrangeiros (*deepfake*, *spoof*, *bonafide*, *pooling*) em itálico no
  documento final.

Decisões da equipe já refletidas aqui:

- **A coleta com colaboradores não aconteceu.** Ela sai do Resumo, da 4.9, da
  5.1, da 5.5, da 5.6, da 7.1, do Marco 2 e da Conclusão.
- **A API e o front-end estão com o Pedro** e ainda não foram integrados ao
  repositório. O texto abaixo trata dos dois só como escopo do sistema. Quando
  chegarem, é preciso completar os Marcos 5 a 7.

---

## RESUMO

Este trabalho investiga a detecção de áudios *deepfake* com fusão de
características e mecanismo de atenção, avaliando o ganho de cada componente
sobre uma linha de base e a robustez dos modelos à degradação de canal. Foram
treinados sete modelos na trilha *Logical Access* da base ASVspoof 2019 e
avaliados em 71.237 áudios gerados por treze ataques ausentes do treino, com
remoção de silêncio. A fusão tardia de LFCC e espectrograma log-mel, com
encoders LCNN, reduziu o EER de 21,56% para 20,18% (−1,38 ponto percentual). A
atenção no *pooling*, acrescentada ao modelo com fusão, não trouxe ganho
(+0,95 p.p.). A combinação de scores de dois modelos de arquiteturas diferentes
atingiu 13,13% de EER, e um grupo de controle com modelos semelhantes rendeu
apenas 0,30 p.p. Isso indica que o ganho vem da diversidade entre os modelos. A
análise por ataque mostrou que modelos com EER global parecido erram em ataques
diferentes, e que os ataques com geração autorregressiva (A10, A12 e A15)
concentram a dificuldade. Sob degradação simulada de canal, o ranking dos
modelos se inverte: o melhor modelo no áudio limpo piorou até 23,42 p.p., contra
7,24 p.p. do modelo com fusão. Sob o mesmo protocolo sem silêncio, os resultados
ficam abaixo dos modelos com CQT reportados na literatura (26% a 27%). Nenhum
modelo atingiu as metas de F1 ≥ 0,85 e EER < 8% fixadas no planejamento, e as
causas dessa diferença são discutidas.

**Palavras-chave:** Deepfake. Áudio. Fusão de características. LFCC. ASVspoof.
Robustez.

## ABSTRACT

This work investigates audio deepfake detection with feature fusion and an
attention mechanism, measuring the gain of each component over a baseline and
the models' robustness to channel degradation. Seven models were trained on
the ASVspoof 2019 Logical Access track and evaluated on 71,237 utterances
produced by thirteen attacks unseen in training, with leading and trailing
silence removed. Late fusion of LFCC and log-mel spectrogram with LCNN encoders
reduced the EER from 21.56% to 20.18% (−1.38 percentage points). Attentive
pooling added to the fused model did not help (+0.95 p.p.). Score-level fusion
of two architecturally different models reached 13.13% EER, whereas a control
pair of similar models gained only 0.30 p.p., indicating that the gain comes
from model diversity. Per-attack analysis showed that models with similar pooled
EER fail on different attacks, and that autoregressive waveform generation (A10,
A12, A15) concentrates the difficulty. Under simulated channel degradation the
model ranking inverts: the best clean-condition model degraded by up to 23.42
p.p., against 7.24 p.p. for the fused model. Under the same silence-removed
protocol, results are below those reported for CQT-based models (26% to 27%).
No model reached the planned targets of F1 ≥ 0.85 and EER < 8%, and the reasons
for this gap are discussed.

**Keywords:** Deepfake. Audio. Feature fusion. LFCC. ASVspoof. Robustness.

> Contagem: o Resumo tem cerca de 230 palavras, dentro da faixa de 150 a 500
> que a NBR 6028 dá para trabalhos acadêmicos.

---

## 3.4 Inteligência Artificial Aplicada — trecho a substituir

Troca da frase que atribui a atenção a Vaswani et al.:

> Além disso, mecanismos de atenção (*Attention Mechanisms*) têm sido utilizados
> para direcionar o foco do modelo para as regiões mais relevantes do sinal. Em
> tarefas de voz, uma forma difundida é o *Attentive Statistics Pooling* (OKABE;
> KOSHINAKA; SHINODA, 2018). Nele, cada quadro temporal recebe um peso
> aprendido, e a média e o desvio-padrão ponderados resumem o enunciado. É essa
> a forma adotada neste trabalho. Os mecanismos de autoatenção de múltiplas
> cabeças, popularizados por Vaswani et al. (2017), pertencem a outra família e
> não foram avaliados aqui.

Frase a acrescentar sobre a LCNN, no parágrafo das CNNs:

> Entre as arquiteturas convolucionais aplicadas à detecção de *spoofing*,
> destaca-se a *Light CNN* (LCNN), cuja ativação *Max-Feature-Map* (MFM) mantém,
> para cada par de mapas de características, apenas o maior valor. A LCNN esteve
> entre os melhores sistemas do desafio ASVspoof 2019 (LAVRENTYEVA et al., 2019).

## 4.2 Revisão Bibliográfica — trecho a substituir

Na frase "Entre os trabalhos de maior relevância...":

> Entre os trabalhos de maior relevância para o desenvolvimento deste projeto
> destacam-se Goodfellow, Bengio e Courville (2016), sobre aprendizado profundo;
> Jurafsky e Martin (2023), sobre processamento de fala; Wang et al. (2020), sobre
> a base ASVspoof 2019 e seus ataques; Todisco et al. (2019), sobre o protocolo e
> os sistemas de referência do desafio; Lavrentyeva et al. (2019), sobre a LCNN;
> Okabe, Koshinaka e Shinoda (2018), sobre o *pooling* com atenção; e Müller et
> al. (2021), sobre o efeito do silêncio na base ASVspoof.

## 4.3 Levantamento de Requisitos — nota sobre os requisitos

Parágrafo a acrescentar ao fim da 4.6:

> A avaliação dos requisitos não funcionais, feita na seção 5.6, mostrou que o
> RNF04 está mal especificado para esta base. No conjunto de avaliação, 89,7% dos
> áudios são *spoof*. Um classificador que atribui *spoof* a toda entrada obtém,
> portanto, F1 = 0,9456 sem analisar o sinal, valor que atende ao requisito. Os
> modelos treinados, no limiar calibrado na validação, ficam abaixo de 0,85. O F1
> da classe majoritária não mede capacidade de detecção. Para esta tarefa, a
> métrica adequada é o EER ou o F1 macro.

## 4.7 Ciclo de Vida do Projeto — segundo parágrafo, substituir

> O desenvolvimento foi organizado em três incrementos. No primeiro, foi
> construída a linha de base: LFCC com coeficientes delta e delta-delta e uma
> rede convolucional. No segundo, foi acrescentado um ramo de espectrograma
> log-mel, combinado ao ramo LFCC por fusão tardia. No terceiro, o *pooling* de
> cada ramo passou a ser ponderado por atenção. Cada incremento é comparado com
> o modelo imediatamente anterior, de modo que só um componente muda por vez.
>
> O próprio histórico do projeto ilustra o modelo incremental. A linha de base
> passou por quatro versões antes dos incrementos 2 e 3, e cada versão respondeu
> a uma limitação observada na anterior (Quadro 1). A versão 2 acrescentou
> aumento de dados, recorte aleatório, *pooling* de estatísticas e calibração do
> limiar na validação. A versão 3a isolou a resolução espectral, com 70 filtros
> no lugar de 20. A versão 4 trocou a CNN convencional pela LCNN e passou a
> preservar a estrutura em frequência no *pooling*. Os incrementos 2 e 3 foram
> construídos sobre a versão 4, que é a linha de base dos contrastes controlados
> da seção 5.3.

**Quadro 1 – Evolução dos modelos**

| versão | alteração em relação à anterior | modelo |
|---|---|---|
| v1 | CNN convencional, LFCC (20 filtros) com delta e delta-delta, *pooling* médio, sem aumento de dados | `baseline_lfcc_cnn` |
| v2 | aumento de dados, recorte aleatório, *pooling* de estatísticas, pesos de classe, limiar calibrado | `baseline_lfcc_cnn_v2` |
| v3a | 70 filtros lineares (ablação da resolução espectral) | `baseline_lfcc_cnn_v3a` |
| v3 | v3a com encoder mais largo | `baseline_lfcc_cnn_v3` |
| v4 | encoder LCNN e *pooling* que preserva a frequência | `baseline_lcnn_v4` |
| incremento 2 | + ramo de espectrograma log-mel (fusão) | `fusion_lcnn_v4` |
| incremento 3 | + atenção no *pooling* | `attention_lcnn_v4` |

Fonte: Autoria própria.

> Observação para o texto: os coeficientes delta e delta-delta **não** pertencem
> ao incremento 2. Eles estão no LFCC desde a v1. Por isso o PDF atual precisa
> mudar na 4.7 ("incorporando espectrogramas e coeficientes delta"), na 5.3 e
> no README do repositório.

## 4.8 Tecnologias Utilizadas — item a acrescentar

> • SoundFile (libsndfile): leitura de áudio e codificação/decodificação Opus,
> usada para simular o canal de voz sobre IP nos testes de robustez.

[PENDENTE: acrescentar FastAPI e React quando o código do Pedro for integrado.]

## 4.9 Base de Dados — substituir por inteiro

> Os experimentos utilizam a trilha *Logical Access* (LA) da base ASVspoof 2019
> (WANG et al., 2020). A base contém áudios autênticos (*bonafide*) e sintéticos
> (*spoof*), gerados por sistemas de síntese de fala (TTS) e de conversão de voz
> (VC). A escolha se justifica pela ampla utilização da base na literatura e
> pelos protocolos padronizados de treino, validação e avaliação, que permitem
> comparar os resultados com outros trabalhos (TODISCO et al., 2019).
>
> A integridade dos arquivos foi verificada antes do treino, com a decodificação
> completa de cada arquivo e não apenas a leitura do cabeçalho. Um arquivo FLAC
> truncado informa a duração original no cabeçalho e só falharia durante o
> treinamento.

**Tabela 1 – Partições da ASVspoof 2019 LA**

| partição | áudios | ataques |
|---|---|---|
| treino | 25.380 | A01–A06 |
| validação | 24.844 | A01–A06 |
| avaliação | 71.237 | A07–A19 |
| **total** | **121.461** | |

Fonte: Autoria própria, a partir dos protocolos da base (WANG et al., 2020).

<!-- RESUMO_TCC §2 -->

> O conjunto de avaliação tem 7.355 áudios *bonafide* e 63.882 *spoof*, sendo
> 4.914 áudios por ataque, o que corresponde a 89,7% de *spoof*.

## 4.10 Divisão dos Dados — substituir por inteiro

> A divisão segue os protocolos oficiais da base. O conjunto de treinamento é
> usado no aprendizado dos pesos. O de validação é usado na escolha da época
> (critério: menor EER), no agendamento da taxa de aprendizado, na parada
> antecipada e na calibração do limiar de decisão. O de avaliação é usado apenas
> uma vez, na medição final.
>
> Os treze ataques do conjunto de avaliação (A07 a A19) não aparecem no
> treinamento. Dois deles, A16 e A19, usam os mesmos algoritmos de A04 e A06, mas
> em outro conjunto de locutores. Todo resultado de avaliação deste trabalho mede,
> portanto, generalização para ataques desconhecidos, e não desempenho na
> distribuição de treino. Esse desenho explica por que os EERs ficam na casa das
> dezenas de pontos percentuais.

## 4.11 Métricas de Avaliação — acrescentar ao fim

> Nas métricas dependentes de limiar (*accuracy*, *precision*, *recall* e F1), a
> classe positiva é a *spoof*. O limiar de decisão é o ponto de EER calibrado no
> conjunto de validação e aplicado sem ajuste ao conjunto de avaliação. O EER não
> depende de limiar.
>
> • *tandem Detection Cost Function* (t-DCF): métrica oficial do desafio
> ASVspoof. Ela avalia a contramedida em conjunto com um sistema de verificação
> automática de locutor e pondera o custo de cada tipo de erro (KINNUNEN et al.,
> 2018). O min t-DCF foi calculado com o *script* e os scores de verificação de
> locutor fornecidos pelos organizadores.
> [PENDENTE: rodar o t-DCF; ver RESUMO_TCC §11. Se não rodar, retirar este item.]

## 4.12 Arquitetura do Modelo — seção nova

> Se preferirem não renumerar o capítulo, esta seção pode entrar como 4.12. Uma
> posição mais natural seria logo depois da 4.8, e aí as seguintes mudam de número.

> **Pré-processamento.** Os áudios são processados a 16 kHz, a taxa nativa da base. O silêncio
> inicial e final é removido com limiar de 30 dB abaixo do pico, e a amplitude é
> normalizada pelo pico. Cada amostra é ajustada a uma janela fixa de 4 s: os
> áudios mais curtos são completados por repetição e os mais longos são
> recortados, com posição aleatória no treinamento.
>
> **Extração de características.** Duas representações são calculadas a partir
> da mesma STFT (FFT de 512 pontos, janela de 25 ms e passo de 10 ms), que é
> calculada uma única vez:
>
> • LFCC: 20 coeficientes obtidos de um banco de 70 filtros lineares, com
> coeficientes delta e delta-delta, totalizando uma matriz de 60 × 401 por
> amostra;
>
> • espectrograma log-mel: 80 bandas na escala Mel, em uma matriz de 80 × 401.
>
> **Encoder.** Cada representação é processada por um encoder LCNN próprio, com
> ativação *Max-Feature-Map* (LAVRENTYEVA et al., 2019). Após o encoder, o
> *pooling* calcula a média e o desvio-padrão temporais em quatro faixas de
> frequência, o que preserva parte da estrutura espectral.
>
> **Fusão tardia.** No modelo de fusão, os vetores produzidos pelos dois ramos
> são concatenados depois do *pooling*, e uma camada de classificação com
> *dropout* de 0,3 decide entre *bonafide* e *spoof*.
>
> **Atenção.** No modelo com atenção, a média e o desvio-padrão de cada ramo são
> ponderados por pesos temporais aprendidos, segundo o *Attentive Statistics
> Pooling* (OKABE; KOSHINAKA; SHINODA, 2018). Os pesos vêm de uma projeção
> linear de cada quadro seguida de *softmax* no tempo. O mecanismo acrescenta
> 258 parâmetros ao modelo de fusão (349.124 contra 348.866).
>
> **Treinamento.** Otimizador Adam (KINGMA; BA, 2015), com taxa de aprendizado
> de 10⁻³, decaimento de pesos de 10⁻⁴ e lotes de 32 amostras, por até 50
> épocas. A taxa de aprendizado cai pela metade após três épocas sem melhora no
> EER de validação. O treino para após doze épocas sem melhora, e o checkpoint
> com menor EER de validação é mantido. A função de perda é a entropia cruzada
> com pesos de classe proporcionais à raiz quadrada do inverso da frequência. O
> gradiente é limitado a norma 5 e a semente é fixa (42).
>
> **Aumento de dados.** Durante o treinamento, cada amostra recebe, com
> probabilidade de 0,5 para cada operação: ruído gaussiano branco com relação
> sinal-ruído entre 10 e 30 dB; ganho entre −6 e +6 dB; e deslocamento temporal
> de até 10% da janela.
>
> **Ambiente.** PyTorch, com treinamento em uma GPU NVIDIA GeForce GTX 1650. Cada
> treinamento levou cerca de 3 horas.

---

## 5. EXPERIMENTOS E RESULTADOS — introdução, substituir

> Esta seção apresenta os resultados dos experimentos. Todos os valores foram
> medidos no conjunto de avaliação completo da ASVspoof 2019 LA (71.237 áudios),
> composto de ataques ausentes do treinamento, com uma execução por modelo
> (semente 42). As limitações decorrentes desse desenho são discutidas na seção
> 5.7.

## 5.1 Configuração Experimental — substituir por inteiro

> Os experimentos seguem os protocolos oficiais da ASVspoof 2019 LA (seção 4.10)
> e a arquitetura descrita na seção 4.12. Foram treinados sete modelos: as quatro
> versões da linha de base com CNN convencional (v1, v2, v3a e v3), a linha de
> base com LCNN (v4) e os dois incrementos construídos sobre ela, com fusão e com
> fusão e atenção. Todos usam o mesmo pré-processamento e as mesmas partições.
> Os modelos v2 em diante usam também o mesmo aumento de dados.

## 5.2 Validação Quantitativa — substituir por inteiro

**Tabela 2 – Desempenho dos modelos isolados no conjunto de avaliação**

| modelo | EER (%) | *precision* | *recall* | F1 | *accuracy* |
|---|---|---|---|---|---|
| baseline_lfcc_cnn_v2 | **18,99** | 0,9838 | 0,7184 | **0,8304** | 0,7368 |
| fusion_lcnn_v4 | 20,18 | 0,9999 | 0,5508 | 0,7103 | 0,5971 |
| baseline_lfcc_cnn | 20,78 | 0,9814 | 0,7004 | 0,8174 | 0,7194 |
| baseline_lfcc_cnn_v3 | 21,10 | 0,9987 | 0,5749 | 0,7297 | 0,6181 |
| attention_lcnn_v4 | 21,13 | 0,9998 | 0,5374 | 0,6990 | 0,5851 |
| baseline_lfcc_cnn_v3a | 21,23 | 0,9986 | 0,6196 | 0,7647 | 0,6581 |
| baseline_lcnn_v4 | 21,56 | 0,9985 | 0,5284 | 0,6911 | 0,5764 |

Fonte: Autoria própria. Classe positiva: *spoof*. *Precision*, *recall*, F1 e
*accuracy* no limiar calibrado no conjunto de validação.

<!-- scripts/make_report.py, outputs/report/comparativo_eval.md (25/09/2026) -->

> Os sete modelos ficam entre 18,99% e 21,56% de EER. O melhor é a linha de
> base v2, uma CNN convencional só com LFCC. O F1 varia de 0,69 a 0,83, e
> nenhum modelo atinge o 0,85 do RNF04.
>
> A decomposição do F1 mostra por quê. A *precision* fica entre 0,98 e 0,9999:
> quase tudo o que os modelos classificam como *spoof* é de fato *spoof*. O
> *recall* fica entre 0,53 e 0,72: de um quarto a metade dos áudios sintéticos
> passa como autêntico. O limiar calibrado na validação é, portanto,
> conservador demais para a avaliação. Na validação, os ataques são os mesmos do
> treino e os scores de *spoof* ficam altos. Nos ataques inéditos, boa parte dos
> scores de *spoof* cai abaixo desse limiar. O efeito é mais forte nos modelos
> LCNN (*recall* de 0,53 a 0,55) do que nas CNNs v1 e v2 (0,70 a 0,72). Isso
> explica por que o fusion_v4 tem EER próximo ao do v2 e F1 bem menor: o EER
> não depende de limiar, e o F1 depende.
>
> [PENDENTE: parágrafo "a validação não prevê a avaliação", com o EER de
> validação de cada modelo (ver comando em RESUMO_TCC §3).]

## 5.3 Comparação com Baseline — substituir por inteiro

> Cada incremento foi avaliado contra o modelo imediatamente anterior, e só o
> componente avaliado muda entre os dois (Tabela 3).

**Tabela 3 – Efeito isolado de cada incremento**

| incremento | comparação | EER (%) | variação |
|---|---|---|---|
| fusão de características | baseline_lcnn_v4 → fusion_lcnn_v4 | 21,56 → 20,18 | −1,38 p.p. |
| atenção no *pooling* | fusion_lcnn_v4 → attention_lcnn_v4 | 20,18 → 21,13 | +0,95 p.p. |

Fonte: Autoria própria.

> A fusão de características reduziu o EER em 1,38 p.p. A atenção, acrescentada
> ao modelo com fusão, aumentou o EER em 0,95 p.p. Esse resultado negativo vale
> para a forma de atenção avaliada: uma única projeção linear por quadro, com
> 258 parâmetros, aplicada a cerca de 25 quadros depois das reduções temporais
> do encoder. Ele não permite concluir sobre mecanismos de atenção mais
> expressivos.
>
> As duas variações têm magnitude menor que a variação entre execuções que
> Müller et al. (2021) observam para modelos semelhantes nesta base (de 1,4 a
> 5,2 p.p.). Como cada modelo foi treinado uma única vez, nenhuma das duas pode
> ser atribuída ao método com segurança. A seção 5.4 apresenta evidência mais
> forte sobre o efeito da fusão.
>
> **Comparação com a literatura.** Os sistemas de referência oficiais do desafio
> alcançam 8,09% (B02, LFCC-GMM) e 9,57% (B01, CQCC-GMM) de EER no mesmo conjunto
> de avaliação (TODISCO et al., 2019). Esses sistemas, porém, processam o áudio
> **com** o silêncio. Müller et al. (2021) mostram que, na ASVspoof 2019, a
> duração do silêncio inicial sozinha separa as classes com 15,12% de EER, e que
> a remoção do silêncio piora o RawNet2 de 3,61% para 15,50%. Como este trabalho
> remove o silêncio, a comparação equivalente é com os modelos que Müller et al.
> (2021) treinaram e avaliaram sob o mesmo protocolo (Tabela 4).

**Tabela 4 – Comparação sob o mesmo protocolo (silêncio removido)**

| sistema | EER (%) |
|---|---|
| ResNet (CQT) — Müller et al. (2021) | 27,23 ± 3,2 |
| LSTM (CQT) — Müller et al. (2021) | 27,28 ± 1,4 |
| CNN (CQT) — Müller et al. (2021) | 26,27 ± 3,5 |
| RawNet2 — Müller et al. (2021) | 15,50 ± 5,2 |
| baseline_lfcc_cnn_v2 (este trabalho) | 18,99 |
| fusion_lcnn_v4 (este trabalho) | 20,18 |
| fusão de scores v2 + fusion_v4 (este trabalho) | 13,13 |

Fonte: Autoria própria; Müller et al. (2021), Tabela 2.

<!-- RESUMO_TCC §5.1 -->

> Os sete modelos ficam abaixo dos três modelos baseados em CQT, e a fusão de
> scores fica abaixo da média do RawNet2. Algumas diferenças de protocolo
> permanecem. Müller et al. (2021) usam limiar de remoção de silêncio de 40 dB,
> contra 30 dB aqui, e o áudio inteiro, contra a janela fixa de 4 s. Eles
> reportam também média e desvio de várias execuções, enquanto aqui há uma
> execução por modelo.

## 5.4 Análise de Erros — substituir por inteiro

> O EER global resume em um único número ataques de dificuldade muito diferente.
> A Tabela 5 separa o desempenho dos dois modelos principais por ataque. Cada
> ataque tem 4.914 áudios e é comparado com os mesmos 7.355 áudios *bonafide*.

**Tabela 5 – EER (%) por ataque**

| ataque | gerador de forma de onda | baseline_v2 | fusion_v4 |
|---|---|---|---|
| A07 | vocoder (WORLD) | 22,43 | **0,02** |
| A08 | *neural source-filter* | 3,88 | **0,03** |
| A09 | vocoder (Vocaine) | 2,32 | **0,12** |
| A10 | WaveRNN (autorregressivo) | **30,63** | 45,54 |
| A11 | Griffin-Lim | **3,85** | 33,74 |
| A12 | WaveNet (autorregressivo) | **36,18** | 47,99 |
| A13 | concatenação e filtragem | 36,50 | **6,54** |
| A14 | vocoder (STRAIGHT) | **12,01** | 17,74 |
| A15 | WaveNet (autorregressivo) | **15,02** | 29,37 |
| A16 | concatenação | 22,75 | **0,03** |
| A17 | filtragem de forma de onda | 11,27 | **0,41** |
| A18 | vocoder (MFCC) | 15,01 | **4,63** |
| A19 | filtragem espectral | 6,98 | **0,00** |
| **global** | | **18,99** | 20,18 |

Fonte: Autoria própria; geradores segundo Wang et al. (2020).

<!-- RESUMO_TCC §7. CONFERIR a coluna de geradores na tabela de ataques de
     Wang et al. (2020) antes de entregar. -->

> **Os modelos erram em ataques diferentes.** Os EERs globais são próximos, mas
> os perfis são quase opostos. O fusion_v4 vence em oito dos treze ataques, e o
> baseline_v2 nos outros cinco, com diferenças de até 30 p.p. nos dois sentidos.
> O A13 é o pior ataque para o baseline_v2 (36,50%) e é quase resolvido pelo
> fusion_v4 (6,54%). Com o A11 acontece o inverso: 3,85% no baseline_v2 e 33,74%
> no fusion_v4. O A11 também mostra o custo da fusão de características. No
> ataque por Griffin-Lim, o ramo de espectrograma acrescentado piora o modelo em
> 30 p.p. em relação ao modelo só com LFCC. O ganho agregado de 1,38 p.p. é,
> portanto, um saldo entre ataques, e não um ganho uniforme.
>
> **A geração autorregressiva concentra a dificuldade.** Os três ataques com
> geradores autorregressivos (A10, A12 e A15) têm EER médio de 27,28% no
> baseline_v2 e de 40,97% no fusion_v4. Nos outros dez ataques, as médias são de
> 13,70% e 6,33%. Um teste de permutação exato, sobre os 286 trios possíveis
> entre os treze ataques, dá p = 0,0070 para o fusion_v4 e p = 0,0490 para o
> baseline_v2. O A08, que usa um gerador neural não autorregressivo, é resolvido
> pelos dois modelos (3,88% e 0,03%). Isso indica que o eixo relevante não é a
> distinção entre gerador neural e clássico. A explicação tem, contudo, um
> limite: A12 e A15 usam o mesmo WaveNet e diferem em 21,16 p.p. no baseline_v2.
> Outras partes do sistema de ataque também pesam. A10 e A12 são os únicos
> ataques em que os dois modelos passam de 30% de EER.
>
> **Fusão de scores.** A complementaridade observada motivou a combinação dos
> scores de modelos treinados separadamente (Tabela 6).

**Tabela 6 – Fusão de scores**

| combinação | EER (%) |
|---|---|
| baseline_v2 + fusion_v4 | **13,13** |
| baseline_v2 + attention_v4 | 13,47 |
| baseline_v2 + baseline_v3 | 15,84 |
| fusion_v4 + attention_v4 (controle) | 19,88 |

Fonte: Autoria própria.

<!-- RESUMO_TCC §4 -->

> A combinação de modelos de arquiteturas diferentes reduziu o EER de 18,99%, o
> do melhor modelo isolado, para 13,13%, um ganho de 5,86 p.p. O par de controle
> (fusion_v4 e attention_v4) reúne dois modelos quase idênticos e rendeu apenas
> 0,30 p.p. O ganho vem, portanto, da diversidade entre os modelos, e não do ato
> de combinar scores. O valor de 13,13% usa a regra de postos, que exige o
> conjunto completo de scores. A média simples, aplicável a um fluxo contínuo de
> áudio, resulta em 14,03%.
>
> **Verificação de atalhos.** Na ASVspoof 2019, variáveis triviais como duração
> e energia separam parcialmente as classes. Após a remoção do silêncio, a
> duração no treino ainda dá EER de 43,80%, significativo em teste de
> permutação. Dentro de cada classe, a correlação entre o score do modelo e
> essas variáveis explica de 1% a 2% da variância do score. O modelo depende
> delas de forma mensurável, mas pequena.

## 5.5 Testes de Robustez — substituir por inteiro

> Para avaliar a robustez ao canal, o conjunto de avaliação completo foi
> reprocessado sob diferentes degradações: ruído gaussiano branco, variação de
> ganho, codificação Opus em taxas típicas de voz sobre IP e limitação de banda
> (reamostragem para 8 kHz e retorno a 16 kHz), como na telefonia. A Tabela 7 mostra as condições
> mais representativas.

**Tabela 7 – EER (%) sob degradação de canal**

| condição | baseline_v2 | fusion_v4 |
|---|---|---|
| limpo | **18,99** | 20,18 |
| Opus, 25 kbps | **20,04** | 22,08 |
| banda estreita (8 kHz) | 35,95 | **25,53** |
| ruído, SNR de 5 dB | 42,41 | **27,42** |
| **pior degradação** | **+23,42 p.p.** | **+7,24 p.p.** |

Fonte: Autoria própria.

<!-- RESUMO_TCC §5. [PENDENTE: tabela completa das 11 condições, que está na
     cópia de agosto.] -->

> Sob degradação, o ranking dos modelos se inverte. O baseline_v2 é o melhor no
> áudio limpo e piora até 23,42 p.p. O fusion_v4 piora no máximo 7,24 p.p. e
> vence por 10,4 p.p. em banda estreita e por 15,0 p.p. com ruído a 5 dB. A
> codificação Opus custa pouco, de 1,2 a 2,8 p.p. até 15 kbps. Já a perda da
> banda alta e o ruído aditivo forte custam muito. Na prática, escolher o modelo
> pelo EER no áudio limpo, como a literatura costuma reportar, levaria à escolha
> errada em uma aplicação com canal degradado.
>
> Duas ressalvas delimitam esse resultado. Primeiro, o treinamento usa ruído
> branco entre 10 e 30 dB e ganho de ±6 dB, gerados pela mesma função dos testes.
> As condições de ruído a 20 e 10 dB e de ganho, portanto, já estavam na
> distribuição de treino. As condições realmente inéditas são o ruído a 5 dB, a
> codificação Opus e a banda estreita, justamente as duas em que a inversão
> aparece. A comparação entre os modelos continua válida, porque ambos tiveram o
> mesmo aumento de dados. Segundo, o limiar calibrado em áudio limpo não se
> transfere para o áudio degradado. Sob Opus a 15 kbps, o *recall* do
> baseline_v2 sobe de 0,72 para 0,86 e o do fusion_v4 cai de 0,55 para 0,39.
>
> As degradações foram simuladas em *software*. Nenhuma avaliação foi feita em
> canal real nem com arquivos MP3, e a avaliação em vozes externas à base, com
> gravações de colaboradores, prevista no planejamento, não foi realizada.

## 5.6 Resultados Esperados e Obtidos — substituir por inteiro

> O planejamento do TC1 estabeleceu três projeções, e nenhuma se confirmou
> integralmente (Quadro 2).

**Quadro 2 – Resultados esperados e obtidos**

| projeção | obtido | situação |
|---|---|---|
| F1 superior a 0,90 | 0,69 a 0,83 | não atingido |
| EER inferior a 8% | 18,99% (modelo isolado); 13,13% (fusão de scores) | não atingido |
| ganho a cada incremento | fusão −1,38 p.p.; atenção +0,95 p.p. | só na fusão, dentro da variância entre execuções |
| processamento em até 30 s por amostra (RNF06) | 2,745 s em CPU para um arquivo de 60 s | atingido |

Fonte: Autoria própria.

> A meta de EER tomou como referência o sistema B02 (8,09%), que processa o
> áudio com o silêncio. Sob o protocolo sem silêncio, que remove um atalho
> conhecido da base (MÜLLER et al., 2021), os resultados deste trabalho ficam
> abaixo dos modelos com CQT da literatura, como mostra a Tabela 4. A meta de F1
> esbarra no desequilíbrio das classes, discutido na seção 4.6: o F1 da classe
> majoritária não mede capacidade de detecção.

## 5.7 Limitações — seção nova

> • **Uma execução por modelo.** Os efeitos dos incrementos (−1,38 e +0,95 p.p.)
> são menores que a variação entre execuções observada na literatura (1,4 a 5,2
> p.p.). Os efeitos grandes (diferenças por ataque de até 30 p.p., inversão de
> ranking de 10,4 e 15,0 p.p. e ganho de 5,86 p.p. da fusão de scores) superam
> essa faixa.
>
> • **Atenção mínima.** O resultado negativo vale para a forma de atenção
> avaliada.
>
> • **Robustez parcialmente dentro da distribuição.** Ver seção 5.5.
>
> • **Canal simulado.** Não houve avaliação em canal real (ASVspoof 2021 LA ou
> chamada gravada).
>
> • **Sem vozes externas.** A coleta com colaboradores não foi realizada, e todos
> os áudios *bonafide* vêm da ASVspoof 2019.
>
> • **Atalhos da base.** O modelo depende de duração e energia em 1% a 2% da
> variância do score.

---

## 6.4 Marcos do Projeto — ajustes

- **Marco 2:** retirar "amostras de colaboradores coletadas".
- **Marco 6:** trocar "Testes de robustez realizados" por "Testes de robustez
  realizados com degradação simulada de canal".
- **Marcos 5 a 7 (API e front-end):** [PENDENTE: status do código do Pedro.]

## 7.1 Privacidade e Utilização dos Dados — substituir por inteiro

> Os experimentos utilizam exclusivamente a base pública ASVspoof 2019, que
> tem finalidade de pesquisa científica e é distribuída sob licença própria dos
> organizadores. Nenhuma voz foi coletada de pessoas externas à base. A coleta
> de amostras com colaboradores, prevista no planejamento, não foi realizada. Se
> for retomada em trabalhos futuros, deverá observar a Lei Geral de Proteção de
> Dados (Lei nº 13.709/2018 – LGPD). Características vocais podem ser
> consideradas dados biométricos e, portanto, dados pessoais sensíveis, o que
> exige consentimento específico, finalidade declarada e descarte ao fim da
> pesquisa.

## 8. CONCLUSÃO — substituir por inteiro

> Este trabalho investigou a detecção de áudios *deepfake* com fusão de
> características e mecanismo de atenção na base ASVspoof 2019 LA, sob um
> protocolo que remove o silêncio e avalia apenas ataques ausentes do
> treinamento. Foram treinados e comparados sete modelos, com um desenho
> incremental em que cada componente é avaliado contra o modelo imediatamente
> anterior.
>
> A fusão tardia de LFCC e espectrograma log-mel reduziu o EER em 1,38 p.p. A
> atenção no *pooling* não trouxe ganho. Ambos os efeitos ficam dentro da
> variação entre execuções reportada na literatura. Quatro achados, porém, têm
> magnitude que supera essa variação.
>
> Primeiro, o desempenho na validação não antecipa o desempenho na avaliação,
> porque a validação contém apenas os ataques do treinamento.
> [PENDENTE: número da tabela validação × avaliação.]
>
> Segundo, a fusão de scores de modelos diferentes reduziu o EER para 13,13%. O
> grupo de controle mostra que o ganho vem da diversidade entre os modelos, que
> erram em ataques diferentes.
>
> Terceiro, o EER agregado esconde os ataques mais fortes. Nos ataques com
> geração autorregressiva A10 e A12, os dois modelos principais ficam entre
> 30,63% e 47,99% de EER, próximos do acaso, enquanto outros ataques são
> resolvidos quase perfeitamente.
>
> Quarto, o ranking dos modelos se inverte sob degradação de canal. O melhor
> modelo no áudio limpo piora até 23,42 p.p., e o modelo com fusão, 7,24 p.p.
> Escolher o modelo pelo desempenho em áudio limpo leva, portanto, à escolha
> errada em cenários reais.
>
> As metas de F1 superior a 0,90 e EER inferior a 8% não foram atingidas. A
> segunda se baseava em sistemas que usam o silêncio como atalho. Sob o mesmo
> protocolo, os resultados ficam abaixo dos modelos com CQT da literatura. A
> primeira revelou que o F1 da classe majoritária não é métrica adequada para
> esta base.
>
> Como trabalhos futuros, destacam-se: repetir os treinamentos com várias
> sementes; avaliar os modelos em canal real (ASVspoof 2021 LA) e em vozes
> externas à base; calcular o t-DCF; investigar mecanismos de atenção mais
> expressivos; e usar encoders pré-treinados em fala, direção apontada pela
> literatura para os ataques autorregressivos.

---

## 9. REFERÊNCIAS — correções e acréscimos

Há **erros nas referências atuais** do PDF. Conferir cada uma na fonte antes da
entrega.

**Corrigir:**

- "TODISCO, Massimiliano et al. *ASVspoof 2019: A Large-scale Public Database of
  Synthesized, Converted and Replayed Speech*. Computer Speech & Language, v.
  64, 2021." Esse artigo é de **WANG, Xin et al.**, publicado em **2020**. O
  artigo de Todisco et al. sobre o desafio de 2019 é outro (ver abaixo). As
  citações "(TODISCO et al., 2021)" no texto precisam ser revistas uma a uma.
- A referência de Wang et al. (2020) sobre *Neural Source-Filter* não é citada
  no texto. Remover, ou citar ao descrever o A08.
- "ZHANG, Bohao et al. AASIST": o primeiro autor do AASIST é **JUNG, Jee-weon**.
- "CHEN, Tao et al. UR Channel-Robust...": o título é *UR Channel-Robust
  Synthetic Speech Detection System for ASVspoof 2021*, e o primeiro autor é
  **CHEN, Xinhui**.
- Vaswani et al. (2017): manter só se continuar citado na 3.4, como outra
  família de atenção.

**Acrescentar (NBR 6023):**

KINGMA, Diederik P.; BA, Jimmy. Adam: a method for stochastic optimization. In:
INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS, 3., 2015, San Diego.
*Proceedings* [...]. San Diego: ICLR, 2015.

KINNUNEN, Tomi et al. t-DCF: a detection cost function for the tandem
assessment of spoofing countermeasures and automatic speaker verification. In:
THE SPEAKER AND LANGUAGE RECOGNITION WORKSHOP (ODYSSEY), 2018, Les Sables
d'Olonne. *Proceedings* [...]. 2018. p. 312-319.
*(Só se o t-DCF entrar.)*

LAVRENTYEVA, Galina et al. STC antispoofing systems for the ASVspoof2019
challenge. In: INTERSPEECH, 2019, Graz. *Proceedings* [...]. 2019.
p. 1033-1037.

MÜLLER, Nicolas M. et al. Speech is silver, silence is golden: what do
ASVspoof-trained models really learn? In: ASVSPOOF 2021 WORKSHOP, 2021.
*Proceedings* [...]. 2021. p. 55-60.

OKABE, Koji; KOSHINAKA, Takafumi; SHINODA, Koichi. Attentive statistics pooling
for deep speaker embedding. In: INTERSPEECH, 2018, Hyderabad. *Proceedings*
[...]. 2018. p. 2252-2256.

TODISCO, Massimiliano et al. ASVspoof 2019: future horizons in spoofed and fake
audio detection. In: INTERSPEECH, 2019, Graz. *Proceedings* [...]. 2019.
p. 1008-1012.

WANG, Xin et al. ASVspoof 2019: a large-scale public database of synthesized,
converted and replayed speech. *Computer Speech & Language*, v. 64, 101114,
2020.

> As páginas, cidades e números de artigo acima foram escritos de memória.
> Conferir cada um na fonte (DOI ou anais) antes da entrega.
