# Redação do relatório do TC2

Texto pronto para o relatório do TC2. Ele parte do documento do TC1 (o PDF da
proposta) e segue a mesma numeração: cada seção abaixo diz se substitui, altera
ou acrescenta algo ao texto do TC1. Todo
número vem de `RESUMO_TCC.md`; a seção de origem aparece entre colchetes, em
comentário, depois de cada tabela ou parágrafo com dado.

Convenções:

- `[PENDENTE: ...]` marca um valor que ainda não existe ou que não está
  versionado. **Não entregar o texto com essas marcas.**
- As tabelas estão numeradas na ordem em que aparecem (Tabelas 1 a 11). As duas
  tabelas do cronograma (hoje Tabela 1 e Tabela 2) passam a ser a **Tabela 12 e
  a Tabela 13**, e as dos apêndices, a **Tabela 14 e a Tabela 15**.
- Números de 25/09/2026: `scripts/make_report.py` (comparativo, por ataque,
  robustez) e `outputs/*_history.json` (EER de validação). Canal real
  (ASVspoof 2021 LA): RESUMO_TCC §5.3, de 28/09/2026. Captura ao vivo e voz
  própria: RESUMO_TCC §10.2.1 a §10.2.6, de 29/09 a 01/10/2026.
- Citações no formato autor-data da ABNT (NBR 10520). As referências novas e as
  correções estão no fim do arquivo.
- Termos estrangeiros (*deepfake*, *spoof*, *bonafide*, *pooling*) em itálico no
  documento final.

Decisões da equipe já refletidas aqui:

- **A coleta com colaboradores não aconteceu.** Ela sai do Resumo, da 4.9, da
  5.1, da 5.5, da 5.6, da 7.1, do Marco 2 e da Conclusão.
- **O sistema entregue é o do repositório:** a interface web (`web/`, FastAPI
  e Jinja2, numa aplicação só, em vez de uma API separada mais um front-end
  React) e o monitor de linha de comando (`monitor.py`), descritos na seção 4.13.
- **A coleta com colaboradores foi substituída por gravações do próprio
  autor**, usadas só como controle pareado (seção 5.5).

---

## RESUMO

Este trabalho investiga a detecção de áudios *deepfake* com fusão de
características e mecanismo de atenção, avaliando o ganho de cada componente
sobre uma linha de base e a robustez dos modelos à degradação de canal. Foram
treinados sete modelos na trilha *Logical Access* da base ASVspoof 2019, com
remoção de silêncio, e avaliados em 71.237 áudios, dos quais 63.882 sintéticos,
gerados por treze ataques: onze inéditos e dois que reutilizam algoritmos do
treino. O desempenho na validação não antecipou o da avaliação: modelos com EER
de validação entre 0,03% e 9,74% ficaram todos entre 18,99% e 21,56% na
avaliação. A fusão tardia de LFCC e espectrograma log-mel, com encoders LCNN,
reduziu o EER de 21,56% para 20,18% (−1,38 ponto percentual). A atenção no
*pooling*, acrescentada ao modelo com fusão, não trouxe ganho (+0,95 p.p.). A
combinação de scores de dois modelos de arquiteturas diferentes atingiu 13,13%
de EER e min t-DCF de 0,3129, e um grupo de controle com modelos semelhantes
rendeu apenas 0,30 p.p. Isso indica que o ganho vem da diversidade entre os
modelos. A análise por ataque mostrou que modelos com EER global parecido erram
em ataques diferentes, e que dois ataques com geração autorregressiva, A10 e
A12, resistem aos sete modelos. Sob degradação simulada de canal, o ranking dos
modelos se inverteu: o melhor modelo no áudio limpo piorou até 23,42 p.p.,
contra 7,24 p.p. do modelo com fusão. Em transmissão real por Opus (ASVspoof
2021 LA), porém, a inversão não se repetiu, e o canal custou de 7 a 11 vezes o
que a simulação do mesmo codec indicava. A degradação simulada não previu qual
modelo resiste ao canal real, enquanto o ganho da fusão de scores se manteve
(24,94% contra 29,43%). Sob o mesmo protocolo sem silêncio, os modelos obtiveram
EER menor que o dos modelos com CQT reportados na literatura (26% a 27%).
Nenhum modelo atingiu as metas do planejamento, F1 de 0,85 (requisito) a 0,90
(projeção) e EER inferior a 8%, e as causas dessa diferença são discutidas. O
modelo foi integrado a um sistema de análise com interface web, que recebe
arquivos ou captura o áudio do computador em tempo real. Na captura, a perda
vinha do filtro de conversão de taxa, que apagava a faixa de 7,6 a 8 kHz, da
qual o modelo depende: com um filtro FIR que a preserva, o custo caiu de 5,68
para 0,39 p.p., sem retreino. Um controle pareado com a voz do autor, refeita
por *vocoders*, mostrou que, em vozes externas à base, o modelo responde às
condições de gravação, e não aos artefatos de síntese.

**Palavras-chave:** Deepfake. Áudio. Fusão de características. LFCC. ASVspoof.
Robustez de canal. Tempo real.

## ABSTRACT

This work investigates audio deepfake detection with feature fusion and an
attention mechanism, measuring the gain of each component over a baseline and
the models' robustness to channel degradation. Seven models were trained on
the ASVspoof 2019 Logical Access track, with leading and trailing silence
removed, and evaluated on 71,237 utterances, 63,882 of them spoofed by thirteen
attacks: eleven unseen and two that reuse training algorithms. Validation
performance did not anticipate evaluation performance: models with validation
EER between 0.03% and 9.74% all scored between 18.99% and 21.56% on
evaluation. Late fusion of LFCC and log-mel spectrogram with LCNN encoders
reduced the EER from 21.56% to 20.18% (−1.38 percentage points). Attentive
pooling added to the fused model did not help (+0.95 p.p.). Score-level fusion
of two architecturally different models reached 13.13% EER and a min t-DCF of
0.3129, whereas a control pair of similar models gained only 0.30 p.p.,
indicating that the gain comes from model diversity. Per-attack analysis showed
that models with similar pooled EER fail on different attacks, and that two
attacks with autoregressive waveform generation, A10 and A12, withstand all
seven models. Under simulated channel degradation the model ranking inverted:
the best clean-condition model degraded by up to 23.42 p.p., against 7.24 p.p.
for the fused model. Under real Opus transmission (ASVspoof 2021 LA), however,
the inversion did not recur, and the channel cost 7 to 11 times what the
simulation of the same codec indicated. Simulated degradation did not predict
which model withstands the real channel, whereas the score-fusion gain held
(24.94% against 29.43%). Under the same silence-removed protocol, the models
achieved lower EER than CQT-based models reported in the literature (26% to
27%). No model reached the planned targets of an F1 of 0.85 (requirement) to
0.90 (projection) and an EER below 8%, and the reasons for this gap are
discussed. The model was integrated into an analysis system with a web
interface that accepts files or captures the computer's audio in real time.
The capture loss came from the sample-rate converter's filter, which removed
the 7.6 to 8 kHz band on which the model depends: with an FIR filter that
preserves it, the cost fell from 5.68 to 0.39 p.p. without retraining. A paired
control with the author's own voice, resynthesized by vocoders, showed that on
voices outside the dataset the model responds to recording conditions rather
than to synthesis artifacts.

**Keywords:** Deepfake. Audio. Feature fusion. LFCC. ASVspoof. Channel robustness.
Real time.

> Contagem: o Resumo tem cerca de 450 palavras, dentro da faixa de 150 a 500
> que a NBR 6028 dá para trabalhos acadêmicos.

---

## Citações "(TODISCO et al., 2021)" — trocar em todo o texto

A obra citada assim é de Wang et al. (2020), a descrição da base (ver
Referências). No texto do TC1 ela aparece em quatro lugares:

- **3.1**, fim do segundo parágrafo: "(TODISCO et al., 2021)" →
  "(WANG et al., 2020)".
- **4.2**, segundo parágrafo: "(TODISCO et al., 2021)" → "(WANG et al., 2020;
  TODISCO et al., 2019)", porque ali se fala dos desafios ASVspoof.
- **4.9** e **introdução do capítulo 5**: já substituídas nas redações abaixo.

## Tempo verbal — trechos do TC1 a passar para o passado

O TC1 descrevia o trabalho como plano. Nos parágrafos que este arquivo não
substitui por inteiro, trocar:

- **4.1**, segundo parágrafo: "o desempenho do sistema será avaliado" →
  "o desempenho do sistema foi avaliado".
- **4.2**, último parágrafo: "que o sistema de detecção proposto deverá ser
  capaz de identificar" → "que o sistema de detecção proposto precisa ser capaz
  de identificar".
- **4.5**, introdução: "as funcionalidades que o sistema deverá executar" →
  "as funcionalidades que o sistema executa".
- **4.7**, primeiro parágrafo: "O desenvolvimento do projeto seguirá um modelo
  incremental" → "O desenvolvimento do projeto seguiu um modelo incremental".
- **4.11**, introdução: "O desempenho do sistema será avaliado" → "O
  desempenho do sistema foi avaliado".
- **4.11**, parágrafo da matriz de confusão: "também será utilizada a matriz de
  confusão para análise detalhada do comportamento do modelo durante o processo
  de classificação" → "também foi utilizada a matriz de confusão, gerada para
  cada modelo no conjunto de avaliação, para análise detalhada do comportamento
  do modelo na classificação" (`src/metrics.py`, `plot_confusion_matrix`).

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

## 4.6 Requisitos Não Funcionais — parágrafo a acrescentar ao fim

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
>
> • FFmpeg: conversão única dos arquivos FLAC da ASVspoof 2021 para WAV, porque
> parte deles não decodifica no libsndfile.

> • SciPy: filtro FIR da conversão de taxa entre a captura (48 kHz) e o modelo
> (16 kHz), seção 4.13.
>
> • SoundCard: captura do áudio de saída do computador (*loopback* do WASAPI,
> no Windows) e do microfone.
>
> • FastAPI e Jinja2: servidor e páginas da interface web, geradas no servidor,
> na mesma linguagem do modelo e sem etapa de compilação de front-end; servida
> pelo Uvicorn.
>
> • SQLite: histórico das análises da interface web.

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
>
> Para medir a robustez a um canal de transmissão real, os modelos treinados
> na ASVspoof 2019 foram avaliados, sem retreino, na trilha LA da ASVspoof 2021.
> Ela contém os mesmos ataques A07 a A19, agora transmitidos por sistemas reais
> de telefonia e voz sobre IP, e, por regra do desafio, não tem partição de
> treino: os sistemas são treinados na base de 2019. Foram usadas duas
> condições da fase de avaliação oficial (14.816 *bonafide* e 133.360 *spoof*):
> a de referência, sem codec nem transmissão, e a de Opus transmitido por redes
> reais. De cada condição foi sorteada uma amostra estratificada de 10.000
> áudios, com a mesma semente, de modo que os modelos veem exatamente os mesmos
> áudios. Nesse tamanho de amostra, o intervalo de confiança de 95% do EER é de
> cerca de ±1,28 p.p.

## 4.10 Divisão dos Dados — substituir por inteiro

> A divisão segue os protocolos oficiais da base. O conjunto de treinamento é
> usado no aprendizado dos pesos. O de validação é usado na escolha da época
> (critério: menor EER), no agendamento da taxa de aprendizado, na parada
> antecipada e na calibração do limiar de decisão. O de avaliação é usado apenas
> uma vez, na medição final.
>
> Dos treze ataques do conjunto de avaliação (A07 a A19), onze são inéditos. Os
> outros dois, A16 e A19, reutilizam os algoritmos de A04 e A06, com sistemas
> treinados em outros dados (WANG et al., 2020). Os resultados de avaliação
> deste trabalho medem, portanto, sobretudo generalização para ataques
> desconhecidos, e não desempenho na distribuição de treino. Esse desenho explica por que os EERs ficam na casa das
> dezenas de pontos percentuais.

## 4.11 Métricas de Avaliação — acrescentar ao fim

> Nas métricas dependentes de limiar (*accuracy*, *precision*, *recall* e F1), a
> classe positiva é a *spoof*. O limiar de decisão é o ponto de EER calibrado no
> conjunto de validação e aplicado sem ajuste ao conjunto de avaliação. O EER não
> depende de limiar e é calculado sobre os *log-odds* da rede (diferença entre
> as duas saídas), e não sobre a probabilidade. Em precisão de 32 bits, a
> probabilidade satura em exatamente 1,0 quando essa diferença passa de cerca
> de 17; se áudios autênticos e sintéticos empatam nesse valor, o EER passa a
> medir o arredondamento, e não o modelo. Os *log-odds* preservam a mesma
> ordenação sem saturar.
>
> • *tandem Detection Cost Function* (t-DCF): métrica oficial do desafio
> ASVspoof. Ela avalia a contramedida em conjunto com um sistema de verificação
> automática de locutor e pondera o custo de cada tipo de erro (KINNUNEN et al.,
> 2018). O min t-DCF foi calculado com os parâmetros de custo do ASVspoof 2019
> e os scores de verificação de locutor fornecidos pelos organizadores, e foi
> reportado para os dois modelos principais e para a fusão de scores. Quanto
> menor, melhor; o valor 1 corresponde a uma contramedida sem utilidade.

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
>
> **Diferenças em relação ao sistema de referência LFCC-LCNN.** Conferido no
> código dos sistemas de referência (o B03 da ASVspoof 2021 LA e a variante
> treinada na 2019 LA por Wang e Yamagishi), o pipeline deste trabalho difere
> em cinco pontos: remove o silêncio, que eles mantêm; normaliza a amplitude
> pelo pico e as características por áudio, enquanto eles normalizam por lote
> dentro da rede; não usa pré-ênfase nem troca o primeiro coeficiente pela
> log-energia; usa janela de 25 ms, contra 20 ms; e usa a faixa de 0 a 8 kHz,
> enquanto o B03 da edição de 2021, voltada a telefonia e voz sobre IP, limita
> o LFCC a 0 a 4 kHz. Nenhum dos dois usa normalização de média e variância
> por coeficiente (CMVN). As duas primeiras diferenças são deliberadas: a
> remoção do silêncio evita o atalho descrito por Müller et al. (2021), e a
> normalização de amplitude evita que o modelo aprenda a energia do sinal, que
> sozinha separa parcialmente as classes nesta base. As demais não foram
> testadas isoladamente.

## 4.13 Sistema de Análise — seção nova

> O modelo é usado por um sistema com duas entradas: o envio de um arquivo e a
> captura em tempo real. A captura pode ler o áudio que o computador reproduz,
> como o de uma chamada ou de um vídeo, ou o microfone, para analisar uma voz
> ao vivo. Todas passam pelo mesmo analisador, de modo que um arquivo e a sua
> reprodução ao vivo recebem o mesmo tratamento.
>
> **Captura.** O áudio de saída é lido por *loopback* do WASAPI, no Windows, e o
> microfone pela mesma interface, os dois à taxa de 48 kHz. A leitura roda numa
> *thread* própria, com *buffer* de 1 s, para que o processamento não provoque
> perda de amostras. Os "aprimoramentos de áudio" do driver precisam estar
> desligados, na saída e no microfone: na saída, eles comprimem a dinâmica do
> sinal e levam todos os scores para perto de 1,0; no microfone, elevam o score
> da voz humana (seção 5.5).
>
> **Conversão de taxa.** O áudio é convertido para os 16 kHz do modelo por um
> filtro FIR de 2047 coeficientes (janela de Kaiser), com corte em 7,9 kHz,
> aplicado em forma polifásica. O conversor padrão das bibliotecas de áudio
> apaga a faixa de 7,6 a 8 kHz, da qual o modelo depende; a seção 5.5 mede o
> efeito dessa escolha. Arquivos gravados acima de 16 kHz passam pelo mesmo
> filtro.
>
> **Janelas e agregação.** O analisador classifica janelas de 4 s com passo de
> 2 s, o mesmo tamanho usado no treinamento. Janelas em silêncio ficam fora das
> médias, assim como todas as janelas de uma sessão identificada como de banda
> estreita (cinco janelas com som e sem energia acima de 4 kHz). As demais entram com peso igual à fração de fala que resta após a remoção de
> silêncio, porque uma janela com pouca fala vira, no pré-processamento, muita
> repetição, fora do domínio de treino. O resultado é a média ponderada dos
> scores e o número de janelas acima do limiar.
>
> **Limiar por tipo de áudio.** O limiar calibrado na validação vale para áudio
> nativo de 16 kHz, o formato da base. Áudio capturado ao vivo, ou gravado
> acima de 16 kHz, passa pela conversão de taxa e usa um segundo limiar,
> calibrado na partição de desenvolvimento submetida à mesma conversão. O
> sistema escolhe o limiar pela taxa de amostragem da entrada e informa qual
> foi usado. O limiar recalibrado fica numa cópia do checkpoint, aceita apenas
> se os pesos forem idênticos aos do original.
>
> **Interface.** A interface web, feita com FastAPI e páginas renderizadas com
> Jinja2, tem quatro abas. Em "Enviar arquivo", o usuário escolhe um áudio
> (WAV, FLAC, MP3, M4A, OGG, Opus ou AAC, até 25 MB). Em "Resultado", vê o score
> médio, o limiar, a decisão (indício ou não de síntese) e um gráfico com o
> score de cada janela ao longo do tempo, com as janelas parciais destacadas;
> o relatório pode ser baixado em JSON. Em "Ao vivo", escolhe a fonte (som do
> computador ou microfone) e o dispositivo, inicia e encerra a captura e
> acompanha os scores enquanto ela ocorre. Em "Histórico", consulta e
> exclui análises anteriores, guardadas num banco SQLite. A interface
> apresenta o resultado como indício, e não como prova, e a página "Sobre"
> descreve as limitações medidas na seção 5.
>
> **Atendimento dos requisitos funcionais.** O envio de arquivos WAV e MP3
> (RF01), o pré-processamento (RF02), a extração de características (RF03), a
> classificação (RF04 e RF05) e a exibição do resultado e do *score* (RF06 e
> RF07) estão implementados na interface web. Na RF07, o valor exibido é o
> *score* do modelo, e não uma probabilidade calibrada: a seção 5.5 mostra que
> a sua escala muda com o canal.

---

## 5. EXPERIMENTOS E RESULTADOS — introdução, substituir

> Esta seção apresenta os resultados dos experimentos. Salvo indicação em
> contrário, os valores foram medidos no conjunto de avaliação completo da
> ASVspoof 2019 LA (71.237 áudios), composto de ataques ausentes do
> treinamento, com uma execução por modelo (semente 42). A seção 5.5 acrescenta
> a avaliação em canal real na ASVspoof 2021 LA, a captura em tempo real pelo
> sistema da seção 4.13 e um controle pareado com a voz do autor. As limitações decorrentes desse desenho são discutidas na seção
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

| modelo | EER validação (%) | EER avaliação (%) | *precision* | *recall* | F1 | *accuracy* |
|---|---|---|---|---|---|---|
| baseline_lfcc_cnn_v2 | 9,35 | **18,99** | 0,9838 | 0,7184 | **0,8304** | 0,7368 |
| fusion_lcnn_v4 | **0,03** | 20,18 | 0,9999 | 0,5508 | 0,7103 | 0,5971 |
| baseline_lfcc_cnn | 9,74 | 20,78 | 0,9814 | 0,7004 | 0,8174 | 0,7194 |
| baseline_lfcc_cnn_v3 | 0,48 | 21,10 | 0,9987 | 0,5749 | 0,7297 | 0,6181 |
| attention_lcnn_v4 | 0,04 | 21,13 | 0,9998 | 0,5374 | 0,6990 | 0,5851 |
| baseline_lfcc_cnn_v3a | 0,71 | 21,23 | 0,9986 | 0,6196 | 0,7647 | 0,6581 |
| baseline_lcnn_v4 | 0,48 | 21,56 | 0,9985 | 0,5284 | 0,6911 | 0,5764 |

Fonte: Autoria própria. EER de validação do *checkpoint* selecionado. Classe
positiva: *spoof*. *Precision*, *recall*, F1 e *accuracy* no limiar calibrado
no conjunto de validação.

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
> **A validação não antecipa a avaliação.** Cinco dos sete modelos praticamente
> resolvem o conjunto de validação (EER de 0,03% a 0,71%), e todos ficam acima
> de 20% na avaliação. Os dois modelos com pior validação (9,35% e 9,74%) são
> o primeiro e o terceiro na avaliação. A correlação de postos entre o EER de
> validação e o de avaliação é ρ = −0,11 (p = 0,84, teste de permutação exato).
> O contraste controlado entre v2 e v3a, que difere essencialmente no número de
> filtros do LFCC (20 e 70), mostra o mecanismo. A resolução maior reduz o EER
> de validação de 9,35% para 0,71% e aumenta o de avaliação de 18,99% para
> 21,23%. A seção 5.4 mostra que a resolução maior resolve os ataques do
> conjunto de avaliação que reutilizam algoritmos do treinamento (A16 e A19
> repetem A04 e A06) e perde nos ataques realmente novos. Como a validação só
> contém os ataques A01 a A06, ela mede o que o modelo aprendeu desses ataques
> e nada informa sobre os demais. O mesmo ocorre com os sistemas de referência
> oficiais: o B01 tem EER de 0,43% na validação e de 9,57% na avaliação, e
> Wang et al. (2020) atribuem a diferença aos ataques novos. A seleção do
> *checkpoint* e a calibração do
> limiar pela validação, prática padrão na área e adotada aqui, escolhem
> portanto por um critério que não prevê o desempenho em ataques inéditos. O
> *recall* baixo discutido acima é uma consequência direta disso.

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
> ser atribuída ao método com segurança pelo EER global. A seção 5.4 mostra
> que, ataque a ataque, os efeitos da fusão são muito maiores (Tabela 6).
>
> **Comparação com a literatura.** Os sistemas de referência oficiais do desafio
> alcançam 8,09% (B02, LFCC-GMM) e 9,57% (B01, CQCC-GMM) de EER no mesmo conjunto
> de avaliação (TODISCO et al., 2019). Esses sistemas, porém, processam o áudio
> **com** o silêncio. Müller et al. (2021) mostram que, na ASVspoof 2019, a
> duração do silêncio inicial sozinha separa as classes com 15,12% de EER, e que
> a remoção do silêncio piora o RawNet2 de 3,61% para 15,50%. Como este trabalho
> remove o silêncio, a comparação equivalente é com os modelos que Müller et al.
> (2021) treinaram e avaliaram sob o mesmo protocolo (Tabela 4).
>
> Na métrica oficial do desafio, o min t-DCF é de 0,5075 para o baseline_v2, de
> 0,3818 para o fusion_v4 e de 0,3129 para a fusão de scores dos dois, contra
> 0,2116 (B02) e 0,2366 (B01), obtidos com o silêncio (TODISCO et al., 2019).
> O t-DCF inverte a ordem dada pelo EER entre os dois modelos: pelo EER o
> baseline_v2 é melhor, e pelo t-DCF o fusion_v4 é bem melhor. As duas métricas
> olham regiões diferentes da curva de erro. O EER usa o ponto em que as duas
> taxas de erro se igualam; o t-DCF, com o custo maior atribuído a aceitar um
> *spoof*, favorece o ponto em que quase nenhum áudio sintético passa, e é
> nessa região que o fusion_v4 se destaca, por resolver quase por completo seis
> dos treze ataques (seção 5.4). A escolha da métrica decide, portanto, qual dos
> dois modelos é o melhor. Não há régua de t-DCF sob o mesmo protocolo sem
> silêncio, porque Müller et al. (2021) reportam apenas o EER.

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

> Os sete modelos têm EER menor que o dos três modelos baseados em CQT, e a
> fusão de scores tem EER menor que a média do RawNet2. Algumas diferenças de protocolo
> permanecem. Müller et al. (2021) usam limiar de remoção de silêncio de 40 dB,
> contra 30 dB aqui, e o áudio inteiro, contra a janela fixa de 4 s. Eles
> reportam também média e desvio de várias execuções, enquanto aqui há uma
> execução por modelo.

## 5.4 Análise de Erros — substituir por inteiro

> O EER global resume em um único número ataques de dificuldade muito diferente.
> A Tabela 5 separa o desempenho dos dois modelos principais por ataque. Cada
> ataque tem 4.914 áudios e é comparado com os mesmos 7.355 áudios *bonafide*.
> A matriz completa dos sete modelos está no Apêndice A.

**Tabela 5 – EER (%) por ataque**

| ataque | gerador de forma de onda | baseline_v2 | fusion_v4 |
|---|---|---|---|
| A07 | WORLD com pós-processamento GAN | 22,43 | **0,02** |
| A08 | *neural source-filter* | 3,88 | **0,03** |
| A09 | vocoder (Vocaine) | 2,32 | **0,12** |
| A10 | WaveRNN (autorregressivo) | **30,63** | 45,54 |
| A11 | Griffin-Lim | **3,85** | 33,74 |
| A12 | WaveNet (autorregressivo) | **36,18** | 47,99 |
| A13 | filtragem de forma de onda | 36,50 | **6,54** |
| A14 | vocoder (STRAIGHT) | **12,01** | 17,74 |
| A15 | WaveNet (autorregressivo) | **15,02** | 29,37 |
| A16 | concatenação | 22,75 | **0,03** |
| A17 | filtragem de forma de onda | 11,27 | **0,41** |
| A18 | vocoder MFCC | 15,01 | **4,63** |
| A19 | filtragem espectral | 6,98 | **0,00** |
| **global** | | **18,99** | 20,18 |

Fonte: Autoria própria; geradores segundo Wang et al. (2020, Tabela 1).

<!-- Geradores conferidos na Tabela 1 de Wang et al. (2020), 25/09/2026. -->

> **Os modelos erram em ataques diferentes.** Os EERs globais são próximos, mas
> os perfis são quase opostos. O fusion_v4 vence em oito dos treze ataques, e o
> baseline_v2 nos outros cinco, com diferenças de até 30 p.p. nos dois sentidos.
> O A13 é o pior ataque para o baseline_v2 (36,50%) e é quase resolvido pelo
> fusion_v4 (6,54%). Com o A11 acontece o inverso: 3,85% no baseline_v2 e 33,74%
> no fusion_v4.
>
> Os sete modelos se agrupam em três famílias de perfil por ataque, com
> correlação de postos de 0,97 dentro de cada família: LFCC com 20 filtros
> (v1 e v2); LFCC com 70 filtros em um ramo (v3a, v3 e baseline_lcnn_v4); e
> LFCC com espectrograma (fusion_v4 e attention_v4). A primeira família falha
> nos ataques que reutilizam algoritmos do treinamento (A07, A16 e A19) e acerta
> o A11. A segunda resolve esses ataques e falha em A12, A13 e A18. A terceira
> resolve também A13, A17 e A18, mas falha em A10, A11, A14 e A15.
>
> A falha da primeira família não depende do classificador. O sistema de
> referência B02 usa a mesma representação (LFCC com 20 filtros lineares) com
> um classificador GMM e falha nos mesmos três ataques: 12,86% no A07, 6,31% no
> A16 e 13,94% no A19, que o B01, baseado em CQCC, resolve (WANG et al., 2020,
> Tabela 8). A correlação de postos entre os perfis por ataque de v1/v2 e do
> B02 é de 0,62 a 0,64. Com a mesma representação e classificadores diferentes,
> as falhas se repetem; com 70 filtros, desaparecem. A causa é a resolução
> espectral da representação.
>
> **Os incrementos, ataque a ataque.** A Tabela 6 decompõe os contrastes
> controlados da Tabela 3 por ataque.
>
**Tabela 6 – Variação do EER por ataque em cada incremento (p.p.)**

| ataque | fusão (baseline_lcnn_v4 → fusion_lcnn_v4) | atenção (fusion_lcnn_v4 → attention_lcnn_v4) |
|---|---|---|
| A13 | −37,65 | −1,03 |
| A18 | −29,48 | +0,93 |
| A17 | −10,09 | −0,02 |
| A12 | −4,29 | +0,12 |
| A11 | +25,78 | −1,71 |
| A15 | +19,83 | +2,96 |
| A10 | +8,04 | −0,93 |
| A14 | +5,60 | +10,75 |
| demais (A07, A08, A09, A16, A19) | −0,72 a −0,17 | −0,04 a +0,06 |
| **global** | **−1,38** | **+0,95** |

Fonte: Autoria própria.

> O ramo de espectrograma reduz o EER em 37,65 p.p. no A13 e em 29,48 p.p. no
> A18, e o aumenta em 25,78 p.p. no A11 e em 19,83 p.p. no A15. O ganho global
> de 1,38 p.p. é o saldo dessas trocas, que são muito maiores que a variação
> entre execuções reportada na literatura. O efeito da fusão sobre *quais*
> ataques o modelo detecta é, portanto, robusto, mesmo que o efeito sobre o
> EER global não seja. A atenção, ao contrário, altera menos de 3 p.p. em doze
> dos treze ataques, e os perfis com e sem atenção têm correlação de 0,97.
>
> Os maiores ganhos do ramo de espectrograma estão em A13, A17 e A18, ataques
> que geram a forma de onda filtrando ou modificando uma fala existente (WANG et
> al., 2020). Nesses três ataques, o EER médio cai de 29,60% (baseline_lcnn_v4)
> para 3,86% (fusion_lcnn_v4). Wang et al. (2020) apontam a filtragem de forma
> de onda como o método mais difícil para os sistemas de referência. Como essa
> associação foi observada depois dos experimentos e envolve três ataques, ela
> é apresentada como hipótese para trabalhos futuros, e não como resultado.
>
> **A geração autorregressiva concentra a dificuldade.** Os três ataques com
> geradores de forma de onda autorregressivos (A10, A12 e A15) têm EER médio de 27,28% no
> baseline_v2 e de 40,97% no fusion_v4. Nos outros dez ataques, as médias são de
> 13,70% e 6,33%. Um teste de permutação exato, sobre os 286 trios possíveis
> entre os treze ataques, dá p = 0,0070 para o fusion_v4 e p = 0,0035 para o
> attention_v4. Nos outros cinco modelos, o valor fica entre 0,0315 e 0,0664,
> no limite da significância. O A08, cujo gerador de forma de onda é neural mas
> não autorregressivo (*neural source-filter*), é resolvido pelos dois modelos
> (3,88% e 0,03%), embora o seu modelo acústico seja um RNN autorregressivo. Isso indica que o eixo relevante não é a
> distinção entre gerador neural e clássico. Wang et al. (2020) chegam a
> conclusão semelhante a partir do par A10 e A11, que tem o mesmo modelo
> acústico e difere só no gerador: o método de geração da forma de onda pesa
> mais que o modelo acústico. A explicação tem, contudo, um
> limite: A12 e A15 usam o mesmo WaveNet e diferem em 21,16 p.p. no baseline_v2.
> Outras partes do sistema de ataque também pesam. A10 e A12 são os únicos
> ataques em que **todos os sete modelos** passam de 30% de EER, com médias de
> 36,86% e 47,80%. Nenhuma das variações testadas (resolução, encoder, segundo
> ramo, atenção) os resolve.
>
> **Fusão de scores.** A complementaridade observada motivou a combinação dos
> scores de modelos treinados separadamente (Tabela 7).

**Tabela 7 – Fusão de scores**

| combinação | correlação entre os perfis por ataque (ρ) | EER (%) |
|---|---|---|
| baseline_v2 + fusion_v4 | 0,34 | **13,13** |
| baseline_v2 + attention_v4 | 0,39 | 13,47 |
| baseline_v2 + baseline_v3 | 0,64 | 15,84 |
| fusion_v4 + attention_v4 (controle) | 0,97 | 19,88 |

Fonte: Autoria própria. ρ: correlação de Spearman entre os EERs dos dois
modelos nos treze ataques.

<!-- RESUMO_TCC §4 -->

> A combinação de modelos de arquiteturas diferentes reduziu o EER de 18,99%, o
> do melhor modelo isolado, para 13,13%, um ganho de 5,86 p.p. O par de controle
> (fusion_v4 e attention_v4) reúne dois modelos quase idênticos e rendeu apenas
> 0,30 p.p. O ganho vem, portanto, da diversidade entre os modelos, e não do ato
> de combinar scores. A Tabela 7 torna essa diversidade mensurável: quanto
> menor a correlação entre os perfis por ataque, maior o ganho, e a ordem se
> mantém nos quatro pares. Com quatro pares, a relação é ilustrativa e não um
> teste estatístico. O valor de 13,13% usa a regra de postos, que exige o
> conjunto completo de scores. A média simples, aplicável a um fluxo contínuo de
> áudio, resulta em 14,03%. Pelo min t-DCF, a fusão por postos também é a
> melhor: 0,3129, contra 0,3818 do melhor modelo isolado nessa métrica.
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
> (reamostragem para 8 kHz e retorno a 16 kHz), como na telefonia. A Tabela 8
> apresenta as onze condições para os dois modelos principais.

**Tabela 8 – EER (%) sob degradação de canal**

| condição | vista no treino | baseline_v2 | fusion_v4 |
|---|---|---|---|
| limpo | — | **18,99** | 20,18 |
| ruído, SNR de 20 dB | sim | 27,75 | **21,27** |
| ruído, SNR de 10 dB | sim | 34,99 | **23,95** |
| ruído, SNR de 5 dB | não | 42,41 | **27,42** |
| ganho de −6 dB | sim | **19,44** | 20,36 |
| ganho de +6 dB | sim | **19,57** | 20,08 |
| Opus, 30 kbps | não | **19,75** | 21,38 |
| Opus, 25 kbps | não | **20,04** | 22,08 |
| Opus, 15 kbps | não | **20,12** | 23,02 |
| banda estreita (8 kHz) | não | 35,95 | **25,53** |
| banda estreita e Opus | não | 32,85 | **25,04** |
| **pior degradação** | | **+23,42 p.p.** | **+7,24 p.p.** |

Fonte: Autoria própria.

<!-- outputs/report/robustez_eval.md (25/09/2026) -->

> Sob degradação simulada, o ranking dos modelos se inverte. O baseline_v2 é o melhor no
> áudio limpo e piora até 23,42 p.p. O fusion_v4 piora no máximo 7,24 p.p. e
> vence em todas as condições de ruído e de banda estreita, com vantagem de
> 6,5 p.p. a 20 dB, 10,4 p.p. em banda estreita e 15,0 p.p. a 5 dB. O
> baseline_v2 vence nas condições de ganho e de codificação Opus, sempre por
> menos de 3 p.p. O ganho de ±6 dB praticamente não altera o EER, porque a
> normalização por pico o desfaz. A codificação Opus custa pouco até 15 kbps:
> de 0,8 a 1,1 p.p. no baseline_v2 e de 1,2 a 2,8 p.p. no fusion_v4. Já a perda
> da banda alta e o ruído aditivo custam muito ao baseline_v2.
>
> Duas ressalvas delimitam esse resultado. Primeiro, o treinamento usa ruído
> branco entre 10 e 30 dB e ganho de ±6 dB, gerados pela mesma função dos testes.
> As condições de ruído a 20 e 10 dB e de ganho, portanto, já estavam na
> distribuição de treino (coluna "vista no treino" da Tabela 8). Ainda assim,
> a inversão já aparece nelas: com o mesmo ruído de 20 dB visto no treino, o
> baseline_v2 piora 8,76 p.p. e o fusion_v4, 1,09 p.p. Como os dois modelos
> receberam o mesmo aumento de dados, a diferença de robustez ao ruído não se
> explica por um ter visto a perturbação e o outro não. Ela está na
> representação e na arquitetura. Segundo, o limiar calibrado em áudio limpo não se
> transfere para o áudio degradado. Sob Opus a 15 kbps, o *recall* do
> baseline_v2 sobe de 0,72 para 0,86 e o do fusion_v4 cai de 0,55 para 0,39.
> O efeito aparece até no F1: na amostra de verificação descrita abaixo, com
> Opus a 15 kbps, o F1 do baseline_v2 sobe de
> 0,83 para 0,91, acima do exigido pelo RNF04, enquanto o seu EER piora. O F1
> melhora porque o limiar passou a classificar mais áudios como *spoof*, não
> porque o modelo discrimine melhor.
>
> Nenhum áudio autêntico recebeu probabilidade saturada em 1,0 em nenhuma das
> condições da Tabela 8, e o EER calculado pela probabilidade coincide com o
> calculado pelos *log-odds*. Uma reavaliação numa amostra estratificada de
> 10.002 áudios reproduziu o eval completo em 21 das 22 medidas dentro do
> intervalo de confiança de 95%, com o mesmo modelo vencedor nas onze
> condições.
>
> **Canal real.** A simulação acima responde se o modelo resiste a cada
> perturbação isolada. Para saber se ela prevê o comportamento em um canal
> real, os dois modelos foram avaliados na ASVspoof 2021 LA (seção 4.9), na
> condição de referência e na condição de Opus transmitido por redes reais
> (Tabela 9).

**Tabela 9 – EER (%) na ASVspoof 2021 LA (amostra de 10.000 áudios por condição)**

| modelo | referência (sem codec) | Opus real | variação | Opus simulado, 25 kbps (Tabela 8) |
|---|---|---|---|---|
| baseline_v2 | 18,15 | **29,43** | +11,28 p.p. | +1,05 p.p. |
| fusion_v4 | 19,51 | 32,62 | +13,11 p.p. | +1,90 p.p. |
| fusão de scores (postos) | — | **24,94** | — | — |

Fonte: Autoria própria. IC de 95% de cerca de ±1,28 p.p. em cada medida.

<!-- RESUMO_TCC §5.3 (28/09/2026) -->

> A condição de referência reproduz a avaliação de 2019 nos dois modelos (18,15%
> contra 18,99%; 19,51% contra 20,18%), dentro do intervalo de confiança da
> amostra. A mudança na condição Opus vem, portanto, do canal, e não de
> diferença entre as bases.
>
> Três resultados se destacam. Primeiro, a simulação subestima o canal real em
> uma ordem de grandeza: o Opus real custa de 7 a 11 vezes o Opus simulado. A
> condição real inclui, além do codec, a transmissão por três redes com taxa e
> perdas desconhecidas, que é justamente o que a simulação não reproduz.
> Segundo, a inversão de ranking observada na simulação não se repete. O
> fusion_v4, que vencia por 6,5 a 15,0 p.p. sob ruído e banda estreita
> simulados, fica 3,19 p.p. atrás do baseline_v2 no canal real. A diferença de
> 3,19 p.p. está acima do intervalo de confiança da amostra, mas dentro da
> variação entre execuções reportada na literatura; o resultado que se sustenta
> não é que o baseline_v2 seja mais robusto, e sim que **a robustez medida na
> degradação simulada não previu qual modelo resiste ao canal real**. Terceiro,
> o ganho da fusão de scores se mantém: a combinação por postos dos dois
> modelos atinge 24,94%, 4,50 p.p. abaixo do melhor modelo isolado, o mesmo
> fenômeno observado na ASVspoof 2019. Os ataques A10 e A12 continuam os mais
> difíceis também no canal real (42,98% e 37,53% após a fusão).
>
> A regra de postos exige o conjunto completo de scores e não se aplica a um
> fluxo contínuo de áudio. A média das probabilidades, que se aplica, dá 29,41%,
> praticamente igual ao baseline_v2 sozinho, porque no canal real o fusion_v4
> atribui probabilidade 1,0 a 97% dos áudios autênticos. Aproveitar a fusão em
> tempo real exigiria calibrar cada modelo no canal de destino.
>
> A análise por ataque (Apêndice B) mostra por que a robustez simulada não se
> transferiu. Na condição de referência, o perfil por ataque dos dois modelos
> reproduz o de 2019 (correlação de postos de 0,99, com nenhum ataque diferindo
> mais de 3,05 p.p.). No Opus real, o perfil do baseline_v2 se preserva
> (correlação de 0,92 com a referência), com um custo distribuído entre os
> ataques. O do fusion_v4 se desfaz (0,67): os quatro ataques que ele detectava
> sem erro (A07, A08, A16 e A19, com 0,00% na referência) passam a 17% a 37%.
> A vantagem do fusion_v4 consistia na detecção quase perfeita de alguns
> ataques, e é essa precisão que o canal real destrói; na média por ataque, ele
> passa de melhor (13,74% contra 16,19%) a pior (31,42% contra 26,75%). A
> vantagem que sobrevive está em A13 e A18, dois dos ataques por filtragem de
> forma de onda discutidos na seção 5.4. Os perfis dos dois modelos também se
> aproximam (correlação de 0,34 em 2019 e de 0,48 no Opus real), o que é
> coerente com o ganho menor da fusão de scores nessa condição (4,50 contra 5,86
> p.p.). Com cerca de 690 áudios sintéticos por ataque, o intervalo de
> confiança por ataque vai de ±1,5 p.p. a ±3,4 p.p., e as comparações acima
> envolvem diferenças bem maiores que isso.
>
> O limiar calibrado também colapsa no canal real: os dois modelos classificam
> todos os áudios como *spoof* (*recall* de 1,0000 e *precision* de 0,8998, a
> proporção de *spoof* da amostra). É o classificador trivial da seção 4.6,
> produzido por um canal real: o fusion_v4 obtém F1 de 0,9473 nessa condição,
> acima do exigido pelo RNF04, sem distinguir nenhum áudio.
>
> Não foram avaliadas as demais condições da ASVspoof 2021 LA (outros codecs e
> telefonia), nem arquivos MP3.
>
> **Captura ao vivo.** O sistema da seção 4.13 acrescenta um canal que nenhuma
> das bases traz: o caminho do áudio dentro do próprio computador, da
> reprodução à captura. Para medi-lo, uma lista de 40 áudios do conjunto de
> avaliação (20 *bonafide* e 20 *spoof*, cobrindo os treze ataques) foi tocada
> no computador e capturada pelo sistema, e cada trecho capturado foi alinhado
> ao arquivo de origem. Antes de atribuir qualquer diferença ao canal, o
> procedimento foi validado em três elos: os recortes da lista, sem passar pelo
> alto-falante, reproduzem os scores da avaliação amostra a amostra; o monitor
> em tempo real dá o mesmo score do processamento do arquivo (diferença abaixo
> de 10⁻⁶); e o caminho de captura, uma vez identificado, foi reproduzido em
> software, com scores iguais aos capturados áudio a áudio.
>
> A primeira captura levou todos os áudios para perto de 1,0, inclusive os
> *bonafide* (de 0,006 para 0,999). A comparação de espectros apontou dois
> efeitos. O primeiro eram os "aprimoramentos de áudio" do driver, que
> comprimiam a dinâmica e elevavam em 31,5 dB o ruído nas pausas; desligados,
> o espectro capturado coincidiu com o original até 7,5 kHz. O segundo
> permaneceu: a faixa de 7,6 a 8 kHz chegava ao modelo atenuada em cerca de
> 50 dB. A captura ocorre a 48 kHz, e o conversor padrão de 48 para 16 kHz
> (soxr) corta essa faixa no seu filtro antialiasing. Como o mesmo conversor é
> usado para abrir arquivos gravados acima de 16 kHz, o efeito atinge também
> quase todo áudio do mundo real, que nasce a 44,1 ou 48 kHz.
>
> A Tabela 10 mede o efeito na amostra estratificada de 10.002 áudios do
> conjunto de avaliação, submetida à ida e volta 16 → 48 → 16 kHz, primeiro com
> o conversor padrão e depois com o filtro FIR da seção 4.13.

**Tabela 10 – Efeito da conversão de taxa da captura (baseline_v2, 10.002 áudios)**

| condição | EER | áudios humanos acima do limiar original |
|---|---|---|
| limpo | 19,02% | 10% |
| captura, conversor padrão (soxr) | 24,70% | 71% |
| captura, filtro FIR | **19,41%** | **12%** |

Fonte: Autoria própria. IC de 95% de cerca de ±1,28 p.p.

<!-- RESUMO_TCC §10.2.1 e §10.2.5 (29 e 30/09/2026) -->

> Com o conversor padrão, a captura custava 5,68 p.p. de EER, mais que o Opus
> simulado na mesma amostra (1,30 p.p.), e o limiar colapsava: 7 em cada 10
> áudios humanos passavam a ser marcados como sintéticos. No fusion_v4, todos
> os 1.032 áudios humanos da amostra chegavam a probabilidade 1,0, o mesmo
> comportamento do Opus real. Com o filtro FIR, o custo cai para 0,39 p.p.,
> dentro do intervalo de confiança, e o limiar recalibrado para a captura fica
> em 0,686, próximo do original (0,654). A degradação atribuída à captura era,
> portanto, quase toda do filtro do conversor, e foi corrigida sem retreino. O
> resultado confirma que o modelo usa a faixa de 7,6 a 8 kHz, a mesma razão que
> levou os organizadores da ASVspoof 2021 LA a limitar o LFCC do seu sistema
> de referência a 0–4 kHz (seção 4.12). O filtro não recupera o que outro
> programa cortou antes da captura: o mesmo arquivo de vozes humanas recebeu
> score médio de 0,03 enviado como arquivo, 0,33 tocado pelo Reprodutor do
> Windows e 0,98 tocado pelo VLC, que altera o áudio. Os áudios sintéticos
> receberam 1,00 nas três formas. Tocados por um alto-falante de celular e
> captados pelo microfone, os mesmos áudios sintéticos caíram para 0,15 a 0,35,
> abaixo do limiar, e a ordem entre os dois grupos dependeu da equalização do
> celular: com ela, os sintéticos ficaram acima dos humanos (0,30 a 0,35
> contra 0,07 a 0,08); sem ela, abaixo (0,15 contra 0,19). A reprodução pelo
> ar corresponde ao cenário *Physical Access* da ASVspoof, fora do treino do
> modelo, e enche as pausas de ruído e reverberação da sala. Numa simulação
> com ruído somado às pausas, os scores dos áudios sintéticos também caíram
> (de 0,999 para 0,766 e de 0,972 para 0,183, em dois exemplos), o que indica
> que o modelo associa pausas ruidosas a voz humana.

<!-- RESUMO_TCC §10.2.4 e §10.2.7 (30/09/2026) -->

> **Vozes externas à base.** Em lugar da coleta com colaboradores, o autor
> gravou cinco frases (48 kHz, microfone comum, duas em inglês e três em
> português) e cada uma foi refeita por *copy-synthesis* com dois vocoders
> usados pelos ataques da base: Griffin-Lim (GRIFFIN; LIM, 1984), do ataque
> A11, e WORLD (MORISE; YOKOMORI; OZAWA, 2016), dos ataques A02, A03, A05 e
> A07. Original e versões sintéticas têm o mesmo locutor, microfone, sala e
> texto; só o vocoder muda. É um controle pareado: se o modelo detecta o
> vocoder, cada versão sintética deve receber score maior que o original da
> mesma frase (Tabela 11).

**Tabela 11 – Scores com a voz do autor (baseline_v2, limiar 0,686)**

| frase | original (humano) | Griffin-Lim | WORLD |
|---|---|---|---|
| 1 (inglês) | 0,901 | 0,446 | 0,443 |
| 2 (inglês) | 0,472 | 0,264 | 0,413 |
| 1 (português) | 0,898 | 0,720 | 0,718 |
| 2 (português) | 0,593 | 0,315 | 0,453 |
| 3 (português) | 0,688 | 0,707 | 0,898 |

Fonte: Autoria própria.

<!-- RESUMO_TCC §10.2.6 (01/10/2026), rodada sem aprimoramentos do microfone -->

> O resultado é negativo. Três dos cinco originais ficaram acima do limiar, e
> em 8 das 10 frases a versão sintética recebeu score menor que o original.
> Numa primeira rodada, gravada com os aprimoramentos do microfone ligados,
> os cinco originais ficaram acima do limiar (média de 0,95, contra 0,71 sem os
> aprimoramentos) e as dez versões sintéticas ficaram abaixo do original; nas
> duas rodadas, 18 de 20. O modelo detecta o ataque A11 na base, mas não o
> mesmo vocoder aplicado a outra voz. O que ele aprendeu não é o artefato
> genérico do vocoder, e sim características do sistema e da base (modelo
> acústico, locutores, condições de gravação), e diante de uma voz nova o score
> responde às condições de gravação. Um teste qualitativo com áudios da
> internet apontou na mesma direção: um audiobook narrado por humano, em banda
> estreita, recebeu score médio maior (0,81) que um vídeo narrado por IA (0,49)
> e que a voz do Google Tradutor (0,47). O sistema funciona de ponta a ponta,
> mas o modelo, treinado só na ASVspoof 2019, não generaliza para vozes e
> geradores fora dela.

## 5.6 Resultados Esperados e Obtidos — substituir por inteiro

> O planejamento do TC1 estabeleceu três projeções, e nenhuma se confirmou
> integralmente (Quadro 2).

**Quadro 2 – Resultados esperados e obtidos**

| projeção | obtido | situação |
|---|---|---|
| F1 superior a 0,90 (projeção) e mínimo de 0,85 (RNF04) | 0,69 a 0,83 | não atingido |
| EER inferior a 8% | 18,99% (modelo isolado); 13,13% (fusão de scores) | não atingido |
| ganho a cada incremento | fusão −1,38 p.p.; atenção +0,95 p.p. | só na fusão; no EER global, dentro da variância entre execuções |
| processamento em até 30 s por amostra (RNF06) | 2,745 s em CPU para um arquivo de 60 s | atingido |
| sistema com envio de arquivo e exibição do resultado (RF01, RF06, RF07) | interface web com envio de arquivo, captura em tempo real e histórico | atingido |
| suporte a WAV e MP3 (RNF05) | aceitos na entrada; sem avaliação de desempenho em MP3 | atingido parcialmente |

Fonte: Autoria própria.

> A meta de EER tomou como referência o sistema B02 (8,09%), que processa o
> áudio com o silêncio. Sob o protocolo sem silêncio, que remove um atalho
> conhecido da base (MÜLLER et al., 2021), os modelos deste trabalho têm EER
> menor que o dos modelos com CQT da literatura, como mostra a Tabela 4. A meta de F1
> esbarra no desequilíbrio das classes, discutido na seção 4.6: o F1 da classe
> majoritária não mede capacidade de detecção.

## 5.7 Limitações — seção nova

> • **Uma execução por modelo.** Os efeitos dos incrementos (−1,38 e +0,95 p.p.)
> são menores que a variação entre execuções observada na literatura (1,4 a 5,2
> p.p.). Os efeitos grandes superam essa faixa: as trocas por ataque da fusão
> de características (de −37,65 a +25,78 p.p.), a diferença entre validação e
> avaliação, a inversão de ranking na simulação (6,5 a 15,0 p.p.), o custo do
> canal real (+11 a +13 p.p.) e o ganho de 5,86 p.p. da fusão de scores. A
> diferença entre os modelos no canal real (3,19 p.p.) e o ganho da fusão nele
> (4,50 p.p.) ficam dentro dessa faixa.
>
> • **Atenção mínima.** O resultado negativo vale para a forma de atenção
> avaliada.
>
> • **Robustez parcialmente dentro da distribuição.** Ruído a 20 e 10 dB e ganho
> estavam no aumento de dados do treino. Ver seção 5.5.
>
> • **Robustez medida em dois modelos.** Os outros cinco não foram avaliados sob
> degradação.
>
> • **Canal real em uma condição.** A ASVspoof 2021 LA foi avaliada só na
> condição Opus, numa amostra de 10.000 áudios, e em dois modelos. O
> processamento de um cliente de conferência (supressão de ruído, cancelamento
> de eco) não foi medido.
>
> • **Vozes externas em pequena escala.** A coleta com colaboradores não foi
> realizada. As únicas vozes externas à base são cinco frases do próprio autor,
> em duas rodadas, suficientes para mostrar a falha de generalização com
> controle pareado, mas não para medir uma taxa de erro.
>
> • **Dependência do caminho do som.** O resultado ao vivo depende de
> configurações fora do sistema: os aprimoramentos de áudio do driver e o
> programa que reproduz o áudio alteram o score. A captura foi medida num
> único computador, com driver Realtek e Windows. A reprodução pelo ar
> (alto-falante e microfone) foi observada em três pares de sessões, sem
> amostra rotulada, e nela o modelo não separou os grupos de forma estável.
>
> • **Limiar escolhido pela taxa do arquivo.** O sistema escolhe o limiar pela
> taxa de amostragem da entrada, não pelo conteúdo: um arquivo de 16 kHz que já
> passou por outra conversão perdeu o topo da banda e recebe o limiar
> original. Áudio abaixo de 16 kHz (telefonia) não foi medido.
>
> • **Atalhos da base.** O modelo depende de duração e energia em 1% a 2% da
> variância do score.
>
> • **Dependência da faixa de 7,6 a 8 kHz.** O filtro FIR do sistema preserva
> a faixa, mas não a recupera quando outro programa já a cortou antes da
> captura: o *player*, o codec da chamada ou uma conversão anterior do arquivo.
>
> • **Teste com áudios da internet qualitativo.** Três áudios, sem rótulo por
> janela; aponta a mesma dependência das condições de gravação, mas não a
> mede.

---

## 6.4 Marcos do Projeto — ajustes

- **Marco 2:** retirar "amostras de colaboradores coletadas".
- **Marco 6:** trocar "Testes de robustez realizados" por "Testes de robustez
  realizados com degradação simulada de canal e com transmissão real (ASVspoof
  2021 LA, condição Opus)".
- **Marcos 5 a 7 (API e front-end):** concluídos. Trocar "API REST funcional" e
  "Sistema integrado (front-end + back-end)" por "Interface web (FastAPI +
  Jinja2) com o pipeline de IA integrado: envio de arquivo, captura em tempo
  real, resultado por janela e histórico" e "Monitor de linha de comando
  integrado ao mesmo analisador". Marco 7: manter.
- **6 (texto) e 6.1:** manter. A divisão descrita é a real: o back-end (modelo,
  pipeline e API) e o front-end foram desenvolvidos em frentes separadas, a
  implementação por Humberto e o front-end por Pedro.

## 7.1 Privacidade e Utilização dos Dados — substituir por inteiro

> O treinamento e a avaliação dos modelos utilizam as bases públicas ASVspoof
> 2019 e ASVspoof 2021, que têm finalidade de pesquisa científica e são distribuídas
> sob licença própria dos organizadores. A única voz externa à base é a do
> próprio autor, gravada por ele para o controle pareado da seção 5.5. A coleta
> de amostras com colaboradores, prevista no planejamento, não foi realizada.
>
> O sistema roda localmente e não envia áudio a serviços externos. O arquivo
> enviado pela interface é apagado logo após a análise, e o histórico guarda
> apenas o nome do arquivo e os scores, com registros que o usuário pode
> excluir. A captura ao vivo grava a sessão em disco, no próprio computador,
> para conferência posterior; ao excluir o registro no histórico, a gravação é
> apagada junto. Como a captura alcança
> tudo o que o computador reproduz, inclusive a voz de terceiros numa chamada,
> o seu uso exige o consentimento dos participantes.
>
> Se a coleta com colaboradores for retomada em trabalhos futuros, deverá
> observar a Lei Geral de Proteção de Dados (Lei nº 13.709/2018 – LGPD). Características vocais podem ser
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
> A fusão tardia de LFCC e espectrograma log-mel reduziu o EER global em 1,38
> p.p., como saldo de trocas de até 37,65 p.p. por ataque: o segundo ramo muda
> quais ataques o modelo detecta. A atenção no *pooling*, na forma mínima
> avaliada, não trouxe ganho e quase não alterou o perfil do modelo. Os efeitos
> sobre o EER global ficam dentro da variação entre execuções reportada na
> literatura. Cinco achados, porém, têm magnitude que supera essa variação.
>
> Primeiro, o desempenho na validação não antecipa o desempenho na avaliação.
> Cinco modelos alcançaram EER de validação abaixo de 0,71% e ficaram todos
> acima de 20% na avaliação. Os dois com pior validação foram o primeiro e o
> terceiro na avaliação. A validação contém apenas os ataques do treinamento, e
> selecionar modelos por ela favorece o que foi aprendido desses ataques. O
> contraste entre 20 e 70 filtros mostra o mecanismo, e o sistema de
> referência B02, com a mesma representação de 20 filtros e outro
> classificador, falha nos mesmos ataques.
>
> Segundo, a fusão de scores de modelos diferentes reduziu o EER para 13,13% e
> o min t-DCF para 0,3129, o melhor resultado do trabalho nas duas métricas. O
> grupo de controle mostra que o ganho vem da diversidade entre os modelos, que
> erram em ataques diferentes. A comparação entre os modelos isolados, por
> outro lado, depende da métrica: o baseline_v2 é o melhor pelo EER, e o
> fusion_v4, pelo t-DCF.
>
> Terceiro, o EER agregado esconde os ataques mais fortes. Nos ataques com
> geração autorregressiva A10 e A12, todos os sete modelos passam de 30% de EER,
> com médias de 36,86% e 47,80%, próximas do acaso, enquanto outros ataques
> são resolvidos quase perfeitamente. Nenhuma das variações testadas os
> resolveu.
>
> Quarto, a degradação simulada não previu o canal real. Na simulação, o
> ranking dos modelos se inverte: o melhor modelo no áudio limpo piora até
> 23,42 p.p., e o modelo com fusão, 7,24 p.p. Em transmissão real por Opus, a
> inversão não se repete, e o canal custa de 7 a 11 vezes o que a simulação do
> mesmo codec indicava. Escolher um modelo pela robustez simulada seria tão
> arriscado quanto escolhê-lo pelo áudio limpo; a avaliação precisa ser feita
> no canal de destino. A análise por ataque indica o motivo: a vantagem do
> modelo com fusão estava na detecção quase perfeita de alguns ataques, e é
> essa precisão que o canal real apaga. O ganho da fusão de scores, por outro
> lado, se manteve no canal real.
>
> Quinto, o modelo foi integrado a um sistema que analisa arquivos e o áudio do
> computador em tempo real, e o uso fora da base expôs duas dependências. A
> primeira é da faixa de 7,6 a 8 kHz: o conversor de taxa padrão a apagava, e
> isso custava 5,68 p.p. na captura; com um filtro FIR que a preserva, o custo
> caiu para 0,39 p.p., sem retreino. A segunda é das condições de gravação: com
> a voz do autor refeita por vocoders, num controle pareado, as versões
> sintéticas receberam scores menores que o original em 18 de 20 casos. O
> modelo detecta os ataques da base, mas não o mesmo vocoder aplicado a uma voz
> nova, e o score passa a responder ao microfone, à sala e ao processamento do
> driver. O mesmo ocorreu na reprodução pelo ar, cenário de *Physical Access*
> ausente do treino: tocados por um alto-falante de celular e captados pelo
> microfone, os áudios sintéticos caíram para abaixo do limiar, e a ordem entre
> humanos e sintéticos passou a depender da equalização do celular. O sistema funciona de ponta a ponta; o limite está no que o modelo
> aprendeu.
>
> As metas de F1 (0,85 no RNF04 e 0,90 na projeção do planejamento) e de EER
> inferior a 8% não foram atingidas. A meta de EER se baseava em sistemas que
> usam o silêncio como atalho; sob o mesmo protocolo, os modelos deste trabalho
> têm EER menor que o dos modelos com CQT da literatura. A meta de F1 revelou
> que o F1 da classe majoritária não é métrica adequada para esta base.
>
> Como trabalhos futuros, destacam-se: repetir os treinamentos com várias
> sementes; estender a avaliação às demais condições da ASVspoof 2021 LA, ao
> processamento de clientes de conferência, à reprodução pelo ar e a vozes
> externas à base, com
> coleta consentida; treinar sem depender do topo da banda, com LFCC limitado
> ou aumento de dados com conversão de taxa; incluir no treino gravações em
> condições variadas (microfones comuns, salas, outras línguas) e ataques
> atuais, como os da ASVspoof 5 (WANG et al., 2024); calibrar os modelos no canal de destino, o
> que permitiria aproveitar a fusão de scores em tempo real; investigar
> mecanismos de atenção mais expressivos; e usar encoders pré-treinados em
> fala, direção apontada pela literatura para os ataques autorregressivos. Fica também como hipótese a
> testar a associação entre o ramo de espectrograma e os ataques por filtragem
> de forma de onda (A13, A17 e A18).

---

## APÊNDICE A – EER (%) por ataque dos sete modelos e dos sistemas de referência

| ataque | B01¹ | B02¹ | v1 | v2 | v3a | v3 | lcnn_v4 | fusion_v4 | attention_v4 |
|---|---|---|---|---|---|---|---|---|---|
| A07 | 0,00 | 12,86 | 22,28 | 22,43 | 0,24 | 0,13 | 0,27 | 0,02 | 0,01 |
| A08 | 0,04 | 0,37 | 2,68 | 3,88 | 0,05 | 0,00 | 0,59 | 0,03 | 0,05 |
| A09 | 0,14 | 0,00 | 1,19 | 2,32 | 0,19 | 0,19 | 0,29 | 0,12 | 0,08 |
| A10 | 15,16 | 18,97 | 31,24 | 30,63 | 32,62 | 35,88 | 37,50 | 45,54 | 44,61 |
| A11 | 0,08 | 0,12 | 4,60 | 3,85 | 4,42 | 11,69 | 7,96 | 33,74 | 32,03 |
| A12 | 4,74 | 4,92 | 43,50 | 36,18 | 56,47 | 50,05 | 52,28 | 47,99 | 48,11 |
| A13 | 26,15 | 9,57 | 34,59 | 36,50 | 59,58 | 48,31 | 44,19 | 6,54 | 5,51 |
| A14 | 10,85 | 1,22 | 10,71 | 12,01 | 0,96 | 2,08 | 12,14 | 17,74 | 28,49 |
| A15 | 1,26 | 2,22 | 19,23 | 15,02 | 8,75 | 12,05 | 9,54 | 29,37 | 32,33 |
| A16 | 0,00 | 6,31 | 22,08 | 22,75 | 0,70 | 0,30 | 0,75 | 0,03 | 0,09 |
| A17 | 19,62 | 7,71 | 11,20 | 11,27 | 10,28 | 8,22 | 10,50 | 0,41 | 0,39 |
| A18 | 3,81 | 3,58 | 21,48 | 15,01 | 30,14 | 32,41 | 34,11 | 4,63 | 5,56 |
| A19 | 0,04 | 13,94 | 8,99 | 6,98 | 0,13 | 0,10 | 0,37 | 0,00 | 0,03 |
| **global** | 9,57 | 8,09 | 20,78 | **18,99** | 21,23 | 21,10 | 21,56 | 20,18 | 21,13 |

Fonte: Autoria própria; B01 e B02: Wang et al. (2020, Tabela 8).

¹ Com o silêncio preservado; comparar a ordem entre ataques, não os valores.

> A figura `outputs/report/eer_por_ataque.png` (gráfico de barras desta matriz)
> e `outputs/report/curvas_comparadas.png` (EER de validação e *loss* por
> época) podem entrar como figuras. No ABNT, legenda acima ("Figura N –
> título") e fonte abaixo. As curvas de validação ilustram bem a seção 5.2:
> cinco modelos ficam colados em zero, e os dois melhores na avaliação são os
> que ficam em torno de 10%.

---

## APÊNDICE B – EER (%) por ataque na ASVspoof 2021 LA

| ataque | baseline_v2 2019 | baseline_v2 referência 2021 | baseline_v2 Opus real | fusion_v4 2019 | fusion_v4 referência 2021 | fusion_v4 Opus real |
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

Fonte: Autoria própria. Amostra estratificada de 10.000 áudios por condição
(cerca de 690 áudios sintéticos por ataque); a coluna 2019 é o eval completo.

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

GRIFFIN, Daniel W.; LIM, Jae S. Signal estimation from modified short-time
Fourier transform. *IEEE Transactions on Acoustics, Speech, and Signal
Processing*, v. 32, n. 2, p. 236-243, 1984.

KINGMA, Diederik P.; BA, Jimmy. Adam: a method for stochastic optimization. In:
INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS, 3., 2015, San Diego.
*Proceedings* [...]. San Diego: ICLR, 2015.

KINNUNEN, Tomi et al. t-DCF: a detection cost function for the tandem
assessment of spoofing countermeasures and automatic speaker verification. In:
THE SPEAKER AND LANGUAGE RECOGNITION WORKSHOP (ODYSSEY), 2018, Les Sables
d'Olonne. *Proceedings* [...]. 2018. p. 312-319.
*(Citada na 4.11: o t-DCF foi calculado.)*

LAVRENTYEVA, Galina et al. STC antispoofing systems for the ASVspoof2019
challenge. In: INTERSPEECH, 2019, Graz. *Proceedings* [...]. 2019.
p. 1033-1037.

MORISE, Masanori; YOKOMORI, Fumiya; OZAWA, Kenji. WORLD: a vocoder-based
high-quality speech synthesis system for real-time applications. *IEICE
Transactions on Information and Systems*, v. E99-D, n. 7, p. 1877-1884, 2016.

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
2020. DOI: 10.1016/j.csl.2020.101114.
*(Conferido em 25/09/2026: volume, número do artigo e DOI.)*

WANG, Xin et al. ASVspoof 5: crowdsourced speech data, deepfakes, and
adversarial attacks at scale. In: THE AUTOMATIC SPEAKER VERIFICATION SPOOFING
COUNTERMEASURES WORKSHOP (ASVSPOOF 2024), 2024. *Proceedings* [...]. 2024.

> As páginas, cidades e números de artigo acima foram escritos de memória.
> Conferir cada um na fonte (DOI ou anais) antes da entrega.
