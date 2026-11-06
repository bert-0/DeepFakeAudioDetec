# Ajustes finais do TC2 (entrega em 08/11/2026)

Trechos para colar no Google Doc "TC2 - Detecção de Deep Fakes em Áudio com Fusão
de Características e Atenção" (versão de 01/11/2026). Cada item diz a seção, o
começo do trecho atual e o trecho novo. Decisão de 04/11/2026: a calibração com
vozes novas e o ensaio 6 não serão feitos antes da entrega, e os trechos
`[PENDENTE]` do ensaio 6 passam a dizer isso, como o plano prevê (o Incremento 4
entra como resultado parcial).

Os números novos daqui já estão no próprio documento ou no `RESUMO_TCC.md`
(seções indicadas).

---

## 1. Ensaio 6: os quatro trechos pendentes

### 1.1 Seção 5.9, último parágrafo

**Atual:** "Faltam duas etapas, com regras registradas antes: a calibração do
limiar com vozes de mais pessoas, [...] \[PENDENTE: resultado da calibração com
vozes novas e do ensaio 6.\]"

**Novo:**

> Duas etapas tinham regras registradas antes. A primeira, a calibração do limiar
> com vozes de mais pessoas, gravadas no modo de captura atual e no microfone
> embutido do notebook, foi feita depois do fechamento dos resultados deste
> texto, para a demonstração; nela, o limiar é o maior score das vozes reais de
> calibração, e quem será testado fica de fora. A segunda, um sexto ensaio, com
> um locutor fora da calibração e o critério da seção 5.7, não coube no prazo.
> Os resultados deste texto usam, portanto, o limiar tirado das dez vozes de um
> único voluntário, gravadas noutro modo de captura, e a meta do incremento com
> o modelo final não teve um teste com critério fixado antes.

(Atualizado em 06/11/2026: a calibração com três pessoas foi feita nesse dia,
RESUMO_TCC §10.2.46. Este trecho não põe os números dela no texto; se o grupo
quiser incluí-los, eles estão lá, com o achado da fala livre.)

### 1.2 Quadro 3, última linha, coluna "situação"

**Atual:** "atingido parcialmente; o teste com critério é o ensaio 6 \[PENDENTE:
ensaio 6\]"

**Novo:** "atingido parcialmente; o ensaio com critério do modelo final não foi
realizado"

### 1.3 Seção 5.13, item "Vozes externas em pequena escala"

**Atual:** "[...] A coleta com 8 a 10 locutores prevista no plano não foi feita.
\[PENDENTE: ensaio 6, com vozes novas no microfone embutido.\]"

**Novo:**

> **Vozes externas em pequena escala.** Os ensaios do quarto incremento usam dois
> locutores (um integrante do grupo e um voluntário), uma sala e um microfone,
> com dez frases por ensaio. A coleta com 8 a 10 locutores prevista no plano não
> foi feita, nem o ensaio com critério do modelo final, com vozes de calibração
> novas e um locutor fora delas (seção 5.9).

### 1.4 Seção 8 (Conclusão), fim do parágrafo "Por fim, o quarto incremento..."

**Atual:** "[...] precisa ser calibrado com vozes reais do próprio canal.
\[PENDENTE: resultado do ensaio 6, com vozes novas no microfone embutido.\]"

**Novo (trocar só a marca pelo período abaixo):**

> O teste com critério do modelo final, com um locutor novo e o limiar calibrado
> em vozes de outras pessoas, não coube no prazo e é o primeiro dos trabalhos
> futuros.

### 1.5 Consentimento do L01 e do L02 (seções 5.7 e 7.1)

Os dois concordaram de boca em outubro e assinam o termo antes da entrega, com
o item que cobre as gravações já feitas. O texto dizia que gravaram "com termo
de consentimento assinado", o que não era o caso na época.

**Seção 5.7, primeira frase. Atual:** "[...] gravadas para este trabalho com
termo de consentimento (seção 7.1)."

**Novo:** "[...] gravadas para este trabalho com o consentimento dos locutores
(seção 7.1)."

**Seção 7.1, segundo parágrafo, primeira frase. Atual:** "As vozes reais dos
ensaios foram gravadas com termo de consentimento assinado, que inclui um item
próprio e opcional para a geração de clones da voz."

**Novo:**

> Os locutores dos ensaios consentiram verbalmente antes de gravar e depois
> assinaram um termo de consentimento que cobre também essas gravações e inclui
> um item próprio e opcional para a geração de clones da voz.

(Só usar depois que os dois assinarem. Se algum não autorizar, os resultados
dele saem do texto, como o termo promete.)

---

## 2. Outras marcas pendentes

### 2.1 Seção 5.13, item "Uma semente também no quarto incremento"

**Atual:** "As três sementes previstas no plano para os modelos finalistas não
foram rodadas. \[PENDENTE: confirmar.\]"

**Novo:** apagar só a marca. Confirmado: nenhum modelo do quarto incremento foi
treinado com mais de uma semente.

### 2.2 Seção 7.1, licença do BUT ReverbDB

**Atual:** "[...] e o BUT ReverbDB \[PENDENTE: conferir a licença\]."

**Novo:** "[...] e o BUT ReverbDB, distribuído gratuitamente pelos autores sob
licença não restritiva (SZŐKE et al., 2019)."

O pacote baixado não traz arquivo de licença. A versão publicada do artigo diz
"non-restrictive license"; a versão preliminar no arXiv dizia Apache 2.0.

### 2.3 Seção 6.4, Marco 8

**Atual:** "Marco 8 – Nov/2026: Modelo final, relatório TC2 finalizado e
apresentação final do TC2. \[PENDENTE: datas de entrega e da banca.\]"

**Novo:** "Marco 8 – Nov/2026: Modelo final e relatório TC2 finalizado, com
entrega do documento em 8 de novembro de 2026, e apresentação à banca."

(Se a data da banca já estiver marcada, acrescentar.)

### 2.4 Seção 7.1, Comitê de Ética

**Atual:** "\[PENDENTE: informar se a UNIP exigiu parecer do Comitê de Ética em
Pesquisa.\]"

**Novo:** depende da resposta do grupo. Se não houve exigência: "A UNIP não
exigiu parecer do Comitê de Ética em Pesquisa para a coleta." Se houve: dizer o
parecer.

---

## 3. Correções de consistência

### 3.1 Seção 4.13, parágrafo "Interface"

A seção 4.14 e o Quadro 2 já falam da aba Clonar e da conferência, mas este
parágrafo diz "quatro abas" e descreve o gráfico só por janela. Na interface, as
abas são cinco (Enviar arquivo, Resultado, Ao vivo, Clonar e Histórico); a
conferência e o "Sobre" são links no cabeçalho.

**Novo (substitui o parágrafo inteiro):**

> **Interface.** A interface web, feita com FastAPI e páginas renderizadas com
> Jinja2, tem cinco abas. Em "Enviar arquivo", o usuário escolhe um áudio (WAV,
> FLAC, MP3, M4A, OGG, Opus ou AAC, até 25 MB). Em "Resultado", vê o score, o
> limiar, a decisão (indício ou não de síntese) e um gráfico com o score de cada
> trecho ao longo do tempo: as frases, no modelo final, ou as janelas de 4 s, nos
> modelos anteriores. O áudio de uma sessão ao vivo ou de um clone pode ser
> ouvido na mesma página, e o relatório pode ser baixado em JavaScript Object
> Notation (JSON). Em "Ao vivo", escolhe a fonte (som do computador ou microfone)
> e o dispositivo, inicia e encerra a captura e acompanha o resultado de cada
> frase; a tela avisa quando há som sem fala reconhecida ou quando o microfone
> satura. Em "Histórico", consulta e exclui análises anteriores, guardadas num
> banco SQLite. Em "Clonar", gera, no próprio computador, o clone de uma voz cujo
> dono autorizou a clonagem no termo de consentimento; o clone é analisado como
> arquivo e apagado junto com o registro (seção 7.1). No cabeçalho, a situação
> da conferência, feita automaticamente quando o sistema é iniciado, leva à
> página "Conferir": o modelo e o limiar, a placa de vídeo, o microfone, a saída
> de som, as ferramentas de clonagem e o espaço em disco. A interface apresenta o
> resultado como indício, e não como prova, e a página "Sobre", também no
> cabeçalho, descreve as limitações medidas na seção 5.

### 3.1b Quadro 2, linha do UC08

**Atual:** "UC08 – Conferir o sistema | — | aba Conferir"

**Novo:** "UC08 – Conferir o sistema | — | página Conferir, no cabeçalho"

### 3.2 Quadro 1, última linha

**Atual:** "modelo final | XLS-R 300M congelado, com pesos aprendidos por camada e
ramo LFCC, treinado em cinco domínios | xlsr\_congelado"

**Novo (duas linhas no lugar de uma):**

| versão | alteração em relação à anterior | modelo |
|---|---|---|
| primeira cabeça | XLS-R 300M congelado, com pesos aprendidos por camada e ramo LFCC, treinado em cinco domínios | xlsr\_congelado |
| modelo final | a mesma cabeça, treinada de novo com o In-the-Wild como sexto domínio | xlsr\_congelado\_itw |

---

## 4. Declaração do uso de assistente de IA

Não há no texto. Sugestão: novo parágrafo ao fim da seção 4.1 (Tipo de
Pesquisa). O grupo deve conferir se a descrição corresponde ao uso que fez.

> **Uso de assistente de inteligência artificial.** O desenvolvimento contou com
> um assistente de programação baseado em inteligência artificial (Claude, da
> Anthropic, por meio do Claude Code). Ele foi usado na escrita e na revisão do
> código e dos testes automatizados, na preparação e na análise dos experimentos
> e na revisão do texto. As decisões de escopo e de desenho, os critérios de
> sucesso de cada experimento, fixados antes dos resultados, e a revisão final
> ficaram com o grupo. Os números citados vêm dos registros de experimentos
> mantidos no repositório do projeto.

---

## 5. Resumo, Abstract e palavras-chave

O Resumo e o Abstract atuais são os da versão de reserva e não citam o quarto
incremento. As versões abaixo têm cerca de 400 palavras (a NBR 6028 pede de 150
a 500 em trabalhos acadêmicos).

### 5.1 RESUMO

> Este trabalho investiga a detecção de áudios *deepfake* com fusão de
> características e mecanismo de atenção, avaliando o ganho de cada componente
> sobre uma linha de base, a robustez ao canal e a generalização para vozes e
> geradores fora da base de treino. Sete modelos foram treinados na trilha
> *Logical Access* da ASVspoof 2019, com remoção de silêncio, e avaliados em
> 71.237 áudios de treze ataques, onze deles inéditos. A validação não antecipou
> a avaliação: modelos com EER de validação entre 0,03% e 9,74% ficaram todos
> entre 18,99% e 21,56% na avaliação. A fusão tardia de LFCC e espectrograma
> log-mel reduziu o EER de 21,56% para 20,18%, a atenção no *pooling* não trouxe
> ganho (+0,95 p.p.) e a fusão de scores de dois modelos de arquiteturas
> diferentes chegou a 13,13%. Em transmissão real por Opus (ASVspoof 2021 LA), o
> canal custou de 7 a 11 vezes o que a simulação do mesmo codec indicava, e a
> robustez simulada não previu qual modelo resiste. O modelo foi integrado a um
> sistema com interface web, que analisa arquivos e o áudio do computador ou do
> microfone em tempo real; na captura, um filtro FIR que preserva a faixa de 7,6
> a 8 kHz reduziu o custo de 5,68 para 0,39 p.p., sem retreino. Diante de uma voz
> externa à base, porém, o modelo respondeu às condições de gravação, e não ao
> *vocoder*. Um quarto incremento atacou esse limite. A degradação de canal no
> treino deixou o modelo quase insensível ao Opus real. Em ensaios com vozes
> reais em português contra clones de duas ferramentas, com critério fixado
> antes, a meta de 8 de 10 acertos em cada classe foi atingida pelo arquivo com
> os clones do XTTS, mas não com uma ferramenta mais recente em todos os
> locutores, nem pelo ar. O XLS-R 300M congelado, usado como instrumento de
> medida, mostrou que a pista que generaliza está na dinâmica da voz no tempo. O
> modelo final, uma cabeça com pesos aprendidos por camada sobre o XLS-R
> congelado e um ramo LFCC, treinada em seis domínios, passou no critério fixado
> antes em todas as réguas, com AUC de 0,975 na avaliação da ASVspoof 5, 0,931 em
> português (BRSpeech-DF), 0,977 em áudio tocado e regravado (ReplayDF) e 0,957
> em vozes da internet (In-the-Wild) de pessoas fora do treino. Em todas, o que
> falha é o ponto de corte, e não a ordem: ele depende do canal e precisa ser
> calibrado com vozes reais do microfone de uso, e o ensaio com critério dessa
> calibração ficou como trabalho futuro. As metas de F1 e de EER do planejamento
> não foram atingidas.

**Palavras-chave:** Deepfake. Áudio. Fusão de características. LFCC. ASVspoof.
XLS-R. Robustez de canal.

### 5.2 ABSTRACT

> This work investigates audio deepfake detection with feature fusion and an
> attention mechanism, measuring the gain of each component over a baseline,
> the robustness to the transmission channel and the generalization to voices
> and generators outside the training data. Seven models were trained on the
> ASVspoof 2019 Logical Access track, with silence removed, and evaluated on
> 71,237 utterances from thirteen attacks, eleven of them unseen. Validation did
> not anticipate evaluation: models with validation EER between 0.03% and 9.74%
> all scored between 18.99% and 21.56% on evaluation. Late fusion of LFCC and
> log-mel spectrogram reduced the EER from 21.56% to 20.18%, attentive pooling
> did not help (+0.95 p.p.), and score-level fusion of two architecturally
> different models reached 13.13%. Under real Opus transmission (ASVspoof 2021
> LA), the channel cost 7 to 11 times what the simulation of the same codec
> indicated, and simulated robustness did not predict which model withstands it.
> The model was integrated into a system with a web interface that analyzes
> files and the computer's or the microphone's audio in real time; in capture,
> an FIR filter that preserves the 7.6 to 8 kHz band reduced the cost from 5.68
> to 0.39 p.p. without retraining. On a voice outside the dataset, however, the
> model responded to recording conditions rather than to the vocoder. A fourth
> increment addressed this limit. Channel degradation during training made the
> model almost insensitive to real Opus. In trials with real Portuguese voices
> against clones from two tools, with criteria fixed in advance, the target of 8
> out of 10 correct decisions per class was reached through the file path with
> XTTS clones, but not with a more recent tool for every speaker, nor over the
> air. The frozen XLS-R 300M, used as a measuring instrument, showed that the
> cue that generalizes lies in the dynamics of the voice over time. The final
> model, a head with learned per-layer weights over the frozen XLS-R plus an
> LFCC branch, trained on six domains, met the criterion fixed in advance on
> every benchmark, with an AUC of 0.975 on the ASVspoof 5 evaluation set, 0.931
> on Portuguese (BRSpeech-DF), 0.977 on replayed audio (ReplayDF) and 0.957 on
> internet voices (In-the-Wild) from speakers outside training. On all of them,
> what fails is the decision threshold, not the ranking: it depends on the
> channel and must be calibrated with real voices from the microphone in use,
> and the trial with criteria for that calibration is left as future work. The
> planned F1 and EER targets were not reached.

**Keywords:** Deepfake. Audio. Feature fusion. LFCC. ASVspoof. XLS-R. Channel
robustness.

---

## 6. No Google Doc, sem texto novo

- **Sumário:** está desatualizado (ainda lista "5.6 Resultados Esperados" e "5.7
  Limitações"). No Doc: clicar no sumário e em "Atualizar sumário".
- **Figuras 1 a 4 (UML):** o texto tem só os marcadores `[INSERIR AQUI ...]`. Os
  diagramas do Drive são do TC1 (maio) e não têm Clonar, Conferir nem o modelo
  final.
- **Tabela 6.1 (Divisão de Responsabilidades):** atribui partes do sistema a
  cada integrante. Conferir com o grupo se fica assim.
