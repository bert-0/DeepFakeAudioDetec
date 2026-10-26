# Literatura: detecção de áudio sintético (levantamento de 27/10/2026)

Levantamento feito depois do estudo das imperfeições (RESUMO_TCC §10.2.17 a
§10.2.28), para ver o que mais existe além do XLS-R e o que muda nos geradores
atuais. Os números daqui são **os que os autores relatam**, em bases e protocolos
deles; não foram reproduzidos neste projeto. Antes de levar um número para a
redação, conferir no artigo.

## 1. O que muda nos geradores de hoje

- **Clonagem com poucos segundos e sem treino ("zero-shot").** Modelos como o
  F5-TTS (*flow matching* com transformer de difusão, ACL 2025) imitam uma voz a
  partir de uma referência curta. Foi treinado com 100 mil horas multilíngues; os
  próprios autores pedem marca d'água e detectores
  ([arXiv 2410.06885](https://arxiv.org/abs/2410.06885)).
- **Codecs neurais no lugar dos vocoders.** Muitos sistemas novos geram *tokens*
  de um codec neural (EnCodec, DAC etc.) e o decodificador do codec produz o som
  ("CoSG"). Detectores treinados só com falsos de vocoder falham nesses casos.
  - CodecFake: treinar com falsos de codec reduz muito o erro contra o VALL-E
    ([arXiv 2406.07237](https://arxiv.org/abs/2406.07237)).
  - CodecFake+: 31 codecs no treino e 17 geradores com codec no teste
    ([arXiv 2501.08238](https://arxiv.org/abs/2501.08238)). Propõe uma taxonomia
    pelo quantizador, pelo decodificador e pelos objetivos auxiliares.
  - Rastreio de origem do codec
    ([arXiv 2505.12994](https://arxiv.org/abs/2505.12994),
    [arXiv 2506.07294](https://arxiv.org/abs/2506.07294)). O segundo relata que
    modelos treinados só com áudio ressintetizado se apoiam nas regiões sem fala:
    o mesmo atalho das pausas visto na parte 7 (RESUMO_TCC §10.2.23).
- **Detectores envelhecem rápido.** No Deepfake-Eval-2024 (áudio de 2024 tirado de
  redes sociais, 56,5 h, 52 línguas), detectores abertos de ponta perdem 48% de
  AUC em áudio em relação às bases acadêmicas
  ([arXiv 2503.02857](https://arxiv.org/abs/2503.02857)). Um preprint de 2026
  testa TTS recentes (Dia2, Maya1, MeloTTS) e conclui que não há garantia de
  generalização ([arXiv 2601.20510](https://arxiv.org/abs/2601.20510)).
- **ASVspoof 5 (resumo do desafio).** Os melhores usaram modelos auto-supervisionados
  pré-treinados e dados externos. O desempenho cai com ataques adversariais e com
  codecs neurais, e a calibração dos scores ficou como problema em aberto
  ([arXiv 2408.08739](https://arxiv.org/abs/2408.08739),
  [arXiv 2601.03944](https://arxiv.org/abs/2601.03944)).
- **Marca d'água de fábrica.** O Chatterbox marca todo áudio com o Perth
  (Resemble AI), verificável pelo pacote `perth`. Nos nossos 20 clones, a marca
  aparece em todos, e em nenhum dos outros 40 áudios (RESUMO_TCC §10.2.28).

## 2. Linhas de detecção e como se ligam ao que medimos

### 2.1 Camadas do meio de modelos auto-supervisionados
- **SLS** (Zhang et al., ACM MM 2024): usa várias camadas do XLS-R, não só a
  última. Relata EER de 1,92% no ASVspoof 2021 DF e de 7,46% no In-the-Wild.
- **Seleção de camadas por sondas** (preprint de jun./2026,
  [arXiv 2606.30791](https://arxiv.org/abs/2606.30791)). Sondas leves ordenam as
  camadas, e as úteis formam zonas de profundidade.
  - No XLS-R 300M, as melhores incluíram as camadas 6, 7, 17 e 19.
  - As mais profundas (23 e 24) foram as piores no In-the-Wild.
  - Na mesma linha: Serrano et al. 2025
    ([arXiv 2509.12003](https://arxiv.org/abs/2509.12003)).
- **Ligação:** é o mesmo achado da parte 6 (RESUMO_TCC §10.2.22). Ali, a camada 7
  foi a melhor, e as camadas 10 a 18 se inverteram no canal dos ensaios.
- **AntiDeepfake** (NII, [arXiv 2506.21090](https://arxiv.org/abs/2506.21090)):
  modelos auto-supervisionados pós-treinados com 56 mil horas de fala real e 18
  mil horas com artefatos, em mais de 100 línguas. Pesos e código públicos.
  Seriam uma caixa-preta pronta, que o grupo decidiu não adotar como detector.

### 2.2 Modelo só de fala natural ("classe única")
- **Trajetórias no espaço auto-supervisionado** (preprint de ago./2026,
  [arXiv 2608.13817](https://arxiv.org/abs/2608.13817)). Um LSTM, treinado só com
  os humanos da 2019 LA, prevê o próximo quadro de um wav2vec 2.0. O erro de
  previsão vira o score.
  - Os autores relatam EER de 30,35% no Deepfake-Eval-2024 sem ver nenhum falso,
    melhor que um sistema supervisionado de referência (33,36%).
  - Ressalva deles: o *backbone* já tinha sido ajustado com falsos.
- **Ligação com a parte 10 (RESUMO_TCC §10.2.26).** É a mesma ideia de base.
  - O que difere aqui: usamos a perda do próprio pré-treino do XLS-R (sem
    treinar nada), medimos o descasamento de canal (§10.2.27) e a soma com o
    detector.
  - Para o TCC, citar esse trabalho como anterior e não apresentar a ideia como
    inédita.
- **ProSDD** (preprint de 2026): primeiro aprende a variação prosódica da fala
  real (altura, voz, energia) e só depois treina com falsos.

### 2.3 Pistas fisiológicas (explicáveis)
- **Respiração** (Layton et al., 2024,
  [arXiv 2404.15143](https://arxiv.org/abs/2404.15143)). Os geradores lidam mal
  com a respiração.
  - Os autores relatam acerto perfeito em 33,6 h de áudio de notícias, contra
    AUPRC de 0,72 de um modelo com wav2vec.
  - Um trabalho de out./2026, em outro protocolo, relata EER de 21,40% com
    respiração, qualidade de voz e tempo. O resultado perfeito não parece se
    repetir fácil.
- **Trato vocal** (Blue et al., USENIX Security 2022,
  [página](https://www.usenix.org/conference/usenixsecurity22/presentation/blue)).
  Estima, por dinâmica de fluidos, a forma do trato vocal que produziria o som; os
  falsos pedem formas anatomicamente improváveis. Relatam 99,5% de revocação com
  99,9% de precisão, em dados de 2022 (TIMIT e clonadores da época).
- **Prosódia clássica** ("Pitch Imperfect", Warren et al., 2025,
  [arXiv 2502.14726](https://arxiv.org/abs/2502.14726)): altura, jitter, shimmer e
  HNR. Relatam EER de 24,7%, parecido com as referências, mas com mais robustez a
  ataques adversariais.
- **Ligação:** na parte 8 (RESUMO_TCC §10.2.24), as medidas de dinâmica feitas à
  mão não passaram do LFCC no dev, o que é coerente com esses EER de 20–25%.

### 2.4 Detecção pela identidade da pessoa
- **Verificação de locutor como detector** (Pianese et al., Univ. de Nápoles,
  WIFS 2022, [arXiv 2209.14098](https://arxiv.org/abs/2209.14098)). Compara o
  áudio com um conjunto de gravações reais da pessoa e, assim, não precisa de
  falsos no treino. Uma versão de 2024 sem treino nenhum usa modelos
  pré-treinados ([arXiv 2405.02179](https://arxiv.org/abs/2405.02179)).
- **Ligação:** essa linha muda a pergunta de "isto é sintético?" para "isto é
  mesmo esta pessoa?".
- **Descartada pelo grupo (27/10/2026)**, por três motivos:
  1. Só protege quem foi cadastrado antes, e o projeto busca um detector geral.
  2. O reconhecedor de voz precisa ser treinado com milhares de locutores, ou
     ser um modelo pronto (outra caixa-preta).
  3. O golpista clona a partir de gravações reais da mesma pessoa; o clone é
     feito para passar exatamente nessa comparação.

### 2.5 Marca d'água (abordagem proativa)
- **AudioSeal** (Meta, ICML 2024,
  [arXiv 2401.17264](https://arxiv.org/abs/2401.17264)): marca que localiza os
  trechos gerados, amostra a amostra. **Perth** (Resemble AI), no Chatterbox.
- **Limite:** só funciona se o gerador marcar. Nas ferramentas abertas, quem
  quer enganar desliga a marca.

### 2.6 Explicação com modelos de linguagem de áudio
- CoLMbo-DF ([arXiv 2603.28021](https://arxiv.org/abs/2603.28021)) dá ao modelo
  medidas acústicas para ele raciocinar antes do veredito.
- FT-GRPO ([arXiv 2601.02983](https://arxiv.org/abs/2601.02983)) gera
  justificativas em tempo e frequência.
- Um estudo de 2026 ([arXiv 2601.03615](https://arxiv.org/abs/2601.03615)) mostra
  que o raciocínio explícito nem sempre resiste a ataques.

## 3. Bases em português

- **BRSpeech-DF** (2025; conferir a referência completa na ACL Anthology):
  apresentado como a primeira base pública de *deepfake* em português (Brasil e
  Portugal), com mais de 458 mil trechos e 62 locutores reais.
  Detectores treinados em inglês tiveram EER de 30,67% a 63,03%.
- **FakeBrAccent** (CILAMCE 2025,
  [página](https://publicacoes.softaliza.com.br/cilamce-2025/article/view/14339)):
  746 trechos com sotaques regionais; o sotaque muda o desempenho.
- **ParlaSpoof-BR** (preprint): falas da Câmara dos Deputados com versões
  sintéticas; decisões inconsistentes entre as regiões do Brasil.
- **MLAAD** ([arXiv 2401.09512](https://arxiv.org/abs/2401.09512)): dezenas de
  TTS em quase 40 línguas; o real vem do M-AILABS. Conferir no cartão da base se
  o português está incluído.

Licença e acesso dessas bases ainda não foram conferidos.
