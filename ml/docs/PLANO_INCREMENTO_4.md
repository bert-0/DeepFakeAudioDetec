# Plano do Incremento 4

Versão de 02/10/2026 (cópia do documento do grupo). Datas e decisões pendentes
no fim.

## Objetivo e critério de sucesso

Entregar a versão final do TCC com o Incremento 4 no corpo do trabalho e mostrar
à banca o sistema separando a voz real do orientador do clone dela, pelo arquivo
e pelo microfone.

- **Meta do ensaio:** no equipamento e na sala da apresentação, com locutores
  fora do treino, pelo menos 8 de 10 frases corretas em cada classe (real e
  sintética), nos dois caminhos.
- **Data de congelamento:** nenhum experimento novo depois dela. A definir, de 7
  a 10 dias antes da entrega.
- **Versão de reserva:** o TC2 como está hoje, marcado com uma tag antes de o
  Incremento 4 mexer no texto (a `tc2-resultados` é mais antiga). Se a meta não
  for atingida, ela segue como texto final, e o Incremento 4 entra como
  resultado parcial e trabalho futuro.

## Decisões já tomadas

O Incremento 4 ataca os três problemas que os sete modelos revelaram: a
validação não prevê a avaliação, a decisão depende da faixa de 7,6 a 8 kHz e o
modelo reage às condições de gravação.

| decisão | motivo |
|---|---|
| Incremento 4 no corpo do texto (seção 4.7); tabelas dos Incrementos 1 a 3 inalteradas | segue o ciclo incremental do TC1; os sete modelos viram o diagnóstico que motiva o incremento |
| Desenho 2x2: fusion_v4 (LCNN) e XLS-R, cada um treinado na 2019 LA e no recorte da ASVspoof 5 | isola o ganho do front-end do ganho da base |
| Front-end wav2vec 2.0 XLS-R 300M, em fusão com um ramo LFCC; atenção sobre as camadas do XLS-R | com fine-tuning o XLS-R supera o front-end congelado (Wang e Yamagishi, 2022) |
| Treino no recorte da ASVspoof 5 (cerca de 59 mil áudios, 11 GB); validação no dev, com ataques inéditos (A09 a A16) | a validação passa a medir generalização, e não o 0,03% que escolheu os checkpoints |
| Avaliação em: 2019 sem silêncio; Opus real da 2021 LA (10 mil); ASVspoof 5 estratificada (10 mil); controle pareado com vozes reais | mesmas réguas para os quatro modelos do 2x2 |
| Degradação de canal (codec, RawBoost, conversão de taxa, resposta ao impulso) aplicada antes do pré-processamento | hoje ela atinge o trecho já sem silêncio e normalizado |
| Sem dependência de 7,6 a 8 kHz: LFCC de 0 a 4 kHz ou descarte aleatório da banda alta no treino | essa faixa some em qualquer áudio gravado a 44,1 ou 48 kHz |
| Toda condição de captura aparece nas duas classes, no treino e na demonstração | evita um detector de microfone em vez de um detector de síntese |
| Microfone por WASAPI exclusivo (sounddevice), 48 kHz e o FIR do projeto; loopback só para chamada; arquivos analisados sem captura | contorna os aprimoramentos do driver e a conversão de taxa do Windows |
| Aba Gravar na interface, com espectrograma e energia por faixa | coleta e demonstração pelo mesmo caminho; cobre a visualização ao operador (RF10) |
| Treino na nuvem: Kaggle (T4, fp16), uma conta por integrante; Colab como reserva; GTX 1650 para avaliação e XLS-R congelado | cerca de 30 h de GPU por semana por conta |
| Áudios fora do GitHub; versionar scripts, protocolos, listas do recorte e resultados | tamanho, licença da ASVspoof 5 e privacidade das vozes |

## Tarefas por frente

Cronograma de 4 semanas a partir de 05/10/2026. A ética e o gravador vêm
primeiro: nenhuma gravação começa antes deles.

**Ética e coleta**
- Combinar com o orientador a mudança de escopo e perguntar se a UNIP exige
  parecer do Comitê de Ética em Pesquisa.
- Termo de consentimento: gravação, uso no treino e, para o orientador,
  autorização explícita para sintetizar a voz só na demonstração.
- Protocolo de gravação: lista de frases, 8 a 10 locutores, equipamento, sala e
  nível de entrada.
- Pseudonimização: código de locutor no lugar do nome; regra de armazenamento e
  exclusão dos clones depois da banca.

**Captura e gravador**
- Fonte de microfone com `sounddevice` em WASAPI exclusivo, 48 kHz, FIR para
  16 kHz; modo compartilhado com aviso se o driver recusar.
- Aba Gravar: gravar e parar, análise do arquivo inteiro, espectrograma, energia
  em 0–4, 4–7,6 e 7,6–8 kHz e score por janela.
- Checagens logo após gravar: saturação, fração de fala, banda estreita, piso de
  ruído nas pausas.
- Botão "salvar no conjunto de dados" com metadados: locutor, frase, termo,
  dispositivo e modo de captura.
- Testar o microfone da coleta: mesma frase em modo compartilhado e exclusivo, e
  com ruído constante ao fundo, para detectar processamento no hardware.

**Dados**
- Baixar treino e dev da ASVspoof 5 (57,5 GB) e montar o recorte: 19 mil
  humanos, cerca de 5 mil falsos por ataque, trechos de 6 s em int16.
- Avaliação estratificada da ASVspoof 5 (10 mil) lendo os `.tar` em fluxo, sem
  baixar os 94 GB.
- `src/data/asvspoof5.py` para o protocolo TSV de 9 colunas.
- Subir o recorte como dataset privado no Kaggle.
- Gravar o grupo e os voluntários com a aba Gravar.
- Gerar os falsos com 2 ou 3 ferramentas de clonagem, deixando uma fora do treino.
- Coloração de microfone e sala nos falsos (resposta ao impulso e equalização), e
  alguns falsos regravados pelo alto-falante.

**Modelos e treino**
- Gancho de degradação logo depois do `load_audio`, antes do
  `preprocess_waveform`.
- RawBoost, codecs e ida e volta 16, 48 e 16 kHz no treino.
- LFCC com `f_min` e `f_max` no config; experimento com CMN.
- Tipo de característica `waveform` e modelo `fusion_ssl` (XLS-R com soma
  ponderada das camadas, ramo LFCC, fusão tardia).
- Retomada do treino a partir do último checkpoint (`--resume`).
- Medir uma época de fine-tuning no Kaggle antes de fechar o cronograma.
- fusion_v4 no recorte da ASVspoof 5, na GTX 1650.
- XLS-R na 2019 LA e no recorte da ASVspoof 5, no Kaggle.
- Três sementes nos modelos finalistas.
- Latência na CPU com o `bench_latencia.py`; truncar camadas do XLS-R se não
  couber em tempo real.

**Avaliação**
- Os quatro modelos do 2x2 nas quatro réguas, com EER e min t-DCF.
- Cruzamento entre bases: treinado na 2019 e testado na ASVspoof 5, e o contrário.
- Controle pareado com locutores fora do treino, nos caminhos arquivo e microfone.
- Ablação de ramos e oclusão da banda alta, só com avaliação.
- Fusão ao vivo dos dois modelos (média das probabilidades, como no monitor) na
  captura FIR, com o limiar da fusão escolhido no dev
  (`scripts/fusao_ao_vivo.py`). Acrescentado em 03/10/2026, depois que o
  fusion_v4 deixou de saturar com o FIR (RESUMO_TCC §10.2.8).
- Limiar calibrado no caminho da demonstração, com o `calibrar_captura.py`.

**Texto do TCC**
- Guardar a versão de reserva.
- Esqueleto das seções novas com marcas `[PENDENTE]`: 3.4, 4.7, 4.8, 4.9, 4.10,
  4.12, 5.3 a 5.7, nova seção de vozes reais, 6 e 7.1.
- Preencher os números até a data de congelamento.
- Reescrever Resumo, Abstract e Conclusão com o resultado final.
- Referências novas: Tak et al. (2022), Wang e Yamagishi (2022), ASVspoof 5.

**Demonstração**
- Roteiro: playlist da ASVspoof pelo som do computador; orientador ao vivo; clone
  dele; tela de espectrograma e faixas.
- Teste cruzado: voz real também como arquivo, clone também pelo microfone.
- Ensaio no equipamento da banca, com a meta de 8 de 10 em cada classe.
- Plano B: vídeo de um ensaio bem-sucedido e o resultado pareado documentado.

**Repositório**
- Merge da branch `claude/jolly-cerf-is46h9` na principal.
- `soundcard` e `sounddevice` no `web/requirements.txt`.
- `.gitignore` para os áudios coletados e gerados.

## Riscos e mitigação

O maior risco é o parecer de ética consumir o mês; por isso a conversa com o
orientador é a primeira tarefa.

| risco | efeito | mitigação |
|---|---|---|
| A UNIP exige parecer do Comitê de Ética | a coleta não termina no mês | reduzir a coleta às vozes do grupo e do orientador, com termo assinado |
| Fila ou cota esgotada no Kaggle | treinos do XLS-R atrasam | uma conta por integrante; Colab como reserva; medir uma época antes de planejar |
| XLS-R lento demais na CPU | a análise ao vivo não acompanha | truncar camadas, rodar na GPU ou usar o XLS-R só no envio de arquivo |
| O modelo não acerta a voz do orientador | a demonstração falha diante da banca | ensaio com meta antes da data de congelamento; plano B com vídeo e o resultado pareado |
| O modelo aprende o canal, e não a síntese | acerta pelo motivo errado, e o teste cruzado derruba | toda condição de captura nas duas classes; teste cruzado arquivo e microfone |
| O microfone faz processamento no hardware | as gravações mudam de forma imprevisível | testar antes da coleta; trocar de microfone ou usar interface USB com ASIO |
| Experimentos atrasam e o texto fica sem números | entrega com marcas de pendência | data de congelamento fixa; versão de reserva pronta |

## Decisões pendentes do grupo

Vêm antes da semana 1, porque mudam o desenho do treino e da demonstração.

- Como a demonstração vai acontecer: orientador ao vivo no microfone, arquivos
  gravados antes ou clone tocando numa chamada?
- Datas de entrega do texto e da banca, e a data de congelamento.
- Quem cuida de cada frente.
- Qual microfone será usado na coleta e na banca, e se há uma interface USB com
  ASIO disponível.
- Quais ferramentas de clonagem usar, e qual fica fora do treino.
