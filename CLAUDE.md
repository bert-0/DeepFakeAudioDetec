# CLAUDE.md

TCC de Ciência da Computação (UNIP): detecção de áudio *deepfake* com fusão de
características, treinada na ASVspoof 2019 LA, mais um sistema de análise
(monitor ao vivo e interface web). Todo o código fica em `ml/`.

## Trabalho atual: Incremento 4

**Siga `ml/docs/PLANO_INCREMENTO_4.md`.** Antes de propor ou implementar algo,
confira se está no plano; se não estiver, diga isso e pergunte antes de seguir.

- Objetivo: separar a voz real do orientador do clone dela, pelo arquivo e pelo
  microfone. Meta do ensaio: 8 de 10 frases certas em cada classe, nos dois
  caminhos, com locutores fora do treino.
- Desenho 2x2: fusion_v4 (LCNN) e XLS-R 300M + LFCC, cada um treinado na 2019 LA
  e no recorte da ASVspoof 5. As tabelas dos Incrementos 1 a 3 não mudam.
- Ordem: ética e gravador primeiro (nenhuma gravação antes do termo de
  consentimento), depois dados, treino, avaliação, texto e demonstração.
- Depois da data de congelamento, nenhum experimento novo. Se a meta não for
  atingida, o texto de reserva (TC2 atual) segue, e o Incremento 4 entra como
  resultado parcial.
- As decisões pendentes estão no fim do plano. Não as decida sozinho.

## Idioma e estilo

- Tudo em português do Brasil: nomes no código, comentários, documentos,
  mensagens de commit e respostas.
- Comentários curtos, só para o porquê (decisão, medida, armadilha). Nada de
  comentário que repete o código. Siga o estilo do arquivo ao redor.
- Interface web: cantos retos, Tahoma, números em monoespaçada, destaque verde
  (tema escuro preto e verde, padrão; tema claro). Nada de gradiente ou brilho.
  Textos curtos na tela; detalhe técnico (caminhos de pasta, erros internos)
  fica no terminal.

## Comandos (de dentro de `ml/`, PowerShell, ambiente conda ativo)

```powershell
python -m pytest tests -q                 # ~690 testes; rodar antes de todo commit
ruff check --select E9,F .                # o mesmo lint do CI
uvicorn web.app:app                       # interface em http://127.0.0.1:8000
python monitor.py --config configs\baseline_v2.yaml --checkpoint checkpoints\baseline_lfcc_cnn_v2.pt [--fonte microfone]
python evaluate.py --config <yaml> --checkpoint <pt> --partition eval
python scripts\robustness_eval.py --config <yaml> --checkpoint <pt> --amostra 10000 --so clean captura_48k_fir
python scripts\calibrar_captura.py --config <yaml> --checkpoint <pt>
```

Mais comandos explicados em `ml/docs/colinha/colinha.html` (PDF ao lado).

**Treinos e avaliações longas quem roda é o usuário** (GTX 1650 local; Kaggle T4
para o XLS-R). Prepare o comando, peça para ele rodar e analise o log ou a saída
que ele colar. Não rode um treino pelo terminal da sessão.

## Mapa do código

- `src/preprocess/`: `load_audio`, `preprocess_waveform` (remove silêncio,
  normaliza pelo pico, janela de 4 s), `reamostragem.py` (FIR polifásico; o
  `soxr` apagava 7,6–8 kHz e não deve voltar ao caminho do modelo),
  `degradacao.py` (aumento no áudio bruto, antes do pré-processamento,
  `audio.degradacao`; o `Augmenter` antigo entra depois dele), `rawboost.py`
  (LnL, ISD e SSI de Tak et al. 2022, com o `rng` da amostra), `codecs.py`
  (MP3, Vorbis, Opus, G.711, GSM pelo libsndfile).
- `src/features/`, `src/models/`: LFCC, log-mel, CNN, LCNN, fusão e atenção.
- `src/capture/`: fontes de áudio (arquivo, loopback, microfone por WASAPI
  exclusivo com `sounddevice`, ou `soundcard` sem ele), analisador em janelas
  de 4 s com passo de 2 s, peso pela fração de fala; `checagens.py` (depois de
  gravar: saturação, fala, banda, piso de ruído).
- `docs/etica/`: rascunhos do termo de consentimento e do protocolo de gravação.
- `src/data/asvspoof5.py` e `scripts/montar_recorte_asvspoof5.py`: protocolos da
  ASVspoof 5 e o recorte lido direto dos `.tar` (em `D:`, pela junção
  `ml/data/ASVspoof5`); recorte em `ml/data/ASVspoof5_recorte/`, seleção
  versionada em `ml/data/recortes/`.
- `src/limiares.py`: limiar por tipo de áudio (nativo 16 kHz: original; 44,1/48
  kHz e ao vivo: o de captura, só se calibrado com o FIR e com os mesmos pesos).
- `web/`: FastAPI + Jinja2 + SQLite. Abas Enviar arquivo, Resultado, Ao vivo e
  Histórico. O ao vivo roda o `monitor.py` como subprocesso, parado por arquivo
  (`--parar-com`). Com `DETECTOR_LIMIAR_FUSAO` (JSON do `limiar_fusao.py`), as
  duas abas usam os modelos, a regra e o limiar dele; Enviar arquivo usa também a
  taxa comum de 48 kHz, como o ensaio 2. Na tela, o score
  aparece na escala centrada no limiar (`centrado` em `src/limiares.py`): o
  limiar fica em 0,50; a decisão continua no score do modelo.
- `scripts/`: avaliação, robustez, calibração, canal real, demonstração
  (`montar_demo.py`), voz própria (`gerar_sintetico.py`, `avaliar_arquivos.py`;
  o teste real contra clone: `gravar_frases.py`, `limiar_fusao.py`,
  `teste_voz_propria.py`, critério em RESUMO_TCC §10.2.12; clonagem local:
  `preparar_referencia.py` e `clonar_xtts.py`, este no ambiente conda `xtts`).
- `docs/RESUMO_TCC.md`: registro de todos os experimentos e números.
  `docs/TC2_REDACAO.md`: texto do relatório, com a fonte de cada número num
  comentário `<!-- RESUMO_TCC §x -->`.

## Regras

- **Número no texto só com fonte.** Todo número da redação vem do
  `RESUMO_TCC.md` (ou de uma saída que o usuário colou), com a seção citada.
  Não invente nem arredonde a partir da memória.
- **Resultado negativo é relatado** do jeito que saiu, com a limitação em 5.7.
  Combine o critério antes de ver o resultado.
- **Calibração nunca toca no conjunto de avaliação**: limiar no dev, efeito
  medido depois no eval.
- **Fora do Git:** áudio (`*.wav`, `*.flac`, `*.mp3`), `checkpoints/`,
  `outputs/`, bases da ASVspoof e qualquer voz coletada (dado biométrico, LGPD).
  Versione scripts, protocolos, listas do recorte e resultados.
- **Vozes de pessoas:** só com termo de consentimento assinado; código de
  locutor no lugar do nome; clones apagados depois da banca. Clonagem só das
  vozes autorizadas e só para o que o termo diz.
- **Texto do TCC:** não atribuir partes do sistema a integrantes específicos. O
  uso de assistente de IA é declarado na metodologia.
- **Git:** a branch principal é `claude/admiring-albattani-ehJU4`. Não abra pull
  request nem faça push forçado sem pedido. Commits pequenos, mensagem em
  português dizendo o que mudou e por quê.
- Toda mudança de código vem com teste no estilo de `ml/tests/` (sem GPU, sem
  base real: dados sintéticos e falsos de `soundcard`).

## Fatos já medidos (não refaça sem motivo)

- Eval 2019 LA sem silêncio: baseline_v2 18,99%, fusion_v4 20,18%, fusão de
  scores dos dois 13,13%. Validação não prevê avaliação (0,03% → 20,18%).
- Canal real (2021 LA, Opus): baseline_v2 29,43%, fusion_v4 32,62%; no Opus o
  fusion_v4 satura a probabilidade em 1,0.
- Captura com FIR, fusion_v4: 20,16% (limpo 20,25%), sem saturação; o soxr era a
  causa. A cópia `_captura` dele (limiar 0,9984, de um dev com EER 0,13%) deixa
  passar 50% dos sintéticos contra 44% do original: no fusion_v4 vale o limiar
  original (RESUMO_TCC §10.2.8).
- Captura 48 → 16 kHz: soxr 24,70%, FIR 19,41% (limpo 19,02%). Limiar de captura
  do baseline_v2: 0,686 (original 0,654).
- Fusão ao vivo (média) dos dois na captura FIR: EER 14,2%. Limiar escolhido no
  dev com o alarme falso do v2 (0,3473): 10,6% de humanos acima e 16,5% de
  sintéticos passando, contra 10,7% e 28,8% do v2. Limiares acima de 0,5 exigem
  que os dois concordem e deixam passar 42–49% (RESUMO_TCC §10.2.9). Sem
  `--limiar-fusao`, o monitor aplica à fusão o limiar do primeiro checkpoint.
- Fusão dos dois fusion_v4 com degradação (RESUMO_TCC §10.2.12): na média das
  probabilidades o modelo do recorte quase não pesa (limiar dele 0,00017); na
  média dos log-odds, dev da ASVspoof 5 com EER 11,87% (21,48% nas
  probabilidades). Limiar −4,437 (score 0,0117), 10% de alarme falso nos devs.
- Ao vivo: aprimoramentos de áudio do Windows (saída e microfone) precisam estar
  desligados; o VLC altera o áudio; reprodução pelo ar (celular → microfone)
  não separa real de falso. Com um fone com microfone conectado, o driver
  Realtek troca o microfone embutido ("Grupo de microfones") pelo do fone sem
  mudar o nome; os testes de 02/10 (RESUMO_TCC §10.2.10) mediram o do fone.
- Oclusão de 7,6–8 kHz no eval (10 mil): baseline_v2 19,02% → 25,29%,
  fusion_v4 20,25% → 31,12% com todos os humanos em 1,0. A dependência passa
  do critério de 2 pp: a correção entra no treino (RESUMO_TCC §10.2.11).
  Treino completo do baseline_v2 (controle 18,14% limpo, 24,33% no corte):
  LFCC de 0 a 4 kHz 23,34% e 23,34%; descarte aleatório 20,54% e 21,34%.
  Nenhuma passou no critério; o grupo adotou o descarte (23/10/2026).
- fusion_v4 treinado no recorte da ASVspoof 5 com a degradação nova: dev
  20,56%; eval 2019 37,76% (fusion_v4 da 2019: 20,18%), Opus real da 2021
  37,44% (32,62%). Pior em todas as réguas da 2019/2021, mas quase invariante ao
  canal (Opus real +0,67 pp sobre a referência, contra +13,11) (RESUMO_TCC §10.7).
- 2x2 do fusion_v4 (RESUMO_TCC §10.9): a degradação nova na 2019 custa 3,56 pp no
  limpo e ganha 8,57 pp no Opus real (24,05%); a base ASVspoof 5 piora todas as
  réguas, mas erra em ataques diferentes. A fusão pela média dos dois dá 18,16%
  no Opus real e 26,7% no eval da ASVspoof 5 (exploratória, sem critério antes).
- Voz própria com copy-synthesis (Griffin-Lim, WORLD): versões sintéticas abaixo
  do original em 18 de 20; o modelo reage às condições de gravação, não ao
  vocoder. É o diagnóstico que motiva o Incremento 4.
- Teste com a voz própria contra clones do XTTS v2 local (RESUMO_TCC §10.2.12,
  critério combinado antes): reais 10 de 10 nos dois caminhos; clones 0 de 10 no
  arquivo e 2 de 10 pelo microfone. Meta não atingida. O modelo da 2019 inverte
  (voz real do microfone acima do clone limpo); o do recorte separa o arquivo
  (100% dos pares), mas pelo ar nenhum separa. Clones tocaram ~8 dB mais baixo.
- Ensaio 2 (RESUMO_TCC §10.2.13, critério antes): `fusion_lcnn_v4_asv5` sozinho,
  limiar do treino (−8,671 nos log-odds), caminho arquivo com `--taxa-comum
  48000`, frases novas: reais 10 de 10, clones do XTTS 9 de 10, meta atingida.
  Um locutor só (o mesmo do ensaio 1); a ordem é estável, o ponto de corte frágil.
- Ensaio 3 (RESUMO_TCC §10.2.14): mesmo critério, voz nova (L02, de fora do
  grupo), frases do conjunto 3, clones sem o ponto final: reais 10 de 10, clones
  10 de 10. Ensaios 2 e 3 juntos: 20 de 20 e 19 de 20. O clone pelo ar segue sem
  separação. O XTTS corta o fim, trava palavras e lê o ponto como "ponto".
- Microfone embutido na sala: ruído 16–25 dB abaixo da fala na distância normal;
  o teste aceita 15 dB (`--piso-minimo 15`). Tecla e mão no notebook viram
  pancadas graves que a checagem de fala ignora (passa-altas de 150 Hz).
