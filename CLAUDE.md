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
  `soxr` apagava 7,6–8 kHz e não deve voltar ao caminho do modelo).
- `src/features/`, `src/models/`: LFCC, log-mel, CNN, LCNN, fusão e atenção.
- `src/capture/`: fontes de áudio (arquivo, loopback, microfone), analisador
  em janelas de 4 s com passo de 2 s, peso pela fração de fala.
- `src/limiares.py`: limiar por tipo de áudio (nativo 16 kHz: original; 44,1/48
  kHz e ao vivo: o de captura, só se calibrado com o FIR e com os mesmos pesos).
- `web/`: FastAPI + Jinja2 + SQLite. Abas Enviar arquivo, Resultado, Ao vivo e
  Histórico. O ao vivo roda o `monitor.py` como subprocesso, parado por arquivo
  (`--parar-com`).
- `scripts/`: avaliação, robustez, calibração, canal real, demonstração
  (`montar_demo.py`), voz própria (`gerar_sintetico.py`, `avaliar_arquivos.py`).
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
  causa. A cópia `_captura` dele (limiar 0,9984, dev com EER 0,13%) ainda não foi
  medida no eval: não usar antes disso (RESUMO_TCC §10.2.8).
- Captura 48 → 16 kHz: soxr 24,70%, FIR 19,41% (limpo 19,02%). Limiar de captura
  do baseline_v2: 0,686 (original 0,654).
- Ao vivo: aprimoramentos de áudio do Windows (saída e microfone) precisam estar
  desligados; o VLC altera o áudio; reprodução pelo ar (celular → microfone)
  não separa real de falso.
- Voz própria com copy-synthesis (Griffin-Lim, WORLD): versões sintéticas abaixo
  do original em 18 de 20; o modelo reage às condições de gravação, não ao
  vocoder. É o diagnóstico que motiva o Incremento 4.
