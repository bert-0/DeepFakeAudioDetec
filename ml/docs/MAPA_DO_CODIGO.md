# Mapa do código

Para quem vai mexer no projeto: a ordem em que as peças se encaixam, onde fica
cada uma e por onde começar. Os números de cada experimento ficam no
`RESUMO_TCC.md`; este mapa diz só onde está o código.

## A linha de raciocínio

Um áudio entra, vira características, o modelo dá um score, o limiar decide.
Cada etapa abaixo usa a anterior.

| etapa | o que faz | onde |
|---|---|---|
| 1. Dados | bases (ASVspoof 2019, 2021, 5; BRSpeech-DF, Common Voice, ReplayDF, salas) e os recortes | `src/data/`, `scripts/montar_*`, `data/recortes/` |
| 2. Áudio | lê em 16 kHz pelo FIR, corta o silêncio, normaliza, degrada no treino | `src/preprocess/` |
| 3. Características | LFCC, log-mel e as camadas do XLS-R | `src/features/` |
| 4. Modelos | CNN, LCNN, fusão e atenção; a cabeça do modelo final | `src/models/`, `src/xlsr_final.py` |
| 5. Treino | os detectores e o modelo final | `train.py`, `scripts/xlsr_congelado.py` |
| 6. Avaliação | EER, t-DCF, por ataque, robustez, réguas | `evaluate.py`, `scripts/per_attack_eval.py`, `scripts/robustness_eval.py` |
| 7. Limiar | escolhido no dev ou calibrado com vozes reais | `src/limiares.py`, `scripts/limiar_xlsr.py` |
| 8. Captura | microfone, som do computador, janelas, frases, checagens | `src/capture/`, `monitor.py` |
| 9. Vozes e clones | gravar, clonar e o placar dos ensaios (Incremento 4) | `scripts/gravar_frases.py`, `scripts/clonar_xtts.py`, `scripts/teste_voz_propria.py` |
| 10. Interface | Enviar arquivo, Resultado, Ao vivo, Clonar, Histórico e a página Conferir | `web/` |

Os estudos (`scripts/estudo_*.py`) ficam fora da linha: são exploratórios, e
nenhum deles muda um placar.

## O caminho de um áudio no modelo final

As duas abas, Enviar arquivo e Ao vivo, seguem o mesmo caminho, frase por frase
(decisão (b), §10.2.45). A cabeça em uso é `checkpoints/xlsr_congelado_itw.pt`
(§10.2.49).

1. **Ler.** `src/preprocess/audio.py` `load_audio` lê o arquivo e leva a 16 kHz
   pelo FIR (`reamostragem.py`; o `soxr` apagava 7,6–8 kHz). Ao vivo, o
   microfone passa pelo mesmo FIR (`src/capture/sources.py`).
2. **Achar as frases.** `src/capture/frases.py` `AnalisadorFrases` junta a fala
   até uma pausa de 0,8 s:
   - a fala é medida sem graves (passa-altas de 150 Hz);
   - ela começa a 6 dB acima do piso de ruído (o percentil 10 dos últimos 30 s)
     e acaba abaixo de 3 dB;
   - menos de 1 s de fala não vira frase, e uma frase passa de no máximo 15 s.

   No arquivo, quem chama é `ModeloFinal.pontuar_por_frases`; ao vivo, o
   `monitor.py --xlsr`.
3. **Preparar cada frase.** `src/features/xlsr.py` `preparar_wav` tira o
   silêncio das pontas e normaliza pelo pico; `janelas` corta 4 s com passo de
   2 s.
4. **Representar.** `por_janela` tira as 25 camadas do XLS-R congelado, com a
   média de cada uma no tempo; `lfcc_medio` tira o LFCC.
5. **Pontuar.** `src/xlsr_final.py` `ModeloFinal.pontuar_wav`: a cabeça pesa as
   camadas, junta o LFCC e dá o log-odds da frase.
6. **Decidir.**
   - No arquivo, o score é a média dos log-odds das frases; ao vivo, cada frase
     tem o seu veredito.
   - O limiar vem de `ler_limiar` (`outputs/limiar_xlsr/limiar.json`): o da
     frase, calibrado com as vozes reais (`scripts/limiar_xlsr.py`).
   - Na tela, o score aparece na escala centrada (`centrado` em
     `src/limiares.py`), com o limiar em 0,50; a decisão é a mesma.

Os modelos antigos (`baseline_v2`, `fusion_v4`) seguem pelo outro caminho:
janelas de 4 s com passo de 2 s no áudio todo, pesadas pela fração de fala
(`src/capture/analyzer.py`).

## Pastas

| pasta | o que tem | no Git? |
|---|---|---|
| `src/` | a biblioteca: tudo que a interface, o monitor e os scripts importam | sim |
| `scripts/` | um comando por tarefa (lista abaixo) | sim |
| `web/` | a interface: rotas (`app.py`), análise (`analise.py`), clonagem (`clonagem.py`), conferência (`conferencia.py`), histórico (`banco.py`), telas | sim |
| `tests/` | um arquivo por módulo, com dados sintéticos, sem GPU e sem base real | sim |
| `configs/` | um YAML por modelo treinado (`baseline_v2`, `fusion_v4`, ...) | sim |
| `docs/` | `RESUMO_TCC.md` (todos os números), `PLANO_INCREMENTO_4.md`, `TC2_REDACAO.md`, ética e colinha | sim |
| `data/` | bases e vozes coletadas; só as listas de `data/recortes/` vão para o Git | não |
| `checkpoints/` | os modelos treinados | não |
| `outputs/` | scores, resultados, limiares, gravações ao vivo e clones | não |

## Os scripts, por etapa

**Dados**
- `montar_recorte_asvspoof5.py`, `montar_amostra_brspeech.py`,
  `montar_amostra_commonvoice.py`, `montar_amostra_replaydf.py`,
  `montar_amostra_inthewild.py`: as amostras e os recortes, com a lista em
  `data/recortes/`. Os dois últimos também dividem a amostra por montagem ou
  por pessoa (`--dividir`).
- `importar_asvspoof2021.py`, `converter_para_wav.py`, `check_data.py`,
  `rotulo.py`: preparo e conferência das bases.

**Treino**
- `xlsr_congelado.py`: o modelo final (treinar e avaliar a cabeça).
- `destilacao.py`, `destilacao_multi.py`, `combinar_detector_aluno.py`: os
  pilotos de destilação (critérios não atingidos, §10.2.37 a §10.2.41).

**Avaliação**
- `per_attack_eval.py`, `robustness_eval.py`, `tdcf.py`, `score_fusion.py`,
  `make_report.py`, `run_pipeline.py`: as tabelas dos Incrementos 1 a 3.
- `checar_saturacao.py`, `check_score_confound.py`, `check_shortcut.py`:
  conferências de atalhos e saturação.
- `canal_real.py`, `fusao_ao_vivo.py`, `bench_latencia.py`: o canal real, a
  fusão ao vivo e a latência.

**Limiar**
- `limiar_xlsr.py`: o do modelo final, com as vozes reais de calibração.
- `limiar_fusao.py`, `calibrar_captura.py`: os dos modelos antigos.

**Captura**
- `testar_microfone.py`, `diagnostico_captura.py` (pacotes repetidos, e o
  `reparar`), `diagnosticar_captura.py` (deformação no tempo),
  `comparar_sessoes.py`, `comparar_espectro.py`, `simular_caminho.py` (o
  `simular_perdas.py` virou `--efeito perdas`, 08/11/2026).

**Vozes e clones**
- `gravar_frases.py`: grava as frases, com as checagens.
- `preparar_referencia.py`, `clonar_xtts.py`: a referência (da leitura do texto
  ou das frases já gravadas) e o clone (XTTS ou Chatterbox, cada um no seu
  ambiente conda).
- `teste_voz_propria.py`: o placar dos ensaios (`--xlsr` para o modelo final).
- `avaliar_arquivos.py`, `gerar_sintetico.py`, `ensaio_canal_igualado.py`,
  `montar_demo.py`.

**Estudos** (exploratórios, §10.2.17 em diante)
- `estudo_faixas.py`, `estudo_relogio.py`, `estudo_linhas.py`,
  `estudo_pente.py`, `estudo_bases.py`, `estudo_acompanha.py`,
  `estudo_perth.py`, `estudo_estalos.py`: as imperfeições dos sintéticos e da
  captura.
- `estudo_ssl.py` (o XLS-R como instrumento e a extração dos `.npz`),
  `estudo_dinamica.py`, `estudo_estatisticas_tempo.py`, `estudo_surpresa.py`,
  `estudo_corte_fala.py`.

## Onde mexer para...

| tarefa | arquivo |
|---|---|
| trocar o limiar do modelo final | rodar `scripts/limiar_xlsr.py` com as vozes novas |
| mudar o que a tela mostra | `web/templates/` e `web/static/estilo.css` |
| mudar uma rota ou o que ela faz | `web/app.py` |
| mudar como um arquivo é analisado | `web/analise.py` e `src/xlsr_final.py` |
| mudar o corte das frases do ao vivo | `src/capture/frases.py` |
| pôr ou tirar uma voz da aba Clonar | `data/vozes/vozes_autorizadas.json`, só com o termo |
| conferir tudo antes de apresentar | a página Conferir, ou `scripts/conferir.py` |
| mudar a checagem depois de gravar | `src/capture/checagens.py` |
| registrar um experimento | `docs/RESUMO_TCC.md`, com o critério antes de rodar |

## Depois da banca: as pastas

Combinado com o grupo (28/10/2026): o mapa agora; mover os arquivos só depois
da apresentação, para os comandos da colinha não mudarem antes dela. Não houve
pré-banca, então a mudança fica para depois da banca. O plano é pôr os scripts
em subpastas com os nomes das etapas acima (`scripts/dados/`, `treino/`,
`avaliacao/`, `limiares/`, `captura/`, `vozes/`, `estudos/`) e, junto:
- trocar os nomes parecidos (`diagnostico_captura.py` e
  `diagnosticar_captura.py`) e os que estão em inglês (`check_*`);
- atualizar os imports (`scripts.x` vira `scripts.etapa.x`), os testes, a
  colinha e o CLAUDE.md. O RESUMO_TCC fica como está, com os caminhos da época.

Já feito, sem mudar comando da colinha (08/11/2026): o `simular_perdas.py` saiu
(era só um atalho para o `simular_caminho.py`), e a `resample_mono` de
`src/capture/sources.py`, sem uso e com o `soxr` que apagava 7,6–8 kHz, também.
