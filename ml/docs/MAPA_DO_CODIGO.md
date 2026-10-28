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
| 10. Interface | Enviar arquivo, Resultado, Ao vivo, Clonar, Histórico | `web/` |

Os estudos (`scripts/estudo_*.py`) ficam fora da linha: são exploratórios, e
nenhum deles muda um placar.

## O caminho de um áudio no modelo final

1. `src/preprocess/audio.py` `load_audio`: lê e leva a 16 kHz pelo FIR
   (`reamostragem.py`; o `soxr` apagava 7,6–8 kHz).
2. `src/features/xlsr.py` `preparar_wav`: tira o silêncio das pontas e normaliza
   pelo pico; `janelas`: 4 s com passo de 2 s.
3. `src/features/xlsr.py` `por_janela`: as 25 camadas do XLS-R, a média de cada
   uma no tempo; `lfcc_medio`: o LFCC do arquivo.
4. `src/xlsr_final.py` `ModeloFinal.pontuar_wav`: a cabeça pesa as camadas e dá
   o log-odds.
5. `src/xlsr_final.py` `ler_limiar`: o limiar do JSON (`outputs/limiar_xlsr/`),
   um para o arquivo e outro para a frase do ao vivo.
6. Ao vivo, antes do passo 2: `src/capture/frases.py` junta a fala até a pausa.

## Pastas

| pasta | o que tem | no Git? |
|---|---|---|
| `src/` | a biblioteca: tudo que a interface, o monitor e os scripts importam | sim |
| `scripts/` | um comando por tarefa (lista abaixo) | sim |
| `web/` | a interface: rotas (`app.py`), análise (`analise.py`), clonagem (`clonagem.py`), histórico (`banco.py`), telas | sim |
| `tests/` | um arquivo por módulo, com dados sintéticos, sem GPU e sem base real | sim |
| `configs/` | um YAML por modelo treinado (`baseline_v2`, `fusion_v4`, ...) | sim |
| `docs/` | `RESUMO_TCC.md` (todos os números), `PLANO_INCREMENTO_4.md`, `TC2_REDACAO.md`, ética e colinha | sim |
| `data/` | bases e vozes coletadas; só as listas de `data/recortes/` vão para o Git | não |
| `checkpoints/` | os modelos treinados | não |
| `outputs/` | scores, resultados, limiares, gravações ao vivo e clones | não |

## Os scripts, por etapa

**Dados**
- `montar_recorte_asvspoof5.py`, `montar_amostra_brspeech.py`,
  `montar_amostra_commonvoice.py`, `montar_amostra_replaydf.py`: as amostras e
  os recortes, com a lista em `data/recortes/`.
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
  `comparar_sessoes.py`, `comparar_espectro.py`, `simular_caminho.py`,
  `simular_perdas.py`.

**Vozes e clones**
- `gravar_frases.py`: grava as frases, com as checagens.
- `preparar_referencia.py`, `clonar_xtts.py`: a referência e o clone (XTTS ou
  Chatterbox, cada um no seu ambiente conda).
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
| mudar a checagem depois de gravar | `src/capture/checagens.py` |
| registrar um experimento | `docs/RESUMO_TCC.md`, com o critério antes de rodar |

## Depois da pré-banca: as pastas

Combinado com o grupo (28/10/2026): o mapa agora; mover os arquivos só depois
da pré-banca, para os comandos da colinha não mudarem antes da apresentação. O
plano é pôr os scripts em subpastas com os nomes das etapas acima
(`scripts/dados/`, `treino/`, `avaliacao/`, `limiares/`, `captura/`, `vozes/`,
`estudos/`) e, junto:
- trocar os nomes parecidos (`diagnostico_captura.py` e
  `diagnosticar_captura.py`) e os que estão em inglês (`check_*`);
- tirar `simular_perdas.py`, que hoje é só um atalho para o
  `simular_caminho.py`;
- atualizar os imports (`scripts.x` vira `scripts.etapa.x`), os testes, a
  colinha e o CLAUDE.md. O RESUMO_TCC fica como está, com os caminhos da época.
