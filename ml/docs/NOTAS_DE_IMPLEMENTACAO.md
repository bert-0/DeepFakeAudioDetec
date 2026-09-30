# Notas de implementação

Explicações que saíram dos comentários do código ao enxugá-los. Os números e
as decisões continuam valendo; o lugar deles é aqui e no `RESUMO_TCC.md`, não
dentro das funções.

## monitor.py

- Escolha do modelo: degradação simulada (robustness_eval.py, eval completo, 71.237 áudios) favorecia o fusion_v4 — EER v2 / v4: limpo 18,99% / 20,18%; Opus 25 kbps 20,04% / 22,08%; banda estreita 35,95% / 25,53%; ruído 5 dB SNR 42,41% / 27,42%; degradação máx. +23,42 pp / +7,24 pp.
- No canal real (ASVspoof 2021 LA, Opus sobre VoIP, 10 mil áudios) o v2 venceu: 29,43% vs 32,62%. Fusão pela média (regra do monitor com dois `--checkpoint`): 29,41%, empata com o v2 sozinho.
- Por que a média não ajuda: o fusion_v4 dá 1,0 para 97% dos bonafide no Opus, então a média vira o v2 deslocado. Fusão por postos ganha 4,5 pp, mas precisa do conjunto inteiro (não roda janela a janela).
- O que degrada: Opus custa pouco (+1,2 a +2,8 pp, mesmo a 15 kbps, abaixo do Teams); perda da banda alta +5,4 pp; ruído acústico +7,2 pp a 5 dB SNR. Banda larga + ambiente silencioso é viável; banda estreita, não.
- Limiar fora do domínio é imprevisível: sob Opus 15 kbps o recall do v2 sobe (0,72 -> 0,86) e o do v4 cai (0,55 -> 0,39). Daí mostrar score, não veredito; para ponto de operação confiável, recalibrar no canal de destino com `--gravar` tocando áudios de rótulo conhecido.
- Limiar recalibrado: captura ao vivo e arquivos acima de 16 kHz perdem a faixa 7,6–8 kHz na conversão de taxa.
- Barra do score: a marca `|` do limiar existe porque sem ela 0,048 e 0,71 parecem só "duas barras", sem indicar o lado da decisão.
- Rótulo verdadeiro com arquivo do dataset: é o que torna a execução uma verificação; sem rótulo, o score só mostra que o sistema opera.
- Janelas sobrepostas inflam a confiança se contadas como independentes.
- Loopback em silêncio total no Windows é indistinguível de falha do modelo sem a dica impressa ao usuário.

## web/app.py

- Ao vivo: a página consulta o JSON do monitor a cada 2 s.

## src/capture/analyzer.py

- Fusão pela média (eval completo): `baseline_v2` 18,99%, `fusion_v4` 20,18%, média 14,03% de EER. Regra `rank` dá 13,13%, mas exige todos os scores para atribuir postos; inviável ao vivo (uma janela por vez). 0,90 pp é o custo do tempo real.
- Janela = `audio.duration` (fixada pelo `fix_length` do treino); não é parâmetro livre, mudar exige retreinar. Trecho sintético mais curto que a janela sempre aparece misturado com áudio real; hop menor não resolve.
- `SILENCIO_RMS = 1e-3`: abaixo disso, silêncio/ruído; incluir encheria a média de scores arbitrários.
- Canal é binário (exclui banda estreita, não atenua): degradação medida +5,35 pp de EER no `fusion_v4` em banda estreita; peso contínuo exigiria curva EER×qualidade não medida.
- Peso da janela = fração de fala: `preprocess_waveform` remove silêncio e completa por repetição (10% de fala vira 0,48 s repetido 8x, fora do domínio de treino). Sem constante de ajuste. Pesos iguais recaem na média simples (comportamento anterior).
- Canal `indeterminado` conta como confiável: na dúvida, continua medindo.
- Janelas sobrepostas não são independentes: janela 4 s, passo 2 s, 5 janelas cobrem 12 s = 3 independentes. Fórmula: `((n-1)*h + w) / w`.
- Média móvel existe porque janela isolada (4 s, com codec, fora do domínio) é frágil; decisão vem da tendência.
- Features por modelo: `baseline_v2` usa `n_filter: 20` e só LFCC; `fusion_v4` usa 70 e também espectrograma.
- `calibracao` no checkpoint só existe após `scripts/calibrar_captura.py`.
- Canal é da sessão: janela sem agudos pode ser o locutor (vogal); só a acumulação distingue do canal.
- `fracao_fala` divide pelo tamanho da janela, não do array: senão um trecho final de 1 s repetido 4x teria peso 1,00.
- Sem `finalizar()`, áudio mais curto que a janela não gera leitura; a maioria dos enunciados do ASVspoof é mais curta que 4 s.

## src/capture/stream.py

- Hop menor que a janela evita que trecho sintético curto caia na fronteira e seja diluído em duas janelas.
- `finalizar` só emite se houver áudio além da última janela: com passo = metade da janela, o fim de arquivo longo já costuma estar coberto, e repetir inflaria a contagem.
- Sem `finalizar`, o `while` de `alimentar` nunca dispara para arquivo curto e o resto é descartado.

## src/capture/__init__.py

- Loopback em vez de integração com o Teams: esta exigiria bot de mídia em C#/.NET, hospedagem em Azure e consentimento de admin do tenant. Loopback funciona com Teams, Meet, Zoom etc. sem publicar app.

# Notas removidas dos comentários (lote 3)

## src/capture/alinhamento.py
- Por que não um bipe como marcador: a supressão de ruído do Teams remove o que não é fala; um seno puro é o caso canônico e some no caminho.
- Por que o envelope de energia: codec, supressão de ruído e AGC mudam espectro e amplitude, mas não reordenam o áudio no tempo. A normalização (média zero, norma 1) tira o efeito do AGC.
- CORRELACAO_MINIMA = 0,5, medida em tests/test_alinhamento.py (pior caso): canal limpo 0,798; opus + banda estreita 8 kHz 0,798; ruído puro (microfone errado) 0,118. Folga de 1,6x para baixo e 4,2x para cima. O codec praticamente não mexe no envelope.
- Descartar abaixo do corte em vez de recortar assim mesmo: recorte com rótulo errado gera um EER que parece resultado mas não é.
- BUSCA_FINA_S: envelope a 100 Hz deixava erro de até 5 ms (medido: 79 amostras com gravação idêntica à referência). É meio passo do STFT: todos os quadros das features mudam e os scores variaram até 0,17 para o mesmo áudio (sessão "limpo" vs eval 2019).
- CORRELACAO_FINA_MINIMA: abaixo dela (ex.: supressão agressiva) mantém-se a posição do envelope, que continua válida.
- Atraso global pode ser negativo: o monitor leva alguns segundos carregando o modelo antes de capturar; se a reprodução começa nesse intervalo, o início da playlist fica fora da gravação. Medido no primeiro controle real: -7,0 s. Com busca só positiva o alinhamento se perdia inteiro (3 de 40 trechos); liberada, perdem-se só os trechos tocados antes da gravação.
- correlacao_maxima nunca devolve atraso negativo porque, no ajuste local, a janela da gravação sempre começa antes do trecho.
- Latência global típica da chamada: centenas de ms; o ajuste por trecho absorve a deriva de relógio entre as duas placas de som.
- Silêncio entre áudios (gap_s) também evita que a supressão de ruído trate a emenda entre dois áudios como um único fluxo contínuo.

## src/capture/qualidade.py
- Motivação (robustness_eval.py, eval completo, fusion_v4 EER): limpo 20,18%; opus 25 kbps 22,08% (+1,90 pp); banda estreita 8 kHz 25,53% (+5,35 pp). Perder a banda alta custa ~5x mais que o codec; metade do banco de filtros do LFCC fica acima de 4 kHz.
- Por que p90 por quadro e não média da janela: fala alterna vogais (sem alta freq.) e fricativas (muita); a média dilui as fricativas. Medido (média / p90): ruído rosa 11,21% / 14,84%; fala 5,40% / 81,39%; vogal sustentada 0% / 0%; fala em banda estreita 0% / 0%.
- Vogal sustentada é indistinguível de banda estreita numa janela isolada; um portão que reprovasse por ausência rejeitaria fala normal. Daí o veredito de sessão com três estados (banda larga definitiva, indeterminado, banda estreita provável).
- JANELAS_PARA_CONCLUIR: em fala corrida fricativas aparecem o tempo todo; várias janelas seguidas sem elas indicam o canal, não o locutor.
- Quadros sem energia ficam fora do percentil: a fração seria 0/0.
- Veredito de banda larga não volta atrás: o canal da chamada não muda a cada 2 s. `indeterminado` conta como avaliável: na dúvida o sistema segue mostrando score em vez de se calar por uma sequência de vogais.

## src/capture/sources.py
- FileSource existe para que janela, features, modelo e agregação sejam testáveis em qualquer máquina (inclusive CI Linux); também torna um resultado ao vivo reproduzível ao reprocessar o áudio gravado.
- BUFFER_CAPTURA_S = 1 s: sem ele o soundcard usa um período do dispositivo (~10 ms); se a thread de captura para mais que isso (o GIL a segura enquanto a janela anterior vira features), o áudio é descartado como descontinuidade. Primeiro controle real: 58 perdas em 3 min (~1 a cada 3 s), quase todo áudio da playlist levava um clique. 1 s não aumenta a latência: record() devolve assim que os quadros pedidos chegam.
- Captura em thread própria: antes dividia o laço com o processamento; enquanto a janela passava pelo modelo ninguém lia o dispositivo, o buffer transbordava ("data discontinuity in recording", dezenas de vezes em 3 min). Cada perda é um salto na forma de onda, um clique de banda larga que o modelo não viu no treino.
- Reamostragem em streaming (soxr.ResampleStream): reamostrar cada bloco de 100 ms isoladamente cria uma borda a cada bloco.
- Loopback WASAPI pega o mix final (todos os participantes + outros sons); separar por participante só pelo bot de mídia do Teams, que o projeto evita. Consequência para a interface: o veredito é sobre o trecho de áudio, não sobre uma pessoa.
- COM no Windows: soundcard inicializa o COM na thread que o importa; thread nova costuma herdar o apartamento multithread, mas não é garantido. Chamar CoInitializeEx de novo é inofensivo (devolve S_FALSE).
- Reamostragem 48 -> 16 kHz faz parte do desvio de domínio (modelo treinado em áudio nativo de 16 kHz); fica isolada para poder ser trocada e medida.


# Notas preservadas (agente 4)

## train.py
- Pesos de classe "auto" (~8,8x no bonafide no LA) deslocam fortemente o ponto de decisão e aumentam falsos positivos; "sqrt" (~3x) é o meio-termo.
- EpochTimer: separar espera por dados de cálculo decide a otimização (dados -> mais num_workers/cache; cálculo -> modelo menor, batch maior). Sem `loss.item()` sincronizando, o tempo de GPU apareceria todo como "dados".
- Barra do tqdm fora de terminal: cada refresh vira uma linha (~794 por época, ~40 mil em 50 épocas), por isso mininterval=30 s.
- persistent_workers no Windows (spawn): recriar cada worker custa ~2,4 s (reimportação de torch+librosa); 4 workers x 50 épocas ~8 min. Só é seguro porque `set_epoch` grava num tensor em memória compartilhada; com int comum os workers repetiriam o recorte/aumentação da época em que nasceram.
- Workers do dev: antes herdavam num_workers/persistent_workers do treino; 8 processos de ~512 MB vivos ao mesmo tempo, somando ~8,5 GB num notebook de 16 GB e levando à paginação. O dev é 100% cache (memmap) exceto na 1ª época; padrão 2 workers. Recriar os workers do dev custa ~2,4 s por época contra ~2 GB mantidos ociosos.
- O dev_loader recebe worker_init_fn para limitar as threads BLAS por worker.
- pin_memory: cópia por DMA sobrepõe ao cálculo com non_blocking=True; em CPU é custo puro. `pin` não deve ser redefinido dentro do loop, senão anula `pin_memory: false`.
- Falha de RAM no Windows: sem MemoryError, a máquina pagina, congela e exige desligar no botão; não há retomada por checkpoint, então o treino inteiro se perde (motivo do aviso de memória e das gravações atômicas).
- Se nenhum batch sobrevive, `running_loss / max(seen, 1)` daria 0.0000 e pesos não treinados poderiam ser salvos como melhor checkpoint.

## evaluate.py
- Os scores em `.npz` evitam repetir a inferência em per_attack_eval.py e score_fusion.py (71.237 áudios por passada no eval do LA).

## infer.py
- Arquivo nativo de 16 kHz usa o limiar calibrado no dev; gravado acima disso perde 7,6-8 kHz na conversão e usa o recalibrado, se existir.

## src/config.py
- `config_derivado`: copiar o config e trocar só os caminhos do eval sobrescreveria `outputs/<nome>_eval_metrics.json` e o `_eval_scores.npz` reaproveitados por per_attack_eval.py/score_fusion.py, e apagaria o cache de features do eval (~11 GB do eval de 2019) por uma avaliação de 10 mil áudios.
- `output_name`: `checkpoints/` e `outputs/` estão no .gitignore, então um smoke sem sufixo sobrescreveria resultados de um treino real de forma irrecuperável.
- `seed_worker`: com 4 workers em 4 núcleos seriam 16 threads BLAS/OpenMP disputando 4 núcleos; as matrizes (banco de filtros x STFT) são pequenas demais para a paralelização interna compensar. Medido: 14,1 s -> 4,3 s para 512 áudios (36 -> 120 amostras/s), sem alteração numérica.
- `_limit_worker_threads`: variáveis de ambiente de threads só valem se definidas antes do import do numpy.
- `RAM_POR_WORKER_GB`: no Windows o spawn reimporta torch e librosa em cada worker; no Linux (fork) as páginas são compartilhadas com o pai.

## src/metrics.py
- Threshold calibrado no dev evita o corte fixo de 0,5, enviesado com classes desbalanceadas ou perda com pesos de classe.
- O threshold continua em probabilidade (unidade do checkpoint e do monitor); só o EER usa log-odds, pois depende só da ordem.

## src/scores.py
- Motivação: run_pipeline.py rodava evaluate.py e per_attack_eval.py sobre o mesmo modelo e partição, cada um com a passada inteira (71.237 áudios no eval do LA, duas vezes por experimento); score_fusion.py repetia uma passada por modelo.
- O `.txt` estilo ASVspoof continua sendo o artefato auditável; o `.npz` guarda precisão total porque o `.txt` arredonda para 6 casas e criaria empates no EER.
- Se bonafide e spoof empatam em P=1,0 (margem > ~17 em float32), o EER passa a medir o arredondamento.
- `load_scores` devolve o motivo em vez de imprimir porque quem chama decide se é informação útil.
- Fingerprint pelos pesos: um retreino que preserve o mtime não pode passar batido.

## src/limiares.py
- `librosa.load` reamostra com o mesmo tipo de filtro da captura ao vivo (loopback a 48 kHz); medido: 7,8-8 kHz ficam ~35 dB abaixo do resto. O baseline_v2 depende dessa faixa, e sem ela todos os scores sobem.
- Limiar de captura: recalibrado no dev passado por 16 -> 48 -> 16 kHz.
- Medido no eval com captura (RESUMO 10.2.1): no limiar original 71% dos humanos passam por sintéticos; no recalibrado, 17%. Em áudio nativo de 16 kHz, o recalibrado deixa passar 69% dos sintéticos.
- Limite: um arquivo salvo a 16 kHz que já tinha sido convertido de 48 kHz também perdeu o topo da banda e recebe o limiar original.
- Um `_captura.pt` de um treino anterior teria o limiar de outro modelo (por isso a checagem de pesos).

## src/data/asvspoof2021.py

- Formato 2019: `LA_0079 LA_E_1234567 - A07 spoof`; 2021: `LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval` (locutor arquivo codec canal ataque chave trim fase).
- O parser de 2019 (`parse_protocol_with_systems`, ataque em parts[3], chave em parts[4]) aplicado ao arquivo de 2021 aceita os bonafide com o canal no lugar do ataque e descarta todos os spoof: protocolo de uma classe só que parece válido.
- No arquivo real o bonafide traz `bonafide` também na coluna de ataque (`... alaw ita_tx bonafide bonafide notrim eval`).
- A primeira versão exigia uma única ocorrência do token e descartava todo bonafide: `--listar` saiu com 163.114 trials e zero bonafide. Os testes usavam o formato suposto (`-` no ataque) e passavam. Daí o `relatorio` de linhas ignoradas.
- Aceitar o protocolo de 2019 (5 campos) geraria condições inventadas ("-/A07") que pareceriam canais reais.
- Levanta em vez de devolver lista vazia: protocolo vazio viraria um EER calculado sobre nada.
- O metadado traz codec e canal por áudio e os ataques são os mesmos A07–A19 do eval 2019: permite comparar o mesmo ataque limpo vs. transmitido, inclusive contra a simulação de canal (`scripts/robustness_eval.py`, Opus).
- Fases: `eval`, `progress`, `hidden`. Só `eval` é o oficial (14.816 bonafide, 133.360 spoof); `hidden` não deve entrar na média.
- Subamostragem: eval 2021 tem 181.566 trials; no regime do projeto (EER ~20%, ~10% bonafide) 10.000 por condição dão IC 95% de ±1,28 pp (RESUMO_TCC, 12.2). Estratificar evita sub-representar A10 e A12, que dominam o erro.

## src/data/cache.py

- Motivação: antes cada áudio era um `.pt` — ~96 mil arquivos (24.844 dev + 71.237 eval) reabertos a cada época/avaliação.
- Medido (Linux, page cache quente, features do baseline_v4): `torch.load` de um `.pt` 0,32 ms vs. linha de memmap 0,01 ms (30x). No Windows tende a ser maior (NTFS + antivírus por abertura); 96 mil arquivos numa pasta também pesam em backup/cópia/Explorer.
- `meta.json` guarda o hash da lista de ids porque as linhas são indexadas pela posição no protocolo: protocolo diferente (partição, ordem, linhas) reconstrói o cache em vez de devolver features do áudio errado.
- Concorrência: workers do DataLoader gravam ao mesmo tempo, cada um só a sua linha; `np.memmap` usa mapeamento compartilhado, então bytes diferentes da mesma página vão para a mesma página física, sem o risco de sobrescrita de I/O em buffer. `filled` só é marcado após a linha inteira.
- Shapes constantes porque `fix_length` padroniza a duração antes da extração — é o que permite uma matriz única.
- `meta.json` escrito por último: criação interrompida (disco cheio, Ctrl+C) leva à reconstrução, não à leitura de matriz pela metade.

## src/data/dataset.py

- Época em memória compartilhada: com `persistent_workers=True` os workers guardam cópia do dataset; um `int` deixaria recorte e aumentação congelados na época em que o worker nasceu. Tensor compartilhado funciona com `fork` (Linux) e `spawn` (Windows).
- RNG por amostra derivado de (semente, época, índice): um `rng` guardado no objeto daria a mesma sequência em todos os workers e se repetiria a cada época quando os workers são recriados.
- Cache indexado pelo fingerprint das features, não pelo experimento: v3a, v3 e v4 (n_filter=70) compartilham; cada cópia custa ~11 GB.
- `_open_cache`: a primeira extração acontece no processo principal para descobrir o tamanho da linha; o resultado já é gravado na posição 0.
- `buscar_rotulo` ignora protocolo ausente em silêncio; diagnóstico fica com `scripts/rotulo.py`.
- `protocol_ids_and_systems` permite conferir scores salvos sem construir dataset nem cache.
- `_NAO_AFETAM_CACHE` (`augment`, `random_crop`): quando ativas o cache é desligado e dev/eval nunca as recebem; incluí-las só duplicaria pastas. A lista de exclusões é curta de propósito: errar para o lado conservador custa espaço, o outro corromperia o experimento.

## src/features/extractor.py

- Só os tipos ativos entram no fingerprint, para não invalidar o cache ao mudar uma feature fora de uso.
- Espectro compartilhado entre ramos com a mesma janela: em `fusion_v4.yaml` corta 29% da extração.

## src/features/lfcc.py / spectrogram.py / stft.py

- Banco linear era reconstruído por áudio: ~0,85 ms por amostra, o mesmo custo de aplicá-lo; `lru_cache` constrói uma vez por processo. Matriz somente-leitura porque todos recebem o mesmo objeto.
- Banco mel: `librosa.feature.melspectrogram` reconstrói a cada chamada (o `@cache` do librosa é `Memory(location=None)`, no-op sem `LIBROSA_CACHE_DIR`); custava ~0,85 ms por áudio.
- STFT compartilhado: fusion_v4 usa `n_fft: 512`, `win_length: 400`, `hop_length: 160` nos dois ramos; eram 2,09 ms de 7,27 ms por áudio (29%). Como v3/v4 ligam `random_crop` e `augment`, o treino não tem cache e o custo era pago nas 50 épocas.
- LFCC: filtros lineares preservam melhor a alta frequência associada a artefatos de síntese (vs. MFCC, perceptual).
- Deltas: falas sintéticas frequentemente falham em reproduzir a dinâmica temporal (TC1 §3.3).

# Notas do agente 6 (src/models, src/preprocess)

## src/models/blocks.py
- StatsPool fora do float32: sob AMP a variância (soma de quadrados) estoura o fp16, vira inf -> NaN, o NaN entra nas estatísticas do BatchNorm e o modelo passa a produzir NaN para sempre em modo eval.
- FreqStatsPool/AttentiveFreqStatsPool: a média na frequência do StatsPool descarta onde no espectro cada padrão ocorreu; como os artefatos de síntese são específicos de faixa, isso joga fora o que distingue as classes. Sem o AttentiveFreqStatsPool, o Incremento 3 com `freq_stats` descartaria o eixo espectral que a fusão preserva e a comparação entre incrementos deixaria de isolar a atenção.

## src/models/baseline_cnn.py
- O nn.Flatten() inicial do classificador existe para manter os índices das camadas (chaves do state_dict) compatíveis com checkpoints treinados antes do pooling configurável.

## src/preprocess/audio.py (load_audio)
- Um único .flac corrompido entre os 121.461 da base derruba um treino de horas. Quando o soundfile falha, o `librosa.load` tenta o `audioread`; sem ffmpeg termina em `NoBackendError` com mensagem vazia, descartando o erro real do libsndfile (`flac decoder lost sync`, `Internal psf_fseek() failed`). Por isso o caminho vai na mensagem e a exceção original fica em `__cause__`.
- No Linux o libsndfile recusa arquivo truncado; no Windows o `audioread` devolve áudio parcial sem erro. Medido: os testes de truncagem passavam no Linux e falhavam no Windows. Daí a comparação com a duração do cabeçalho (que sobrevive à truncagem: um FLAC cortado continua anunciando a duração original).
- fix_length: o random crop (com `rng`) só no treino faz cada época ver um trecho diferente do áudio (mais diversidade, menos overfitting); sem `rng` o recorte é determinístico para avaliação reprodutível.

## src/preprocess/augment.py
- Augmenter: guardar o `rng` no objeto não funciona com `DataLoader(num_workers>0)`: cada worker recebe uma cópia com o mesmo estado e, como os workers são recriados a cada época, a sequência se repetiria época após época. O `rng` vem derivado de (semente, época, índice).
- Perturbation: a versão anterior devolvia `lambda`, que o pickle não serializa (no Windows o `spawn` serializa o dataset sempre). A robustez ficava presa a `num_workers=0`: um processo extraindo features de 71.237 áudios, seis vezes (uma por condição), sem cache possível.
- Perturbation: antes um gerador único era compartilhado entre chamadas, e o ruído de cada áudio dependia da ordem de processamento; com o `rng` por amostra fica reprodutível com 0, 2 ou 8 workers.
- Com a normalização por instância das features (média 0, desvio 1), a aumentação de ganho tem efeito pequeno; o ganho real é treinar para ser invariante a ela.

## src/preprocess/channel.py
- Contexto: o detector foi treinado no ASVspoof (16 kHz limpo, sem compressão); em Teams/Meet/Zoom o sinal passa por Opus a dezenas de kbps e, em rede ruim, por banda estreita. Com `n_filter: 70`, metade do banco de filtros do LFCC olha acima de 4 kHz.
- Atenuação medida do Opus a ~26 kbps em ruído rosa: 0-1 kHz -0,9 dB; 1-2 kHz -1,8 dB; 2-4 kHz -2,6 dB; 4-6 kHz -3,7 dB; 6-8 kHz -4,4 dB. O codec atenua progressivamente em vez de cortar, mas como o LFCC é DCT dos logs das energias, alguns dB nas bandas altas já deslocam os coeficientes.
- opus_roundtrip: `compression_level` 0 a 1 medido em ruído branco de 4 s dá ~260 a ~7 kbps; por ser VBR, o script de robustez mede e reporta a taxa obtida.
- ida_e_volta_captura: filtro antialiasing de conversão para 16 kHz, medido em ruído branco: -30 dB em 7,7-7,9 kHz e -103 dB acima de 7,9 kHz. No teste ao vivo, o controle sem efeitos do driver bateu áudio a áudio com esta simulação (Seção 10.2.1). O mesmo `soxr` é usado em `canal_real.py tocar`.
- _ajustar_comprimento: a perturbação vem depois do `preprocess_waveform`; um codec que mudasse alguns quadros alteraria o shape das features e derrubaria a inferência no meio de uma avaliação de horas.
- ChannelChain: o caso relevante é banda estreita e codec juntos (o Teams cai para banda estreita e continua comprimindo); aplicar só um subestima a degradação.

# Notas preservadas (lote 7)

## scripts/canal_real.py
- A simulação de canal do `robustness_eval.py` (Opus + limitação de banda) é um limite inferior da degradação: não inclui supressão de ruído, cancelamento de eco nem AGC, que não são simuláveis de forma honesta. A camada 2 (chamada real) mede isso.
- `sortear`: sorteio ao acaso seguiria a proporção do eval (4.914 por ataque contra 7.355 bonafide) e playlists pequenas ficariam sem A10/A12, os ataques que dominam o erro.
- `CONFIG_PADRAO`: modelo recomendado para chamada pelo resultado no canal real do ASVspoof 2021 LA (Seção 5.3 do resumo); checkpoint de outro modelo faz o evaluate falhar ao carregar os pesos.
- `AVISO_MINUTOS`: chamada longa acumula deriva de relógio e aumenta a chance de queda; 80 áudios de ~4 s dão ~7 min.
- `ESPERA_CHROME_S`: o Chrome começa a tocar o arquivo quando a página abre o microfone (na prévia do Meet, antes de entrar na reunião).
- Medido no Chromium: com `%noloop`, ao fim do arquivo o microfone falso repete o último bloco indefinidamente; sem `%noloop` o arquivo recomeça.
- `tocar_audio`: tocar a 16 kHz obrigaria o Windows a converter, e uma conversão errada muda a duração e o tom da gravação inteira (ver scripts/diagnosticar_captura.py). O `soundcard` substitui o VLC.
- `--use-fake-device-for-media-stream` e `--use-file-for-fake-audio-capture` são opções de teste do WebRTC; o alinhamento acha o atraso global sozinho, então não precisa saber da espera inicial.

## scripts/calibrar_captura.py
- Captura 48 kHz -> 16 kHz apaga a faixa de 7,6-8 kHz. Medido (`robustness_eval.py --so clean captura_48k`, 10.002 áudios): EER do `baseline_v2` vai de 19,02% a 24,70%, mas com o limiar do áudio limpo os humanos acima do limiar vão de 10% a 71%. Como a ordenação sobrevive, basta mover o limiar.
- Calibração no dev (A01-A06) para não vazar o eval; efeito medido depois no eval (A07-A19). O original não é tocado; o monitor acha a cópia sozinho.

## scripts/checar_saturacao.py
- No eval de 2019 chegam a ~26 mil spoof em 1.0 num mesmo modelo, nenhum bonafide.
- Sintoma no ASVspoof 2021 LA com Opus: quatro ataques com EER idêntico (48,65%) apesar de números de amostras diferentes — o EER passou a depender só dos bonafide. Correção: `evaluate.py` salva log-odds e calcula o EER sobre eles.

## scripts/check_data.py
- Medido: FLAC com 10% dos bytes reporta "frames=64000, dur=4.00s" no `sf.info()` e passa como OK, mas `sf.read()` falha com `flac decoder lost sync`. Por isso `--deep` decodifica tudo.

## scripts/check_score_confound.py
- Se spoof é mais alto que bonafide, score e energia correlacionam globalmente mesmo que o modelo ignore o nível; só a correlação dentro de cada classe distingue. Spearman: não assume linearidade e é imune a transformações monotônicas do score.
- Piso de ruído depende de n: com 300 pontos por classe |rho| de 0,11 é ruído; com 3.000, é sinal.

## scripts/check_shortcut.py
- Literatura do ASVspoof: a duração do silêncio no início/fim difere entre as classes, e modelos aprendem a contar silêncio.
- Permutação com 500 embaralhamentos; LA tem ~9 spoof por bonafide, então o acaso não dá 50%. EER abaixo do p5 do nulo = sinal real, mas um atalho de ~44% não explica um modelo de ~19%.

## scripts/comparar_sessoes.py
- Com playlist de 40 áudios, um bonafide a mais ou a menos mexe vários pontos de EER.

## scripts/converter_para_wav.py
- Medido nos .flac do ASVspoof 2021: libsndfile falhou em 38 de 50 arquivos ("unknown error in flac decoder"), mesmo na 1.2.2 e com cabeçalho OK; o librosa cai no `audioread`, que no Windows abre um FFmpeg por arquivo.
- Áudios são 16 bits na origem: conversão para PCM16 é exata. Escrita via temporário + rename evita arquivo pela metade que pareça pronto.

## scripts/bench_latencia.py
- `infer.py` usa `fix_length`: um envio de 60 s teria só os primeiros 4 s analisados (6,7% do arquivo).

# Notas preservadas (agente 8)

## scripts/diagnosticar_captura.py
- Três assinaturas de deformação: deslocamento constante = só atraso (alinhamento deveria funcionar); deslocamento crescendo em linha reta = taxa de amostragem errada no caminho (ex.: 44,1 kHz tratado como 48 kHz), a inclinação dá o fator; degraus = amostras perdidas/inseridas (descontinuidades da captura, engasgos da reprodução).
- Decisão pelos pedaços e não pela correlação global: no primeiro controle real a correlação global foi 0,09 com 54 de 54 pedaços bem localizados (gravação começou depois da reprodução, a referência inteira não cabia).
- Mediana de 3 nos deslocamentos: um pedaço que casa no lugar errado (fala parecida em outro ponto da playlist) é ponto fora isolado.

## scripts/gerar_sintetico.py
- Controle pareado: mesmo locutor, microfone, sala e texto; se o monitor separa os dois, reage à síntese e não ao canal (problema do audiobook, RESUMO 10.2.3). Métodos são copy-synthesis.
- `griffinlim` = vocoder do ataque A11 (Tacotron2 + Griffin-Lim), um dos que o baseline_v2 mais acerta no eval; só usa librosa.
- `world` = vocoder dos ataques A02, A03, A05 (treino) e A07 (eval). O pyworld importa `pkg_resources`, removido no setuptools novo, por isso "setuptools<81".
- Saída a 48 kHz: o microfone chega ao modelo sem a faixa de 7,6-8 kHz; salvar a 48 kHz faz o monitor escolher o limiar recalibrado também com `--arquivo` (src/limiares.py).

## scripts/importar_asvspoof2021.py
- O ASVspoof 2021 LA transmite os mesmos ataques A07–A19 do eval de 2019 por redes reais (VoIP e PSTN) com codecs reais, com condição de referência sem codec e sem transmissão: mesmo ataque, canal real (a simulação da Seção 5 só aproxima isso).
- Layout: ml/data/ASVspoof2021_LA/ (no .gitignore); áudio em <Zenodo 4837263>/flac/*.flac sem rótulo; rótulos em keys/LA/CM/trial_metadata.txt (LA-keys-full.tar.gz de www.asvspoof.org/asvspoof2021/). As chaves não estão no repositório asvspoof-challenge/2021 do GitHub (só a descrição do formato).
- Por que gerar o config: artefatos do evaluate.py levam `experiment.name` + partição; copiar fusion_v4.yaml trocando só caminhos do eval sobrescreveria métricas, scores reaproveitados e cache de features de 2019.
- audios_faltando: sem a checagem, caminho errado/extração incompleta aparece só como FileNotFoundError num worker do DataLoader; conferir 10 mil caminhos custa menos de 1 s.
- falhas_de_leitura: `sf.info` (só cabeçalho) passava nos arquivos reais do 2021 (FLAC, 16 kHz, 16 bits); a falha documentada no python-soundfile ("unknown error in flac decoder") é cabeçalho bom com decodificação quebrada; arquivo truncado idem. Quando o libsndfile falha, o librosa cai no audioread, que no Windows abre um FFmpeg por arquivo; o aviso "PySoundFile failed" não diz o motivo e aparece uma vez por processo (com 4 workers não dá para saber quantos falharam).
- Um config por modelo: o protocolo é o mesmo para os dois modelos, mas sem o nome do experimento no config o segundo sobrescreveria o do primeiro.

## scripts/make_report.py
- Motivação: copiar número por número dos JSONs para o texto a cada rodada é onde entra erro (um EER desatualizado numa tabela não dá sinal). Markdown para conferir, LaTeX para colar, CSV para planilha.
- Tabela por ataque: EER global de 20% pode ser 20% em todos ou 0% em doze e 90% em um; lado a lado revela complementaridade entre configurações — foi o que motivou a fusão de scores.

## scripts/montar_demo.py
- Silêncio de 4 s: com menos, uma janela pegava fim de um áudio e começo do outro e o score misturava os dois (aconteceu no controle do teste ao vivo); o silêncio das pontas é removido pelo `trim`.
- Áudios padrão escolhidos entre os acertos, com score medido pelo caminho real de captura (sessão "controle sem aprimoramentos"/controle_sem_efeitos, RESUMO 10.2.1). Na apresentação, dizer que é ilustração e mostrar em seguida o EER e a tabela por ataque.

## scripts/per_attack_eval.py
- Referência: TC1 §5.4 (mantida no título). EER global de 20% pode ser 20% em todos os ataques ou 0% em doze e 90% em um.

# Notas preservadas (lote 9)

## scripts/robustness_eval.py
- Condições de canal respondem a outra pergunta que as acústicas: "o modelo sobrevive ao meio de transmissão?" — é o que decide se o monitor ao vivo (monitor.py) tem base.
- Calibração do compression_level do Opus, medida em ruído rosa de 4 s: 0.90 -> 30 kbps, 0.92 -> 25, 0.94 -> 19, 0.96 -> 15, 1.00 -> 6. Varrer de 130 a 6 kbps testaria sobretudo faixas que uma chamada nunca usa.
- `captura_48k` é medido aqui com amostra que dá IC (o teste ao vivo não dava).
- Com `num_workers=0`, a avaliação de robustez era um único processo extraindo 71.237 áudios seis vezes; por isso as perturbações foram tornadas serializáveis.
- Amostra estratificada: sem estratificar, um ataque raro poderia ficar de fora e o EER da amostra deixaria de ser comparável ao do conjunto inteiro.
- Se o EER pela probabilidade bate com o do log-odds, os números antigos (calculados pela probabilidade) continuam valendo.

## scripts/rotulo.py
- Motivação: procurar o ID "na mão" (findstr) falha em silêncio — não distingue protocolo errado, pasta errada, ID de outra partição ou formato diferente.
- Sem trocar a barra invertida, o `stem` de um caminho do Windows colado num shell Linux devolveria a linha inteira; o caminho quase sempre vem do PowerShell.
- Spoof espalhados entre ataques: pegar os primeiros do protocolo daria todos do mesmo algoritmo e a figura mostraria um só caso, não o intervalo de dificuldade.

## scripts/run_pipeline.py
- PYTHONUNBUFFERED: com stdout num pipe, o filho usa buffer de bloco (~8 KB) e nenhum print de train.py dá flush; o `bufsize=1` do Popen configura o pai, não o filho. Medido: três linhas espaçadas de 0,5 s chegaram juntas em t=1,51 s.
- PYTHONIOENCODING: o `encoding` do Popen só diz como o pai decodifica; o filho codifica pelo locale (cp1252 no Windows). O pai lê cp1252 como UTF-8, acentos viram U+FFFD (perda permanente no log). Se a saída do pipeline for redirecionada para arquivo, imprimir U+FFFD levanta UnicodeEncodeError e o pipeline morre no meio.
- Sem resolver caminhos relativos de config, o pré-check passa e cada etapa morre com FileNotFoundError.
- experiment_name precisa do sufixo `_smoke`, senão o pipeline procura checkpoints/JSONs com nome errado.

## scripts/score_fusion.py
- Diferente da fusão de características do Incremento 2 (dois ramos no mesmo modelo): aqui combinam-se saídas de modelos treinados separadamente, prática padrão no ASVspoof.
- Motivação empírica: a análise por ataque mostrou que configurações diferentes vencem ataques diferentes (baixa resolução melhor no núcleo duro; alta resolução nos ataques semelhantes ao treino).
- Regra max: o sistema mais "desconfiado" decide.
- Reuso do .npz: sem ele a fusão de N modelos faria N passadas completas de inferência.
- Postos: os log-odds não empatam por arredondamento; o posto médio continua valendo para arquivos antigos só com probabilidade.

## scripts/simular_caminho.py
- O controle do teste ao vivo levou todos os áudios a ~1,0, bonafide inclusive, sem perdas de amostra; o script roda sobre a sessão "limpo".
- `perdas` simula descontinuidade do WASAPI.
- `reamostragem` usa o mesmo soxr do `tocar` e da captura. Medido em ruído branco: preserva até 7,3 kHz e apaga 7,6-8 kHz (-30 dB em 7,7-7,9 kHz, -103 dB acima de 7,9 kHz). Nenhuma qualidade do soxr evita: é o filtro antialiasing de qualquer conversão para 16 kHz.
- `passa_baixa` serve para achar a partir de onde o modelo colapsa.
- `piso`: medido no controle real (comparar_espectro.py), o piso das pausas subiu de -62,5 para -30,9 dB do pico.

## scripts/tdcf.py
- Referência de Todisco et al. (2019), Tab. 1: B01 0,2366; B02 0,2116 (também impresso pelo script).
- Empates: o softmax satura em 1,0 em dezenas de milhares de áudios; o script oficial percorre a DET amostra a amostra e, com empates, cria pontos de operação que dependem da ordem do array. Avaliando só limiares distintos, o resultado é o que um limiar real consegue; sem empates, os dois coincidem.
- Formato do --exportar: `utt_id ataque chave score`, score maior = bonafide, para conferência cruzada com o script oficial.
- Arquivo de scores ASV: variantes de 3 e 4 colunas (com ou sem id do locutor na frente).
- Com bonafide e spoof empatados em 1,0 (npz antigo), o t-DCF mediria o arredondamento do float32.

