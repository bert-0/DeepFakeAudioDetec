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

