# Protocolo de gravação (rascunho)

Como coletar as vozes reais e gerar os falsos do Incremento 4. Nenhuma
gravação começa antes do termo assinado (TERMO_CONSENTIMENTO.md). Campos
`[A DEFINIR]` dependem das decisões pendentes do plano.

## Locutores

- 8 a 10 locutores: o grupo, voluntários e o orientador. Cada um vira um
  código (L01, L02, ...) no primeiro contato.
- **Separação por locutor, decidida antes de gravar:** a meta do ensaio é com
  locutores fora do treino. O orientador e pelo menos [A DEFINIR: 2]
  voluntários ficam **só no teste**. Nenhum arquivo deles, real ou clone, entra
  no treino nem na validação.
- A tabela código → nome fica fora do repositório (o `.gitignore` bloqueia
  `*locutores_nomes*`), com [A DEFINIR: responsável].

## Frases

- Português do Brasil, lidas, de **5 a 8 s de fala** cada. O modelo decide em
  janelas de 4 s (RESUMO_TCC §9.1): frase mais curta que isso nunca ocupa uma
  janela inteira e é completada por repetição.
- Lista de treino: [A DEFINIR: número, sugestão 40] frases por locutor de
  treino. Lista de ensaio: 10 frases diferentes, que não aparecem no treino.
- Sugestão: partir de uma lista foneticamente balanceada publicada para o
  português do Brasil e completar com frases longas o bastante; conferir a
  referência antes de citar.
- Os falsos usam **as mesmas frases** dos reais (controle pareado): a única
  diferença entre os pares é real ou sintético.

## Equipamento e sala

- Microfone: na apresentação, o embutido do notebook, junto da webcam ("Grupo
  de microfones (Realtek(R) Audio)", sem fone conectado). Na coleta: [A DEFINIR];
  recomendação, o mesmo da apresentação, para a condição de captura ser a da
  demonstração. Interface USB com ASIO: [A DEFINIR].
- Captura pela aba Gravar: WASAPI compartilhado, 48 ou 96 kHz conforme o
  microfone, FIR para 16 kHz. O arquivo guardado é o da taxa do microfone, em
  24 bits, sem conversão. O exclusivo ficou de fora em 26/10/2026: neste
  notebook ele insere cópias de trechos recentes (cerca de 1 em 3 trechos de
  10 ms) e entrega 16 bits (RESUMO_TCC §10.2.34).
- Nenhum fone com microfone conectado ao notebook: com ele, o driver Realtek
  troca o microfone embutido pelo do fone sem mudar o nome da entrada.
- Aprimoramentos de áudio do Windows desligados no microfone (RESUMO_TCC
  §10.2.1). Antes da coleta, a mesma frase em modo compartilhado e exclusivo, e
  com ruído constante ao fundo, para ver se o hardware processa o áudio.
- Sala: [A DEFINIR]. Distância boca–microfone de 15 a 30 cm, a mesma para
  todos.
- Nível: no modo compartilhado vale o volume (e o reforço) do microfone no
  Windows. No "Grupo de microfones", o ruído da sala ficou +40 dB acima do
  exclusivo; no exclusivo, a fala do embutido tinha pico de −25 dBFS (L01) e
  −29 dBFS (L02) (RESUMO_TCC §10.2.34). Com o mesmo ganho, a fala saturaria. O modelo normaliza pelo pico, então o que
  importa é a distância entre fala e ruído, não o nível; guardar em 24 bits
  para não perder resolução. Ajuste pelo volume do microfone no Windows, com
  uma frase de teste que não entra no conjunto: pico entre −12 e −3 dBFS, sem
  saturar. O mesmo nível para todos os locutores e na apresentação.

## Toda condição nas duas classes

O modelo já mostrou que reage às condições de gravação, não ao vocoder
(RESUMO_TCC §10.2.6). Para não treinar um detector de microfone:

- Cada condição de captura (microfone, sala, modo) tem reais e falsos.
- Os falsos recebem coloração de microfone e sala (resposta ao impulso e
  equalização), e uma parte é regravada pelo alto-falante no mesmo microfone.
- Ferramentas de clonagem: [A DEFINIR: 2 ou 3]. Uma fica fora do treino e só
  aparece no teste.

## Depois de cada gravação

A aba Gravar mostra as checagens; se alguma falhar, regravar a frase.

| checagem | falha quando |
|---|---|
| saturação | 3 ou mais amostras seguidas no limite do conversor |
| cópias | algum trecho de 10 ms idêntico a um anterior: a captura repetiu áudio (RESUMO_TCC §10.2.34) |
| fala | menos de 4 s de fala, ou menos de 0,5 s de silêncio antes ou depois |
| banda | nada acima de 4 kHz (o caminho perdeu a banda alta) |
| piso de ruído | ruído nas pausas a menos de 35 dB da fala: o corte de silêncio do modelo (30 dB) não o remove |

Os limites ficam no código (`src/capture/checagens.py`) e são os mesmos para
todos os locutores. Começar a falar cerca de 1 s depois de iniciar a gravação
e esperar 1 s antes de parar.

## Metadados de cada arquivo

Locutor (código), frase (id), classe (real ou falso), ferramenta de clonagem
(nos falsos), número do termo, dispositivo, modo de captura (exclusivo ou
compartilhado), sala, data e resultado das checagens.

## Armazenamento e exclusão

- Vozes em `ml/data/vozes/`, clones em `ml/data/clones/`, termos digitalizados
  em `ml/data/termos/`; as três pastas ficam fora do Git.
- Cópia de segurança: [A DEFINIR: onde, com acesso restrito].
- Se o treino rodar no Kaggle, as vozes sobem como dataset **privado**, e o
  termo precisa dizer isso (item 3, serviços externos).
- Clones apagados logo depois da banca. Vozes reais apagadas até
  [A DEFINIR: data do termo]. Pedido de retirada: apagar gravações, clones e
  features em cache do locutor, e anotar a data.
