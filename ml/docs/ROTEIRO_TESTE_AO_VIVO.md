# Roteiro do teste ao vivo

Passo a passo para rodar o monitor numa chamada real (Meet ou Teams) com áudios
de rótulo conhecido, e sair com números comparáveis e uma figura para o texto.
Tempo total: **1 a 2 horas**, contando a preparação. As chamadas em si duram
~4 minutos cada.

Todos os comandos rodam de dentro de `ml/`, no PowerShell.

---

## 0. O que este teste mostra — e o que não mostra

**Mostra:**

- que o sistema funciona de ponta a ponta numa chamada de verdade: captura a
  saída de som, analisa janela a janela e mostra o score (RF04–RF07);
- **para onde o canal empurra o score** de cada áudio, comparando o mesmo
  áudio em três sessões: limpo, controle (captura sem chamada) e chamada.

**Não mostra:**

- um EER com precisão. São 40 áudios (20 bonafide, 20 spoof): cada bonafide
  mal ordenado move o EER em ~2,5 pp. O número de canal real com precisão é o
  do ASVspoof 2021 LA (29,43%, Seção 5.3 do resumo). Aqui ele é **indicativo**
  e deve ser escrito assim.

Por isso são três sessões e não uma: a diferença entre **limpo** e **controle**
mede o próprio procedimento (captura, volume, reamostragem); a diferença entre
**controle** e **chamada** é o que o Meet/Teams faz com o áudio. Sem o controle,
um erro de volume pareceria efeito do canal.

---

## 1. Pré-requisitos (uma vez só)

1. **Pacote de áudio** (o monitor já usa; o `tocar` também):
   ```powershell
   pip install soundcard
   ```
2. **Google Chrome ou Microsoft Edge** instalado. Não precisa de VB-Cable nem
   de VLC: o Chrome tem um modo de teste do WebRTC que usa um arquivo WAV como
   microfone, e o próprio script toca a playlist no alto-falante.
3. **Conta no Meet ou no Teams.** A segunda ponta pode entrar como convidado.
4. Confira que o dispositivo de saída padrão do Windows é o seu alto-falante ou
   fone, e anote o nome exato:
   ```powershell
   python monitor.py --listar-dispositivos
   ```
5. **Desligue os aprimoramentos de áudio** durante o teste — **obrigatório** (medido: com eles ligados, a Realtek comprimiu a dinâmica, o piso das pausas subiu 31,5 dB e todo áudio foi a score ~1,0): *Configurações →
   Sistema → Som → (alto-falante) → Aprimoramentos de áudio → Desativado*.
   Efeitos do driver (Realtek, por exemplo) mexem no áudio antes de ele sair,
   e o controle deixaria de ser neutro. Anote o formato do dispositivo (ex.:
   24 bits, 48000 Hz) e volte o ajuste depois, se quiser.

## 2. Montar a playlist rotulada

```powershell
python scripts/canal_real.py preparar --config configs/baseline_v2.yaml --n-por-classe 20 --saida outputs/canal_real
```

Sai `outputs/canal_real/referencia.wav` (~3–4 min: 20 bonafide e 20 spoof
cobrindo os 13 ataques, com 1 s de silêncio entre eles) e `mapa.json` (onde
cada áudio começa). **Use a mesma playlist nas três sessões** — não rode o
`preparar` de novo no meio.

---

## 3. Sessão "limpo" — sem captura nenhuma (2 min)

É o ponto de partida: os mesmos 40 áudios, sem passar por lugar nenhum.

```powershell
python scripts/canal_real.py alinhar --pasta outputs/canal_real --gravacao outputs/canal_real/referencia.wav --sessao limpo
python evaluate.py --config outputs/canal_real/limpo/config_canal_real.yaml --checkpoint checkpoints/baseline_lfcc_cnn_v2.pt --partition eval
```

O `alinhar` deve dizer `Recuperados: 40/40` e `Ajuste fino à amostra: 40/40`.

**Conferência do procedimento:** os recortes da sessão "limpo" têm de dar
exatamente os scores do eval de 2019 para os mesmos arquivos:

```powershell
python scripts/comparar_sessoes.py configs/baseline_v2.yaml outputs/canal_real/limpo/config_canal_real.yaml
```

As duas colunas devem sair iguais (a do eval completo aparece como `configs`).
Se diferirem, o recorte está deslocado — foi assim que se achou o erro de até
5 ms do alinhamento por envelope, hoje corrigido pelo ajuste fino.

---

## 4. Sessão "controle" — captura, mas sem chamada (10 min)

Mesmo caminho de captura da chamada, só que sem o Meet/Teams no meio.

1. Abra **dois** terminais em `ml/`.
2. No terminal 1, inicie o monitor **antes** de tocar:
   ```powershell
   python monitor.py --config configs/baseline_v2.yaml --checkpoint checkpoints/baseline_lfcc_cnn_v2.pt --gravar outputs/canal_real/controle.wav --json outputs/canal_real/controle.json
   ```
3. No terminal 2, toque a playlist no **alto-falante padrão**, com o volume do
   Windows num nível médio e fixo (não mexa até o fim):
   ```powershell
   python scripts/canal_real.py tocar --pasta outputs/canal_real
   ```
   Ele espera um Enter para começar. **Só dê Enter quando o terminal 1 já
   mostrar o cabeçalho da tabela** (`t  score  média ...`): o monitor leva
   alguns segundos carregando o modelo, e o que tocar antes disso não é
   gravado. (No primeiro controle real a gravação começou 7 s atrasada; o
   `alinhar` agora aguenta isso, mas os trechos do começo se perdem.)
4. Quando a playlist acabar, espere ~5 s e pare o monitor com **Ctrl+C**.
5. Alinhe e avalie:
   ```powershell
   python scripts/canal_real.py alinhar --pasta outputs/canal_real --gravacao outputs/canal_real/controle.wav --sessao controle
   python evaluate.py --config outputs/canal_real/controle/config_canal_real.yaml --checkpoint checkpoints/baseline_lfcc_cnn_v2.pt --partition eval
   ```

**Se o controle já divergir muito do limpo**, pare aqui e veja a Seção 8: o
problema é do procedimento, e uma chamada por cima só vai somar ruído.

---

## 5. Sessão "chamada" — pelo Meet ou Teams (20 min)

A ponta **A** toca a playlist como se fosse o microfone; a ponta **B** recebe
pela chamada e o monitor escuta o que sai no alto-falante de B.

### Opção 1 — um computador só, sem instalar nada (recomendada)

A ponta A é um Chrome **separado** cujo microfone é a playlist; a ponta B é o
seu navegador de sempre. O áudio **passa pelos servidores** do Meet/Teams e
pelo codec deles; o que muda em relação a dois PCs é só que a rede de ida e
volta é a mesma.

1. Terminal 1, **antes de tudo**, inicie a gravação da ponta B:
   ```powershell
   python monitor.py --config configs/baseline_v2.yaml --checkpoint checkpoints/baseline_lfcc_cnn_v2.pt --gravar outputs/canal_real/chamada.wav --json outputs/canal_real/chamada.json
   ```
2. **Ponta B — seu navegador normal:** crie a reunião no Meet (ou Teams).
   **Microfone desligado** (evita eco); alto-falante = o **padrão** do
   Windows, o mesmo que aparece em `--listar-dispositivos`. Copie o link.
3. Terminal 2, abra a ponta A:
   ```powershell
   python scripts/canal_real.py chrome --pasta outputs/canal_real --url "<link da reunião>"
   ```
   Abre uma janela **nova** do Chrome (perfil próprio, dentro de
   `outputs/canal_real/perfil_navegador`), com o microfone trocado pelo
   arquivo. Se o Chrome não for achado, ele tenta o Edge; ou passe
   `--navegador "C:\caminho\chrome.exe"`.
4. **Nessa janela (ponta A):** permita o microfone, **desligue a câmera**
   (ela vira uma imagem de teste), entre na reunião e, se for convidado, admita
   pela ponta B. Não precisa escolher microfone.
5. **A playlist começa sozinha** 60 s depois de o Meet abrir o microfone (na
   tela de prévia) — é o tempo para entrar. Se precisar de mais, use
   `--espera 120`. O alinhamento acha o atraso sozinho; a espera não atrapalha.
6. Com a playlist tocando, **tire um print do terminal 1** com as barras do
   score — é a figura do teste funcional.
7. Quando a playlist acabar (~3–4 min depois da espera), espere ~5 s e pare o
   monitor com **Ctrl+C**. Pode fechar a janela da ponta A.
8. **Não mexa** nas opções de supressão de ruído do Meet/Teams. O teste mede o
   canal como ele vem de fábrica — só **anote** qual estava ligada.

Como funciona: o Chrome é aberto com `--use-fake-device-for-media-stream` e
`--use-file-for-fake-audio-capture=<arquivo>%noloop`, opções de teste do próprio
WebRTC. Conferido no Chromium: o arquivo começa a tocar quando a página abre o
microfone, toca uma vez só, e depois o microfone fica em silêncio. O perfil
próprio é necessário: com o Chrome já aberto, sem ele, as opções seriam
ignoradas sem aviso.

### Opção 2 — com VB-Cable (se a opção 1 não funcionar)

Instale o VB-Cable (`vb-audio.com/Cable`, reinicie o Windows), escolha
`CABLE Output` como microfone da ponta A no Meet e toque a playlist nele:

```powershell
python scripts/canal_real.py tocar --pasta outputs/canal_real --dispositivo "CABLE Input"
```

### Opção 3 — dois computadores

Qualquer uma das opções acima, com a ponta A num PC e a ponta B (Meet +
monitor) no outro. Mede também a rede entre dois lugares.

### Alinhar e avaliar

```powershell
python scripts/canal_real.py alinhar --pasta outputs/canal_real --gravacao outputs/canal_real/chamada.wav --sessao chamada
python evaluate.py --config outputs/canal_real/chamada/config_canal_real.yaml --checkpoint checkpoints/baseline_lfcc_cnn_v2.pt --partition eval
```

Se quiser comparar Meet com Teams, repita esta seção com `--sessao teams`
(gravando em `teams.wav`). Cada sessão fica na própria pasta e nenhuma
sobrescreve a outra.

---

## 6. Comparar as sessões

```powershell
python scripts/comparar_sessoes.py outputs/canal_real/limpo/config_canal_real.yaml outputs/canal_real/controle/config_canal_real.yaml outputs/canal_real/chamada/config_canal_real.yaml --csv outputs/canal_real/comparacao.csv
```

Sai:

- por sessão: EER e **acertos no limiar** (quantos dos 40 o limiar do
  checkpoint classificou certo);
- **por áudio**: o score em cada sessão, lado a lado. O CSV vira tabela do
  texto.

O resumo que o monitor imprime ao dar Ctrl+C mistura bonafide e spoof da
playlist inteira — ele prova que o sistema operou, mas **não é o resultado**.
O resultado é o do passo 6.

---

## 7. O que anotar (vai para a metodologia)

| item | exemplo |
|---|---|
| data e hora | 30/09, 21h |
| plataforma e versão | Google Meet, Chrome 1xx |
| topologia | 1 PC, Chrome (A) + Edge (B) |
| rede | Wi-Fi doméstico |
| supressão de ruído | padrão (ligada) |
| volume do Windows | 50% |
| alinhamento de cada sessão | correlação mediana, `Recuperados: x/40` |
| resultado de cada sessão | EER e acertos (saída do passo 6) |

**Como ler:**

- **limpo ≈ controle**: o procedimento é neutro, e a diferença até a chamada é
  do canal. É o resultado esperado.
- **chamada pior que controle**: é o custo do Meet/Teams. Compare a direção com
  o ASVspoof 2021 LA (limpo 18,15% → Opus real 29,43%).
- **scores descendo em bloco** na chamada, sem perder a ordem: o EER mal se
  move, mas os acertos no limiar caem. É o "o limiar não transfere" da Seção 5,
  agora numa chamada real.

---

## 8. Se algo der errado

| sintoma | causa provável | o que fazer |
|---|---|---|
| monitor só mostra `(silêncio)` | capturando o dispositivo errado | `--listar-dispositivos` e `--dispositivo-audio "<nome>"` |
| B não ouve nada (opção 1) | a janela da ponta A não é a aberta pelo `chrome`, ou o microfone foi negado | feche e reabra pelo comando; permita o microfone |
| B não ouve nada (opção 1) e havia um Chrome aberto | as opções foram ignoradas | o comando já usa perfil próprio; confira se a janela nova apareceu separada |
| a playlist acabou antes de entrar | espera curta demais | reabra com `--espera 120` e grave de novo |
| B não ouve nada (opção 2) | microfone de A não é o `CABLE Output` | configurações de áudio da ponta A |
| `Recuperados` abaixo de 32/40 | gravação começou depois do play, ou volume mudou | refaça a sessão, iniciando o monitor antes |
| `só sobrou uma classe` | quase nada alinhou | idem; confira a correlação impressa |
| eco ou microfonia | microfone de B ligado | desligue o microfone da ponta B |
| controle muito diferente do limpo | volume muito baixo ou saturando, ou aprimoramentos ligados | volume em ~50%, aprimoramentos desativados, repita o controle |
| `alinhar` recupera poucos trechos | gravação começou tarde, taxa errada ou buracos | `python scripts/diagnosticar_captura.py --pasta outputs/canal_real --gravacao <arquivo>` diz qual |

Depois de rodar, mande a saída do passo 6 e as anotações da Seção 7. A partir
disso a Seção 10.2 do resumo passa de "não executada" para uma execução
reduzida, com o resultado registrado.
