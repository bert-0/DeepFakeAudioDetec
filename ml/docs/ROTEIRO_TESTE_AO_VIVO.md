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

1. **Pacote de captura:**
   ```powershell
   pip install soundcard
   ```
2. **VB-Cable** (cabo de áudio virtual, gratuito): baixe em
   `vb-audio.com/Cable`, instale como administrador e **reinicie o Windows**.
   Ele cria dois dispositivos: `CABLE Input` (saída — onde se toca) e
   `CABLE Output` (entrada — o que a chamada usa como microfone).
3. **VLC** (ou outro player que deixe escolher o dispositivo de saída).
4. **Conta no Meet ou no Teams.** Duas contas facilitam, mas não são
   obrigatórias: a segunda ponta pode entrar como convidado.
5. Confira que o dispositivo de saída padrão do Windows é o seu alto-falante ou
   fone, e anote o nome exato:
   ```powershell
   python monitor.py --listar-dispositivos
   ```

---

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

O `alinhar` deve dizer `Recuperados: 40/40` com correlação perto de 1.

---

## 4. Sessão "controle" — captura, mas sem chamada (10 min)

Mesmo caminho de captura da chamada, só que sem o Meet/Teams no meio.

1. Abra **dois** terminais em `ml/`.
2. No terminal 1, inicie o monitor **antes** de tocar:
   ```powershell
   python monitor.py --config configs/baseline_v2.yaml --checkpoint checkpoints/baseline_lfcc_cnn_v2.pt --gravar outputs/canal_real/controle.wav --json outputs/canal_real/controle.json
   ```
3. Toque `outputs/canal_real/referencia.wav` no VLC pelo **alto-falante
   padrão**, com o volume do Windows num nível médio e fixo. Não mexa no volume
   até o fim.
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

### Opção 1 — um computador só (mais simples)

As duas pontas no mesmo PC, em navegadores diferentes. O áudio **passa pelos
servidores** do Meet/Teams e pelo codec deles; o que muda em relação a dois PCs
é só que a rede de ida e volta é a mesma.

1. **Ponta A — Chrome:** crie a reunião. Em *Configurações → Áudio*:
   microfone = `CABLE Output (VB-Audio Virtual Cable)`; alto-falante = qualquer
   (A não vai ouvir nada, porque B fica mudo).
2. **Ponta B — Edge** (ou janela anônima): entre na mesma reunião (como
   convidado, se for preciso admitir pela ponta A). Em *Configurações → Áudio*:
   **microfone desligado** (evita eco); alto-falante = o **padrão** do Windows,
   o mesmo que aparece em `--listar-dispositivos`.
3. **VLC:** *Áudio → Dispositivo de áudio →* `CABLE Input`. Assim a playlist
   vai **só** para o microfone de A, e não para o seu alto-falante.
4. **Não mexa** nas opções de supressão de ruído do Meet/Teams. O teste mede o
   canal como ele vem de fábrica — só **anote** qual estava ligada.
5. Terminal 1, **antes** de tocar:
   ```powershell
   python monitor.py --config configs/baseline_v2.yaml --checkpoint checkpoints/baseline_lfcc_cnn_v2.pt --gravar outputs/canal_real/chamada.wav --json outputs/canal_real/chamada.json
   ```
6. Dê play no VLC. Com o monitor rodando, **tire um print da tela do
   terminal** com as barras do score — é a figura do teste funcional.
7. Fim da playlist + ~5 s → **Ctrl+C**.

### Opção 2 — dois computadores

Igual, mas a ponta A (Chrome + VB-Cable + VLC) fica num PC e a ponta B (Meet +
monitor) no outro. Mede também a rede entre dois lugares. Se tiver como, é a
versão mais realista; a opção 1 já é válida.

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
| a playlist toca no alto-falante na sessão chamada | VLC não está no `CABLE Input` | *Áudio → Dispositivo de áudio* no VLC |
| B não ouve nada | microfone de A não é o `CABLE Output` | configurações de áudio da ponta A |
| `Recuperados` abaixo de 32/40 | gravação começou depois do play, ou volume mudou | refaça a sessão, iniciando o monitor antes |
| `só sobrou uma classe` | quase nada alinhou | idem; confira a correlação impressa |
| eco ou microfonia | microfone de B ligado | desligue o microfone da ponta B |
| controle muito diferente do limpo | volume muito baixo ou saturando | volume do Windows em ~50% e repita o controle |

Depois de rodar, mande a saída do passo 6 e as anotações da Seção 7. A partir
disso a Seção 10.2 do resumo passa de "não executada" para uma execução
reduzida, com o resultado registrado.
