# `ml/data/` — Dados

Esta pasta guarda os dados usados no treino e na avaliação. **Nenhum áudio é
versionado no git** (veja `.gitignore`) — tanto pelo tamanho da base quanto,
principalmente, porque voz é dado biométrico sensível segundo a LGPD.

## 1. ASVspoof 2019 LA (base principal)

Trilha **LA (Logical Access)** — ataques de TTS e Voice Conversion, que é o
escopo deste projeto.

- Página oficial: https://www.asvspoof.org/index2019.html
- Download (Edinburgh DataShare): https://datashare.ed.ac.uk/handle/10283/3336

Após baixar e extrair, organize assim (é o layout padrão da base):

```
ml/data/LA/
├── ASVspoof2019_LA_train/flac/        # .flac de treino
├── ASVspoof2019_LA_dev/flac/          # .flac de validação
├── ASVspoof2019_LA_eval/flac/         # .flac de teste
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

Os caminhos podem ser ajustados em `ml/configs/*.yaml` (seção `data`).

### Formato do protocolo

Cada linha dos arquivos `.txt` tem o formato:

```
SPEAKER_ID  AUDIO_FILE_NAME  -  SYSTEM_ID  KEY
```

onde `KEY` é `bonafide` (áudio autêntico) ou `spoof` (sintético). O loader usa
apenas `AUDIO_FILE_NAME` e `KEY`.

## 2. Amostras de voluntários (testes de robustez)

Gravações coletadas voluntariamente de colaboradores, **todas bonafide**,
usadas apenas nos testes de robustez (TC1 §5.5).

```
ml/data/volunteers/        # ignorado pelo git — NÃO versionar
```

### Cuidados de LGPD (obrigatório)

- Coletar **termo de consentimento assinado** de cada colaborador antes de gravar.
- Usar os dados **exclusivamente** para fins acadêmicos deste projeto.
- **Descartar** as gravações ao término do projeto.
- Nunca commitar os áudios nem dados pessoais que identifiquem os voluntários.

## 3. Cache de features

Quando `cache_features: true`, as features extraídas são salvas em
`ml/data/cache/` para acelerar épocas seguintes. Essa pasta também é ignorada
pelo git e pode ser apagada a qualquer momento.

## 4. Bases do quarto incremento (no `D:`, por junções)

As pastas abaixo, em `ml/data/`, são junções para o `D:` (SSD SATA). Em
05/11/2026, os pacotes originais foram apagados para liberar espaço (78 GB):
ficaram só as amostras usadas, que são as que as listas de `ml/data/recortes/`
apontam. As características do XLS-R tiradas delas estão em `ml/outputs/ssl/`.

| pasta | o que ficou | o que saiu (baixar de novo para montar outra amostra) |
|---|---|---|
| `ASVspoof5` | os protocolos (`.tsv`) | os `.tar` de treino e dev (Zenodo); o recorte usado está em `ASVspoof5_recorte/`, no `E:` |
| `BRSpeech-DF` | `amostra/` (treino, validação, teste) | `bonafide/` e `spoof/` da base inteira |
| `CommonVoice_pt` | `amostra/` e `tabelas/` | o pacote `cv-corpus-27.0` |
| `BUT_ReverbDB` | `extraido/` (salas e ruídos) | o pacote `RIR-Only.tgz` |
| `InTheWild` | a base inteira | — |
| `ReplayDF` | a base inteira | — |

Sem os pacotes, `montar_recorte_asvspoof5.py`, `montar_amostra_brspeech.py` e
`montar_amostra_commonvoice.py` só rodam depois de baixar de novo.
