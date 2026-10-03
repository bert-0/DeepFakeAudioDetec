# Detecção de Deepfakes em Áudio

[![Testes](https://github.com/bert-0/DeepFakeAudioDetec/actions/workflows/testes.yml/badge.svg)](https://github.com/bert-0/DeepFakeAudioDetec/actions/workflows/testes.yml)

Sistema de detecção de *deepfakes* em áudio com **fusão de características** e
**mecanismos de atenção**, desenvolvido como Trabalho de Conclusão de Curso
(Ciência da Computação — UNIP).

A proposta combina múltiplas representações acústicas do sinal (LFCC,
espectrograma, coeficientes delta) e mecanismos de atenção para distinguir
vozes humanas reais (*bonafide*) de vozes sintéticas (*spoof*) geradas por
TTS e Voice Conversion, usando a base **ASVspoof 2019 LA**.

## Organização do repositório

O projeto é um monorepo dividido em três áreas, alinhadas ao cronograma do TCC:

| Pasta  | Fase            | Conteúdo                                                        |
|--------|-----------------|----------------------------------------------------------------|
| `ml/`  | TC1 + Férias    | Pipeline de IA: pré-processamento → extração → modelo → métricas |
| `ml/web/` | TC2 (Ago–Out) | Interface web (FastAPI + Jinja2): análise, resultado, histórico |

A interface web fica dentro de `ml/` porque usa o mesmo código de análise do
monitor. Foi feita com FastAPI e páginas renderizadas no servidor (Jinja2), em
vez de uma API separada mais um front-end React: uma aplicação só, na mesma
linguagem do modelo, sem etapa de build. Veja [`ml/README.md`](ml/README.md)
para instruções de uso.

## Incrementos do modelo (TC1 §4.7)

1. **Baseline** — LFCC + CNN convencional.
2. **Fusão de características** — adiciona o espectrograma log-mel (os deltas do LFCC já estão no baseline).
3. **Atenção** — *pooling* com atenção em cada ramo, antes da concatenação.

Cada incremento tem um arquivo de configuração em `ml/configs/` e pode ser
treinado/avaliado de forma independente para comparação (TC1 §5.3).

## Início rápido

```bash
cd ml
pip install -r requirements.txt

# Roda o pipeline inteiro ponta-a-ponta com dados sintéticos (sem dataset):
python train.py --config configs/baseline.yaml --smoke
python evaluate.py --config configs/baseline.yaml --checkpoint checkpoints/baseline_lfcc_cnn.pt --smoke
```
