"""Onde o modelo vê o sintético: tira uma faixa de frequência e mede o score.

Para cada gravação real e cada clone dos ensaios, retira uma faixa de cada vez
(filtro de fase zero, já a 16 kHz) e compara o log-odds com o do áudio inteiro.
Se tirar uma faixa derruba o score dos clones, a pista do sintético está nela;
se o efeito aparece igual nas gravações reais, a faixa só muda o "tipo de som".
Estudo exploratório (RESUMO_TCC §10.2.17): não muda nenhum placar.

Uso (de dentro de ml/, ambiente do projeto):
    python scripts/estudo_faixas.py --limiar outputs/limiar_fusao/limiar_asv5_treino.json \\
        --conjunto L01_xtts=data/vozes/ensaio2:clone \\
        --conjunto L02_xtts=data/vozes/L02:clone \\
        --conjunto L01_chatterbox=data/vozes/ensaio2:clone_chatterbox \\
        --conjunto L02_chatterbox=data/vozes/L02:clone_chatterbox
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Faixas retiradas, em Hz. Mais finas embaixo, onde a voz concentra energia.
FAIXAS = ((0, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000),
          (4000, 5000), (5000, 6000), (6000, 7000), (7000, 8000))
TAXA_COMUM = 48000


def sem_faixa(wav: np.ndarray, sr: int, faixa: tuple[int, int]) -> np.ndarray:
    """Retira a faixa com um Butterworth de ordem 8, ida e volta (fase zero)."""
    from scipy.signal import butter, sosfiltfilt

    baixo, alto = faixa
    nyq = sr / 2
    if baixo <= 0:
        sos = butter(8, alto, "highpass", fs=sr, output="sos")
    elif alto >= nyq:
        sos = butter(8, baixo, "lowpass", fs=sr, output="sos")
    else:
        sos = butter(8, [baixo, alto], "bandstop", fs=sr, output="sos")
    return sosfiltfilt(sos, wav).astype(np.float32)


def logodds(wav16: np.ndarray, analisador_fabrica, regra: str) -> float | None:
    """Média das janelas ponderada pela fala, nos log-odds, como no placar."""
    from src.capture.analyzer import Agregador

    analisador = analisador_fabrica()
    agregador = Agregador(janela_s=analisador.janela.tamanho / analisador.sample_rate,
                          passo_s=analisador.janela.passo / analisador.sample_rate, regra=regra)
    for i in range(0, len(wav16), 1600):
        for leitura in analisador.processar(wav16[i:i + 1600]):
            agregador.adicionar(leitura)
    for leitura in analisador.finalizar():
        agregador.adicionar(leitura)
    return agregador.resumo()["logodds_medio"]


def estudar(arquivos: dict[str, list[Path]], analisador_fabrica, regra: str,
            faixas=FAIXAS) -> dict:
    """{grupo: {"inteiro": [...], "sem a-b": [...]}} com o log-odds de cada arquivo."""
    from src.capture import FileSource

    saida = {}
    for grupo, lista in arquivos.items():
        r = {"inteiro": []} | {f"sem {a}-{b}": [] for a, b in faixas}
        for arq in lista:
            with FileSource(arq, 16000, taxa_comum=TAXA_COMUM) as fonte:
                wav = fonte._wav
            r["inteiro"].append(logodds(wav, analisador_fabrica, regra))
            for a, b in faixas:
                r[f"sem {a}-{b}"].append(logodds(sem_faixa(wav, 16000, (a, b)),
                                                 analisador_fabrica, regra))
        saida[grupo] = r
    return saida


def resumir(estudo: dict, limiar: float) -> dict:
    """Variação média do log-odds por faixa e quantos passam do limiar sem ela."""
    tabela = {}
    for grupo, r in estudo.items():
        base = np.array(r["inteiro"], dtype=float)
        linhas = {"inteiro": {"media": float(base.mean()),
                              "acima": int((base >= limiar).sum()), "n": len(base)}}
        for nome, valores in r.items():
            if nome == "inteiro":
                continue
            v = np.array(valores, dtype=float)
            linhas[nome] = {"variacao": float((v - base).mean()),
                            "acima": int((v >= limiar).sum()), "n": len(v)}
        tabela[grupo] = linhas
    return tabela


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Onde o modelo vê o sintético, faixa por faixa")
    p.add_argument("--limiar", required=True, help="JSON do limiar_fusao.py (modelos e regra)")
    p.add_argument("--conjunto", action="append", required=True, metavar="NOME=PASTA:SUBPASTA",
                   help="clones de PASTA/SUBPASTA; as gravações reais de PASTA/real entram uma vez")
    p.add_argument("--device", default="cpu")
    p.add_argument("--json", default="outputs/estudo_faixas.json")
    args = p.parse_args(argv)

    from src.capture.analyzer import AnalisadorContinuo
    from src.config import load_config, resolve_device
    from src.limiares import ler_limiar_fusao

    escolha = ler_limiar_fusao(args.limiar)
    config = load_config(escolha.config)
    device = resolve_device(args.device)

    def fabrica():
        return AnalisadorContinuo(config, list(escolha.checkpoints), device, regra=escolha.regra)

    arquivos: dict[str, list[Path]] = {}
    for item in args.conjunto:
        nome, caminho = item.split("=", 1)
        pasta, sub = caminho.rsplit(":", 1)
        locutor = nome.split("_")[0]
        arquivos.setdefault(f"{locutor}_real", sorted(Path(pasta, "real").glob("*.wav")))
        arquivos[nome] = sorted(Path(pasta, sub).glob("*.wav"))

    estudo = estudar(arquivos, fabrica, escolha.regra)
    tabela = resumir(estudo, escolha.limiar)
    grupos = list(tabela)
    print(f"{'faixa retirada':18s} " + " ".join(f"{g:>16s}" for g in grupos))
    print(f"{'(inteiro, média)':18s} " + " ".join(f"{tabela[g]['inteiro']['media']:>+16.2f}" for g in grupos))
    for a, b in FAIXAS:
        nome = f"sem {a}-{b}"
        print(f"{nome + ' Hz':18s} " + " ".join(
            f"{tabela[g][nome]['variacao']:>+10.2f} ({tabela[g][nome]['acima']:>2d})" for g in grupos))
    print(f"\nVariação média do log-odds ao tirar a faixa; entre parênteses, quantos ficam "
          f"acima do limiar ({escolha.limiar:.3f}). Inteiro: "
          + ", ".join(f"{g} {tabela[g]['inteiro']['acima']}/{tabela[g]['inteiro']['n']}" for g in grupos))
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({"estudo": estudo, "tabela": tabela}, indent=2,
                                          ensure_ascii=False), encoding="utf-8")
    print(f"\nResultado: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
