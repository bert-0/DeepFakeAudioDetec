"""A marca d'água Perth do Chatterbox: está nos clones? Explica as linhas fixas?

O Chatterbox põe em todo áudio gerado a marca Perth (Resemble AI), verificável
com o próprio pacote `perth`. Duas perguntas:

1. Ela aparece nos nossos clones do Chatterbox e não nos reais nem no XTTS?
2. As linhas fixas atribuídas ao vocoder do Chatterbox (6.550, 7.450 e 7.850 Hz,
   RESUMO_TCC §10.2.19 e §10.2.20) poderiam ser a marca, e não o vocoder?
   Teste: aplica a marca em áudios sem ela (reais e XTTS) e conta as linhas
   antes e depois.

Roda no ambiente conda `chatterbox` (o `perth` só está lá). Os áudios marcados
ficam numa pasta temporária, apagada no fim: voz é dado biométrico.
Estudo exploratório (RESUMO_TCC §10.2.28).

Uso (de dentro de ml/, ambiente chatterbox):
    python scripts/estudo_perth.py --grupo L01_real=data/vozes/ensaio2/real \\
        --grupo L01_chatterbox=data/vozes/ensaio2/clone_chatterbox --marcar L01_real \\
        --linhas 6550,7450,7850
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def ler(arq: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    y, sr = sf.read(arq, dtype="float32")
    return (y.mean(axis=1) if y.ndim > 1 else y), sr


def detectar(marcador, arquivos: list[Path]) -> list[float]:
    saida = []
    for a in arquivos:
        y, sr = ler(a)
        saida.append(float(marcador.get_watermark(y, sample_rate=sr)))
    return saida


def marcar(marcador, arquivos: list[Path], destino: Path) -> list[Path]:
    import soundfile as sf

    destino.mkdir(parents=True, exist_ok=True)
    saida = []
    for a in arquivos:
        y, sr = ler(a)
        m = marcador.apply_watermark(y, watermark=None, sample_rate=sr)
        sf.write(destino / a.name, np.asarray(m, dtype=np.float32), sr)
        saida.append(destino / a.name)
    return saida


def contar_linhas(arquivos: list[Path], linhas: list[float]) -> dict[str, int]:
    from scripts.estudo_linhas import matriz, presenca

    m = matriz(arquivos)
    return {f"{h:g}": presenca(m, h) for h in linhas}


def estudar(marcador, grupos: dict[str, list[Path]], a_marcar: list[str], linhas: list[float]) -> dict:
    saida: dict = {}
    pasta = Path(tempfile.mkdtemp(prefix="perth_"))
    try:
        for nome, arquivos in grupos.items():
            saida[nome] = {"n": len(arquivos), "marca": detectar(marcador, arquivos),
                           "linhas": contar_linhas(arquivos, linhas)}
            if nome in a_marcar:
                marcados = marcar(marcador, arquivos, pasta / nome)
                saida[f"{nome}+perth"] = {"n": len(marcados), "marca": detectar(marcador, marcados),
                                          "linhas": contar_linhas(marcados, linhas)}
    finally:
        shutil.rmtree(pasta, ignore_errors=True)
    return saida


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Marca Perth do Chatterbox e as linhas fixas")
    p.add_argument("--grupo", action="append", required=True, metavar="NOME=PASTA")
    p.add_argument("--marcar", action="append", default=[], metavar="NOME",
                   help="grupos sem a marca que recebem a marca para o controle")
    p.add_argument("--linhas", default="6550,7450,7850")
    p.add_argument("--json", default="outputs/estudo_perth.json")
    args = p.parse_args(argv)

    import perth

    grupos = {}
    for item in args.grupo:
        nome, pasta = item.split("=", 1)
        grupos[nome] = sorted(Path(pasta).glob("*.wav"))
    linhas = [float(x) for x in args.linhas.split(",")]
    r = estudar(perth.PerthImplicitWatermarker(), grupos, args.marcar, linhas)
    print(f"{'grupo':18s} {'marca':>7s} " + " ".join(f"{h:>8g}" for h in linhas))
    for nome, x in r.items():
        print(f"{nome:18s} {sum(v > 0.5 for v in x['marca']):>3d}/{x['n']:<3d} "
              + " ".join(f"{x['linhas'][f'{h:g}']:>5d}/{x['n']:<2d}" for h in linhas))
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultado: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
