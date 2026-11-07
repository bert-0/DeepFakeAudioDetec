"""Score de cada arquivo, lado a lado, com o mesmo caminho da interface web.

Feito para a demonstração com a própria voz (`gerar_sintetico.py`): agrupa os
arquivos `<frase>_original`, `<frase>_griffinlim` e `<frase>_world` e mostra se
as versões sintéticas ficam acima do original da mesma frase.

Uso:
    python scripts/avaliar_arquivos.py outputs/demo_propria
    python scripts/avaliar_arquivos.py gravacoes/frase1.wav outputs/demo/reais.wav
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web.analise import CHECKPOINT_PADRAO, CONFIG_PADRAO, Detector  # noqa: E402

EXTENSOES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
VERSOES = ("original", "griffinlim", "world")


def listar(entradas: list[str]) -> list[Path]:
    arquivos = []
    for e in map(Path, entradas):
        if e.is_dir():
            arquivos += sorted(p for p in e.iterdir() if p.suffix.lower() in EXTENSOES
                               and p.stem != "pareado")
        else:
            arquivos.append(e)
    return arquivos


def frase_e_versao(nome: str) -> tuple[str, str | None]:
    for v in VERSOES:
        if nome.endswith("_" + v):
            return nome[: -len(v) - 1], v
    return nome, None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Score de cada arquivo, lado a lado")
    p.add_argument("entradas", nargs="+", help="arquivos ou pastas")
    p.add_argument("--config", default=CONFIG_PADRAO)
    p.add_argument("--checkpoint", default=CHECKPOINT_PADRAO)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    arquivos = listar(args.entradas)
    if not arquivos:
        print("[ERRO] nenhum arquivo de áudio encontrado.")
        return 1
    detector = Detector(args.config, args.checkpoint, args.device)

    print(f"{'arquivo':34s} {'dur.':>6s} {'score':>6s} {'máx':>6s} {'janelas acima':>14s} "
          f"{'limiar':>7s}  resultado")
    grupos: dict[str, dict[str, float]] = defaultdict(dict)
    for arq in arquivos:
        r = detector.analisar(arq)
        uteis = [j for j in r.janelas if j["util"]]
        acima = sum(1 for j in uteis if r.limiar is not None and j["score"] >= r.limiar)
        if r.score_medio is None:
            resultado, score = "sem fala", "—"
        else:
            score = f"{r.score_medio:6.3f}"
            resultado = "indício de síntese" if r.indicio_de_sintese else "sem indício"
        limiar = f"{r.limiar:7.4f}" if r.limiar is not None else "     —"
        maximo = f"{r.score_maximo:6.3f}" if r.score_maximo is not None else "     —"
        print(f"{arq.stem[:34]:34s} {r.duracao_s:5.1f}s {score:>6s} {maximo} "
              f"{acima:>6d} de {len(uteis):<5d} {limiar}  {resultado}")
        frase, versao = frase_e_versao(arq.stem)
        if versao and r.score_medio is not None:
            grupos[frase][versao] = r.score_medio

    pares = {f: v for f, v in grupos.items() if "original" in v and len(v) > 1}
    if pares:
        print(f"\n{'frase':20s} " + " ".join(f"{v:>11s}" for v in VERSOES) + "   sintética acima?")
        acertos = total = 0
        for frase, v in sorted(pares.items()):
            linha = " ".join(f"{v[x]:11.3f}" if x in v else f"{'—':>11s}" for x in VERSOES)
            sint = [v[x] for x in VERSOES[1:] if x in v]
            ok = sum(s > v["original"] for s in sint)
            acertos += ok
            total += len(sint)
            print(f"{frase[:20]:20s} {linha}   {ok} de {len(sint)}")
        print(f"\nVersões sintéticas com score acima do original da mesma frase: "
              f"{acertos} de {total}.")
        print("É a comparação que importa: mesmo locutor, microfone e texto; só o "
              "vocoder muda.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
