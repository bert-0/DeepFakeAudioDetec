"""Arquivo inteiro ou só a fala? A decisão pendente do RESUMO_TCC §10.2.44, nos devs.

(a) o arquivo inteiro, o caminho das réguas: os scores saem dos .npz do
    `estudo_ssl.py extrair`, pela cabeça do modelo final;
(b) o caminho do ao vivo aplicado ao arquivo (`ModeloFinal.pontuar_por_frases`):
    frases cortadas pela pausa, média dos log-odds.

Só os devs (os que escolheram a época do modelo final), para a escolha não tocar
nas réguas. Descritivo, sem critério (RESUMO_TCC §10.2.45): informa a decisão do grupo.

Uso (de dentro de ml/, na GPU):
    python scripts/estudo_corte_fala.py --dev asv5=outputs/ssl/asv5_dev.npz \\
        --dev salas=outputs/ssl/sala_dev.npz --sala-de salas --sala data/BUT_ReverbDB/extraido
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def deslocamento(a: np.ndarray, b: np.ndarray) -> dict | None:
    if not len(a):
        return None
    d = np.asarray(b) - np.asarray(a)
    return {"mediana": float(np.median(d)), "p10": float(np.percentile(d, 10)), "p90": float(np.percentile(d, 90))}


def resumir(a: np.ndarray, b: np.ndarray, rotulo: np.ndarray, n_frases: np.ndarray) -> dict:
    """`b` é NaN onde o caminho (b) não achou frase; as comparações usam só os outros."""
    from scripts.estudo_pente import auc
    from scripts.estudo_ssl import eer

    ok = ~np.isnan(b)
    r = {"n": int(len(a)), "sem_frase": int((~ok).sum()), "mais_de_uma_frase": int((n_frases > 1).sum()),
         "reais": deslocamento(a[ok & (rotulo == 0)], b[ok & (rotulo == 0)]),
         "falsos": deslocamento(a[ok & (rotulo == 1)], b[ok & (rotulo == 1)])}
    if (rotulo == 0).any() and (rotulo == 1).any():
        for nome, s, m in (("a", a, np.ones_like(ok)), ("a_mesmos", a, ok), ("b", b, ok)):
            h, f = s[m & (rotulo == 0)], s[m & (rotulo == 1)]
            r[f"auc_{nome}"] = auc(f, h)
            r[f"eer_{nome}"] = eer(h, f)[0]
    return r


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Arquivo inteiro contra só a fala, nos devs")
    p.add_argument("--cabeca", default="checkpoints/xlsr_congelado.pt")
    p.add_argument("--dev", action="append", required=True, metavar="NOME=NPZ")
    p.add_argument("--sala-de", action="append", default=[], metavar="NOME",
                   help="dev passado pela sala simulada, como na extração (rng [42, k, 3])")
    p.add_argument("--sala", default=None, help="pasta do BUT ReverbDB extraído")
    p.add_argument("--dispositivo", default="cuda")
    p.add_argument("--json", default="outputs/estudo_corte_fala.json")
    args = p.parse_args(argv)

    from scripts.xlsr_congelado import ler, pontuar
    from src.features.xlsr import preparar
    from src.preprocess.audio import load_audio
    from src.xlsr_final import SR, FalaCurta, ModeloFinal

    modelo = ModeloFinal(args.cabeca, args.dispositivo)
    sala = None
    if args.sala_de:
        from src.preprocess.sala import Sala

        sala = Sala(args.sala, SR)
    saida = {}
    for item in args.dev:
        nome, _, npz = item.partition("=")
        d = ler(npz)
        a = pontuar(modelo.cabeca, d, modelo.dev)
        b = np.full(len(a), np.nan)
        n_frases = np.zeros(len(a), int)
        for k, arq in enumerate(d["arquivo"]):
            arq = Path(str(arq).replace("\\", "/"))
            if nome in args.sala_de:
                # Como no `estudo_ssl.extrair --sala`: a sala entra depois do preparo.
                wav = sala(preparar(arq)[0], np.random.default_rng([42, k, 3]))
            else:
                wav = load_audio(arq, SR)
            try:
                pt = modelo.pontuar_por_frases(wav)
            except FalaCurta:
                continue
            b[k], n_frases[k] = pt.logodds, len(pt.janelas)
            if (k + 1) % 200 == 0:
                print(f"  {nome}: {k + 1}/{len(a)}", flush=True)
        saida[nome] = resumir(a, b, d["rotulo"], n_frases)
        print(f"{nome}: {json.dumps(saida[nome], ensure_ascii=False)}", flush=True)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Resultado: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
