"""Fusão ao vivo: média das probabilidades de dois modelos, como faz o monitor.

Lê os scores por áudio que o `robustness_eval.py` salva para cada condição e
mede a fusão no mesmo conjunto: EER e ponto de operação (humanos acima do
limiar, sintéticos que passam). A regra de postos entra só como referência: ela
exige o conjunto inteiro e não roda ao vivo.

Uso (os dois modelos na mesma amostra e condição):
    python scripts/robustness_eval.py --config configs/baseline_v2.yaml \\
        --checkpoint checkpoints/baseline_lfcc_cnn_v2.pt --amostra 10000 --so captura_48k_fir
    python scripts/robustness_eval.py --config configs/fusion_v4.yaml \\
        --checkpoint checkpoints/fusion_lcnn_v4.pt --amostra 10000 --so captura_48k_fir
    python scripts/fusao_ao_vivo.py \\
        outputs/baseline_lfcc_cnn_v2_amostra10000_captura_48k_fir_eval_captura_48k_fir_scores.npz \\
        outputs/fusion_lcnn_v4_amostra10000_captura_48k_fir_eval_captura_48k_fir_scores.npz

Com `--dev` (os mesmos dois comandos com `--partition dev`), o limiar da fusão é
calibrado no dev, como no `calibrar_captura.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.score_fusion import combine  # noqa: E402
from src.metrics import compute_eer, compute_eer_with_threshold  # noqa: E402


def carregar(caminho: str | Path) -> dict:
    with np.load(caminho, allow_pickle=False) as d:
        dados = {k: d[k] for k in d.files}
    dados["nome"] = Path(caminho).name.split("_amostra")[0].split("_eval")[0].split("_dev")[0]
    return dados


def alinhar(a: dict, b: dict) -> tuple[dict, dict]:
    """Mesmos áudios, na mesma ordem. Recusa conjuntos diferentes: a fusão de
    áudios distintos mediria outra coisa."""
    if a["ids"].shape == b["ids"].shape and (a["ids"] == b["ids"]).all():
        return a, b
    comuns = sorted(set(a["ids"].tolist()) & set(b["ids"].tolist()))
    if len(comuns) < 0.99 * max(len(a["ids"]), len(b["ids"])):
        raise ValueError(f"os arquivos têm conjuntos diferentes ({len(a['ids'])} e "
                         f"{len(b['ids'])} áudios, {len(comuns)} em comum): rode os dois "
                         "com a mesma --amostra, condição e partição")

    def filtrar(d):
        pos = {i: k for k, i in enumerate(d["ids"].tolist())}
        idx = [pos[i] for i in comuns]
        return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == len(d["ids"])
                     else v) for k, v in d.items()}

    return filtrar(a), filtrar(b)


def ponto_de_operacao(labels: np.ndarray, scores: np.ndarray, limiar: float) -> dict:
    """Fração de humanos acima do limiar e de sintéticos abaixo dele."""
    humanos, sinteticos = labels == 0, labels == 1
    return {"limiar": float(limiar),
            "humanos_acima": float((scores[humanos] >= limiar).mean()),
            "sinteticos_passam": float((scores[sinteticos] < limiar).mean())}


def fundir(a: dict, b: dict, dev: tuple[dict, dict] | None = None,
           extras: dict[str, float] | None = None) -> dict:
    """Métricas de cada modelo e da fusão pela média, em vários limiares."""
    a, b = alinhar(a, b)
    if not (a["labels"] == b["labels"]).all():
        raise ValueError("rótulos diferentes para os mesmos áudios")
    labels = a["labels"]
    media = combine([a["scores"], b["scores"]], "mean")
    postos = combine([a["scores"], b["scores"]], "rank", [a["logodds"], b["logodds"]])

    saida = {"n": int(len(labels)), "modelos": {}, "fusao": {}}
    for d in (a, b):
        lim = float(d["threshold"])
        saida["modelos"][d["nome"]] = {
            "eer": compute_eer(labels, d["logodds"]),
            "ponto": None if np.isnan(lim) else ponto_de_operacao(labels, d["scores"], lim)}

    limiares = {}
    la, lb = float(a["threshold"]), float(b["threshold"])
    if not np.isnan(la):
        limiares[f"limiar de {a['nome']} (o que o monitor usa hoje)"] = la
    if not (np.isnan(la) or np.isnan(lb)):
        limiares["média dos dois limiares"] = (la + lb) / 2
    if dev is not None:
        da, db = alinhar(*dev)
        eer_dev, lim_dev = compute_eer_with_threshold(
            da["labels"], combine([da["scores"], db["scores"]], "mean"))
        limiares["calibrado no dev (ponto de EER da fusão)"] = lim_dev
        saida["eer_dev_fusao"] = eer_dev
    limiares.update(extras or {})

    saida["fusao"] = {
        "eer_media": compute_eer(labels, media),
        "eer_postos": compute_eer(labels, postos),
        "pontos": {nome: ponto_de_operacao(labels, media, lim) for nome, lim in limiares.items()},
    }
    return saida


def _pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def relatar(r: dict) -> None:
    print(f"Áudios: {r['n']}\n")
    print(f"{'modelo':28s} {'EER':>7s}  {'limiar':>7s}  {'humanos acima':>14s}  {'sintéticos passam':>18s}")
    for nome, m in r["modelos"].items():
        p = m["ponto"]
        lim = f"{p['limiar']:.4f}" if p else "—"
        ha = _pct(p["humanos_acima"]) if p else "—"
        sp = _pct(p["sinteticos_passam"]) if p else "—"
        print(f"{nome:28s} {_pct(m['eer']):>7s}  {lim:>7s}  {ha:>14s}  {sp:>18s}")
    f = r["fusao"]
    print(f"{'fusão (média, ao vivo)':28s} {_pct(f['eer_media']):>7s}")
    print(f"{'fusão (postos, referência)':28s} {_pct(f['eer_postos']):>7s}")
    if "eer_dev_fusao" in r:
        print(f"\nEER da fusão no dev: {_pct(r['eer_dev_fusao'])}")
    print("\nFusão pela média, por limiar:")
    for nome, p in f["pontos"].items():
        print(f"  {p['limiar']:.4f}  humanos acima {_pct(p['humanos_acima'])}  "
              f"sintéticos passam {_pct(p['sinteticos_passam'])}  — {nome}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fusão ao vivo (média das probabilidades)")
    p.add_argument("scores", nargs=2, help="os dois .npz do robustness_eval.py")
    p.add_argument("--dev", nargs=2, default=None, help="os mesmos .npz na partição dev")
    p.add_argument("--limiar", type=float, action="append", default=[],
                   help="limiar extra a avaliar na fusão (repita a opção)")
    p.add_argument("--json", default=None, help="grava o resultado neste arquivo")
    args = p.parse_args(argv)

    dev = tuple(carregar(c) for c in args.dev) if args.dev else None
    extras = {f"pedido ({v:.4f})": v for v in args.limiar}
    r = fundir(carregar(args.scores[0]), carregar(args.scores[1]), dev, extras)
    relatar(r)
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResultado: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
