"""Parte 9: estatísticas temporais dos quadros locais bastam?

A parte 6 (RESUMO_TCC §10.2.22) mostrou que a pista do XLS-R está na ordem dos
quadros: a convolução de entrada (quadros de 25 ms) resumida pela média separa
pouco, e as camadas do transformer separam quase tudo. A parte 8 mostrou que
medidas globais de dinâmica feitas à mão não chegam lá. O degrau do meio: para
cada dimensão dos quadros locais, média, desvio, variação quadro a quadro e
autocorrelação em vários atrasos. É o que o nosso modelo ganharia trocando o
pooling de média e desvio por estatísticas temporais, sem um modelo de sequência.

Dois tipos de quadro: a convolução do XLS-R (512 dimensões, 20 ms; atrasos de 1,
2, 4 e 8 quadros = 20 a 160 ms) e o LFCC do nosso detector (20 coeficientes, 10
ms; atrasos de 1, 2, 4, 8 e 16 = 10 a 160 ms). Arquivos, classificador e réguas
da parte 6.

Critério combinado antes de rodar: a estatística temporal pega a pista se
ganhar pelo menos 0,05 de AUC sobre a média e o desvio das mesmas
características, no dev da ASVspoof 5 e na 2019 LA. Se nenhuma ganhar, fica
indicado que o nosso modelo precisa de um modelo de sequência para chegar ao que
o XLS-R vê. Estudo exploratório (RESUMO_TCC §10.2.25): não muda nenhum placar.

Uso (de dentro de ml/, com HF_HOME no cache do XLS-R):
    python scripts/estudo_estatisticas_tempo.py extrair --npz outputs/ssl/asv5_train.npz \\
        --saida outputs/estat_tempo/asv5_train.npz
    python scripts/estudo_estatisticas_tempo.py sondar --treino outputs/estat_tempo/asv5_train.npz \\
        --dev outputs/estat_tempo/asv5_dev.npz --avaliar outputs/estat_tempo/asv19_train.npz ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ATRASOS_CNN = (1, 2, 4, 8)
ATRASOS_LFCC = (1, 2, 4, 8, 16)
MARGEM = 0.05
#: Pares (temporal, base) comparados pelo critério.
PARES = (("cnn_tempo", "cnn_media_desvio"), ("lfcc_tempo", "lfcc_media_desvio"))


def estatisticas(quadros: np.ndarray, atrasos: tuple[int, ...]) -> dict[str, np.ndarray]:
    """quadros (T, D) -> média, desvio, |Δ| médio e autocorrelação por atraso, cada um (D,)."""
    x = quadros.astype(np.float64)
    media = x.mean(axis=0)
    c = x - media
    var = (c ** 2).sum(axis=0) + 1e-12
    saida = {"media": media, "desvio": x.std(axis=0),
             "delta": np.abs(np.diff(x, axis=0)).mean(axis=0) if len(x) > 1 else np.zeros(x.shape[1])}
    for k in atrasos:
        saida[f"ac{k}"] = (c[:-k] * c[k:]).sum(axis=0) / var if len(x) > k else np.zeros(x.shape[1])
    return saida


def _juntar(est: dict[str, np.ndarray], chaves) -> np.ndarray:
    return np.concatenate([est[k] for k in chaves]).astype(np.float32)


def quadros_cnn(rede, dev, x: np.ndarray) -> np.ndarray:
    """Saída normalizada da convolução do XLS-R (a mesma `extract_features`), (T, 512)."""
    import torch

    with torch.inference_mode():
        f = rede.feature_extractor(torch.from_numpy(x).float()[None].to(dev)).transpose(1, 2)
        _, normal = rede.feature_projection(f)
    return normal[0].float().cpu().numpy()


def por_arquivo(wav: np.ndarray, rede, dev) -> dict[str, np.ndarray]:
    from scripts.estudo_ssl import LFCC_CFG, SR, janelas
    from src.features.lfcc import compute_lfcc

    ests = []
    for x in janelas(wav):
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        ests.append(estatisticas(quadros_cnn(rede, dev, x), ATRASOS_CNN))
    cnn = {k: np.mean([e[k] for e in ests], axis=0) for k in ests[0]}
    lf = compute_lfcc(wav, SR, LFCC_CFG)          # (60, T): 20 estáticos + deltas
    est_l = estatisticas(lf[:20].T, ATRASOS_LFCC)
    tempo_cnn = ["media", "desvio", "delta"] + [f"ac{k}" for k in ATRASOS_CNN]
    tempo_lfcc = ["media", "desvio", "delta"] + [f"ac{k}" for k in ATRASOS_LFCC]
    return {"cnn_media": _juntar(cnn, ["media"]),
            "cnn_media_desvio": _juntar(cnn, ["media", "desvio"]),
            "cnn_tempo": _juntar(cnn, tempo_cnn),
            "lfcc_media_desvio": _juntar(est_l, ["media", "desvio"]),
            "lfcc_tempo": _juntar(est_l, tempo_lfcc)}


def extrair(origem: Path, destino: Path, rede, dev) -> None:
    from scripts.estudo_ssl import ler_npz, preparar

    d = ler_npz(origem)
    reps: dict[str, list] = {}
    for i, arq in enumerate(d["arquivo"]):
        wav, _ = preparar(Path(str(arq)))
        for k, v in por_arquivo(wav, rede, dev).items():
            reps.setdefault(k, []).append(v)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(d['arquivo'])}", flush=True)
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destino, arquivo=d["arquivo"], classe=d["classe"], rotulo=d["rotulo"],
                        lfcc=d["lfcc"], **{k: np.array(v) for k, v in reps.items()})


def sondar(treino: dict, dev: dict, avaliar: dict[str, dict]) -> dict:
    from scripts.estudo_ssl import aplicar, eer, medir, treinar_sonda

    nomes = ["cnn_media", "cnn_media_desvio", "cnn_tempo", "lfcc", "lfcc_media_desvio", "lfcc_tempo"]
    saida = {}
    for rep in nomes:
        sonda = treinar_sonda(treino[rep].astype(np.float64), treino["rotulo"])
        s = aplicar(sonda, dev[rep].astype(np.float64))
        e, limiar = eer(s[dev["rotulo"] == 0], s[dev["rotulo"] == 1])
        r = {"C": sonda["C"], "eer_dev": e, "dim": int(treino[rep].shape[1]),
             "dev": medir(s, dev["classe"], dev["rotulo"], limiar)}
        r["auc_media_dev"] = float(np.mean([c["auc"] for c in r["dev"]["classes"].values() if c["falso"]]))
        r["avaliar"] = {n: medir(aplicar(sonda, d[rep].astype(np.float64)), d["classe"], d["rotulo"], limiar)
                        for n, d in avaliar.items()}
        saida[rep] = r
    return saida


def criterio(r: dict, regua_2019: str) -> dict:
    saida = {}
    for tempo, base in PARES:
        g_dev = r[tempo]["auc_media_dev"] - r[base]["auc_media_dev"]
        g_19 = r[tempo]["avaliar"][regua_2019]["auc"] - r[base]["avaliar"][regua_2019]["auc"]
        saida[tempo] = {"base": base, "ganho_dev": g_dev, "ganho_2019": g_19,
                        "passa": bool(g_dev >= MARGEM and g_19 >= MARGEM)}
    return saida


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Estatísticas temporais dos quadros locais (parte 9)")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extrair")
    e.add_argument("--npz", required=True, help=".npz da parte 6 (lista de arquivos)")
    e.add_argument("--saida", required=True)
    e.add_argument("--dispositivo", default="cuda")
    s = sub.add_parser("sondar")
    s.add_argument("--treino", required=True)
    s.add_argument("--dev", required=True)
    s.add_argument("--avaliar", action="append", default=[])
    s.add_argument("--regua-2019", default="asv19_train")
    s.add_argument("--json", default="outputs/estat_tempo/sondas.json")
    args = p.parse_args(argv)

    from scripts.estudo_ssl import ler_npz

    if args.cmd == "extrair":
        from scripts.estudo_ssl import carregar_rede

        extrair(Path(args.npz), Path(args.saida), *carregar_rede(args.dispositivo))
        print(f"Pronto: {args.saida}")
        return 0
    avaliar = {Path(a).stem: ler_npz(Path(a)) for a in args.avaliar}
    r = sondar(ler_npz(Path(args.treino)), ler_npz(Path(args.dev)), avaliar)
    print(f"{'representação':18s} {'dim':>5s} {'EER dev':>8s} {'AUC dev':>8s} "
          + " ".join(f"{'AUC ' + n:>16s}" for n in avaliar) + "     A12")
    for rep, x in r.items():
        a12 = x["dev"]["classes"].get("A12", {}).get("auc")
        print(f"{rep:18s} {x['dim']:5d} {100 * x['eer_dev']:7.2f}% {x['auc_media_dev']:8.3f} "
              + " ".join(f"{x['avaliar'][n]['auc']:16.3f}" for n in avaliar)
              + (f" {a12:7.3f}" if a12 is not None else ""))
    c = None
    if args.regua_2019 in avaliar:
        c = criterio(r, args.regua_2019)
        print(f"\nCritério (ganho mínimo de {MARGEM} de AUC sobre a média e o desvio, no dev e na 2019):")
        for rep, x in c.items():
            print(f"  {rep:12s} sobre {x['base']:18s} dev {x['ganho_dev']:+.3f}, 2019 {x['ganho_2019']:+.3f} -> "
                  f"{'passa' if x['passa'] else 'não passa'}")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({"sondas": r, "criterio": c}, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    print(f"\nResultado: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
