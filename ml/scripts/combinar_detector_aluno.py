"""O aluno destilado somado ao nosso detector: ajuda mesmo?

No eval da ASVspoof 5 (RESUMO_TCC §10.2.29), o aluno ganha do detector no geral,
mas o detector ainda ganha em alguns ataques (A17, A21, A24, A25, A28). Aqui os
dois scores, arquivo a arquivo, são combinados de dois jeitos:

- principal, fixado antes: pesos iguais, cada score em z pelos humanos do dev
  (como a surpresa da parte 10);
- secundário, exploratório: uma regressão logística de dois pesos ajustada no
  dev; medida só fora dele.

Critério combinado antes de rodar (RESUMO_TCC §10.2.30): a soma de pesos iguais
"ajuda" se superar o melhor dos dois sozinhos em pelo menos 0,02 de AUC no eval
da ASVspoof 5 e na 2019 LA, sem perder mais de 0,02 nos ensaios. Limiar de cada
score: ponto de EER do dev. Por ataque, por codec, humanos acusados e clones dos
ensaios que passam.

Uso (de dentro de ml/):
    python scripts/combinar_detector_aluno.py --aluno outputs/destilacao/avaliacao_com_eval.json \\
        --detector outputs/ssl/detector.json --detector outputs/ssl/detector_eval.json \\
        --npz dev=outputs/ssl/asv5_dev.npz --npz asv19_train=outputs/ssl/asv19_train.npz \\
        --npz ensaios=outputs/ssl/ensaios.npz --npz asv5_eval=outputs/ssl/asv5_eval.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MARGEM, PERDA_MAX_ENSAIOS = 0.02, 0.02
#: Nomes dos conjuntos no avaliacao.json do aluno -> nomes usados aqui.
NOMES_ALUNO = {"dev": "dev", "asv19_train": "asv19_train", "ensaios": "ensaios", "asv5_eval_aluno": "asv5_eval"}


def juntar(aluno: dict, detectores: list[dict], npz: dict[str, dict]) -> dict[str, dict]:
    """{conjunto: {"arquivo", "classe", "rotulo", "detector", "aluno"}}, só os arquivos com os dois scores."""
    det: dict[str, dict[str, float]] = {}
    for d in detectores:
        for nome, scores in d["scores"].items():
            if nome in npz:
                det.setdefault(nome, {}).update(zip([str(a) for a in npz[nome]["arquivo"]], scores))
    sondas = aluno["sondas"]["nova"]
    saida = {}
    for nome_aluno, nome in NOMES_ALUNO.items():
        if nome not in npz or nome_aluno not in sondas["scores"]:
            continue
        al = dict(zip(aluno["sondas"]["nova"]["arquivos"][nome_aluno], sondas["scores"][nome_aluno]))
        d = npz[nome]
        sel = [i for i, a in enumerate(d["arquivo"]) if str(a) in al and str(a) in det.get(nome, {})]
        arquivos = [str(d["arquivo"][i]) for i in sel]
        saida[nome] = {"arquivo": arquivos, "classe": d["classe"][sel], "rotulo": d["rotulo"][sel],
                       "detector": np.array([det[nome][a] for a in arquivos]),
                       "aluno": np.array([al[a] for a in arquivos])}
    return saida


def combinar(conj: dict[str, dict]) -> dict[str, dict[str, np.ndarray]]:
    """Scores de cada conjunto: detector, aluno, soma (pesos iguais) e logística (ajustada no dev)."""
    from sklearn.linear_model import LogisticRegression

    dev = conj["dev"]
    h = dev["rotulo"] == 0
    norma = {k: (dev[k][h].mean(), dev[k][h].std() + 1e-9) for k in ("detector", "aluno")}

    def z(c, k):
        return (c[k] - norma[k][0]) / norma[k][1]

    x_dev = np.column_stack([z(dev, "detector"), z(dev, "aluno")])
    lr = LogisticRegression(C=1.0).fit(x_dev, dev["rotulo"])
    saida = {}
    for nome, c in conj.items():
        x = np.column_stack([z(c, "detector"), z(c, "aluno")])
        saida[nome] = {"detector": c["detector"], "aluno": c["aluno"], "soma": x.sum(axis=1),
                       "logistica": lr.decision_function(x)}
    return saida, {"norma": {k: [float(v[0]), float(v[1])] for k, v in norma.items()},
                   "pesos_logistica": lr.coef_[0].tolist()}


def medir_tudo(conj: dict, scores: dict, codecs: dict[str, str] | None = None) -> dict:
    from scripts.estudo_ssl import eer, medir, por_codec

    dev = conj["dev"]
    limiares = {k: eer(scores["dev"][k][dev["rotulo"] == 0], scores["dev"][k][dev["rotulo"] == 1])[1]
                for k in scores["dev"]}
    saida = {}
    for nome, c in conj.items():
        saida[nome] = {}
        for k, s in scores[nome].items():
            m = medir(s, c["classe"], c["rotulo"], limiares[k])
            if codecs is not None and nome == "asv5_eval":
                m["por_codec"] = por_codec(s, c["arquivo"], c["rotulo"], codecs)
            saida[nome][k] = m
    return saida


def criterio(m: dict) -> dict:
    def auc(nome, k):
        return m[nome][k]["auc"]

    saida = {}
    for nome in ("asv5_eval", "asv19_train"):
        melhor = max(auc(nome, "detector"), auc(nome, "aluno"))
        saida[nome] = {"soma": auc(nome, "soma"), "melhor_sozinho": melhor, "ganho": auc(nome, "soma") - melhor}
    melhor_e = max(auc("ensaios", "detector"), auc("ensaios", "aluno"))
    saida["ensaios"] = {"soma": auc("ensaios", "soma"), "melhor_sozinho": melhor_e,
                        "perda": melhor_e - auc("ensaios", "soma")}
    saida["passa"] = bool(saida["asv5_eval"]["ganho"] >= MARGEM and saida["asv19_train"]["ganho"] >= MARGEM
                          and saida["ensaios"]["perda"] <= PERDA_MAX_ENSAIOS)
    return saida


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Aluno destilado + detector: ajuda mesmo?")
    p.add_argument("--aluno", required=True, help="avaliacao.json do destilacao.py (com scores do dev)")
    p.add_argument("--detector", action="append", required=True, help="detector.json do estudo_ssl.py")
    p.add_argument("--npz", action="append", required=True, metavar="NOME=NPZ")
    p.add_argument("--tsv", default="data/recortes/asvspoof5_eval_amostra10000_codec_s42.tsv")
    p.add_argument("--json", default="outputs/destilacao/combinacao.json")
    args = p.parse_args(argv)

    from scripts.estudo_ssl import ler_codecs, ler_npz

    npz = {}
    for item in args.npz:
        nome, caminho = item.split("=", 1)
        npz[nome] = ler_npz(Path(caminho))
    conj = juntar(json.loads(Path(args.aluno).read_text(encoding="utf-8")),
                  [json.loads(Path(d).read_text(encoding="utf-8")) for d in args.detector], npz)
    scores, info = combinar(conj)
    codecs = ler_codecs(Path(args.tsv)) if Path(args.tsv).exists() else None
    m = medir_tudo(conj, scores, codecs)
    chaves = ["detector", "aluno", "soma", "logistica"]
    print("Arquivos com os dois scores: " + ", ".join(f"{n} {len(c['arquivo'])}" for n, c in conj.items()))
    print(f"Pesos da logística (ajustada no dev; z do detector, z do aluno): {info['pesos_logistica']}")
    print(f"\n{'conjunto':12s} " + " ".join(f"{k:>10s}" for k in chaves) + "   (AUC)")
    for nome in m:
        print(f"{nome:12s} " + " ".join(f"{m[nome][k]['auc']:10.3f}" for k in chaves))
    for nome in ("asv5_eval", "asv19_train", "dev"):
        if nome in m:
            print(f"\n{nome}, por ataque (AUC): detector / aluno / soma / logística")
            for c, v in m[nome]["soma"]["classes"].items():
                if v["falso"]:
                    print(f"  {c:5s} " + " / ".join(f"{m[nome][k]['classes'][c]['auc']:.2f}" for k in chaves))
            h = next(c for c, v in m[nome]["soma"]["classes"].items() if not v["falso"])
            print("  humanos abaixo do limiar do dev: " + " / ".join(
                f"{m[nome][k]['classes'][h]['certos']}/{m[nome][k]['classes'][h]['n']}" for k in chaves))
    if "asv5_eval" in m and "por_codec" in m["asv5_eval"]["soma"]:
        print("\nasv5_eval por codec (AUC): detector / aluno / soma")
        for c, v in m["asv5_eval"]["soma"]["por_codec"].items():
            print(f"  {c:4s} {v['nome']:28s} " + " / ".join(
                f"{m['asv5_eval'][k]['por_codec'][c]['auc']:.3f}" for k in ("detector", "aluno", "soma")))
    if "ensaios" in m:
        print("\nensaios, certos pelo limiar do dev: detector / aluno / soma")
        for c in m["ensaios"]["soma"]["classes"]:
            print(f"  {c:16s} " + " / ".join(
                f"{m['ensaios'][k]['classes'][c]['certos']}/{m['ensaios'][k]['classes'][c]['n']}"
                for k in ("detector", "aluno", "soma")))
    crit = None
    if all(n in m for n in ("asv5_eval", "asv19_train", "ensaios")):
        crit = criterio(m)
        print(f"\nCritério (soma >= melhor sozinho + {MARGEM} no eval e na 2019; ensaios sem perder mais de "
              f"{PERDA_MAX_ENSAIOS}): eval {crit['asv5_eval']['ganho']:+.3f}, 2019 {crit['asv19_train']['ganho']:+.3f}, "
              f"ensaios {-crit['ensaios']['perda']:+.3f} -> {'passa' if crit['passa'] else 'não passa'}")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({"info": info, "medidas": m, "criterio": crit}, ensure_ascii=False,
                                          indent=1, default=float), encoding="utf-8")
    print(f"Resultado: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
