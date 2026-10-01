"""min t-DCF (ASVspoof 2019) a partir dos scores que o `evaluate.py` já salvou.

O t-DCF (Kinnunen et al., 2018) avalia a contramedida em série com o ASV fixo
dos organizadores; é a métrica oficial do ASVspoof 2019. Lê
`outputs/<modelo>_eval_scores.npz` (sem refazer inferência); os scores ASV vêm
no pacote LA da base:

    data/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt

Uso (de dentro de `ml/`):

    python scripts/tdcf.py --asv-scores <arquivo ASV> \\
        --scores outputs/baseline_lfcc_cnn_v2_eval_scores.npz \\
                 outputs/fusion_lcnn_v4_eval_scores.npz

    # fusão de scores (mesma regra de postos do score_fusion.py):
    python scripts/tdcf.py --asv-scores <arquivo ASV> --fundir \\
        --scores outputs/baseline_lfcc_cnn_v2_eval_scores.npz \\
                 outputs/fusion_lcnn_v4_eval_scores.npz

O score de spoof é negado (o t-DCF espera maior = bonafide) e a curva DET só é
avaliada nos limiares distintos, para que empates não criem pontos de operação
falsos. `--exportar` grava os scores no formato do script oficial.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Parâmetros do ASVspoof 2019 (plano de avaliação; os mesmos do script oficial).
CUSTOS_2019 = {
    "Pspoof": 0.05,
    "Cmiss_asv": 1.0,
    "Cfa_asv": 10.0,
    "Cmiss_cm": 1.0,
    "Cfa_cm": 10.0,
}
CUSTOS_2019["Ptar"] = (1 - CUSTOS_2019["Pspoof"]) * 0.99
CUSTOS_2019["Pnon"] = (1 - CUSTOS_2019["Pspoof"]) * 0.01

CHAVES_ASV = ("target", "nontarget", "spoof")


def ler_scores_asv(caminho: str | Path) -> dict[str, np.ndarray]:
    """Lê o arquivo de scores ASV da base: {'target', 'nontarget', 'spoof'}.

    Chave pelo token, score no último: aceita as variantes de 3 e 4 colunas.
    """
    grupos: dict[str, list[float]] = {k: [] for k in CHAVES_ASV}
    with open(caminho, encoding="utf-8") as fh:
        for n, linha in enumerate(fh, 1):
            partes = linha.split()
            if not partes:
                continue
            chave = next((p for p in partes if p in CHAVES_ASV), None)
            if chave is None:
                raise ValueError(f"{caminho}:{n}: linha sem chave target/nontarget/spoof")
            grupos[chave].append(float(partes[-1]))
    vazios = [k for k, v in grupos.items() if not v]
    if vazios:
        raise ValueError(f"{caminho}: nenhum score ASV do tipo {', '.join(vazios)}")
    return {k: np.asarray(v, dtype=float) for k, v in grupos.items()}


def curva_det(positivos: np.ndarray, negativos: np.ndarray):
    """(Pmiss, Pfa, limiares), com positivo quando score >= limiar.

    Só nos limiares distintos, mais +inf: nenhum limiar separa scores idênticos.
    """
    positivos = np.sort(np.asarray(positivos, dtype=float))
    negativos = np.sort(np.asarray(negativos, dtype=float))
    limiares = np.append(np.unique(np.concatenate([positivos, negativos])), np.inf)
    pmiss = np.searchsorted(positivos, limiares, side="left") / positivos.size
    pfa = 1.0 - np.searchsorted(negativos, limiares, side="left") / negativos.size
    return pmiss, pfa, limiares


def eer_e_limiar(positivos: np.ndarray, negativos: np.ndarray) -> tuple[float, float]:
    pmiss, pfa, limiares = curva_det(positivos, negativos)
    i = int(np.argmin(np.abs(pmiss - pfa)))
    return float((pmiss[i] + pfa[i]) / 2), float(limiares[i])


def min_tdcf(cm_bonafide: np.ndarray, cm_spoof: np.ndarray,
             asv: dict[str, np.ndarray], custos: dict | None = None) -> dict:
    """min t-DCF normalizado (versão de 2019), com o ASV no seu limiar de EER."""
    c = custos or CUSTOS_2019
    _, limiar_asv = eer_e_limiar(asv["target"], asv["nontarget"])
    pfa_asv = float(np.mean(asv["nontarget"] >= limiar_asv))
    pmiss_asv = float(np.mean(asv["target"] < limiar_asv))
    pmiss_spoof_asv = float(np.mean(asv["spoof"] < limiar_asv))

    c1 = c["Ptar"] * (c["Cmiss_cm"] - c["Cmiss_asv"] * pmiss_asv) \
        - c["Pnon"] * c["Cfa_asv"] * pfa_asv
    c2 = c["Cfa_cm"] * c["Pspoof"] * (1 - pmiss_spoof_asv)
    if c1 < 0 or c2 < 0:
        raise ValueError("custos negativos: o ASV fornecido não é um ASV razoável "
                         "(confira se o arquivo de scores ASV é o certo)")

    pmiss_cm, pfa_cm, _ = curva_det(cm_bonafide, cm_spoof)
    tdcf = (c1 * pmiss_cm + c2 * pfa_cm) / min(c1, c2)
    eer_cm, _ = eer_e_limiar(cm_bonafide, cm_spoof)
    return {
        "min_tdcf": float(tdcf.min()),
        "eer_cm": eer_cm,
        "asv": {"pfa": pfa_asv, "pmiss": pmiss_asv, "pmiss_spoof": pmiss_spoof_asv},
    }


def postos_medios(x: np.ndarray) -> np.ndarray:
    """Postos em [0, 1] com empate recebendo o posto médio (como no score_fusion)."""
    ordem = np.argsort(x, kind="mergesort")
    ordenado = x[ordem]
    postos = np.empty(x.size, dtype=float)
    inicio = 0
    while inicio < x.size:
        fim = inicio
        while fim + 1 < x.size and ordenado[fim + 1] == ordenado[inicio]:
            fim += 1
        postos[ordem[inicio:fim + 1]] = (inicio + fim) / 2
        inicio = fim + 1
    return postos / max(x.size - 1, 1)


def ler_npz(caminho: str | Path):
    """(ids, labels, score de spoof, systems) de um `.npz` do `evaluate.py`.

    Usa o `logodds` quando existe. Arquivo antigo, só com a probabilidade, é
    recusado se algum bonafide saturou em 1,0 (mesma regra de `src/scores.py`).
    """
    dados = np.load(caminho, allow_pickle=False)
    labels = np.asarray(dados["labels"]).astype(int)
    if "logodds" in dados.files:
        score = np.asarray(dados["logodds"], dtype=float)
    else:
        score = np.asarray(dados["scores"], dtype=float)
        if (score[labels == 0] >= 1.0).any():
            raise ValueError(
                f"{caminho}: formato antigo com bonafide saturado em 1,0; "
                "rode o evaluate.py de novo para gravar os log-odds")
    return dados["ids"], labels, score, dados["systems"]


def score_bonafide(score_spoof: np.ndarray) -> np.ndarray:
    """Converte score de spoof para a convenção do t-DCF (maior = bonafide)."""
    return -np.asarray(score_spoof, dtype=float)


def exportar_formato_oficial(caminho, ids, systems, labels, score_bona) -> None:
    with open(caminho, "w", encoding="utf-8") as fh:
        for uid, sis, lab, s in zip(ids, systems, labels, score_bona):
            chave = "spoof" if lab == 1 else "bonafide"
            fh.write(f"{uid} {sis} {chave} {s:.10f}\n")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--asv-scores", required=True,
                   help="ASVspoof2019.LA.asv.eval.gi.trl.scores.txt (vem no LA.zip)")
    p.add_argument("--scores", nargs="+", required=True,
                   help="um ou mais outputs/<modelo>_eval_scores.npz")
    p.add_argument("--fundir", action="store_true",
                   help="calcula também a fusão por postos dos modelos dados")
    p.add_argument("--exportar", metavar="PASTA",
                   help="grava os scores no formato do script oficial nesta pasta")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    asv = ler_scores_asv(args.asv_scores)
    print(f"ASV: {asv['target'].size} target, {asv['nontarget'].size} nontarget, "
          f"{asv['spoof'].size} spoof")

    modelos = []
    for caminho in args.scores:
        ids, labels, s_spoof, systems = ler_npz(caminho)
        nome = Path(caminho).name.removesuffix("_eval_scores.npz").removesuffix(".npz")
        modelos.append((nome, ids, labels, s_spoof, systems))

    if args.fundir:
        if len(modelos) < 2:
            raise SystemExit("--fundir precisa de pelo menos dois --scores")
        ref = modelos[0][1]
        for nome, ids, *_ in modelos[1:]:
            if not np.array_equal(ids, ref):
                raise SystemExit(f"{nome} não avaliou a mesma lista de áudios")
        fundido = np.mean([postos_medios(m[3]) for m in modelos], axis=0)
        nome = " + ".join(m[0] for m in modelos) + " (postos)"
        modelos.append((nome, ref, modelos[0][2], fundido, modelos[0][4]))

    print(f"\n{'modelo':52s} {'EER CM':>8s} {'min t-DCF':>10s}")
    for nome, ids, labels, s_spoof, systems in modelos:
        s = score_bonafide(s_spoof)
        r = min_tdcf(s[labels == 0], s[labels == 1], asv)
        print(f"{nome[:52]:52s} {100 * r['eer_cm']:7.2f}% {r['min_tdcf']:10.4f}")
        if args.exportar:
            pasta = Path(args.exportar)
            pasta.mkdir(parents=True, exist_ok=True)
            arquivo = pasta / f"{nome.split(' ')[0]}_cm_scores_oficial.txt"
            exportar_formato_oficial(arquivo, ids, systems, labels, s)

    e = r["asv"]
    print(f"\nASV no limiar de EER: Pfa={e['pfa']:.4f}  Pmiss={e['pmiss']:.4f}  "
          f"Pmiss_spoof={e['pmiss_spoof']:.4f}")
    print("Referência (Todisco et al., 2019, Tab. 1): B02 0,2116 · B01 0,2366")


if __name__ == "__main__":
    main()
