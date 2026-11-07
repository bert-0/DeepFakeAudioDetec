"""XLS-R congelado com pesos por camada e ramo LFCC: o modelo final (RESUMO_TCC §10.2.43).

O grupo decidiu (27/10/2026) que o modelo final usa o XLS-R. Nas réguas, a melhor
camada muda com o canal: quase todas na ASVspoof, a 4 no português de audiolivro,
da 10 à 17 na regravação (§10.2.40, §10.2.42). Aqui o XLS-R fica congelado e uma
cabeça pequena aprende quanto pesar cada uma das 25 camadas (soma ponderada, como
no `fusion_ssl` do plano), com um ramo LFCC somado no fim. Ela treina sobre as
médias no tempo que o `estudo_ssl.py extrair` já grava, em minutos.

Cada domínio de treino pesa o mesmo, e as duas classes pesam o mesmo dentro dele
(um domínio só de vozes reais, como o Common Voice, entra só com elas). A época
escolhida é a de menor perda média nos devs, cada dev com as classes equilibradas.

Subcomandos (de dentro de ml/):
    treinar   --treino NOME=NPZ ... --dev NOME=NPZ ... --saida checkpoints/xlsr_congelado.pt
    avaliar   --modelo ... --regua NOME=NPZ ... [--calibrar NPZ] [--ensaio NOME]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Fixados antes do treino (§10.2.43).
LR, DECAIMENTO, LOTE, EPOCAS, PACIENCIA, DROPOUT = 3e-3, 1e-2, 256, 60, 8, 0.2


def ler(caminho: str | Path) -> dict:
    from scripts.estudo_ssl import ler_npz

    d = ler_npz(Path(caminho))
    return {"camadas": d["camadas"], "lfcc": d["lfcc"].astype(np.float32), "rotulo": d["rotulo"].astype(np.int64),
            "classe": d["classe"], "arquivo": d["arquivo"]}


def pesos_por_dominio(dominios: list[dict]) -> list[np.ndarray]:
    """Peso de cada amostra: cada domínio soma 1, dividido igualmente entre as classes que ele tem."""
    saida = []
    for d in dominios:
        y = d["rotulo"]
        classes = np.unique(y)
        w = np.zeros(len(y), np.float32)
        for c in classes:
            w[y == c] = 1.0 / (len(classes) * (y == c).sum())
        saida.append(w)
    return saida


def criar_cabeca(n_camadas: int = 25, dim: int = 1024, dim_lfcc: int = 120, dropout: float = DROPOUT):
    """A cabeça mora em src/xlsr_final.py, que a interface usa."""
    from src.xlsr_final import criar_cabeca as criar

    return criar(n_camadas, dim, dim_lfcc, dropout)


def _normalizacao(treino: list[dict]):
    """Média e desvio por camada e dimensão, nas amostras de treino de todos os domínios."""
    soma = soma2 = None
    n = 0
    for d in treino:
        for i in range(0, len(d["rotulo"]), 2048):
            x = d["camadas"][i:i + 2048].astype(np.float64)
            soma = x.sum(axis=0) if soma is None else soma + x.sum(axis=0)
            soma2 = (x ** 2).sum(axis=0) if soma2 is None else soma2 + (x ** 2).sum(axis=0)
            n += len(x)
    media = soma / n
    escala = np.sqrt(np.maximum(soma2 / n - media ** 2, 0)) + 1e-4
    lfcc = np.concatenate([d["lfcc"] for d in treino])
    return media.astype(np.float32), escala.astype(np.float32), lfcc.mean(axis=0), lfcc.std(axis=0) + 1e-4


def _perda(cabeca, d: dict, w: np.ndarray, dispositivo, lote: int = 1024) -> float:
    import torch

    cabeca.eval()
    soma = 0.0
    with torch.no_grad():
        for i in range(0, len(d["rotulo"]), lote):
            c = torch.from_numpy(d["camadas"][i:i + lote].astype(np.float32)).to(dispositivo)
            z = cabeca(c, torch.from_numpy(d["lfcc"][i:i + lote]).to(dispositivo))
            y = torch.from_numpy(d["rotulo"][i:i + lote].astype(np.float32)).to(dispositivo)
            ww = torch.from_numpy(w[i:i + lote]).to(dispositivo)
            soma += float((torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none") * ww).sum())
    return soma


def treinar(treino: dict[str, dict], devs: dict[str, dict], saida: Path, semente: int = 42,
            dispositivo: str = "cuda", epocas: int = EPOCAS) -> dict:
    import torch

    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    dev_t = torch.device(dispositivo if torch.cuda.is_available() or dispositivo == "cpu" else "cpu")
    nomes = list(treino)
    dominios = [treino[n] for n in nomes]
    w_treino = pesos_por_dominio(dominios)
    devs_l = list(devs.values())
    w_devs = pesos_por_dominio(devs_l)
    d0 = dominios[0]
    cabeca = criar_cabeca(d0["camadas"].shape[1], d0["camadas"].shape[2], d0["lfcc"].shape[1]).to(dev_t)
    media, escala, media_l, escala_l = _normalizacao(dominios)
    for nome, v in (("media", media), ("escala", escala), ("media_lfcc", media_l), ("escala_lfcc", escala_l)):
        getattr(cabeca, nome).copy_(torch.from_numpy(v))
    # Sem decaimento nos pesos das camadas: ele os puxava de volta para o uniforme.
    otim = torch.optim.AdamW([{"params": [cabeca.pesos], "weight_decay": 0.0},
                              {"params": [q for n, q in cabeca.named_parameters() if n != "pesos"]}],
                             lr=LR, weight_decay=DECAIMENTO)
    indice = [(k, i) for k, d in enumerate(dominios) for i in range(len(d["rotulo"]))]
    historico, melhor, sem_melhora = [], float("inf"), 0
    for epoca in range(1, epocas + 1):
        cabeca.train()
        ordem = rng.permutation(len(indice))
        for b in range(0, len(ordem), LOTE):
            sel = ordem[b:b + LOTE]
            por_dom: dict[int, list[int]] = {}
            for j in sel:
                k, i = indice[j]
                por_dom.setdefault(k, []).append(i)
            cs, ls, ys, ws = [], [], [], []
            for k, idx in por_dom.items():
                idx = np.sort(idx)
                cs.append(dominios[k]["camadas"][idx].astype(np.float32))
                ls.append(dominios[k]["lfcc"][idx])
                ys.append(dominios[k]["rotulo"][idx].astype(np.float32))
                ws.append(w_treino[k][idx])
            c = torch.from_numpy(np.concatenate(cs)).to(dev_t)
            z = cabeca(c, torch.from_numpy(np.concatenate(ls)).to(dev_t))
            y = torch.from_numpy(np.concatenate(ys)).to(dev_t)
            ww = torch.from_numpy(np.concatenate(ws)).to(dev_t)
            perda = (torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none") * ww).sum()
            perda = perda / ww.sum()
            otim.zero_grad(set_to_none=True)
            perda.backward()
            otim.step()
        perdas_dev = [_perda(cabeca, d, w, dev_t) for d, w in zip(devs_l, w_devs, strict=True)]
        perda_dev = float(np.mean(perdas_dev))
        pesos_camadas = torch.softmax(cabeca.pesos.detach(), 0).cpu().numpy()
        historico.append({"epoca": epoca, "perda_dev": perda_dev, "perdas_dev": dict(zip(devs, perdas_dev)),
                          "camadas_mais_pesadas": [int(i) for i in np.argsort(pesos_camadas)[::-1][:5]]})
        if perda_dev < melhor:
            melhor, sem_melhora = perda_dev, 0
            saida.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"estado": cabeca.state_dict(), "config": cabeca.config, "epoca": epoca, "perda_dev": perda_dev,
                        "treino": nomes, "devs": list(devs)}, saida)
        else:
            sem_melhora += 1
        print(f"época {epoca:3d}: dev {perda_dev:.4f} ({', '.join(f'{n} {p:.3f}' for n, p in zip(devs, perdas_dev))}); "
              f"camadas mais pesadas {historico[-1]['camadas_mais_pesadas']}", flush=True)
        if sem_melhora >= PACIENCIA:
            break
    return {"historico": historico, "melhor_perda_dev": melhor}


def carregar(caminho: Path, dispositivo: str = "cuda"):
    from src.xlsr_final import carregar_cabeca

    return carregar_cabeca(caminho, dispositivo)


def pontuar(cabeca, d: dict, dispositivo, lote: int = 1024) -> np.ndarray:
    import torch

    saida = []
    with torch.no_grad():
        for i in range(0, len(d["rotulo"]), lote):
            c = torch.from_numpy(d["camadas"][i:i + lote].astype(np.float32)).to(dispositivo)
            saida.append(cabeca(c, torch.from_numpy(d["lfcc"][i:i + lote]).to(dispositivo)).cpu().numpy())
    return np.concatenate(saida)


def veredito_calibrado(scores: np.ndarray, grupos: np.ndarray, calibracao: np.ndarray, real: str = "real") -> dict:
    """Regra do §10.2.39: o limiar é o maior score das vozes reais de outro locutor no
    mesmo canal; acima dele, sintético. Devolve os acertos por grupo e o limiar."""
    limiar = float(np.max(calibracao))
    r = {"limiar": limiar}
    for g in sorted(set(grupos.tolist())):
        s = scores[grupos == g]
        r[g] = int((s <= limiar).sum()) if g == real else int((s > limiar).sum())
    return r


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="XLS-R congelado com pesos por camada (modelo final)")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("treinar")
    t.add_argument("--treino", action="append", required=True, metavar="NOME=NPZ")
    t.add_argument("--dev", action="append", required=True, metavar="NOME=NPZ")
    t.add_argument("--saida", required=True)
    t.add_argument("--dispositivo", default="cuda")
    t.add_argument("--epocas", type=int, default=EPOCAS)
    v = sub.add_parser("avaliar")
    v.add_argument("--modelo", required=True)
    v.add_argument("--dev", required=True, help="dá o limiar genérico (ponto de EER)")
    v.add_argument("--regua", action="append", default=[], metavar="NOME=NPZ")
    v.add_argument("--calibrar", help="NPZ com as vozes reais de outro locutor no canal do ensaio (L02)")
    v.add_argument("--ensaio", default="ensaio5", help="a régua em que a calibração é aplicada")
    v.add_argument("--dispositivo", default="cuda")
    v.add_argument("--json", default="outputs/xlsr_congelado.json")
    args = p.parse_args(argv)

    par = lambda itens: {i.split("=", 1)[0]: ler(i.split("=", 1)[1]) for i in itens}  # noqa: E731
    if args.cmd == "treinar":
        r = treinar(par(args.treino), par(args.dev), Path(args.saida), dispositivo=args.dispositivo, epocas=args.epocas)
        Path(args.saida).with_suffix(".json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(f"\nMelhor perda no dev: {r['melhor_perda_dev']:.4f}. Modelo: {args.saida}")
        return 0

    import torch

    from scripts.estudo_ssl import eer, medir

    cabeca, ck, dev_t = carregar(Path(args.modelo), args.dispositivo)
    dev = ler(args.dev)
    s_dev = pontuar(cabeca, dev, dev_t)
    e, limiar = eer(s_dev[dev["rotulo"] == 0], s_dev[dev["rotulo"] == 1])
    pesos = torch.softmax(cabeca.pesos.detach(), 0).cpu().numpy()
    r = {"epoca": ck["epoca"], "eer_dev": e, "limiar": limiar, "pesos_camadas": pesos.tolist(), "reguas": {},
         "scores": {}, "arquivos": {}}
    print(f"Modelo da época {ck['epoca']}: EER no dev {100 * e:.2f}%; camadas mais pesadas: "
          + ", ".join(f"{i} ({pesos[i]:.2f})" for i in np.argsort(pesos)[::-1][:6]))
    reguas = par(args.regua)
    for nome, d in reguas.items():
        s = pontuar(cabeca, d, dev_t)
        r["reguas"][nome] = medir(s, d["classe"], d["rotulo"], limiar)
        r["scores"][nome], r["arquivos"][nome] = s.tolist(), [str(a) for a in d["arquivo"]]
        m = r["reguas"][nome]
        print(f"  {nome}: AUC {m['auc'] if m['auc'] is None else round(m['auc'], 3)}; " + ", ".join(
            f"{c} {x['certos']}/{x['n']}" + ("" if x["auc"] is None else f" ({x['auc']:.2f})")
            for c, x in m["classes"].items()))
    if args.calibrar and args.ensaio in reguas:
        cal = pontuar(cabeca, ler(args.calibrar), dev_t)
        classes = np.asarray(reguas[args.ensaio]["classe"])
        r["calibrado"] = veredito_calibrado(np.array(r["scores"][args.ensaio]), classes, cal,
                                            real=next(c for c in sorted(set(classes.tolist())) if "real" in c))
        print(f"  {args.ensaio} com o limiar do L02: {r['calibrado']}")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    print(f"Resultado: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
