"""Parte 10: o modelo de fala natural se "surpreende" com a fala sintética?

O XLS-R aprendeu as camadas que separam os geradores (parte 6) sem nunca ver um
áudio falso: no pré-treino, ele esconde trechos de fala humana e tenta
reconhecer o trecho verdadeiro entre distratores do mesmo áudio. Se a fala
sintética viola regularidades da fala natural, essa tarefa deve dar errado de
um jeito diferente nela. Aqui a tarefa do pré-treino vira medida, com o próprio
XLS-R (pesos da Meta, congelados, instrumento de pesquisa):

- máscaras como no pré-treino (início em 7,5% dos quadros, 10 quadros cada),
  repetidas com sementes fixas;
- por quadro escondido: perda contrastiva contra 100 distratores do mesmo
  áudio, acerto (o verdadeiro em 1º) e similaridade de cosseno com o verdadeiro;
- por arquivo, a média; separação pela AUC (falsos contra humanos).

Critério combinado antes de rodar: o XLS-R "estranha" a fala sintética de forma
consistente se a perda separar falsos de humanos com |AUC - 0,5| >= 0,2 no dev
da ASVspoof 5 e na 2019 LA, no mesmo sentido nas duas. Se valer, o princípio
(um modelo só de fala natural detecta o que foge dela) vira proposta para um
modelo nosso. Estudo exploratório (RESUMO_TCC §10.2.26): não muda nenhum placar.

Uso (de dentro de ml/, com HF_HOME no cache do XLS-R):
    python scripts/estudo_surpresa.py --npz outputs/ssl/asv5_dev.npz \\
        --npz outputs/ssl/asv19_train.npz --npz outputs/ssl/ensaios.npz --n-por-classe 80
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELO_HF = "facebook/wav2vec2-xls-r-300m"
REPETICOES = 6
DISTANCIA = 0.2   # |AUC - 0,5| exigido pelo critério


def carregar(dispositivo: str = "cuda", modelo: str = MODELO_HF):
    import torch
    from transformers import Wav2Vec2ForPreTraining

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    dev = torch.device(dispositivo if torch.cuda.is_available() or dispositivo == "cpu" else "cpu")
    return Wav2Vec2ForPreTraining.from_pretrained(modelo).eval().to(dev), dev


def surpresa_janela(rede, dev, x: np.ndarray, rng: np.random.Generator,
                    repeticoes: int = REPETICOES) -> dict[str, float]:
    """Perda, acerto e similaridade médios nos quadros escondidos de uma janela."""
    import torch
    from transformers.models.wav2vec2.modeling_wav2vec2 import (_compute_mask_indices,
                                                                _sample_negative_indices)

    cfg = rede.config
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    entrada = torch.from_numpy(x).float()[None].to(dev)
    t = int(rede.wav2vec2._get_feat_extract_output_lengths(len(x)))
    perdas, acertos, sims, contagem, perda_modelo = 0.0, 0.0, 0.0, 0, 0.0
    for _ in range(repeticoes):
        semente = int(rng.integers(2 ** 31))
        np.random.seed(semente)  # as duas funções do transformers sorteiam pelo numpy global
        mascara = _compute_mask_indices((1, t), cfg.mask_time_prob, cfg.mask_time_length, min_masks=2)
        negativos = _sample_negative_indices((1, t), cfg.num_negatives, mascara)
        m = torch.from_numpy(mascara).to(dev)
        with torch.inference_mode():
            s = rede(entrada, mask_time_indices=m,
                     sampled_negative_indices=torch.from_numpy(negativos).long().to(dev))
        c = s.projected_states[0].float()           # (T, D): o que o contexto prevê
        q = s.projected_quantized_states[0].float()  # (T, D): o trecho verdadeiro
        idx = np.where(mascara[0])[0]
        neg = q[torch.from_numpy(negativos[0, idx]).long().to(dev)]   # (M, K, D)
        alvo = torch.cat([q[idx][:, None], neg], dim=1)                 # (M, 1+K, D)
        logits = torch.nn.functional.cosine_similarity(c[idx][:, None], alvo, dim=-1)
        igual = (neg == q[idx][:, None]).all(-1)
        logits[:, 1:][igual] = float("-inf")
        logits = logits / cfg.contrastive_logits_temperature
        perdas += float(torch.nn.functional.cross_entropy(logits, torch.zeros(len(idx), dtype=torch.long,
                                                                               device=dev), reduction="sum"))
        acertos += float((logits.argmax(dim=1) == 0).sum())
        sims += float(torch.nn.functional.cosine_similarity(c[idx], q[idx], dim=-1).sum())
        contagem += len(idx)
        perda_modelo += float(s.contrastive_loss)
    return {"perda": perdas / contagem, "acerto": acertos / contagem, "similaridade": sims / contagem,
            "perda_modelo": perda_modelo / contagem, "quadros": contagem}


def surpresa_arquivo(rede, dev, caminho: Path, rng: np.random.Generator) -> dict[str, float]:
    from scripts.estudo_ssl import janelas, preparar

    wav, _ = preparar(caminho)
    partes = [surpresa_janela(rede, dev, x, rng) for x in janelas(wav)]
    total = sum(p["quadros"] for p in partes)
    return {k: sum(p[k] * p["quadros"] for p in partes) / total
            for k in ("perda", "acerto", "similaridade", "perda_modelo")}


def resumir(registros: list[dict]) -> dict:
    """AUC (falsos contra humanos do mesmo conjunto) de cada medida, no geral e por classe."""
    from scripts.estudo_pente import auc

    saida = {}
    for conj in sorted({r["conjunto"] for r in registros}):
        rs = [r for r in registros if r["conjunto"] == conj]
        humanos = [r for r in rs if r["rotulo"] == 0]
        if not humanos:
            continue
        linha = {}
        for medida in ("perda", "acerto", "similaridade"):
            h = [r[medida] for r in humanos]
            falsos = [r[medida] for r in rs if r["rotulo"] == 1]
            linha[medida] = {"auc": auc(falsos, h), "humanos": float(np.median(h)),
                             "classes": {c: {"auc": auc([r[medida] for r in rs if r["classe"] == c], h),
                                             "mediana": float(np.median([r[medida] for r in rs if r["classe"] == c]))}
                                         for c in sorted({r["classe"] for r in rs if r["rotulo"] == 1})}}
        saida[conj] = linha
    return saida


def criterio(resumo: dict, dev: str = "asv5_dev", regua_2019: str = "asv19_train") -> dict | None:
    if dev not in resumo or regua_2019 not in resumo:
        return None
    a, b = resumo[dev]["perda"]["auc"], resumo[regua_2019]["perda"]["auc"]
    mesmo = (a - 0.5) * (b - 0.5) > 0
    return {"auc_dev": a, "auc_2019": b,
            "passa": bool(mesmo and abs(a - 0.5) >= DISTANCIA and abs(b - 0.5) >= DISTANCIA),
            "sentido": "falsos surpreendem mais" if a > 0.5 else "falsos surpreendem menos"}


def combinar(registros: list[dict], detector: dict[str, float], ref: str = "asv5_dev") -> dict:
    """Detector + surpresa com pesos iguais, fixados antes: cada score vira z pelos
    humanos do dev (calibração só no dev) e os dois são somados."""
    from scripts.estudo_pente import auc

    regs = [r for r in registros if r["arquivo"] in detector]
    humanos_ref = [r for r in regs if r["conjunto"] == ref and r["rotulo"] == 0]
    norma = {}
    for chave, valores in (("detector", [detector[r["arquivo"]] for r in humanos_ref]),
                           ("perda", [r["perda"] for r in humanos_ref])):
        norma[chave] = (float(np.mean(valores)), float(np.std(valores)) + 1e-9)

    def z(chave, v):
        return (v - norma[chave][0]) / norma[chave][1]

    for r in regs:
        r["z_detector"], r["z_perda"] = z("detector", detector[r["arquivo"]]), z("perda", r["perda"])
        r["fusao"] = r["z_detector"] + r["z_perda"]
    saida = {"norma": norma, "conjuntos": {}}
    for conj in sorted({r["conjunto"] for r in regs}):
        rs = [r for r in regs if r["conjunto"] == conj]
        h = [r for r in rs if r["rotulo"] == 0]
        linha = {}
        for chave in ("z_detector", "z_perda", "fusao"):
            linha[chave] = {"auc": auc([r[chave] for r in rs if r["rotulo"] == 1], [r[chave] for r in h]),
                            "classes": {c: auc([r[chave] for r in rs if r["classe"] == c], [r[chave] for r in h])
                                        for c in sorted({r["classe"] for r in rs if r["rotulo"] == 1})}}
        saida["conjuntos"][conj] = linha
    return saida


def scores_do_detector(caminho_json: Path, conjuntos: dict[str, Path]) -> dict[str, float]:
    """{arquivo: log-odds} a partir do detector.json da parte 6 e dos .npz na mesma ordem."""
    from scripts.estudo_ssl import ler_npz

    r = json.loads(Path(caminho_json).read_text(encoding="utf-8"))["scores"]
    saida = {}
    for nome, npz in conjuntos.items():
        arquivos = [str(a) for a in ler_npz(npz)["arquivo"]]
        saida.update(dict(zip(arquivos, r[nome])))
    return saida


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "combinar":
        return main_combinar(argv[1:])
    p = argparse.ArgumentParser(description="Surpresa do modelo de fala natural (parte 10)")
    p.add_argument("--npz", action="append", required=True, help=".npz da parte 6 (lista de arquivos)")
    p.add_argument("--n-por-classe", type=int, default=80)
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--dispositivo", default="cuda")
    p.add_argument("--json", default="outputs/surpresa/surpresa.json")
    args = p.parse_args(argv)

    from scripts.estudo_ssl import ler_npz

    rng = np.random.default_rng(args.semente)
    lista = []
    for caminho in args.npz:
        d = ler_npz(Path(caminho))
        for c in sorted(set(d["classe"].tolist())):
            idx = np.where(d["classe"] == c)[0]
            for i in rng.permutation(idx)[:args.n_por_classe]:
                lista.append((Path(str(d["arquivo"][i])), str(c), int(d["rotulo"][i]), Path(caminho).stem))
    print(f"Surpresa: {len(lista)} arquivos", flush=True)
    rede, dev = carregar(args.dispositivo)
    registros = []
    for k, (arq, classe, rotulo, conj) in enumerate(lista):
        registros.append({"arquivo": str(arq), "classe": classe, "rotulo": rotulo, "conjunto": conj,
                          **surpresa_arquivo(rede, dev, arq, rng)})
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(lista)}", flush=True)
    resumo = resumir(registros)
    for conj, linha in resumo.items():
        print(f"\n{conj}:")
        for medida, x in linha.items():
            print(f"  {medida:12s} AUC {x['auc']:.3f} (humanos, mediana {x['humanos']:.3f}); "
                  + ", ".join(f"{c} {v['auc']:.2f}" for c, v in x["classes"].items()))
    c = criterio(resumo)
    if c:
        print(f"\nCritério (|AUC - 0,5| >= {DISTANCIA} na perda, no dev e na 2019, mesmo sentido): "
              f"dev {c['auc_dev']:.3f}, 2019 {c['auc_2019']:.3f} -> {'passa' if c['passa'] else 'não passa'} "
              f"({c['sentido']})")
    dif = [abs(r["perda"] - r["perda_modelo"]) for r in registros]
    print(f"Conferência com a perda do próprio modelo: maior diferença {max(dif):.2e}")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({"resumo": resumo, "criterio": c, "registros": registros},
                                          ensure_ascii=False), encoding="utf-8")
    print(f"Resultado: {args.json}")
    return 0


def main_combinar(argv) -> int:
    p = argparse.ArgumentParser(prog="estudo_surpresa.py combinar",
                                description="Detector + surpresa, pesos iguais, normalização no dev")
    p.add_argument("--surpresa", default="outputs/surpresa/surpresa.json")
    p.add_argument("--detector", default="outputs/ssl/detector.json")
    p.add_argument("--conjunto", action="append", required=True, metavar="NOME=NPZ",
                   help="nomes do detector.json e os .npz da parte 6 na mesma ordem")
    p.add_argument("--json", default="outputs/surpresa/combinar.json")
    args = p.parse_args(argv)
    conjuntos = {}
    for item in args.conjunto:
        nome, caminho = item.split("=", 1)
        conjuntos[nome] = Path(caminho)
    registros = json.loads(Path(args.surpresa).read_text(encoding="utf-8"))["registros"]
    r = combinar(registros, scores_do_detector(Path(args.detector), conjuntos))
    for conj, linha in r["conjuntos"].items():
        print(f"\n{conj}:")
        for chave, x in linha.items():
            print(f"  {chave:11s} AUC {x['auc']:.3f}; " + ", ".join(f"{c} {v:.2f}" for c, v in x["classes"].items()))
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nResultado: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
