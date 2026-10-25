"""Limiar da fusão pela média de dois modelos, escolhido nos devs.

O limiar sai só dos humanos dos devs, na captura FIR, com uma taxa-alvo de
alarme falso: o lado humano se mantém do dev para o eval, o dos ataques não
(RESUMO_TCC §10.2.9). Os áudios do teste com voz real nunca entram aqui.

A regra padrão é a média dos log-odds, escolhida nos devs em 24/10/2026: na
média das probabilidades o modelo do recorte quase não pesa (RESUMO_TCC
§10.2.12). `--regra probabilidade` refaz a escolha com a regra antiga.

Dois passos:
- `preparar`: para cada modelo e cada dev dos modelos (o da 2019 e o do recorte
  da ASVspoof 5), escreve um config derivado que avalia o modelo naquele dev e
  mostra os comandos do `robustness_eval.py` (quem roda é o usuário).
- `escolher`: lê os scores que eles salvaram, funde pela média, escolhe o limiar
  com os humanos dos dois devs juntos e grava o resultado, com a impressão dos
  pesos de cada checkpoint (o `teste_voz_propria.py` confere).

Uso (de dentro de ml/):
    python scripts/limiar_fusao.py preparar
    (os quatro comandos que ele mostrar)
    python scripts/limiar_fusao.py escolher
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.fusao_ao_vivo import alinhar, limiar_por_alarme_falso, ponto_de_operacao  # noqa: E402
from scripts.score_fusion import combine  # noqa: E402
from src.config import config_derivado, load_config, salvar_config  # noqa: E402
from src.limiares import LimiarFusao  # noqa: E402
from src.metrics import compute_eer  # noqa: E402

MODELOS_PADRAO = (
    ("configs/fusion_v4_2019_degr.yaml", "checkpoints/fusion_lcnn_v4_2019_degr.pt"),
    ("configs/fusion_v4_asv5.yaml", "checkpoints/fusion_lcnn_v4_asv5.pt"),
)
PASTA = Path("outputs/limiar_fusao")
CONDICAO = "captura_48k_fir"
#: 10 mil, como as outras réguas; o dev do recorte (9.999) entra inteiro.
AMOSTRA = 10000
#: Perto do que a fusão ao vivo anterior deu no eval (10,6%, §10.2.9).
ALVO_PADRAO = 0.10
REGRA_PADRAO = "logodds"


def rotulo_do_dev(protocolo: str) -> str:
    # Sem "_dev" no rótulo: o `carregar` do fusao_ao_vivo corta o nome ali.
    if "ASVspoof5" in protocolo:
        return "calib_asv5"
    if "2019" in protocolo:
        return "calib_2019"
    return "calib_" + "".join(c if c.isalnum() else "-" for c in Path(protocolo).stem)


def caminho_scores(nome: str, amostra: int, condicao: str) -> Path:
    """O .npz que o robustness_eval.py grava com --partition eval, --amostra e --so."""
    return Path("outputs") / f"{nome}_amostra{amostra}_{condicao}_eval_{condicao}_scores.npz"


def preparar(modelos, pasta: Path = PASTA, amostra: int = AMOSTRA,
             condicao: str = CONDICAO) -> dict:
    configs = [(load_config(c), c, k) for c, k in modelos]
    devs: dict[str, tuple[str, str]] = {}
    for cfg, _, _ in configs:
        proto = cfg["data"]["protocols"]["dev"]
        devs.setdefault(rotulo_do_dev(proto), (proto, cfg["data"]["audio_dir"]["dev"]))

    preparo = {"amostra": amostra, "condicao": condicao, "modelos": [], "devs": {}}
    for rotulo, (proto, audio) in devs.items():
        preparo["devs"][rotulo] = {"protocolo": proto, "audio": audio, "execucoes": []}
        for cfg, caminho, checkpoint in configs:
            derivado = config_derivado(cfg, rotulo, proto, audio)
            destino = salvar_config(derivado, pasta / f"{derivado['experiment']['name']}.yaml")
            preparo["devs"][rotulo]["execucoes"].append({
                "config": destino.as_posix(), "checkpoint": checkpoint,
                "scores": caminho_scores(derivado["experiment"]["name"], amostra,
                                         condicao).as_posix()})
    preparo["modelos"] = [{"config": c, "checkpoint": k} for _, c, k in configs]
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "preparo.json").write_text(json.dumps(preparo, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    return preparo


def comandos(preparo: dict) -> list[str]:
    return [f"python scripts\\robustness_eval.py --config {Path(e['config'])} "
            f"--checkpoint {Path(e['checkpoint'])} --amostra {preparo['amostra']} "
            f"--so {preparo['condicao']}"
            for d in preparo["devs"].values() for e in d["execucoes"]]


def _carregar(caminho: str | Path) -> dict:
    with np.load(caminho, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def escolher(pares: dict[str, tuple[dict, ...]], alvo: float,
             regra: str = REGRA_PADRAO) -> dict:
    """Funde os modelos de cada dev pela média e escolhe o limiar com os humanos de
    todos os devs. Com um modelo só, a "fusão" é o próprio modelo.

    O limiar fica na escala da regra; `limiar_probabilidade` é o mesmo ponto na
    escala do score das leituras ao vivo, em [0, 1]."""
    chave = {"logodds": "logodds", "probabilidade": "scores"}[regra]
    fundidos = {}
    for rotulo, modelos in pares.items():
        if len(modelos) == 2:
            modelos = alinhar(*modelos)
            if not (modelos[0]["labels"] == modelos[1]["labels"]).all():
                raise ValueError(f"{rotulo}: rótulos diferentes para os mesmos áudios")
        elif len(modelos) != 1:
            raise ValueError("a fusão é de um ou dois modelos")
        fundidos[rotulo] = (modelos, combine([d[chave] for d in modelos], "mean"))
    labels = np.concatenate([m[0]["labels"] for m, _ in fundidos.values()])
    media = np.concatenate([f for _, f in fundidos.values()])
    limiar = limiar_por_alarme_falso(labels, media, alvo)

    devs = {}
    for rotulo, (modelos, m) in fundidos.items():
        rotulos = modelos[0]["labels"]
        devs[rotulo] = {"n": int(len(m)), "humanos": int((rotulos == 0).sum()),
                        "eer_modelos": [compute_eer(rotulos, d["logodds"]) for d in modelos],
                        "eer_fusao": compute_eer(rotulos, m),
                        **ponto_de_operacao(rotulos, m, limiar)}
    return {"regra": regra, "limiar": limiar,
            "limiar_probabilidade": LimiarFusao("", (), regra, limiar, alvo, "").probabilidade,
            "alvo": alvo, "humanos": int((labels == 0).sum()),
            "humanos_acima": float((media[labels == 0] >= limiar).mean()), "devs": devs}


def do_treino(config: str, checkpoint: str) -> dict:
    """O limiar que o treino gravou no checkpoint (ponto de EER no dev), em log-odds.

    O ensaio 2 usa este (RESUMO_TCC §10.2.13): no microfone da sala as vozes reais
    ficam bem abaixo dos humanos dos devs, e o limiar do alarme falso nos devs
    deixava passar quase todos os clones."""
    import math

    import torch

    p = float(torch.load(checkpoint, map_location="cpu", weights_only=False)["threshold"])
    return {"regra": "logodds", "limiar": math.log(p) - math.log1p(-p),
            "limiar_probabilidade": p, "alvo": None, "condicao": "dev do treino, sem canal",
            "origem": f"limiar do treino de {Path(checkpoint).stem} (ponto de EER no dev)",
            "data": date.today().isoformat(),
            "modelos": [{"config": config, "checkpoint": checkpoint,
                         "impressao": impressoes([{"checkpoint": checkpoint}])[0]}]}


def impressoes(modelos: list[dict]) -> list[str]:
    import torch

    from src.scores import checkpoint_fingerprint

    return [checkpoint_fingerprint(torch.load(m["checkpoint"], map_location="cpu",
                                              weights_only=False)["model_state"])
            for m in modelos]


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def relatar(r: dict, modelos: list[dict]) -> None:
    nomes = [Path(m["checkpoint"]).stem for m in modelos]
    regra = "média dos log-odds" if r["regra"] == "logodds" else "média das probabilidades"
    print(f"Fusão: {regra}")
    print(f"Limiar da fusão: {r['limiar']:.4f} (score ao vivo {r['limiar_probabilidade']:.4f}; "
          f"alvo de {_pct(r['alvo'])} de alarme falso, {r['humanos']} humanos dos devs; "
          f"acima dele: {_pct(r['humanos_acima'])})\n")
    for rotulo, d in r["devs"].items():
        print(f"{rotulo}: {d['n']} áudios, {d['humanos']} humanos")
        for nome, eer in zip(nomes, d["eer_modelos"], strict=True):
            print(f"  EER de {nome}: {_pct(eer)}")
        print(f"  EER da fusão: {_pct(d['eer_fusao'])}")
        print(f"  no limiar: humanos acima {_pct(d['humanos_acima'])}, "
              f"sintéticos passam {_pct(d['sinteticos_passam'])}")
    print("\nO limiar sai só dos humanos; os sintéticos são referência. No dev da 2019 os "
          "ataques são os do treino;\nno da ASVspoof 5 (A09 a A16), nenhum dos dois modelos "
          "os viu.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Limiar da fusão pela média, escolhido nos devs")
    sub = p.add_subparsers(dest="passo", required=True)
    pp = sub.add_parser("preparar", help="configs derivados e comandos de avaliação")
    pp.add_argument("--modelo", nargs=2, action="append", metavar=("CONFIG", "CHECKPOINT"),
                    help="repita para os dois modelos (padrão: os dois fusion_v4 com degradação)")
    pe = sub.add_parser("escolher", help="escolhe o limiar com os scores já salvos")
    pe.add_argument("--alarme-falso", type=float, default=ALVO_PADRAO,
                    help="fração-alvo de humanos dos devs acima do limiar")
    pe.add_argument("--regra", choices=("logodds", "probabilidade"), default=REGRA_PADRAO)
    pe.add_argument("--so", default=None, metavar="CHECKPOINT",
                    help="usa só este modelo (parte do nome do checkpoint), com os mesmos "
                         "scores dos devs")
    pe.add_argument("--saida", default=str(PASTA / "limiar.json"))
    pt = sub.add_parser("treino", help="grava o limiar que veio do treino de um modelo")
    pt.add_argument("--modelo", nargs=2, required=True, metavar=("CONFIG", "CHECKPOINT"))
    pt.add_argument("--saida", required=True)
    for s in (pp, pe, pt):
        s.add_argument("--pasta", default=str(PASTA))
    args = p.parse_args(argv)
    pasta = Path(args.pasta)

    if args.passo == "treino":
        r = do_treino(*args.modelo)
        Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
        Path(args.saida).write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{r['origem']}: {r['limiar']:.4f} nos log-odds (score {r['limiar_probabilidade']:.6f})")
        print(f"Resultado: {args.saida}")
        return 0

    if args.passo == "preparar":
        modelos = [tuple(m) for m in args.modelo] if args.modelo else list(MODELOS_PADRAO)
        if len(modelos) != 2:
            p.error("a fusão é de dois modelos: passe --modelo duas vezes")
        preparo = preparar(modelos, pasta)
        print(f"Configs em {pasta}. Rode, de dentro de ml/ (um por vez ou encadeados):\n")
        for c in comandos(preparo):
            print(c)
        print("\nDepois: python scripts\\limiar_fusao.py escolher")
        return 0

    preparo = json.loads((pasta / "preparo.json").read_text(encoding="utf-8"))
    faltam = [e for d in preparo["devs"].values() for e in d["execucoes"]
              if not Path(e["scores"]).is_file()]
    if faltam:
        print("[ERRO] faltam os scores destas avaliações:")
        for e in faltam:
            print(f"  {e['scores']}")
        return 1
    modelos = preparo["modelos"]
    if args.so:
        modelos = [m for m in modelos if args.so in Path(m["checkpoint"]).stem]
        if len(modelos) != 1:
            p.error(f"--so {args.so} precisa apontar para exatamente um dos checkpoints do preparo")
    usados = {m["checkpoint"] for m in modelos}
    pares = {rotulo: tuple(_carregar(e["scores"]) for e in d["execucoes"]
                           if e["checkpoint"] in usados)
             for rotulo, d in preparo["devs"].items()}
    r = escolher(pares, args.alarme_falso, args.regra)
    r.update(condicao=preparo["condicao"], amostra=preparo["amostra"], data=date.today().isoformat(),
             modelos=[{**m, "impressao": i} for m, i in
                      zip(modelos, impressoes(modelos), strict=True)])
    relatar(r, modelos)
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(args.saida).write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultado: {args.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
