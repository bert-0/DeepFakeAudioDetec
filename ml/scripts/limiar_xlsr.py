"""Limiar do modelo final (XLS-R congelado), num JSON que a interface e o monitor leem.

Duas origens:
- `--real NOME=PASTA` (repetível): vozes reais de outros locutores, no mesmo
  microfone e modo de captura da demonstração. Regra do RESUMO_TCC §10.2.39,
  fixada antes: o limiar é o maior score dessas vozes. Sai um limiar por caminho,
  das mesmas gravações e pelo mesmo código da interface: o do arquivo inteiro
  (`src/xlsr_final.py`, Enviar arquivo) e o da frase cortada pelo ao vivo
  (`src/capture/frases.py`). Nas gravações do microfone os dois scores diferem:
  o arquivo leva segundos de ruído da sala que o corte de silêncio não tira.
- `--dev NPZ`: o ponto de EER no dev (um .npz do `estudo_ssl.py extrair`), sem canal
  e sem calibração por frase.

O JSON guarda todos os scores, para outra regra poder ser medida depois sem
gravar de novo.

Uso (de dentro de ml/, com HF_HOME no cache do XLS-R):
    python scripts/limiar_xlsr.py --real L02=data/vozes/reparadas/L02_embutido
    python scripts/limiar_xlsr.py --dev outputs/ssl/asv5_dev.npz --saida outputs/limiar_xlsr/limiar_dev.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAIDA_PADRAO = "outputs/limiar_xlsr/limiar.json"


def pontuar_pastas(modelo, pastas: dict[str, Path]) -> dict[str, dict[str, dict]]:
    """{locutor: {arquivo: {"arquivo": log-odds, "frases": [log-odds de cada frase]}}}.
    Arquivo sem fala suficiente fica de fora, com aviso."""
    from scripts.estudo_ssl import EXTENSOES_AUDIO
    from src.capture.frases import frases_de
    from src.preprocess.audio import load_audio
    from src.xlsr_final import SR, FalaCurta

    saida = {}
    for nome, pasta in pastas.items():
        arquivos = sorted(a for a in Path(pasta).iterdir() if a.suffix.lower() in EXTENSOES_AUDIO)
        if not arquivos:
            raise ValueError(f"nenhum áudio em {pasta}")
        saida[nome] = {}
        for a in arquivos:
            bruto = load_audio(a, SR)
            try:
                inteiro = modelo.pontuar_wav(bruto).logodds
            except FalaCurta as erro:
                print(f"  pulado {nome}/{a.name}: {erro}", flush=True)
                continue
            frases = [x.logodds for x in frases_de(modelo, bruto, SR)]
            if not frases:
                print(f"  {nome}/{a.name}: o ao vivo não achou frase; entra só no arquivo", flush=True)
            saida[nome][a.name] = {"arquivo": inteiro, "frases": frases}
    return saida


def limiar_por_reais(scores: dict[str, dict[str, dict]]) -> tuple[float, float | None, dict]:
    """Maior score entre todas as vozes reais (§10.2.39), em cada caminho, e o resumo
    por locutor."""
    locutores = {}
    for nome, s in scores.items():
        if not s:
            continue
        frases = [z for x in s.values() for z in x["frases"]]
        locutores[nome] = {"n": len(s), "maximo": max(x["arquivo"] for x in s.values()),
                           "n_frases": len(frases), "maximo_frase": max(frases) if frases else None,
                           "scores": s}
    if not locutores:
        raise ValueError("nenhuma voz real pontuada")
    frases = [x["maximo_frase"] for x in locutores.values() if x["maximo_frase"] is not None]
    return max(x["maximo"] for x in locutores.values()), (max(frases) if frases else None), locutores


def limiar_do_dev(cabeca: str, npz: str, dispositivo: str = "cpu") -> float:
    from scripts.estudo_ssl import eer
    from scripts.xlsr_congelado import ler, pontuar
    from src.xlsr_final import carregar_cabeca

    c, _, dev = carregar_cabeca(cabeca, dispositivo)
    d = ler(npz)
    s = pontuar(c, d, dev)
    return float(eer(s[d["rotulo"] == 0], s[d["rotulo"] == 1])[1])


def main(argv=None) -> int:
    from src.xlsr_final import CABECA_PADRAO, gravar_limiar

    p = argparse.ArgumentParser(description="Limiar do XLS-R congelado")
    p.add_argument("--cabeca", default=CABECA_PADRAO)
    origem = p.add_mutually_exclusive_group(required=True)
    origem.add_argument("--real", action="append", metavar="NOME=PASTA",
                        help="vozes reais de um locutor de calibração (repita para mais locutores)")
    origem.add_argument("--dev", metavar="NPZ", help="ponto de EER neste dev")
    p.add_argument("--saida", default=SAIDA_PADRAO)
    p.add_argument("--dispositivo", default="cuda")
    args = p.parse_args(argv)

    if args.dev:
        limiar = limiar_do_dev(args.cabeca, args.dev, args.dispositivo)
        destino = gravar_limiar(args.saida, args.cabeca, limiar, regra="ponto de EER no dev",
                                origem=f"limiar do dev ({Path(args.dev).stem}), sem o canal da demonstração")
        print(f"Limiar {limiar:.4f} (ponto de EER em {args.dev}) -> {destino}")
        return 0

    from src.xlsr_final import ModeloFinal

    pastas = {}
    for item in args.real:
        nome, _, pasta = item.partition("=")
        if not pasta:
            p.error(f"use NOME=PASTA em --real (veio {item})")
        pastas[nome] = Path(pasta)
    modelo = ModeloFinal(args.cabeca, args.dispositivo)
    limiar, limiar_frase, locutores = limiar_por_reais(pontuar_pastas(modelo, pastas))
    for nome, x in locutores.items():
        inteiros = [s["arquivo"] for s in x["scores"].values()]
        frase = "—" if x["maximo_frase"] is None else f"{x['maximo_frase']:.3f}"
        print(f"  {nome}: {x['n']} arquivos, de {min(inteiros):.3f} a {x['maximo']:.3f}; "
              f"{x['n_frases']} frases, a maior {frase}")
        x["pasta"] = str(pastas[nome]).replace("\\", "/")
    destino = gravar_limiar(args.saida, args.cabeca, limiar, regra="maior voz real (§10.2.39)",
                            origem="calibrado com as vozes reais de " + ", ".join(locutores) + " (maior score)",
                            locutores=locutores, limiar_frase=limiar_frase)
    frase = "—" if limiar_frase is None else f"{limiar_frase:.4f}"
    print(f"Limiar do arquivo {limiar:.4f}; da frase (ao vivo) {frase} -> {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
