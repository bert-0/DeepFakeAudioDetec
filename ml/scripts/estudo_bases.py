"""Parte 5: as marcas das partes 3 e 4 nas bases ASVspoof.

Nas partes 3 e 4 (RESUMO_TCC §10.2.19 e §10.2.20), os humanos vinham todos do
mesmo microfone, que tem marcas próprias. Aqui os humanos são os da base e os
sintéticos vêm de vários geradores, alguns que o estudo nunca viu. Para cada
classe do protocolo (bonafide e cada ataque), uma amostra fixa de arquivos,
cortados nos primeiros 3 s: todos com o mesmo número de segmentos no espectro,
porque o ruído da medida depende disso.

- Linhas fixas da classe: escolhidas na metade A dos arquivos e contadas na
  metade B e na metade B dos humanos (a réplica da parte 3).
- Duas medidas por arquivo que não sabem o gerador: a fração da banda de 2 a
  7,6 kHz com linha (6 dB acima do contorno) e o pente varrido de 40 a 200 Hz
  em 4 a 7,6 kHz (parte 4). A separação é a AUC contra os humanos da base.

Só partições de treino e dev; o eval fica para um teste com critério.
Estudo exploratório (RESUMO_TCC §10.2.21): não muda nenhum placar.

Uso (de dentro de ml/):
    python scripts/estudo_bases.py --rotulo asv19_train \\
        --protocolo data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \\
        --pasta data/LA/ASVspoof2019_LA_train/flac
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.estudo_linhas import GRADE, escolher_linhas, presenca  # noqa: E402
from scripts.estudo_pente import FAIXA_MODELO, auc, varrido  # noqa: E402

SR = 16000
SEGUNDOS = 3.0
FAIXA_LINHAS = (2000.0, 7600.0)
LINHA_DB = 6.0
#: 2 Hz de resolução: 3 s dão 11 segmentos de 0,5 s com metade de sobreposição.
NPERSEG = SR // 2
#: Contorno em 21 pontos de 2 Hz, o mesmo alcance (~41 Hz) das partes 3 e 4.
CONTORNO_PONTOS = 21


def ler_protocolo(caminho: Path) -> dict[str, list[str]]:
    """{classe: [arquivo, ...]} no formato da 2019 e do recorte da ASVspoof 5
    (locutor, arquivo, -, ataque, chave)."""
    classes: dict[str, list[str]] = {}
    for linha in Path(caminho).read_text(encoding="utf-8").splitlines():
        partes = linha.split()
        if len(partes) < 5:
            continue
        classe = "bonafide" if partes[-1] == "bonafide" else partes[-2]
        classes.setdefault(classe, []).append(partes[1])
    return classes


def ler_corte(arq: Path, segundos: float = SEGUNDOS) -> np.ndarray | None:
    """Os primeiros `segundos` do arquivo, ou None se ele for mais curto."""
    import soundfile as sf

    info = sf.info(arq)
    if info.samplerate != SR:
        raise ValueError(f"{arq.name}: {info.samplerate} Hz, o estudo espera {SR}")
    n = int(segundos * SR)
    if info.frames < n:
        return None
    wav, _ = sf.read(arq, frames=n, dtype="float64")
    return wav.mean(axis=1) if wav.ndim > 1 else wav


def medidas(wav: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
    """(excesso na GRADE de 1 Hz, fração da banda com linha, (base, força) do pente)."""
    from scipy.ndimage import median_filter
    from scipy.signal import welch

    f, p = welch(wav, SR, nperseg=NPERSEG, noverlap=NPERSEG // 2)
    db = 10 * np.log10(p + 1e-30)
    ex = db - median_filter(db, size=CONTORNO_PONTOS, mode="nearest")
    banda = (f >= FAIXA_LINHAS[0]) & (f <= FAIXA_LINHAS[1])
    fracao = float((ex[banda] > LINHA_DB).mean())
    grade = np.interp(GRADE, f, ex, right=np.nan)
    return grade, fracao, varrido(grade, FAIXA_MODELO)


def estudar(classes: dict[str, list[str]], pasta: Path, n: int, semente: int,
            extensao: str = "flac", n_linhas: int = 5) -> dict:
    rng = np.random.default_rng(semente)
    saida: dict = {}
    matrizes: dict[str, np.ndarray] = {}
    for classe in sorted(classes):
        ids = classes[classe]
        usados, grades, fracoes, pentes = [], [], [], []
        for i in rng.permutation(len(ids)):
            if len(usados) == n:
                break
            wav = ler_corte(pasta / f"{ids[i]}.{extensao}")
            if wav is None:
                continue
            g, fr, pt = medidas(wav)
            usados.append(ids[i])
            grades.append(g)
            fracoes.append(fr)
            pentes.append(pt)
        matrizes[classe] = np.array(grades)
        saida[classe] = {"arquivos": usados, "linhas_fracao": fracoes,
                         "pente_base": [b for b, _ in pentes], "pente": [f for _, f in pentes]}
    humanos = matrizes.get("bonafide")
    for classe, m in matrizes.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # colunas acima de 8 kHz são NaN
            escolhidas = escolher_linhas(m[::2], n_linhas) if len(m) >= 2 else []
        saida[classe]["linhas"] = [
            {"hz": hz, "db": db, "metade_b": [presenca(m[1::2], hz), len(m[1::2])],
             "humanos": ([presenca(humanos[1::2], hz), len(humanos[1::2])] if humanos is not None else None)}
            for hz, db in escolhidas]
        if humanos is not None and classe != "bonafide":
            saida[classe]["auc_linhas"] = auc(saida[classe]["linhas_fracao"], saida["bonafide"]["linhas_fracao"])
            saida[classe]["auc_pente"] = auc(saida[classe]["pente"], saida["bonafide"]["pente"])
    return saida


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Linhas e pente nas classes de uma base ASVspoof")
    p.add_argument("--protocolo", required=True)
    p.add_argument("--pasta", required=True, help="pasta dos áudios do protocolo")
    p.add_argument("--rotulo", required=True, help="nome curto da base e partição (ex.: asv19_train)")
    p.add_argument("--n", type=int, default=100, help="arquivos por classe")
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--extensao", default="flac")
    p.add_argument("--json", default=None, help="padrão: outputs/estudo_bases_<rotulo>.json")
    args = p.parse_args(argv)

    classes = ler_protocolo(Path(args.protocolo))
    r = estudar(classes, Path(args.pasta), args.n, args.semente, args.extensao)
    print(f"{args.rotulo}: {args.n} arquivos por classe, primeiros {SEGUNDOS:.0f} s; "
          f"AUC contra os humanos da base")
    print(f"{'classe':9s} {'n':>4s} {'linhas ‰':>9s} {'AUC':>5s} {'pente dB':>9s} {'AUC':>5s}  "
          f"bases mais comuns do pente")
    for classe, c in r.items():
        comuns = ", ".join(f"{b:g} ({k})" for b, k in Counter(c["pente_base"]).most_common(2))
        print(f"{classe:9s} {len(c['arquivos']):4d} {1000 * np.median(c['linhas_fracao']):9.2f} "
              f"{c.get('auc_linhas', float('nan')):5.2f} {np.median(c['pente']):+9.2f} "
              f"{c.get('auc_pente', float('nan')):5.2f}  {comuns}")
    print("\nLinhas fixas de cada classe (escolhidas na metade A): Hz +dB "
          "(arquivos com a linha na metade B; nos humanos, metade B)")
    for classe, c in r.items():
        texto = "; ".join(
            f"{ln['hz']:.0f} +{ln['db']:.1f} ({ln['metade_b'][0]}/{ln['metade_b'][1]}"
            + (f"; {ln['humanos'][0]}/{ln['humanos'][1]})" if ln["humanos"] else ")")
            for ln in c["linhas"])
        print(f"{classe:9s} {texto}")
    destino = Path(args.json or f"outputs/estudo_bases_{args.rotulo}.json")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({"rotulo": args.rotulo, "protocolo": args.protocolo, "n": args.n,
                                   "semente": args.semente, "classes": r}, indent=1, ensure_ascii=False),
                       encoding="utf-8")
    print(f"\nResultado: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
