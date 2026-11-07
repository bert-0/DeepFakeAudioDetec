"""Parte 8: a dinâmica da voz no tempo, com características nossas (sem o XLS-R).

A parte 6 (RESUMO_TCC §10.2.22) mostrou que o XLS-R separa geradores que nunca
viu pela ordem dos quadros no tempo, com a evidência em 0-500 Hz e 1-2 kHz:
altura e vozeação, formantes. Aqui a mesma pergunta vai para medidas que a
gente calcula e entende: o contorno da altura (pYIN), a vozeação, as
trajetórias de energia por faixa, o acoplamento altura-energia e a qualidade
de voz (H1-H2) ao longo do tempo. Os arquivos, o classificador (sonda linear,
C por validação cruzada só no treino) e as réguas são os da parte 6.

Critério combinado antes de rodar: a direção vale para o nosso modelo se a
sonda na dinâmica superar a do LFCC em pelo menos 0,05 de AUC nas duas réguas
de generalização: a média dos ataques do dev da ASVspoof 5 e a 2019 LA. A
combinação LFCC + dinâmica é medida do mesmo jeito. Estudo exploratório
(RESUMO_TCC §10.2.24): não muda nenhum placar.

Uso (de dentro de ml/):
    python scripts/estudo_dinamica.py extrair --npz outputs/ssl/asv5_train.npz \\
        --saida outputs/dinamica/asv5_train.npz
    python scripts/estudo_dinamica.py sondar --treino outputs/dinamica/asv5_train.npz \\
        --dev outputs/dinamica/asv5_dev.npz --avaliar outputs/dinamica/ensaios.npz \\
        --avaliar outputs/dinamica/asv19_train.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SR = 16000
SALTO = 160          # 10 ms
QUADRO_F0 = 1024     # 64 ms: duas voltas de 60 Hz cabem
FMIN, FMAX = 60.0, 500.0
FAIXAS = ((0, 500), (500, 1000), (1000, 2000), (2000, 4000))
MARGEM = 0.05        # o critério
JANELA_MICRO = 5     # 50 ms


def _semitons(f0: np.ndarray) -> np.ndarray:
    ref = np.nanmedian(f0)
    return 12 * np.log2(f0 / ref)


def _segmentos(mascara: np.ndarray) -> list[tuple[int, int]]:
    """Trechos contíguos True: [(início, fim exclusivo)]."""
    m = np.concatenate([[False], mascara, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _ac(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    if len(x) <= lag or np.dot(x, x) == 0:
        return float("nan")
    return float(np.dot(x[:-lag], x[lag:]) / np.dot(x, x))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def dinamica(wav: np.ndarray, f0: np.ndarray | None = None, vozeado: np.ndarray | None = None,
             prob: np.ndarray | None = None) -> dict[str, float]:
    """As medidas de um arquivo a 16 kHz. f0/vozeado/prob vêm do pYIN (10 ms);
    passá-los evita recalcular (e permite testar com contornos conhecidos)."""
    import librosa

    if f0 is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f0, vozeado, prob = librosa.pyin(wav, fmin=FMIN, fmax=FMAX, sr=SR,
                                             frame_length=QUADRO_F0, hop_length=SALTO)
    nan = float("nan")
    m: dict[str, float] = {}
    n = len(f0)
    vozeado = np.asarray(vozeado, dtype=bool)

    # Altura (só quadros vozeados, em semitons em torno da mediana do arquivo)
    st = np.full(n, np.nan)
    if vozeado.sum() >= 5:
        st[vozeado] = _semitons(f0[vozeado])
    v = st[~np.isnan(st)]
    pares = ~np.isnan(st[1:]) & ~np.isnan(st[:-1])
    d1 = np.diff(st)[pares]
    trios = pares[1:] & pares[:-1]
    d2 = np.diff(st, 2)[trios]
    m["f0_desvio_st"] = float(np.std(v)) if len(v) >= 5 else nan
    m["f0_iqr_st"] = float(np.subtract(*np.percentile(v, [75, 25]))) if len(v) >= 5 else nan
    m["f0_delta_abs"] = float(np.mean(np.abs(d1))) if len(d1) >= 5 else nan
    m["f0_delta_desvio"] = float(np.std(d1)) if len(d1) >= 5 else nan
    m["f0_delta2_abs"] = float(np.mean(np.abs(d2))) if len(d2) >= 5 else nan
    micro, inclinacoes, suaves = [], [], []
    for a, b in _segmentos(~np.isnan(st)):
        s = st[a:b]
        if len(s) >= 2 * JANELA_MICRO:
            liso = np.convolve(s, np.ones(JANELA_MICRO) / JANELA_MICRO, mode="valid")
            micro.append(s[JANELA_MICRO // 2: JANELA_MICRO // 2 + len(liso)] - liso)
            inclinacoes.append(np.polyfit(np.arange(len(s)) * SALTO / SR, s, 1)[0])
            suaves.append(np.diff(s))
    m["f0_micro"] = float(np.std(np.concatenate(micro))) if micro else nan
    m["f0_inclinacao_st_s"] = float(np.median(inclinacoes)) if inclinacoes else nan
    m["f0_suavidade"] = _ac(np.concatenate(suaves), 1) if suaves else nan

    # Vozeação
    dur = n * SALTO / SR
    segs = _segmentos(vozeado)
    m["vozeada_fracao"] = float(vozeado.mean()) if n else nan
    m["transicoes_por_s"] = float(np.abs(np.diff(vozeado.astype(int))).sum() / dur) if dur > 0 else nan
    tam = np.array([(b - a) * SALTO / SR for a, b in segs])
    m["seg_vozeado_media_s"] = float(tam.mean()) if len(tam) else nan
    m["seg_vozeado_desvio_s"] = float(tam.std()) if len(tam) > 1 else nan
    p = np.nan_to_num(np.asarray(prob, dtype=float))
    m["prob_voz_media"] = float(p.mean())
    m["prob_voz_delta_abs"] = float(np.mean(np.abs(np.diff(p)))) if n > 1 else nan
    m["prob_voz_incerta"] = float(((p > 0.2) & (p < 0.8)).mean())

    # Trajetórias de energia por faixa (25 ms, passo de 10 ms)
    espec = np.abs(librosa.stft(wav, n_fft=512, hop_length=SALTO, win_length=400)) ** 2
    freqs = np.linspace(0, SR / 2, espec.shape[0])
    total = 10 * np.log10(espec.sum(axis=0) + 1e-10)
    for a, b in FAIXAS:
        e = 10 * np.log10(espec[(freqs >= a) & (freqs < b)].sum(axis=0) + 1e-10)
        nome = f"{a}_{b}"
        m[f"fluxo_{nome}"] = float(np.mean(np.abs(np.diff(e)))) if len(e) > 1 else nan
        m[f"ac1_{nome}"] = _ac(e, 1)
        m[f"ac5_{nome}"] = _ac(e, 5)
        if len(e) >= 64:
            pm = np.abs(np.fft.rfft((e - e.mean()) * np.hanning(len(e)))) ** 2
            fm = np.fft.rfftfreq(len(e), SALTO / SR)
            base = pm[(fm >= 1) & (fm <= 50)].sum()
            m[f"mod_4_16_{nome}"] = float(pm[(fm >= 4) & (fm <= 16)].sum() / base) if base > 0 else nan
        else:
            m[f"mod_4_16_{nome}"] = nan
    baixo = (freqs > 0) & (freqs < 2000)
    pb = espec[baixo]
    centroide = (freqs[baixo, None] * pb).sum(axis=0) / (pb.sum(axis=0) + 1e-12)
    m["centroide_baixo_delta"] = float(np.mean(np.abs(np.diff(centroide)))) if len(centroide) > 1 else nan
    m["centroide_baixo_desvio"] = float(np.std(centroide))

    # Acoplamento altura-energia (quadros vozeados; o pYIN e o STFT têm o mesmo passo)
    k = min(n, len(total))
    sel = ~np.isnan(st[:k])
    m["f0_energia_corr"] = _corr(st[:k][sel], total[:k][sel])
    dsel = sel[1:] & sel[:-1]
    m["f0_energia_delta_corr"] = _corr(np.diff(st[:k])[dsel], np.diff(total[:k])[dsel])

    # Qualidade de voz: H1-H2 pela janela longa do pYIN
    longo = np.abs(librosa.stft(wav, n_fft=QUADRO_F0, hop_length=SALTO)) + 1e-10
    fl = np.linspace(0, SR / 2, longo.shape[0])
    h = []
    for i in np.where(vozeado[:longo.shape[1]])[0]:
        if np.isfinite(f0[i]):
            a1 = longo[np.argmin(np.abs(fl - f0[i])), i]
            a2 = longo[np.argmin(np.abs(fl - 2 * f0[i])), i]
            h.append(20 * np.log10(a1 / a2))
    h = np.array(h)
    m["h1h2_media"] = float(h.mean()) if len(h) >= 5 else nan
    m["h1h2_desvio"] = float(h.std()) if len(h) >= 5 else nan
    m["h1h2_delta_abs"] = float(np.mean(np.abs(np.diff(h)))) if len(h) >= 5 else nan
    return m


# --------------------------------------------------------------- extrair

def _uma(caminho: str) -> dict[str, float] | None:
    from scripts.estudo_ssl import preparar

    try:
        wav, _ = preparar(Path(caminho))
        return dinamica(wav.astype(np.float64))
    except Exception as erro:  # arquivo ilegível ou curto demais: registra e segue
        print(f"  pulado {Path(caminho).name}: {erro}", flush=True)
        return None


def _iniciar_processo():
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = "1"


def extrair(origem: Path, destino: Path, processos: int = 6) -> None:
    """Mesmos arquivos, classes e rótulos do .npz da parte 6; junta o LFCC dele."""
    from multiprocessing import Pool

    from scripts.estudo_ssl import ler_npz

    d = ler_npz(origem)
    arquivos = [str(a) for a in d["arquivo"]]
    if processos > 1:
        with Pool(processos, initializer=_iniciar_processo) as pool:
            medidas = pool.map(_uma, arquivos, chunksize=8)
    else:
        medidas = [_uma(a) for a in arquivos]
    nomes = next(list(x) for x in medidas if x)
    matriz = np.array([[x[n] for n in nomes] if x else [np.nan] * len(nomes) for x in medidas],
                      dtype=np.float32)
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destino, arquivo=d["arquivo"], classe=d["classe"], rotulo=d["rotulo"],
                        lfcc=d["lfcc"], dinamica=matriz, nomes=np.array(nomes))


# --------------------------------------------------------------- sondar

def _completar(x: np.ndarray, mediana: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64).copy()
    i = np.where(np.isnan(x))
    x[i] = mediana[i[1]]
    return x


def sondar(treino: dict, dev: dict, avaliar: dict[str, dict]) -> dict:
    """Sondas em LFCC, dinâmica e os dois juntos; mesmas réguas da parte 6."""
    from scripts.estudo_ssl import aplicar, eer, medir, treinar_sonda

    mediana = np.nanmedian(treino["dinamica"], axis=0)

    def reps(d):
        din = _completar(d["dinamica"], mediana)
        return {"lfcc": d["lfcc"].astype(np.float64), "dinamica": din,
                "lfcc+dinamica": np.column_stack([d["lfcc"].astype(np.float64), din])}

    rt, rd = reps(treino), reps(dev)
    ra = {n: reps(d) for n, d in avaliar.items()}
    saida = {}
    for rep in rt:
        sonda = treinar_sonda(rt[rep], treino["rotulo"])
        s = aplicar(sonda, rd[rep])
        e, limiar = eer(s[dev["rotulo"] == 0], s[dev["rotulo"] == 1])
        r = {"C": sonda["C"], "eer_dev": e, "limiar": limiar, "dev": medir(s, dev["classe"], dev["rotulo"], limiar)}
        r["auc_media_dev"] = float(np.mean([c["auc"] for c in r["dev"]["classes"].values() if c["falso"]]))
        r["avaliar"] = {n: medir(aplicar(sonda, ra[n][rep]), d["classe"], d["rotulo"], limiar)
                        for n, d in avaliar.items()}
        if rep == "dinamica":
            r["pesos"] = dict(zip([str(x) for x in treino["nomes"]], sonda["coef"].tolist()))
        saida[rep] = r
    return saida


def criterio(r: dict, regua_2019: str) -> dict:
    """Ganho sobre o LFCC nas duas réguas e se passa a margem combinada."""
    base = r["lfcc"]
    saida = {}
    for rep in ("dinamica", "lfcc+dinamica"):
        g_dev = r[rep]["auc_media_dev"] - base["auc_media_dev"]
        g_19 = r[rep]["avaliar"][regua_2019]["auc"] - base["avaliar"][regua_2019]["auc"]
        saida[rep] = {"ganho_dev": g_dev, "ganho_2019": g_19, "passa": bool(g_dev >= MARGEM and g_19 >= MARGEM)}
    return saida


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="A dinâmica da voz no tempo, sem o XLS-R (parte 8)")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extrair")
    e.add_argument("--npz", required=True, help=".npz da parte 6 (lista de arquivos e o LFCC)")
    e.add_argument("--saida", required=True)
    e.add_argument("--processos", type=int, default=6)
    s = sub.add_parser("sondar")
    s.add_argument("--treino", required=True)
    s.add_argument("--dev", required=True)
    s.add_argument("--avaliar", action="append", default=[])
    s.add_argument("--regua-2019", default="asv19_train", help="nome (sem .npz) do conjunto da 2019")
    s.add_argument("--json", default="outputs/dinamica/sondas.json")
    args = p.parse_args(argv)

    if args.cmd == "extrair":
        extrair(Path(args.npz), Path(args.saida), args.processos)
        print(f"Pronto: {args.saida}")
        return 0

    from scripts.estudo_ssl import ler_npz

    avaliar = {Path(a).stem: ler_npz(Path(a)) for a in args.avaliar}
    r = sondar(ler_npz(Path(args.treino)), ler_npz(Path(args.dev)), avaliar)
    print(f"{'representação':14s} {'EER dev':>8s} {'AUC dev':>8s} " + " ".join(f"{'AUC ' + n:>16s}" for n in avaliar))
    for rep, x in r.items():
        print(f"{rep:14s} {100 * x['eer_dev']:7.2f}% {x['auc_media_dev']:8.3f} "
              + " ".join(f"{x['avaliar'][n]['auc']:16.3f}" for n in avaliar))
    for rep in r:
        print(f"\n{rep}, por classe (AUC; certos pelo limiar do dev):")
        for conj, mm in [("dev", r[rep]["dev"])] + list(r[rep]["avaliar"].items()):
            print(f"  {conj}: " + ", ".join(f"{c} {v['certos']}/{v['n']}" + (f" ({v['auc']:.2f})" if v["auc"] is not None else "")
                                           for c, v in mm["classes"].items()))
    pesos = sorted(r["dinamica"]["pesos"].items(), key=lambda kv: -abs(kv[1]))[:12]
    print("\nPesos mais fortes da sonda na dinâmica (medidas padronizadas; + puxa para falso):")
    for nome, w in pesos:
        print(f"  {nome:24s} {w:+.3f}")
    c = None
    if args.regua_2019 in avaliar:
        c = criterio(r, args.regua_2019)
        print(f"\nCritério (ganho mínimo de {MARGEM} de AUC sobre o LFCC no dev e na 2019):")
        for rep, x in c.items():
            print(f"  {rep:14s} dev {x['ganho_dev']:+.3f}, 2019 {x['ganho_2019']:+.3f} -> "
                  f"{'passa' if x['passa'] else 'não passa'}")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({"sondas": r, "criterio": c}, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    print(f"\nResultado: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
