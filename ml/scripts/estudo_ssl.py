"""Parte 6: o que o XLS-R (pré-treinado pela Meta, congelado) enxerga.

Instrumento de pesquisa, não detector: o modelo não é ajustado e não entra no
sistema. Para cada arquivo, a média no tempo de cada representação interna (a
convolução de entrada e as 25 saídas do transformer) vira um vetor. Uma
regressão logística por representação ("sonda") é treinada no treino do recorte
da ASVspoof 5 e medida em geradores que ela não viu: o dev do recorte (A09 a
A16), a 2019 LA e os clones dos ensaios. A mesma sonda sobre o LFCC médio do
nosso detector é o controle: se o XLS-R generaliza melhor com o mesmo
classificador, a diferença está no que ele representa.

Regra fixada antes dos resultados: a melhor camada é a de maior AUC média nos
ataques do dev da ASVspoof 5; os clones dos ensaios e a 2019 LA só medem.
Estudo exploratório (RESUMO_TCC §10.2.22): não muda nenhum placar.

Subcomandos (de dentro de ml/, com HF_HOME apontando para o cache do modelo):
    extrair   embeddings, LFCC e medidas simples por arquivo, num .npz (outputs/ssl/)
    sondar    sonda por representação; AUC e EER nos conjuntos de avaliação
    explicar  o que as medidas simples explicam do score da melhor camada
    faixas    retira uma faixa por vez e mede a sonda da melhor camada (como a parte 1)
    tempo     reordena os quadros antes do transformer; decompõe o score por tipo de quadro
    detector  o nosso detector nos mesmos arquivos, pelo caminho da avaliação
    codecs    AUC por codec no eval da ASVspoof 5 (só medida; nada é escolhido nele)

Exemplos:
    python scripts/estudo_ssl.py extrair --saida outputs/ssl/asv5_train.npz \\
        --protocolo data/ASVspoof5_recorte/train/protocolo.txt \\
        --pasta data/ASVspoof5_recorte/train/flac --n-humanos 2000 --n-ataque 250
    python scripts/estudo_ssl.py extrair --saida outputs/ssl/ensaios.npz \\
        --grupo L01_real=data/vozes/ensaio2/real:bonafide \\
        --grupo L01_xtts=data/vozes/ensaio2/clone:spoof
    python scripts/estudo_ssl.py sondar --treino outputs/ssl/asv5_train.npz \\
        --dev outputs/ssl/asv5_dev.npz --avaliar outputs/ssl/ensaios.npz
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

MODELO_HF = "facebook/wav2vec2-xls-r-300m"
SR = 16000
#: Mesma janela do detector (4 s, passo de 2 s); áudio curto entra inteiro, sem repetir.
JANELA_S, PASSO_S = 4.0, 2.0
TOP_DB = 30
#: Mesmo LFCC dos configs/fusion_v4*.yaml.
LFCC_CFG = {"n_lfcc": 20, "n_fft": 512, "win_length": 400, "hop_length": 160,
            "n_filter": 70, "deltas": True}
CS = (1e-3, 1e-2, 1e-1, 1.0)


# --------------------------------------------------------------- áudio

def preparar(caminho: Path) -> tuple[np.ndarray, float]:
    """Como no treino: 16 kHz pelo FIR, sem silêncio nas pontas, pico em 1.
    Devolve (áudio, segundos de silêncio retirados)."""
    from src.preprocess.audio import load_audio, peak_normalize, trim_silence

    bruto = load_audio(caminho, SR)
    wav = trim_silence(bruto, TOP_DB)
    return peak_normalize(wav).astype(np.float32), (len(bruto) - len(wav)) / SR


def janelas(wav: np.ndarray) -> list[np.ndarray]:
    n, passo = int(JANELA_S * SR), int(PASSO_S * SR)
    if len(wav) <= n:
        return [wav]
    inicios = list(range(0, len(wav) - n + 1, passo))
    return [wav[i:i + n] for i in inicios]


# --------------------------------------------------------------- representações

def carregar_rede(dispositivo: str = "cuda", modelo: str = MODELO_HF):
    import torch
    from transformers import Wav2Vec2Model

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    dev = torch.device(dispositivo if torch.cuda.is_available() or dispositivo == "cpu" else "cpu")
    return Wav2Vec2Model.from_pretrained(modelo).eval().to(dev), dev


def carregar_xlsr(dispositivo: str = "cuda", modelo: str = MODELO_HF):
    """Função wav -> (cnn (512,), camadas (25, 1024)), médias no tempo, em fp32."""
    return representar_com(*carregar_rede(dispositivo, modelo))


def representar_com(rede, dev):
    import torch

    def rodar(wav: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cnns, camadas = [], []
        for x in janelas(wav):
            x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)  # do_normalize do XLS-R
            with torch.inference_mode():
                s = rede(torch.from_numpy(x).float()[None].to(dev), output_hidden_states=True)
            cnns.append(s.extract_features[0].float().mean(0).cpu().numpy())
            camadas.append(torch.stack([h[0].float().mean(0) for h in s.hidden_states]).cpu().numpy())
        return np.mean(cnns, axis=0), np.mean(camadas, axis=0)
    return rodar


def lfcc_medio(wav: np.ndarray) -> np.ndarray:
    """Média e desvio no tempo do LFCC com deltas (120 valores)."""
    from src.features.lfcc import compute_lfcc

    c = compute_lfcc(wav, SR, LFCC_CFG)
    return np.concatenate([c.mean(axis=1), c.std(axis=1)]).astype(np.float32)


# --------------------------------------------------------------- medidas simples

def medidas_simples(wav: np.ndarray, silencio_s: float) -> dict[str, float]:
    """Medidas que a gente sabe interpretar, para comparar com o que o XLS-R usa."""
    from scipy.signal import welch

    from scripts.estudo_bases import medidas as linhas_e_pente
    from scripts.estudo_faixas import FAIXAS

    f, p = welch(wav, SR, nperseg=512)
    total = p.sum() + 1e-20
    m = {"silencio_s": float(silencio_s), "duracao_s": len(wav) / SR}
    for a, b in FAIXAS:
        m[f"energia_{a}_{b}"] = float(10 * np.log10(p[(f >= a) & (f < b)].sum() / total + 1e-12))
    sel = (f >= 500) & (f <= 7600)
    m["inclinacao_db_khz"] = float(np.polyfit(f[sel] / 1000, 10 * np.log10(p[sel] + 1e-20), 1)[0])
    alta = p[(f >= 4000) & (f <= 7600)] + 1e-20
    m["planura_alta"] = float(np.exp(np.log(alta).mean()) / alta.mean())
    rms = np.sqrt(np.mean(wav ** 2)) + 1e-12
    m["crista_db"] = float(20 * np.log10(np.abs(wav).max() / rms + 1e-12))
    m["hnr_db"] = hnr(wav)
    if len(wav) >= 3 * SR:
        _, fracao, (_, forca) = linhas_e_pente(wav[:3 * SR].astype(np.float64))
        m["linhas_fracao"], m["pente_forca"] = fracao, forca
    else:
        m["linhas_fracao"] = m["pente_forca"] = float("nan")
    return m


def hnr(wav: np.ndarray, quadro_s: float = 0.04) -> float:
    """HNR mediano (dB) dos quadros mais fortes, pela autocorrelação em 80-400 Hz."""
    n = int(quadro_s * SR)
    if len(wav) < 2 * n:
        return float("nan")
    quadros = wav[: len(wav) // n * n].reshape(-1, n).astype(np.float64)
    energia = (quadros ** 2).sum(axis=1)
    valores = []
    for q in quadros[energia >= np.median(energia)]:
        q = q - q.mean()
        ac = np.correlate(q, q, "full")[n - 1:]
        if ac[0] <= 0:
            continue
        r = ac[SR // 400: SR // 80].max() / ac[0]
        r = min(max(r, 1e-4), 1 - 1e-4)
        valores.append(10 * np.log10(r / (1 - r)))
    return float(np.median(valores)) if valores else float("nan")


# --------------------------------------------------------------- extrair

def selecionar_protocolo(protocolo: Path, pasta: Path, n_humanos: int, n_ataque: int,
                         semente: int, extensao: str = "flac") -> list[tuple[Path, str, int]]:
    from scripts.estudo_bases import ler_protocolo

    rng = np.random.default_rng(semente)
    saida = []
    for classe, ids in sorted(ler_protocolo(protocolo).items()):
        n = n_humanos if classe == "bonafide" else n_ataque
        for i in rng.permutation(len(ids))[:n]:
            saida.append((pasta / f"{ids[i]}.{extensao}", classe, int(classe != "bonafide")))
    return saida


EXTENSOES_AUDIO = {".wav", ".flac", ".mp3"}
#: Menos que isso de fala não é uma frase (o mínimo de `checagens.TRECHO_MINIMO_S`).
MIN_AMOSTRAS = int(0.25 * 16000)


def selecionar_grupos(grupos: list[str]) -> list[tuple[Path, str, int]]:
    saida = []
    for item in grupos:
        nome, resto = item.split("=", 1)
        pasta, rotulo = resto.rsplit(":", 1)
        if rotulo not in ("bonafide", "spoof"):
            raise ValueError(f"--grupo {item}: o rótulo depois de ':' é bonafide ou spoof")
        # O BRSpeech-DF vem em FLAC, o Common Voice em MP3 (10.2.40).
        for arq in sorted(a for a in Path(pasta).iterdir() if a.suffix.lower() in EXTENSOES_AUDIO):
            saida.append((arq, nome, int(rotulo == "spoof")))
    return saida


def ler_lista(caminho: Path) -> list[tuple[Path, str, int]]:
    """(arquivo, classe, rótulo) de um TSV com essas colunas; as outras são ignoradas."""
    import csv

    with open(caminho, encoding="utf-8", newline="") as f:
        return [(Path(r["arquivo"]), r["classe"], int(r["rotulo"])) for r in csv.DictReader(f, delimiter="	")]


def extrair(lista: list[tuple[Path, str, int]], representar, destino: Path,
            avisar_a_cada: int = 200) -> None:
    cnns, camadas, lfccs, medidas, guardados = [], [], [], [], []
    for k, (arq, classe, rotulo) in enumerate(lista):
        try:
            wav, silencio = preparar(arq)
        except Exception as erro:  # arquivo ilegível: registra e segue
            print(f"  pulado {arq.name}: {erro}", flush=True)
            continue
        if len(wav) < MIN_AMOSTRAS:
            # Um clone do ToucanTTS no BRSpeech-DF sobra com 17 ms depois do corte de
            # silêncio, menor que a primeira convolução do XLS-R (10.2.40).
            print(f"  pulado {arq.name}: {len(wav) / SR:.3f} s depois do corte de silêncio", flush=True)
            continue
        cnn, cam = representar(wav)
        cnns.append(cnn.astype(np.float16))
        camadas.append(cam.astype(np.float16))
        lfccs.append(lfcc_medio(wav))
        medidas.append(medidas_simples(wav, silencio))
        guardados.append((str(arq), classe, rotulo))
        if (k + 1) % avisar_a_cada == 0:
            print(f"  {k + 1}/{len(lista)}", flush=True)
    nomes = list(medidas[0])
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destino, arquivo=np.array([g[0] for g in guardados]), classe=np.array([g[1] for g in guardados]),
        rotulo=np.array([g[2] for g in guardados], dtype=np.int8), cnn=np.array(cnns),
        camadas=np.array(camadas), lfcc=np.array(lfccs),
        medidas=np.array([[m[n] for n in nomes] for m in medidas], dtype=np.float32),
        nomes_medidas=np.array(nomes))


# --------------------------------------------------------------- sondar

def representacoes(d) -> dict[str, np.ndarray]:
    """{nome: (N, D)} na ordem: cnn, camada_00 ... camada_24, lfcc."""
    reps = {"cnn": d["cnn"].astype(np.float32)}
    for i in range(d["camadas"].shape[1]):
        reps[f"camada_{i:02d}"] = d["camadas"][:, i].astype(np.float32)
    reps["lfcc"] = d["lfcc"].astype(np.float32)
    return reps


def treinar_sonda(x: np.ndarray, y: np.ndarray, semente: int = 42) -> dict:
    """Padronização + regressão logística; C escolhido por validação cruzada só no treino."""
    from sklearn.linear_model import LogisticRegressionCV

    media, escala = x.mean(axis=0), x.std(axis=0) + 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegressionCV(Cs=list(CS), cv=5, max_iter=3000, scoring="roc_auc",
                                  random_state=semente, n_jobs=-1).fit((x - media) / escala, y)
    return {"media": media, "escala": escala, "coef": lr.coef_[0], "intercepto": float(lr.intercept_[0]),
            "C": float(lr.C_[0])}


def aplicar(sonda: dict, x: np.ndarray) -> np.ndarray:
    return ((x - sonda["media"]) / sonda["escala"]) @ sonda["coef"] + sonda["intercepto"]


def eer(scores_humanos, scores_falsos) -> tuple[float, float]:
    """(EER, limiar) com o falso como positivo."""
    h, f = np.sort(np.asarray(scores_humanos)), np.sort(np.asarray(scores_falsos))
    limiares = np.concatenate([h, f])
    far = 1 - np.searchsorted(h, limiares, side="left") / len(h)   # humanos >= limiar
    frr = np.searchsorted(f, limiares, side="left") / len(f)       # falsos < limiar
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(limiares[i])


def medir(scores: np.ndarray, classes: np.ndarray, rotulos: np.ndarray, limiar: float) -> dict:
    """AUC de cada classe falsa contra todos os humanos do conjunto, e quantos ficam
    do lado certo do limiar (escolhido no dev)."""
    from scripts.estudo_pente import auc

    humanos = scores[rotulos == 0]
    r = {"auc": auc(scores[rotulos == 1], humanos) if (rotulos == 1).any() and (rotulos == 0).any() else None,
         "classes": {}}
    for c in sorted(set(classes.tolist())):
        s = scores[classes == c]
        falso = bool(rotulos[classes == c][0])
        r["classes"][c] = {"n": int(len(s)), "falso": falso,
                           "certos": int((s >= limiar).sum() if falso else (s < limiar).sum()),
                           "auc": auc(s, humanos) if falso and len(humanos) else None}
    return r


def sondar(treino: dict, dev: dict, avaliar: dict[str, dict]) -> dict:
    """Uma sonda por representação. Limiar de cada uma: o ponto de EER do dev."""
    reps_t, reps_d = representacoes(treino), representacoes(dev)
    reps_a = {nome: representacoes(d) for nome, d in avaliar.items()}
    saida = {}
    for rep in reps_t:
        sonda = treinar_sonda(reps_t[rep], treino["rotulo"])
        s_dev = aplicar(sonda, reps_d[rep])
        e, limiar = eer(s_dev[dev["rotulo"] == 0], s_dev[dev["rotulo"] == 1])
        r = {"C": sonda["C"], "limiar": limiar, "eer_dev": e,
             "dev": medir(s_dev, dev["classe"], dev["rotulo"], limiar)}
        aucs = [c["auc"] for c in r["dev"]["classes"].values() if c["falso"]]
        r["auc_media_dev"] = float(np.mean(aucs))
        r["avaliar"] = {nome: medir(aplicar(sonda, reps_a[nome][rep]), d["classe"], d["rotulo"], limiar)
                        for nome, d in avaliar.items()}
        r["sonda"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sonda.items()}
        saida[rep] = r
    camadas = [k for k in saida if k not in ("lfcc",)]
    melhor = max(camadas, key=lambda k: saida[k]["auc_media_dev"])
    return {"melhor": melhor, "representacoes": saida}


def ler_npz(caminho: Path) -> dict:
    with np.load(caminho, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


# --------------------------------------------------------------- explicar

def explicar(resultado: dict, treino: dict, conjuntos: dict[str, dict]) -> dict:
    """Quanto as medidas simples explicam do score da melhor camada e quanto
    separam sozinhas (mesma sonda, treinada no mesmo treino)."""
    from scipy.stats import spearmanr

    rep = resultado["melhor"]
    sonda = {k: np.asarray(v) if isinstance(v, list) else v
             for k, v in resultado["representacoes"][rep]["sonda"].items()}
    nomes = [str(n) for n in treino["nomes_medidas"]]
    mediana = np.nanmedian(treino["medidas"], axis=0)

    def completar(m):
        m = m.astype(np.float64).copy()
        idx = np.where(np.isnan(m))
        m[idx] = mediana[idx[1]]
        return m

    sonda_medidas = treinar_sonda(completar(treino["medidas"]), treino["rotulo"])
    limiar_med = 0.0  # sem dev, log-odds 0 (probabilidade 0,5)
    if "dev" in conjuntos:
        d = conjuntos["dev"]
        s = aplicar(sonda_medidas, completar(d["medidas"]))
        _, limiar_med = eer(s[d["rotulo"] == 0], s[d["rotulo"] == 1])
    saida = {"representacao": rep, "medidas": nomes, "limiar_medidas": limiar_med, "conjuntos": {}}
    for nome, d in conjuntos.items():
        score = aplicar(sonda, representacoes(d)[rep])
        m = completar(d["medidas"])
        corr = {}
        for j, n in enumerate(nomes):
            linha = {}
            for rotulo, chave in ((0, "humanos"), (1, "falsos")):
                sel = d["rotulo"] == rotulo
                if sel.sum() > 5 and np.std(m[sel, j]) > 0:
                    linha[chave] = float(spearmanr(score[sel], m[sel, j]).statistic)
            corr[n] = linha
        x = np.column_stack([(m - m.mean(0)) / (m.std(0) + 1e-9), np.ones(len(m))])
        coef, *_ = np.linalg.lstsq(x, score, rcond=None)
        r2 = 1 - np.var(score - x @ coef) / np.var(score)
        saida["conjuntos"][nome] = {"correlacao": corr, "r2_medidas": float(r2),
                                    "so_medidas": medir(aplicar(sonda_medidas, m), d["classe"], d["rotulo"],
                                                        limiar_med),
                                    "efeitos": efeitos(d)}
    return saida


def efeitos(d: dict) -> dict[str, dict[str, float]]:
    """Tamanho do efeito (d de Cohen) de cada classe falsa contra os humanos, por medida."""
    nomes = [str(n) for n in d["nomes_medidas"]]
    m, classes = d["medidas"].astype(np.float64), d["classe"]
    humanos = m[d["rotulo"] == 0]
    saida = {}
    for c in sorted(set(classes[d["rotulo"] == 1].tolist())):
        x = m[classes == c]
        linha = {}
        for j, n in enumerate(nomes):
            a, b = x[:, j][~np.isnan(x[:, j])], humanos[:, j][~np.isnan(humanos[:, j])]
            if len(a) > 1 and len(b) > 1:
                linha[n] = float((a.mean() - b.mean()) / (np.sqrt((a.var() + b.var()) / 2) + 1e-12))
        saida[c] = linha
    return saida


# --------------------------------------------------------------- faixas

def faixas(resultado: dict, lista: list[tuple[Path, str, int]], representar) -> dict:
    """Variação do score da melhor camada ao retirar cada faixa (parte 1)."""
    from scripts.estudo_faixas import FAIXAS, sem_faixa

    rep = resultado["melhor"]
    sonda = {k: np.asarray(v) if isinstance(v, list) else v
             for k, v in resultado["representacoes"][rep]["sonda"].items()}
    indice = None if rep == "cnn" else int(rep.split("_")[1])

    def score(wav):
        cnn, cam = representar(wav)
        x = cnn if indice is None else cam[indice]
        return float(aplicar(sonda, x[None].astype(np.float32))[0])

    saida: dict = {}
    for arq, classe, rotulo in lista:
        wav, _ = preparar(arq)
        base = score(wav)
        r = saida.setdefault(classe, {"falso": bool(rotulo), "inteiro": [], **{f"sem {a}-{b}": [] for a, b in FAIXAS}})
        r["inteiro"].append(base)
        for a, b in FAIXAS:
            r[f"sem {a}-{b}"].append(score(sem_faixa(wav, SR, (a, b))) - base)
    return saida


# --------------------------------------------------------------- tempo

#: Quadros da convolução do XLS-R: 400 amostras (25 ms) a cada 320 (20 ms).
QUADRO, SALTO = 400, 320
VARIANTES = ("original", "embaralhado", "embaralhado_100ms", "invertido")
BLOCO_QUADROS = 5  # 5 quadros de 20 ms = 100 ms


def ordem(variante: str, t: int, rng: np.random.Generator) -> np.ndarray | None:
    """Nova ordem dos t quadros da convolução, ou None para manter."""
    if variante == "original":
        return None
    if variante == "invertido":
        return np.arange(t)[::-1].copy()
    if variante == "embaralhado":
        return rng.permutation(t)
    idx = np.arange(t)
    for i in range(0, t, BLOCO_QUADROS):
        idx[i:i + BLOCO_QUADROS] = rng.permutation(idx[i:i + BLOCO_QUADROS])
    return idx


def energia_dos_quadros(x: np.ndarray, t: int) -> np.ndarray:
    """Energia em dB de cada quadro de 25 ms alinhado com a saída da convolução."""
    e = np.array([np.sum(x[i * SALTO:i * SALTO + QUADRO] ** 2) for i in range(t)])
    return 10 * np.log10(e + 1e-10)


def tempo(resultado: dict, lista: list[tuple[Path, str, int, str]], rede, dev, semente: int = 0) -> list[dict]:
    """Score da melhor camada com os quadros reordenados antes do transformer, e a
    decomposição do score original por tipo de quadro (a sonda é linear: o score
    da média é a média dos scores dos quadros)."""
    import torch

    rep = resultado["melhor"]
    if rep == "cnn":
        raise ValueError("a decomposição no tempo precisa de uma camada do transformer")
    k = int(rep.split("_")[1])
    sonda = {c: np.asarray(v) if isinstance(v, list) else v
             for c, v in resultado["representacoes"][rep]["sonda"].items()}
    rng = np.random.default_rng(semente)
    atual = {"variante": "original"}

    def gancho(_modulo, _entrada, saida):
        p = ordem(atual["variante"], saida.shape[-1], rng)
        return saida if p is None else saida[:, :, torch.as_tensor(p, device=saida.device)]

    alca = rede.feature_extractor.register_forward_hook(gancho)
    saida = []
    try:
        for arq, classe, rotulo, conjunto in lista:
            wav, _ = preparar(arq)
            r = {"arquivo": str(arq), "classe": classe, "rotulo": rotulo, "conjunto": conjunto,
                 "scores": {}, "quadros": {"pausa": [], "meio": [], "fala": [], "transicao": []}}
            for variante in VARIANTES:
                atual["variante"] = variante
                por_janela = []
                for x in janelas(wav):
                    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
                    with torch.inference_mode():
                        s = rede(torch.from_numpy(x).float()[None].to(dev), output_hidden_states=True)
                    h = s.hidden_states[k][0].float().cpu().numpy()
                    q = ((h - sonda["media"]) / sonda["escala"]) @ sonda["coef"] + sonda["intercepto"]
                    por_janela.append(float(q.mean()))
                    if variante == "original":
                        e = energia_dos_quadros(x, len(q))
                        de = np.abs(np.diff(e, prepend=e[0]))
                        grupos = {"pausa": e <= np.percentile(e, 20), "fala": e >= np.percentile(e, 60),
                                  "transicao": de >= np.percentile(de, 80)}
                        grupos["meio"] = ~(grupos["pausa"] | grupos["fala"])
                        for nome, sel in grupos.items():
                            if sel.any():
                                r["quadros"][nome].append(float(q[sel].mean()))
                r["scores"][variante] = float(np.mean(por_janela))
            r["quadros"] = {n: (float(np.mean(v)) if v else None) for n, v in r["quadros"].items()}
            saida.append(r)
    finally:
        alca.remove()
        atual["variante"] = "original"
    return saida


def resumir_tempo(registros: list[dict]) -> dict:
    """AUC de cada conjunto (falsos contra humanos do mesmo conjunto) por variante e
    por tipo de quadro."""
    from scripts.estudo_pente import auc

    saida = {}
    for conj in sorted({r["conjunto"] for r in registros}):
        rs = [r for r in registros if r["conjunto"] == conj]
        h = [r for r in rs if r["rotulo"] == 0]
        f = [r for r in rs if r["rotulo"] == 1]
        if not h or not f:
            continue
        linha = {v: auc([r["scores"][v] for r in f], [r["scores"][v] for r in h]) for v in VARIANTES}
        for tipo in ("pausa", "meio", "fala", "transicao"):
            a = [r["quadros"][tipo] for r in f if r["quadros"][tipo] is not None]
            b = [r["quadros"][tipo] for r in h if r["quadros"][tipo] is not None]
            linha[f"quadros_{tipo}"] = auc(a, b) if a and b else None
        saida[conj] = linha
    return saida


# --------------------------------------------------------------- por codec (eval da ASVspoof 5)

#: Códigos do `ASVspoof5.codec.config.csv`; "-" = sem codec.
NOMES_CODEC = {"-": "sem codec", "C01": "Opus banda larga", "C02": "AMR banda larga",
               "C03": "Speex banda larga", "C04": "EnCodec (neural)", "C05": "MP3", "C06": "M4A",
               "C07": "MP3 + EnCodec", "C08": "Opus banda estreita", "C09": "AMR banda estreita",
               "C10": "Speex banda estreita", "C11": "aparelhos (Bluetooth, cabo)"}


def ler_codecs(tsv: Path) -> dict[str, str]:
    """{id do arquivo: codec} a partir da amostra do eval (4ª coluna)."""
    saida = {}
    for linha in Path(tsv).read_text(encoding="utf-8").splitlines():
        partes = linha.split()
        if len(partes) >= 4:
            saida[partes[1]] = partes[3]
    return saida


def por_codec(scores: np.ndarray, arquivos, rotulos: np.ndarray, codecs: dict[str, str]) -> dict:
    """AUC dos falsos contra os humanos que passaram pelo mesmo codec."""
    from scripts.estudo_pente import auc

    cod = np.array([codecs.get(Path(str(a)).stem, "?") for a in arquivos])
    saida = {}
    for c in sorted(set(cod.tolist())):
        h, f = scores[(cod == c) & (rotulos == 0)], scores[(cod == c) & (rotulos == 1)]
        if len(h) and len(f):
            saida[c] = {"nome": NOMES_CODEC.get(c, c), "auc": auc(f, h), "humanos": int(len(h)), "falsos": int(len(f))}
    return saida


# --------------------------------------------------------------- o nosso detector

def pontuar_detector(arquivos: list[str], config_caminho: str, checkpoint: str,
                     dispositivo: str = "cuda") -> np.ndarray:
    """Log-odds do detector pelo caminho da avaliação (infer.py): 4 s desde o início."""
    import torch

    from src.config import load_config
    from src.features import FeatureExtractor
    from src.models import build_model
    from src.preprocess import load_audio, preprocess_waveform

    config = load_config(config_caminho)
    dev = torch.device(dispositivo if torch.cuda.is_available() or dispositivo == "cpu" else "cpu")
    extrator = FeatureExtractor(config["audio"], config["features"])
    modelo = build_model(config["model"]).to(dev)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
    modelo.load_state_dict(ckpt["model_state"])
    modelo.eval()
    saida = []
    for arq in arquivos:
        wav = preprocess_waveform(load_audio(arq, config["audio"]["sample_rate"]), config["audio"])
        with torch.no_grad():
            z = modelo({k: v.unsqueeze(0).to(dev) for k, v in extrator(wav).items()})[0]
        saida.append(float(z[1] - z[0]))
    return np.array(saida)


def comparar_detector(scores: dict[str, np.ndarray], conjuntos: dict[str, dict]) -> dict:
    """Mesma medida das sondas; limiar no ponto de EER do conjunto chamado dev."""
    d = conjuntos["dev"]
    e, limiar = eer(scores["dev"][d["rotulo"] == 0], scores["dev"][d["rotulo"] == 1])
    return {"limiar": limiar, "eer_dev": e,
            "conjuntos": {n: medir(scores[n], c["classe"], c["rotulo"], limiar) for n, c in conjuntos.items()}}


# --------------------------------------------------------------- linha de comando

def _imprimir_sondas(r: dict) -> None:
    reps = r["representacoes"]
    conjuntos = list(next(iter(reps.values()))["avaliar"])
    print(f"{'representação':14s} {'C':>6s} {'EER dev':>8s} {'AUC dev':>8s} "
          + " ".join(f"{'AUC ' + c:>16s}" for c in conjuntos))
    for nome, x in reps.items():
        marca = " <- melhor" if nome == r["melhor"] else ""
        print(f"{nome:14s} {x['C']:6g} {100 * x['eer_dev']:7.2f}% {x['auc_media_dev']:8.3f} "
              + " ".join(f"{(x['avaliar'][c]['auc'] or float('nan')):16.3f}" for c in conjuntos) + marca)
    for nome in (r["melhor"], "lfcc"):
        x = reps[nome]
        print(f"\n{nome}: por classe (AUC contra os humanos do conjunto; certos pelo limiar do dev)")
        for conj, m in [("dev", x["dev"])] + list(x["avaliar"].items()):
            texto = ", ".join(f"{c} {v['certos']}/{v['n']}" + (f" ({v['auc']:.2f})" if v["auc"] is not None else "")
                              for c, v in m["classes"].items())
            print(f"  {conj}: {texto}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="O que o XLS-R congelado enxerga (parte 6)")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extrair")
    e.add_argument("--saida", required=True)
    e.add_argument("--protocolo")
    e.add_argument("--pasta")
    e.add_argument("--n-humanos", type=int, default=400)
    e.add_argument("--n-ataque", type=int, default=100)
    e.add_argument("--grupo", action="append", default=[], metavar="NOME=PASTA:bonafide|spoof")
    e.add_argument("--lista", help="TSV com arquivo, classe e rotulo (a amostra do ReplayDF)")
    e.add_argument("--semente", type=int, default=42)
    e.add_argument("--dispositivo", default="cuda")
    e.add_argument("--modelo", default=MODELO_HF)

    s = sub.add_parser("sondar")
    s.add_argument("--treino", required=True)
    s.add_argument("--dev", required=True)
    s.add_argument("--avaliar", action="append", default=[], metavar="NPZ")
    s.add_argument("--json", default="outputs/ssl/sondas.json")

    x = sub.add_parser("explicar")
    x.add_argument("--sondas", default="outputs/ssl/sondas.json")
    x.add_argument("--treino", required=True)
    x.add_argument("--conjunto", action="append", required=True, metavar="NOME=NPZ",
                   help="o conjunto chamado dev dá o limiar das medidas simples")
    x.add_argument("--json", default="outputs/ssl/explicar.json")

    f = sub.add_parser("faixas")
    f.add_argument("--sondas", default="outputs/ssl/sondas.json")
    f.add_argument("--npz", action="append", required=True, help="arquivos já extraídos")
    f.add_argument("--n-por-classe", type=int, default=30)
    f.add_argument("--semente", type=int, default=42)
    f.add_argument("--dispositivo", default="cuda")
    f.add_argument("--modelo", default=MODELO_HF)
    f.add_argument("--json", default="outputs/ssl/faixas.json")

    q = sub.add_parser("tempo")
    q.add_argument("--sondas", default="outputs/ssl/sondas.json")
    q.add_argument("--npz", action="append", required=True, help="arquivos já extraídos")
    q.add_argument("--n-por-classe", type=int, default=30)
    q.add_argument("--semente", type=int, default=42)
    q.add_argument("--dispositivo", default="cuda")
    q.add_argument("--modelo", default=MODELO_HF)
    q.add_argument("--json", default="outputs/ssl/tempo.json")

    k = sub.add_parser("codecs")
    k.add_argument("--sondas", default="outputs/ssl/sondas.json")
    k.add_argument("--npz", required=True, help="o eval da ASVspoof 5 extraído")
    k.add_argument("--tsv", default="data/recortes/asvspoof5_eval_amostra10000_codec_s42.tsv")
    k.add_argument("--rep", action="append", default=[], help="representações (padrão: a melhor e o lfcc)")
    k.add_argument("--detector", default=None, help="detector.json com scores deste conjunto")
    k.add_argument("--nome", default="asv5_eval", help="nome do conjunto no detector.json")
    k.add_argument("--json", default="outputs/ssl/codecs.json")

    t = sub.add_parser("detector")
    t.add_argument("--config", default="configs/fusion_v4_asv5.yaml")
    t.add_argument("--checkpoint", default="checkpoints/fusion_lcnn_v4_asv5.pt")
    t.add_argument("--conjunto", action="append", required=True, metavar="NOME=NPZ",
                   help="o conjunto chamado dev dá o limiar")
    t.add_argument("--dispositivo", default="cuda")
    t.add_argument("--json", default="outputs/ssl/detector.json")
    args = p.parse_args(argv)

    if args.cmd == "extrair":
        if args.protocolo:
            lista = selecionar_protocolo(Path(args.protocolo), Path(args.pasta), args.n_humanos,
                                         args.n_ataque, args.semente)
        elif args.lista:
            lista = ler_lista(Path(args.lista))
        else:
            lista = selecionar_grupos(args.grupo)
        if not lista:
            p.error("nada para extrair: use --protocolo e --pasta, ou --grupo")
        print(f"Extraindo {len(lista)} arquivos -> {args.saida}", flush=True)
        extrair(lista, carregar_xlsr(args.dispositivo, args.modelo), Path(args.saida))
        print(f"Pronto: {args.saida}")
    elif args.cmd == "sondar":
        r = sondar(ler_npz(Path(args.treino)), ler_npz(Path(args.dev)),
                   {Path(a).stem: ler_npz(Path(a)) for a in args.avaliar})
        _imprimir_sondas(r)
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        print(f"\nResultado: {args.json}")
    elif args.cmd == "explicar":
        r = json.loads(Path(args.sondas).read_text(encoding="utf-8"))
        conjuntos = {}
        for item in args.conjunto:
            nome, caminho = item.split("=", 1)
            conjuntos[nome] = ler_npz(Path(caminho))
        ex = explicar(r, ler_npz(Path(args.treino)), conjuntos)
        print(f"Representação: {ex['representacao']}")
        for nome, c in ex["conjuntos"].items():
            a = c["so_medidas"]["auc"]
            print(f"\n{nome}: R² das medidas simples sobre o score = {c['r2_medidas']:.2f}; "
                  f"só as medidas: AUC {'-' if a is None else f'{a:.3f}'}")
            print("  " + ", ".join(f"{cl} {v['certos']}/{v['n']}" + (f" ({v['auc']:.2f})" if v["auc"] is not None else "")
                                   for cl, v in c["so_medidas"]["classes"].items()))
            fortes = sorted(c["correlacao"].items(),
                            key=lambda kv: -max(abs(v) for v in kv[1].values()) if kv[1] else 0)[:6]
            for n, v in fortes:
                print(f"  {n:20s} " + ", ".join(f"{k} {x:+.2f}" for k, x in v.items()))
            if c["efeitos"]:
                classes = list(c["efeitos"])
                print(f"  efeito (d de Cohen) contra os humanos: {'':4s}" + " ".join(f"{k:>8s}" for k in classes))
                for n in ex["medidas"]:
                    print(f"    {n:20s}{'':19s}" + " ".join(f"{c['efeitos'][k].get(n, float('nan')):+8.2f}" for k in classes))
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(ex, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nResultado: {args.json}")
    elif args.cmd == "tempo":
        r = json.loads(Path(args.sondas).read_text(encoding="utf-8"))
        rng = np.random.default_rng(args.semente)
        lista = []
        for caminho in args.npz:
            d = ler_npz(Path(caminho))
            for c in sorted(set(d["classe"].tolist())):
                idx = np.where(d["classe"] == c)[0]
                for i in rng.permutation(idx)[:args.n_por_classe]:
                    lista.append((Path(str(d["arquivo"][i])), str(c), int(d["rotulo"][i]), Path(caminho).stem))
        print(f"Tempo: {len(lista)} arquivos, representação {r['melhor']}", flush=True)
        registros = tempo(r, lista, *carregar_rede(args.dispositivo, args.modelo), semente=args.semente)
        resumo = resumir_tempo(registros)
        colunas = list(next(iter(resumo.values())))
        print(f"{'conjunto':12s} " + " ".join(f"{c:>18s}" for c in colunas))
        for conj, linha in resumo.items():
            print(f"{conj:12s} " + " ".join(f"{(v if v is not None else float('nan')):18.3f}" for v in linha.values()))
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"representacao": r["melhor"], "resumo": resumo,
                                               "registros": registros}, ensure_ascii=False), encoding="utf-8")
        print(f"\nResultado: {args.json}")
    elif args.cmd == "codecs":
        r = json.loads(Path(args.sondas).read_text(encoding="utf-8"))
        d = ler_npz(Path(args.npz))
        codecs = ler_codecs(Path(args.tsv))
        reps = representacoes(d)
        nomes = args.rep or [r["melhor"], "lfcc"]
        scores = {}
        for nome in nomes:
            sonda = {c: np.asarray(v) if isinstance(v, list) else v
                     for c, v in r["representacoes"][nome]["sonda"].items()}
            scores[nome] = aplicar(sonda, reps[nome].astype(np.float64))
        if args.detector:
            scores["detector"] = np.asarray(json.loads(Path(args.detector).read_text(encoding="utf-8"))["scores"][args.nome])
        saida = {n: por_codec(s, d["arquivo"], d["rotulo"], codecs) for n, s in scores.items()}
        todos = sorted({c for x in saida.values() for c in x})
        print(f"{'codec':30s} {'humanos':>7s} {'falsos':>6s} " + " ".join(f"{n:>12s}" for n in saida))
        for c in todos:
            x0 = next(x[c] for x in saida.values() if c in x)
            print(f"{c + ' ' + x0['nome']:30s} {x0['humanos']:7d} {x0['falsos']:6d} "
                  + " ".join(f"{saida[n][c]['auc']:12.3f}" if c in saida[n] else f"{'':12s}" for n in saida))
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nResultado: {args.json}")
    elif args.cmd == "detector":
        conjuntos = {}
        for item in args.conjunto:
            nome, caminho = item.split("=", 1)
            conjuntos[nome] = ler_npz(Path(caminho))
        if "dev" not in conjuntos:
            p.error("um dos conjuntos precisa se chamar dev (é dele o limiar)")
        scores = {n: pontuar_detector([str(a) for a in c["arquivo"]], args.config, args.checkpoint,
                                      args.dispositivo) for n, c in conjuntos.items()}
        r = comparar_detector(scores, conjuntos)
        print(f"Detector {args.checkpoint}: EER no dev {100 * r['eer_dev']:.2f}%")
        for nome, m in r["conjuntos"].items():
            texto = ", ".join(f"{c} {v['certos']}/{v['n']}" + (f" ({v['auc']:.2f})" if v["auc"] is not None else "")
                              for c, v in m["classes"].items())
            print(f"  {nome}: AUC {m['auc'] if m['auc'] is None else round(m['auc'], 3)}; {texto}")
        r["scores"] = {n: s.tolist() for n, s in scores.items()}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        print(f"\nResultado: {args.json}")
    else:
        r = json.loads(Path(args.sondas).read_text(encoding="utf-8"))
        rng = np.random.default_rng(args.semente)
        lista = []
        for caminho in args.npz:
            d = ler_npz(Path(caminho))
            for c in sorted(set(d["classe"].tolist())):
                idx = np.where(d["classe"] == c)[0]
                for i in rng.permutation(idx)[:args.n_por_classe]:
                    lista.append((Path(str(d["arquivo"][i])), str(c), int(d["rotulo"][i])))
        print(f"Faixas: {len(lista)} arquivos, representação {r['melhor']}", flush=True)
        saida = faixas(r, lista, carregar_xlsr(args.dispositivo, args.modelo))
        humanos = {nome: v for nome, v in saida.items() if not v["falso"]}
        print(f"{'classe':12s} {'inteiro':>8s} " + " ".join(f"{k[4:]:>10s}" for k in next(iter(saida.values())) if k.startswith("sem")))
        for nome, v in saida.items():
            print(f"{nome:12s} {np.mean(v['inteiro']):+8.2f} " + " ".join(
                f"{np.mean(v[k]):+10.2f}" for k in v if k.startswith("sem")))
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"representacao": r["melhor"], "humanos": list(humanos),
                                               "classes": saida}, ensure_ascii=False), encoding="utf-8")
        print(f"\nResultado: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
