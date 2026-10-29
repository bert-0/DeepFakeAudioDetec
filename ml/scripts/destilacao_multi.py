"""Piloto 4 de destilação: um aluno com vários professores (RESUMO_TCC §10.2.41).

Cada régua mostrou um professor forte num lugar diferente (§10.2.36 a §10.2.40):
a camada 7 do XLS-R na ASVspoof e nos clones do L01 pelo arquivo; a camada 4 no
português de audiolivro; o nosso detector nos clones do BRSpeech-DF; e só os
rótulos conhecem o F5-TTS, onde todos erram. O aluno 4 parte do aluno 3b e ganha
uma cabeça por professor, mais uma cabeça de decisão treinada com os rótulos, que
é o detector final. Cada janela ensina o que tiver: alvo que falta não entra
naquela parte da perda (máscara).

Subcomandos (de dentro de ml/, HF_HOME no cache do XLS-R):
    pca          PCA da camada extra numa amostra de janelas de uma pasta preparada
    acrescentar  grava, ao lado de uma pasta preparada, a camada extra e o score do
                 detector nas mesmas janelas (as do `destilacao.py preparar`)
    treinar      treina o aluno 4 a partir de um aluno anterior
    avaliar      o score da cabeça de decisão nas réguas
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.destilacao import (  # noqa: E402
    MAX_ALVO,
    _Juntos,
    _lote,
    carregar_dados,
    juntar,
    perda,
)

CAMADA_EXTRA = 4
#: Pesos fixados antes do treino (§10.2.41): a imitação das duas camadas pesa 1 cada.
PESO_DETECTOR, PESO_CLASSE = 0.5, 1.0


# --------------------------------------------------------------- professores extras

def janelas_da_pasta(meta: dict, transformar=None, semente: int = 42):
    """(índice do arquivo, janela) na mesma ordem do `destilacao.py preparar`, com a
    mesma sala sorteada (o rng da sala é [semente, i, 3])."""
    from scripts.estudo_ssl import janelas
    from scripts.estudo_ssl import preparar as preparar_audio

    for i, a in enumerate(meta["arquivo"]):
        wav, _ = preparar_audio(Path(str(a)))
        if transformar is not None:
            wav = transformar(wav, np.random.default_rng([semente, i, 3]))
        for x in janelas(wav):
            yield i, x


def quadros_da_camada(rede, dev, x: np.ndarray, camada: int) -> np.ndarray:
    import torch

    xn = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    with torch.inference_mode():
        s = rede(torch.from_numpy(xn).float()[None].to(dev), output_hidden_states=True)
    return s.hidden_states[camada][0].float().cpu().numpy()[:MAX_ALVO]


def pca_camada(pasta: Path, rede, dev, camada: int = CAMADA_EXTRA, n_janelas: int = 600, k: int = 256,
               semente: int = 0) -> tuple[dict, float]:
    """PCA por soma acumulada (média e X'X), sem guardar os quadros na memória."""
    meta = dict(np.load(pasta / "meta.npz", allow_pickle=False))
    rng = np.random.default_rng(semente)
    escolhidos = set(rng.permutation(len(meta["n_alvo"]))[:n_janelas].tolist())
    s1 = s2 = None
    n = 0
    for k_janela, (_, x) in enumerate(janelas_da_pasta(meta)):
        if k_janela not in escolhidos:
            continue
        a = quadros_da_camada(rede, dev, x, camada).astype(np.float64)
        s1 = a.sum(axis=0) if s1 is None else s1 + a.sum(axis=0)
        s2 = a.T @ a if s2 is None else s2 + a.T @ a
        n += len(a)
    media = s1 / n
    cov = s2 / n - np.outer(media, media)
    val, vec = np.linalg.eigh(cov)
    ordem = np.argsort(val)[::-1]
    comp = vec[:, ordem[:k]].T.astype(np.float32)
    return {"componentes": comp, "media": media.astype(np.float32)}, float(val[ordem[:k]].sum() / val.sum())


class Detector:
    """O nosso detector numa janela já preparada (4 s, 16 kHz): log-odds de sintético."""

    def __init__(self, config: str, checkpoint: str, dev):
        import torch

        from src.config import load_config
        from src.features import FeatureExtractor
        from src.models import build_model

        self.config = load_config(config)
        self.extrator = FeatureExtractor(self.config["audio"], self.config["features"])
        self.modelo = build_model(self.config["model"]).to(dev)
        self.modelo.load_state_dict(torch.load(checkpoint, map_location=dev, weights_only=False)["model_state"])
        self.modelo.eval()
        self.dev = dev

    def __call__(self, xs: list[np.ndarray]) -> np.ndarray:
        import torch

        from src.preprocess import preprocess_waveform

        feats = [self.extrator(preprocess_waveform(x, self.config["audio"])) for x in xs]
        lote = {k: torch.stack([f[k] for f in feats]).to(self.dev) for k in feats[0]}
        with torch.no_grad():
            z = self.modelo(lote).float()
        return (z[:, 1] - z[:, 0]).cpu().numpy()


def acrescentar(pasta: Path, rede=None, dev=None, pca: dict | None = None, detector=None,
                camada: int = CAMADA_EXTRA, transformar=None, semente: int = 42) -> dict:
    """Grava `alvos_c<camada>.npy` (com a PCA) e `detector.npy` nas janelas da pasta."""
    meta = dict(np.load(pasta / "meta.npz", allow_pickle=False))
    n = len(meta["n_alvo"])
    alv = None
    if rede is not None:
        dim = pca["componentes"].shape[0] if pca else int(rede.config.hidden_size)
        alv = np.lib.format.open_memmap(pasta / f"alvos_c{camada}.npy", "w+", np.float16, (n, MAX_ALVO, dim))
    det = np.full(n, np.nan, np.float32) if detector is not None else None
    k, por_arquivo, atual = 0, [], None
    for i, x in janelas_da_pasta(meta, transformar, semente):
        if alv is not None:
            a = quadros_da_camada(rede, dev, x, camada)
            if pca:
                a = (a - pca["media"]) @ pca["componentes"].T
            if len(a) != int(meta["n_alvo"][k]):
                raise ValueError(f"janela {k}: {len(a)} quadros, o preparo tinha {meta['n_alvo'][k]}")
            alv[k, :len(a)], alv[k, len(a):] = a, 0
        if det is not None:
            if atual is not None and i != atual:
                det[k - len(por_arquivo):k] = detector(por_arquivo)
                por_arquivo = []
            por_arquivo.append(x)
            atual = i
        k += 1
        if k % 2000 == 0:
            print(f"  {k}/{n} janelas", flush=True)
    if k != n:
        raise ValueError(f"{k} janelas agora, {n} no preparo: a pasta não é desta origem")
    if det is not None and por_arquivo:
        det[k - len(por_arquivo):k] = detector(por_arquivo)
    if alv is not None:
        alv.flush()
    if det is not None:
        np.save(pasta / "detector.npy", det)
    return {"janelas": n}


# --------------------------------------------------------------- dados

class _Vazio:
    """Alvo que a pasta não tem: zeros do formato certo (a máscara o tira da perda)."""

    def __init__(self, n: int, forma: tuple):
        self.shape = (n, *forma)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, chave):
        idx = chave[0] if isinstance(chave, tuple) else chave
        return np.zeros((len(np.atleast_1d(idx)), *self.shape[1:]), np.float16)


def carregar(pasta: Path, camada: int = CAMADA_EXTRA) -> dict:
    d = carregar_dados(pasta)
    c = pasta / f"alvos_c{camada}.npy"
    d["alvos_extra"] = np.load(c, mmap_mode="r") if c.exists() else None
    det = pasta / "detector.npy"
    d["detector"] = np.load(det) if det.exists() else None
    return d


def juntar_multi(conjuntos: list[dict]) -> dict:
    """`juntar` mais os alvos extras, com máscara de quem os tem."""
    base = juntar(conjuntos)
    forma = next((c["alvos_extra"].shape[1:] for c in conjuntos if c.get("alvos_extra") is not None), None)
    n = [len(c["n_alvo"]) for c in conjuntos]
    if forma is None:
        base["alvos_extra"], base["tem_extra"] = None, np.zeros(sum(n), bool)
    else:
        partes = [c["alvos_extra"] if c.get("alvos_extra") is not None else _Vazio(k, forma)
                  for c, k in zip(conjuntos, n, strict=True)]
        base["alvos_extra"] = partes[0] if len(partes) == 1 else _Juntos(partes)
        base["tem_extra"] = np.concatenate([np.full(k, c.get("alvos_extra") is not None)
                                            for c, k in zip(conjuntos, n, strict=True)])
    det = np.concatenate([c["detector"] if c.get("detector") is not None else np.full(k, np.nan, np.float32)
                          for c, k in zip(conjuntos, n, strict=True)])
    base["detector"], base["tem_detector"] = np.nan_to_num(det), ~np.isnan(det)
    return base


def lote_multi(dados, idx, est, dispositivo, rng=None):
    """O lote do `destilacao._lote` mais os alvos extras, o detector e o rótulo da janela."""
    import torch

    idx = np.sort(idx)
    # A PCA é a do treino: os devs antigos guardam a camada 7 com 1.024 dimensões.
    x, y7, n_e, n_a = _lote(dados, idx, est["media7"], est["desvio7"], dispositivo, rng=rng, pca=est.get("pca"))
    y4 = None
    if dados.get("alvos_extra") is not None:
        y4 = torch.from_numpy((np.asarray(dados["alvos_extra"][idx], np.float32) - est["media4"]) / est["desvio4"])
        y4 = y4.to(dispositivo)
    t = lambda a, tipo: torch.from_numpy(np.asarray(a)).to(dispositivo, tipo)  # noqa: E731
    det = t((dados["detector"][idx] - est["media_det"]) / est["desvio_det"], torch.float32)
    rotulo = t(np.asarray(dados["rotulo"])[np.asarray(dados["janela_arquivo"])[idx]], torch.long)
    return (x, n_e, n_a, y7, y4, t(dados["tem_extra"][idx], torch.bool), det,
            t(dados["tem_detector"][idx], torch.bool), rotulo)


def estatisticas(dados: dict, amostra: int = 500, semente: int = 0, media7=None, desvio7=None) -> dict:
    from scripts.destilacao import estatisticas_alvo

    if media7 is None:
        media7, desvio7 = estatisticas_alvo(dados, amostra, semente)
    est = {"media7": media7, "desvio7": desvio7}
    com = np.where(dados["tem_extra"])[0]
    if len(com):
        rng = np.random.default_rng(semente)
        sel = np.sort(rng.permutation(com)[:amostra])
        q = np.concatenate([np.asarray(dados["alvos_extra"][i, :dados["n_alvo"][i]], np.float32) for i in sel])
        est["media4"], est["desvio4"] = q.mean(axis=0), q.std(axis=0) + 1e-4
    else:
        est["media4"] = est["desvio4"] = None
    d = dados["detector"][dados["tem_detector"]]
    est["media_det"], est["desvio_det"] = (float(d.mean()), float(d.std() + 1e-6)) if len(d) else (0.0, 1.0)
    return est


# --------------------------------------------------------------- o aluno 4

def criar_aluno4(dim_entrada: int = 140, dim: int = 384, camadas: int = 6, cabecas: int = 4, dim_alvo: int = 256,
                 dropout: float = 0.1, dim_extra: int = 256):
    """O mesmo corpo do aluno da destilação (os pesos do 3b carregam por nome) e três
    cabeças novas: a camada extra, o score do detector e a decisão real ou falso."""
    import torch
    from torch import nn

    class Aluno4(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {"dim_entrada": dim_entrada, "dim": dim, "camadas": camadas, "cabecas": cabecas,
                           "dim_alvo": dim_alvo, "dropout": dropout, "dim_extra": dim_extra}
            self.entrada = nn.Linear(dim_entrada, dim)
            self.reduz = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)
            self.posicao = nn.Conv1d(dim, dim, kernel_size=31, padding=15, groups=max(1, dim // 16))
            camada = nn.TransformerEncoderLayer(dim, cabecas, 4 * dim, dropout, batch_first=True,
                                                activation="gelu", norm_first=True)
            self.encoder = nn.TransformerEncoder(camada, camadas, enable_nested_tensor=False)
            self.norma = nn.LayerNorm(dim)
            self.saida = nn.Linear(dim, dim_alvo)
            self.saida_extra = nn.Linear(dim, dim_extra)
            self.cab_detector = nn.Linear(2 * dim, 1)
            self.cab_classe = nn.Linear(2 * dim, 2)

        def forward(self, x, n_entrada):
            h = self.entrada(x)
            h = nn.functional.gelu(self.reduz(h.transpose(1, 2))).transpose(1, 2)
            h = h + nn.functional.gelu(self.posicao(h.transpose(1, 2))).transpose(1, 2)
            n_saida = torch.div(n_entrada + 1, 2, rounding_mode="floor")
            vazio = torch.arange(h.shape[1], device=h.device)[None] >= n_saida[:, None]
            h = self.norma(self.encoder(h, src_key_padding_mask=vazio))
            peso = (~vazio).float().unsqueeze(-1)
            n = peso.sum(dim=1).clamp(min=1)
            media = (h * peso).sum(dim=1) / n
            desvio = (((h - media.unsqueeze(1)) ** 2 * peso).sum(dim=1) / n).clamp(min=1e-6).sqrt()
            pool = torch.cat([media, desvio], dim=-1)
            return (self.saida(h), self.saida_extra(h), self.cab_detector(pool).squeeze(-1),
                    self.cab_classe(pool), n_saida)

    return Aluno4()


def perda_total(saidas, alvos) -> tuple:
    """(total, partes) do lote; cada parte só nas janelas que têm o alvo dela."""
    import torch

    pred7, pred4, pdet, logits, n_s = saidas
    _, _, n_a, y7, y4, tem4, det, temdet, rotulo = alvos
    n_val = torch.minimum(n_s, n_a)
    l7 = perda(pred7.float(), y7, n_val)
    zero = pred7.sum() * 0
    l4 = perda(pred4[tem4].float(), y4[tem4], n_val[tem4]) if y4 is not None and bool(tem4.any()) else zero
    ld = (torch.nn.functional.smooth_l1_loss(pdet[temdet].float(), det[temdet])
          if bool(temdet.any()) else zero)
    lc = torch.nn.functional.cross_entropy(logits.float(), rotulo)
    total = l7 + l4 + PESO_DETECTOR * ld + PESO_CLASSE * lc
    return total, {"c7": float(l7), "c4": float(l4), "detector": float(ld), "classe": float(lc)}


def avaliar_perda4(aluno, dados, est, lote, dispositivo) -> float:
    import torch

    aluno.eval()
    soma, cont = 0.0, 0
    with torch.inference_mode():
        for b in range(0, len(dados["n_alvo"]), lote):
            a = lote_multi(dados, np.arange(b, min(b + lote, len(dados["n_alvo"]))), est, dispositivo)
            total, _ = perda_total(aluno(a[0], a[1]), a)
            soma, cont = soma + float(total) * len(a[0]), cont + len(a[0])
    return soma / cont


def treinar4(treino: dict, devs: list[dict], saida: Path, inicial: Path, epocas: int = 12, lote: int = 16,
             lr: float = 3e-4, semente: int = 42, dispositivo: str = "cuda") -> dict:
    import torch

    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    dev_t = torch.device(dispositivo if torch.cuda.is_available() or dispositivo == "cpu" else "cpu")
    ck0 = torch.load(inicial, map_location=dev_t, weights_only=False)
    aluno = criar_aluno4(**{k: v for k, v in ck0["config"].items() if k != "dim_extra"},
                         dim_extra=treino["alvos_extra"].shape[2]).to(dev_t)
    faltam, sobram = aluno.load_state_dict(ck0["estado"], strict=False)
    if sobram:
        raise ValueError(f"o aluno inicial tem pesos que o aluno 4 não tem: {sobram}")
    # A normalização da camada 7 é a do aluno inicial: a cabeça herdada fala nessa escala.
    est = estatisticas(treino, media7=ck0["media_alvo"], desvio7=ck0["desvio_alvo"])
    est["pca"] = treino.get("pca")
    otim = torch.optim.AdamW(aluno.parameters(), lr=lr, weight_decay=0.01)
    passos = epocas * int(np.ceil(len(treino["n_alvo"]) / lote))
    aquecimento = min(500, max(1, passos // 10))
    agenda = torch.optim.lr_scheduler.LambdaLR(
        otim, lambda p: min(1.0, (p + 1) / aquecimento) * 0.5 * (1 + np.cos(np.pi * min(p, passos) / passos)))
    n_param = sum(p.numel() for p in aluno.parameters())
    print(f"Aluno 4: {n_param / 1e6:.2f} milhões de parâmetros, a partir de {inicial} (novos: {', '.join(faltam)}); "
          f"{len(treino['n_alvo'])} janelas, {int(treino['tem_extra'].sum())} com a camada extra, "
          f"{int(treino['tem_detector'].sum())} com o detector; {epocas} épocas", flush=True)
    historico, melhor = [], float("inf")
    for epoca in range(1, epocas + 1):
        inicio = time.time()
        aluno.train()
        ordem = rng.permutation(len(treino["n_alvo"]))
        soma, cont, partes = 0.0, 0, {}
        for b in range(0, len(ordem), lote):
            a = lote_multi(treino, ordem[b:b + lote], est, dev_t, rng=rng)
            total, p = perda_total(aluno(a[0], a[1]), a)
            otim.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(aluno.parameters(), 5.0)
            otim.step()
            agenda.step()
            soma, cont = soma + float(total.detach()) * len(a[0]), cont + len(a[0])
            for k, v in p.items():
                partes[k] = partes.get(k, 0.0) + v * len(a[0])
        perdas_dev = [avaliar_perda4(aluno, d, est, lote, dev_t) for d in devs]
        perda_dev = float(np.mean(perdas_dev))
        historico.append({"epoca": epoca, "perda_treino": soma / cont, "perda_dev": perda_dev,
                          "perdas_dev": perdas_dev, "partes_treino": {k: v / cont for k, v in partes.items()}})
        marca = ""
        if perda_dev < melhor:
            melhor = perda_dev
            saida.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"estado": aluno.state_dict(), "config": aluno.config, "estatisticas": est,
                        "epoca": epoca, "perda_dev": perda_dev, "camada": int(treino["camada"]),
                        "camada_extra": CAMADA_EXTRA, "pca": treino.get("pca"), "inicial": str(inicial)}, saida)
            marca = "  <- salvo"
        print(f"época {epoca:3d}: perda treino {soma / cont:.4f} "
              f"({', '.join(f'{k} {v / cont:.3f}' for k, v in partes.items())}), dev {perda_dev:.4f} "
              f"({time.time() - inicio:.0f} s){marca}", flush=True)
    return {"historico": historico, "melhor_perda_dev": melhor, "parametros": n_param}


# --------------------------------------------------------------- avaliar

def pontuar(aluno, dados, dispositivo, lote: int = 32) -> np.ndarray:
    """Por arquivo: a média, nas janelas, dos log-odds da cabeça de decisão."""
    import torch

    from scripts.destilacao import _lote as lote_simples

    por_janela = []
    with torch.inference_mode():
        for b in range(0, len(dados["n_alvo"]), lote):
            x, _, n_e, _ = lote_simples(dados, np.arange(b, min(b + lote, len(dados["n_alvo"]))), None, None,
                                        dispositivo, com_alvo=False)
            logits = aluno(x, n_e)[3].float()
            por_janela.extend((logits[:, 1] - logits[:, 0]).cpu().numpy().tolist())
    por_janela = np.array(por_janela)
    return np.array([por_janela[dados["janela_arquivo"] == i].mean() for i in range(len(dados["arquivo"]))])


def avaliar4(caminho: Path, dev: dict, reguas: dict[str, dict], dispositivo: str = "cuda") -> dict:
    import torch

    from scripts.estudo_ssl import eer, medir

    dev_t = torch.device(dispositivo if torch.cuda.is_available() or dispositivo == "cpu" else "cpu")
    ck = torch.load(caminho, map_location=dev_t, weights_only=False)
    aluno = criar_aluno4(**ck["config"]).to(dev_t)
    aluno.load_state_dict(ck["estado"])
    aluno.eval()
    s_dev = pontuar(aluno, dev, dev_t)
    e, limiar = eer(s_dev[dev["rotulo"] == 0], s_dev[dev["rotulo"] == 1])
    r = {"epoca": ck["epoca"], "perda_dev": ck["perda_dev"], "eer_dev": e, "limiar": limiar,
         "dev": medir(s_dev, dev["classe"], dev["rotulo"], limiar), "reguas": {}, "scores": {"dev": s_dev.tolist()},
         "arquivos": {"dev": [str(a) for a in dev["arquivo"]]}}
    r["auc_media_dev"] = float(np.mean([c["auc"] for c in r["dev"]["classes"].values() if c["falso"]]))
    for nome, d in reguas.items():
        s = pontuar(aluno, d, dev_t)
        r["reguas"][nome] = medir(s, d["classe"], d["rotulo"], limiar)
        r["scores"][nome] = s.tolist()
        r["arquivos"][nome] = [str(a) for a in d["arquivo"]]
    return r


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Piloto 4: aluno com vários professores")
    sub = p.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("pca")
    pc.add_argument("--pasta", required=True, help="pasta preparada (as janelas saem dela)")
    pc.add_argument("--n-janelas", type=int, default=600)
    pc.add_argument("--saida", required=True)
    pc.add_argument("--dispositivo", default="cuda")
    ac = sub.add_parser("acrescentar")
    ac.add_argument("--pasta", required=True)
    ac.add_argument("--pca", help="PCA da camada extra (subcomando pca)")
    ac.add_argument("--sem-camada", action="store_true", help="só o detector")
    ac.add_argument("--sem-detector", action="store_true", help="só a camada extra")
    ac.add_argument("--config", default="configs/fusion_v4_asv5.yaml")
    ac.add_argument("--checkpoint", default="checkpoints/fusion_lcnn_v4_asv5.pt")
    ac.add_argument("--sala", help="a mesma pasta do BUT ReverbDB do preparo (as salas se repetem pelo rng)")
    ac.add_argument("--semente", type=int, default=42, help="a do preparo")
    ac.add_argument("--dispositivo", default="cuda")
    t = sub.add_parser("treinar")
    t.add_argument("--treino", action="append", required=True)
    t.add_argument("--dev", action="append", required=True)
    t.add_argument("--inicial", required=True, help="o aluno de partida (o 3b)")
    t.add_argument("--saida", required=True)
    t.add_argument("--epocas", type=int, default=12)
    t.add_argument("--lote", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--dispositivo", default="cuda")
    v = sub.add_parser("avaliar")
    v.add_argument("--aluno", required=True)
    v.add_argument("--dev", required=True, help="dá o limiar (ponto de EER)")
    v.add_argument("--regua", action="append", default=[])
    v.add_argument("--dispositivo", default="cuda")
    v.add_argument("--json", default="outputs/destilacao/avaliacao_piloto4.json")
    args = p.parse_args(argv)

    if args.cmd in ("pca", "acrescentar"):
        import torch

        from scripts.estudo_ssl import carregar_rede
    if args.cmd == "pca":
        rede, dev = carregar_rede(args.dispositivo)
        pca, fracao = pca_camada(Path(args.pasta), rede, dev, n_janelas=args.n_janelas)
        Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.saida, **pca, fracao=fracao)
        print(f"PCA da camada {CAMADA_EXTRA}: {100 * fracao:.1f}% da variância em {len(pca['componentes'])} "
              f"componentes -> {args.saida}")
        return 0
    if args.cmd == "acrescentar":
        rede = dev = pca = None
        if not args.sem_camada:
            rede, dev = carregar_rede(args.dispositivo)
            if args.pca:
                z = np.load(args.pca)
                pca = {"componentes": z["componentes"], "media": z["media"]}
        dev_t = dev or torch.device(args.dispositivo if torch.cuda.is_available() else "cpu")
        detector = None if args.sem_detector else Detector(args.config, args.checkpoint, dev_t)
        transformar = None
        if args.sala:
            from src.preprocess.sala import Sala

            transformar = Sala(args.sala, 16000)
        r = acrescentar(Path(args.pasta), rede, dev, pca, detector, transformar=transformar, semente=args.semente)
        print(f"Pronto: {r['janelas']} janelas em {args.pasta}")
        return 0
    if args.cmd == "treinar":
        r = treinar4(juntar_multi([carregar(Path(x)) for x in args.treino]),
                     [juntar_multi([carregar(Path(x))]) for x in args.dev], Path(args.saida), Path(args.inicial),
                     args.epocas, args.lote, args.lr, dispositivo=args.dispositivo)
        Path(args.saida).with_suffix(".json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(f"\nMelhor perda no dev: {r['melhor_perda_dev']:.4f}. Aluno: {args.saida}")
        return 0
    r = avaliar4(Path(args.aluno), carregar_dados(Path(args.dev)),
                 {Path(x).name: carregar_dados(Path(x)) for x in args.regua}, args.dispositivo)
    print(f"Aluno 4 da época {r['epoca']}: EER no dev {100 * r['eer_dev']:.2f}%, AUC média no dev "
          f"{r['auc_media_dev']:.3f}, limiar {r['limiar']:.3f}")
    for nome, m in r["reguas"].items():
        auc = "-" if m["auc"] is None else f"{m['auc']:.3f}"
        print(f"  {nome}: AUC {auc}; " + ", ".join(
            f"{c} {x['certos']}/{x['n']}" + ("" if x["auc"] is None else f" ({x['auc']:.2f})")
            for c, x in m["classes"].items()))
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    print(f"Resultado: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
