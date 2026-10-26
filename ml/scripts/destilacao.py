"""Piloto de destilação: um modelo pequeno, nosso, aprende a imitar a camada 7 do XLS-R.

A parte 6 (RESUMO_TCC §10.2.22) mostrou que a camada 7 do XLS-R separa ataques
que ele nunca viu, pela ordem da fala no tempo, e as partes 8 e 9 mostraram que
medidas feitas à mão não chegam lá. Aqui o XLS-R entra como professor, só no
treino: responde, quadro a quadro (20 ms), sobre os áudios que já usamos, e um
aluno pequeno (transformer de 4 camadas, ~3,5 milhões de parâmetros) aprende a
dar a mesma resposta a partir do LFCC e do log-mel do nosso detector. O detector
final é o aluno e roda sem o XLS-R.

Critério combinado antes de rodar (RESUMO_TCC §10.2.29): o piloto passa se uma
sonda linear nas representações do aluno (média no tempo, treinada no mesmo
treino da parte 6) superar a sonda no LFCC em pelo menos 0,05 de AUC no dev da
ASVspoof 5 (média dos ataques) e na 2019 LA. O checkpoint do aluno é escolhido
pela perda de destilação no dev, que não usa rótulo.

Subcomandos (de dentro de ml/):
    preparar  o professor responde sobre os arquivos de um .npz da parte 6 (GPU;
              HF_HOME no cache do XLS-R); grava entradas e alvos em outputs/destilacao/
    treinar   treina o aluno (quem roda é o grupo, GTX 1650)
    avaliar   as réguas da parte 6 e o critério

    python scripts/destilacao.py preparar --npz outputs/ssl/asv5_train.npz --saida outputs/destilacao/asv5_train
    python scripts/destilacao.py treinar --treino outputs/destilacao/asv5_train --dev outputs/destilacao/asv5_dev \\
        --saida checkpoints/aluno_xlsr_l7.pt
    python scripts/destilacao.py avaliar --aluno checkpoints/aluno_xlsr_l7.pt --treino outputs/destilacao/asv5_train \\
        --dev outputs/destilacao/asv5_dev --avaliar outputs/destilacao/asv19_train --avaliar outputs/destilacao/ensaios
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SR = 16000
CAMADA = 7
#: 4 s: 401 quadros de 10 ms nas entradas (STFT centrado) e 199 de 20 ms no professor.
MAX_ENTRADA, MAX_ALVO = 401, 199
MEL_CFG = {"n_fft": 512, "win_length": 400, "hop_length": 160, "n_mels": 80}
MARGEM = 0.05


# --------------------------------------------------------------- entradas do aluno

def entradas(x: np.ndarray) -> np.ndarray:
    """LFCC com deltas (60) + log-mel (80) por quadro de 10 ms, cada dimensão com
    média 0 e desvio 1 na janela: (T, 140)."""
    from scripts.estudo_ssl import LFCC_CFG
    from src.features.lfcc import compute_lfcc
    from src.features.spectrogram import compute_log_mel

    f = np.concatenate([compute_lfcc(x, SR, LFCC_CFG), compute_log_mel(x, SR, MEL_CFG)]).T
    return ((f - f.mean(axis=0)) / (f.std(axis=0) + 1e-5)).astype(np.float32)


# --------------------------------------------------------------- preparar (professor)

def preparar(origem: Path, destino: Path, rede, dev, camada: int = CAMADA) -> dict:
    """Uma amostra por janela de 4 s (as mesmas janelas da parte 6). Duas passadas:
    a primeira só conta as janelas, para gravar direto em disco (memmap)."""
    import torch

    from scripts.estudo_ssl import janelas, ler_npz
    from scripts.estudo_ssl import preparar as preparar_audio

    d = ler_npz(origem)
    arquivos = [Path(str(a)) for a in d["arquivo"]]
    contagem = [len(janelas(preparar_audio(a)[0])) for a in arquivos]
    n = sum(contagem)
    dim_alvo = int(rede.config.hidden_size)
    destino.mkdir(parents=True, exist_ok=True)
    ent = np.lib.format.open_memmap(destino / "entradas.npy", "w+", np.float16, (n, MAX_ENTRADA, 140))
    alv = np.lib.format.open_memmap(destino / "alvos.npy", "w+", np.float16, (n, MAX_ALVO, dim_alvo))
    janela_arquivo = np.zeros(n, np.int32)
    n_ent, n_alv = np.zeros(n, np.int32), np.zeros(n, np.int32)
    k = 0
    for i, arq in enumerate(arquivos):
        wav, _ = preparar_audio(arq)
        for x in janelas(wav):
            e = entradas(x)[:MAX_ENTRADA]
            xn = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
            with torch.inference_mode():
                s = rede(torch.from_numpy(xn).float()[None].to(dev), output_hidden_states=True)
            a = s.hidden_states[camada][0].float().cpu().numpy()[:MAX_ALVO]
            ent[k, :len(e)], ent[k, len(e):] = e, 0
            alv[k, :len(a)], alv[k, len(a):] = a, 0
            janela_arquivo[k], n_ent[k], n_alv[k] = i, len(e), len(a)
            k += 1
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(arquivos)}", flush=True)
    ent.flush()
    alv.flush()
    np.savez(destino / "meta.npz", arquivo=d["arquivo"], classe=d["classe"], rotulo=d["rotulo"],
             janela_arquivo=janela_arquivo, n_entrada=n_ent, n_alvo=n_alv, camada=camada)
    return {"arquivos": len(arquivos), "janelas": n}


def carregar_dados(pasta: Path) -> dict:
    meta = dict(np.load(pasta / "meta.npz", allow_pickle=False))
    meta["entradas"] = np.load(pasta / "entradas.npy", mmap_mode="r")
    meta["alvos"] = np.load(pasta / "alvos.npy", mmap_mode="r")
    return meta


# --------------------------------------------------------------- o aluno

def criar_aluno(dim_entrada: int = 140, dim: int = 256, camadas: int = 4, cabecas: int = 4,
                dim_alvo: int = 1024, dropout: float = 0.1):
    import torch
    from torch import nn

    class Aluno(nn.Module):
        """Projeção, convolução de passo 2 (10 ms -> 20 ms, o passo do professor),
        posição por convolução (como no wav2vec 2.0) e um transformer pequeno."""

        def __init__(self):
            super().__init__()
            self.config = {"dim_entrada": dim_entrada, "dim": dim, "camadas": camadas,
                           "cabecas": cabecas, "dim_alvo": dim_alvo, "dropout": dropout}
            self.entrada = nn.Linear(dim_entrada, dim)
            self.reduz = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)
            self.posicao = nn.Conv1d(dim, dim, kernel_size=31, padding=15, groups=max(1, dim // 16))
            camada = nn.TransformerEncoderLayer(dim, cabecas, 4 * dim, dropout, batch_first=True,
                                                activation="gelu", norm_first=True)
            self.encoder = nn.TransformerEncoder(camada, camadas, enable_nested_tensor=False)
            self.norma = nn.LayerNorm(dim)
            self.saida = nn.Linear(dim, dim_alvo)

        def forward(self, x, n_entrada):
            h = self.entrada(x)
            h = nn.functional.gelu(self.reduz(h.transpose(1, 2))).transpose(1, 2)
            h = h + nn.functional.gelu(self.posicao(h.transpose(1, 2))).transpose(1, 2)
            n_saida = torch.div(n_entrada + 1, 2, rounding_mode="floor")
            vazio = torch.arange(h.shape[1], device=h.device)[None] >= n_saida[:, None]
            h = self.encoder(h, src_key_padding_mask=vazio)
            return self.saida(self.norma(h)), n_saida

    return Aluno()


def perda(pred, alvo, n_valido):
    """L1 + (-log sigmoide do cosseno) por quadro, como no DistilHuBERT, só nos válidos."""
    import torch

    t = min(pred.shape[1], alvo.shape[1])
    pred, alvo = pred[:, :t], alvo[:, :t]
    valido = (torch.arange(t, device=pred.device)[None] < n_valido[:, None]).float()
    l1 = (pred - alvo).abs().mean(dim=-1)
    cos = torch.nn.functional.cosine_similarity(pred, alvo, dim=-1)
    por_quadro = l1 - torch.nn.functional.logsigmoid(cos)
    return (por_quadro * valido).sum() / valido.sum().clamp(min=1)


def _lote(dados, idx, media, desvio, dispositivo):
    import torch

    idx = np.sort(idx)
    x = torch.from_numpy(np.asarray(dados["entradas"][idx], dtype=np.float32)).to(dispositivo)
    y = np.asarray(dados["alvos"][idx], dtype=np.float32)
    y = torch.from_numpy((y - media) / desvio).to(dispositivo)
    n_e = torch.from_numpy(dados["n_entrada"][idx].astype(np.int64)).to(dispositivo)
    n_a = torch.from_numpy(dados["n_alvo"][idx].astype(np.int64)).to(dispositivo)
    return x, y, n_e, n_a


def estatisticas_alvo(dados, amostra: int = 500, semente: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Média e desvio por dimensão dos alvos (quadros válidos de uma amostra de janelas)."""
    rng = np.random.default_rng(semente)
    idx = np.sort(rng.permutation(len(dados["n_alvo"]))[:amostra])
    quadros = np.concatenate([np.asarray(dados["alvos"][i, :dados["n_alvo"][i]], dtype=np.float32) for i in idx])
    return quadros.mean(axis=0), quadros.std(axis=0) + 1e-4


def treinar(treino: dict, dev: dict, saida: Path, epocas: int = 30, lote: int = 16, lr: float = 5e-4,
            semente: int = 42, dispositivo: str = "cuda", arquitetura: dict | None = None) -> dict:
    import torch

    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    dev_t = torch.device(dispositivo if torch.cuda.is_available() or dispositivo == "cpu" else "cpu")
    media, desvio = estatisticas_alvo(treino)
    aluno = criar_aluno(dim_alvo=treino["alvos"].shape[2], **(arquitetura or {})).to(dev_t)
    otim = torch.optim.AdamW(aluno.parameters(), lr=lr, weight_decay=0.01)
    passos = epocas * int(np.ceil(len(treino["n_alvo"]) / lote))
    aquecimento = min(500, max(1, passos // 10))
    agenda = torch.optim.lr_scheduler.LambdaLR(
        otim, lambda p: min(1.0, (p + 1) / aquecimento) * 0.5 * (1 + np.cos(np.pi * min(p, passos) / passos)))
    usar_amp = dev_t.type == "cuda"
    escala = torch.amp.GradScaler("cuda", enabled=usar_amp)
    n_param = sum(p.numel() for p in aluno.parameters())
    print(f"Aluno: {n_param / 1e6:.2f} milhões de parâmetros; {len(treino['n_alvo'])} janelas de treino, "
          f"{len(dev['n_alvo'])} de dev; {epocas} épocas; {dev_t}", flush=True)
    historico, melhor = [], float("inf")
    for epoca in range(1, epocas + 1):
        inicio = time.time()
        aluno.train()
        ordem = rng.permutation(len(treino["n_alvo"]))
        soma, cont = 0.0, 0
        for b in range(0, len(ordem), lote):
            x, y, n_e, n_a = _lote(treino, ordem[b:b + lote], media, desvio, dev_t)
            with torch.autocast(dev_t.type, enabled=usar_amp):
                pred, n_s = aluno(x, n_e)
            p = perda(pred.float(), y, torch.minimum(n_s, n_a))
            otim.zero_grad(set_to_none=True)
            escala.scale(p).backward()
            escala.unscale_(otim)
            torch.nn.utils.clip_grad_norm_(aluno.parameters(), 5.0)
            escala.step(otim)
            escala.update()
            agenda.step()
            soma, cont = soma + float(p.detach()) * len(x), cont + len(x)
        perda_dev = avaliar_perda(aluno, dev, media, desvio, lote, dev_t)
        historico.append({"epoca": epoca, "perda_treino": soma / cont, "perda_dev": perda_dev})
        marca = ""
        if perda_dev < melhor:
            melhor = perda_dev
            saida.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"estado": aluno.state_dict(), "config": aluno.config, "media_alvo": media,
                        "desvio_alvo": desvio, "epoca": epoca, "perda_dev": perda_dev,
                        "camada": int(treino["camada"])}, saida)
            marca = "  <- salvo"
        print(f"época {epoca:3d}: perda treino {soma / cont:.4f}, dev {perda_dev:.4f} "
              f"({time.time() - inicio:.0f} s){marca}", flush=True)
    return {"historico": historico, "melhor_perda_dev": melhor, "parametros": n_param}


def avaliar_perda(aluno, dados, media, desvio, lote, dispositivo) -> float:
    import torch

    aluno.eval()
    soma, cont = 0.0, 0
    with torch.inference_mode():
        for b in range(0, len(dados["n_alvo"]), lote):
            x, y, n_e, n_a = _lote(dados, np.arange(b, min(b + lote, len(dados["n_alvo"]))), media, desvio, dispositivo)
            pred, n_s = aluno(x, n_e)
            soma += float(perda(pred.float(), y, torch.minimum(n_s, n_a))) * len(x)
            cont += len(x)
    return soma / cont


# --------------------------------------------------------------- avaliar

def carregar_aluno(caminho: Path, dispositivo: str = "cuda"):
    import torch

    dev_t = torch.device(dispositivo if torch.cuda.is_available() or dispositivo == "cpu" else "cpu")
    ck = torch.load(caminho, map_location=dev_t, weights_only=False)
    aluno = criar_aluno(**ck["config"]).to(dev_t)
    aluno.load_state_dict(ck["estado"])
    return aluno.eval(), ck, dev_t


def representar(aluno, ck, dados, dispositivo, lote: int = 32) -> np.ndarray:
    """Por arquivo: média no tempo das respostas do aluno (na escala do professor),
    média simples das janelas, como na extração da parte 6."""
    import torch

    media, desvio = ck["media_alvo"], ck["desvio_alvo"]
    por_janela = []
    with torch.inference_mode():
        for b in range(0, len(dados["n_alvo"]), lote):
            x, _, n_e, _ = _lote(dados, np.arange(b, min(b + lote, len(dados["n_alvo"]))), media, desvio, dispositivo)
            pred, n_s = aluno(x, n_e)
            pred = pred.float().cpu().numpy() * desvio + media
            for j, n in enumerate(n_s.cpu().numpy()):
                por_janela.append(pred[j, :max(1, int(n))].mean(axis=0))
    por_janela = np.array(por_janela)
    saida = np.zeros((len(dados["arquivo"]), por_janela.shape[1]), np.float32)
    for i in range(len(dados["arquivo"])):
        saida[i] = por_janela[dados["janela_arquivo"] == i].mean(axis=0)
    return saida


def avaliar(aluno_caminho: Path, sondas: dict, treino: dict, dev: dict, avaliar_: dict[str, dict],
            dispositivo: str = "cuda", regua_2019: str = "asv19_train") -> dict:
    from scripts.estudo_ssl import aplicar, eer, medir, treinar_sonda

    aluno, ck, dev_t = carregar_aluno(aluno_caminho, dispositivo)
    camada = f"camada_{int(ck['camada']):02d}"
    conjuntos = {"treino": treino, "dev": dev, **avaliar_}
    emb = {n: representar(aluno, ck, d, dev_t) for n, d in conjuntos.items()}
    saida = {"camada": camada, "epoca": ck["epoca"], "perda_dev": ck["perda_dev"], "sondas": {}}

    # 1) a sonda do professor (parte 6) aplicada às respostas do aluno
    prof = {k: np.asarray(v) if isinstance(v, list) else v
            for k, v in sondas["representacoes"][camada]["sonda"].items()}
    # 2) uma sonda nova, treinada nas respostas do aluno (a régua justa, como a do LFCC)
    nova = treinar_sonda(emb["treino"].astype(np.float64), treino["rotulo"])
    for nome_sonda, sonda in (("professor", prof), ("nova", nova)):
        s_dev = aplicar(sonda, emb["dev"].astype(np.float64))
        e, limiar = eer(s_dev[dev["rotulo"] == 0], s_dev[dev["rotulo"] == 1])
        r = {"eer_dev": e, "dev": medir(s_dev, dev["classe"], dev["rotulo"], limiar)}
        r["auc_media_dev"] = float(np.mean([c["auc"] for c in r["dev"]["classes"].values() if c["falso"]]))
        scores = {n: aplicar(sonda, emb[n].astype(np.float64)) for n in avaliar_}
        r["avaliar"] = {n: medir(scores[n], d["classe"], d["rotulo"], limiar) for n, d in avaliar_.items()}
        r["scores"] = {n: s.tolist() for n, s in scores.items()}
        saida["sondas"][nome_sonda] = r

    lfcc = sondas["representacoes"]["lfcc"]
    nova_r = saida["sondas"]["nova"]
    if regua_2019 in avaliar_:
        g_dev = nova_r["auc_media_dev"] - lfcc["auc_media_dev"]
        g_19 = nova_r["avaliar"][regua_2019]["auc"] - lfcc["avaliar"][regua_2019]["auc"]
        saida["criterio"] = {"ganho_dev": g_dev, "ganho_2019": g_19,
                             "passa": bool(g_dev >= MARGEM and g_19 >= MARGEM)}
    saida["referencias"] = {nome: {"auc_media_dev": sondas["representacoes"][nome]["auc_media_dev"],
                                   **{n: sondas["representacoes"][nome]["avaliar"].get(n, {}).get("auc")
                                      for n in avaliar_}}
                            for nome in ("lfcc", camada)}
    return saida


# --------------------------------------------------------------- linha de comando

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Piloto de destilação do XLS-R num aluno pequeno")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("preparar")
    a.add_argument("--npz", required=True, help=".npz da parte 6 (a lista de arquivos)")
    a.add_argument("--saida", required=True)
    a.add_argument("--camada", type=int, default=CAMADA)
    a.add_argument("--dispositivo", default="cuda")
    t = sub.add_parser("treinar")
    t.add_argument("--treino", required=True)
    t.add_argument("--dev", required=True)
    t.add_argument("--saida", required=True, help="checkpoint do aluno (.pt)")
    t.add_argument("--epocas", type=int, default=30)
    t.add_argument("--lote", type=int, default=16)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--semente", type=int, default=42)
    t.add_argument("--dispositivo", default="cuda")
    t.add_argument("--dim", type=int, default=256)
    t.add_argument("--camadas", type=int, default=4)
    v = sub.add_parser("avaliar")
    v.add_argument("--aluno", required=True)
    v.add_argument("--sondas", default="outputs/ssl/sondas.json")
    v.add_argument("--treino", required=True)
    v.add_argument("--dev", required=True)
    v.add_argument("--avaliar", action="append", default=[])
    v.add_argument("--dispositivo", default="cuda")
    v.add_argument("--tsv", default=None,
                   help="codecs da amostra do eval da ASVspoof 5: quebra a AUC por codec (só medida)")
    v.add_argument("--json", default="outputs/destilacao/avaliacao.json")
    args = p.parse_args(argv)

    if args.cmd == "preparar":
        from scripts.estudo_ssl import carregar_rede

        r = preparar(Path(args.npz), Path(args.saida), *carregar_rede(args.dispositivo), camada=args.camada)
        print(f"Pronto: {r['arquivos']} arquivos, {r['janelas']} janelas em {args.saida}")
        return 0
    if args.cmd == "treinar":
        r = treinar(carregar_dados(Path(args.treino)), carregar_dados(Path(args.dev)), Path(args.saida),
                    args.epocas, args.lote, args.lr, args.semente, args.dispositivo,
                    {"dim": args.dim, "camadas": args.camadas})
        Path(args.saida).with_suffix(".json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(f"\nMelhor perda no dev: {r['melhor_perda_dev']:.4f}. Aluno: {args.saida}")
        return 0
    sondas = json.loads(Path(args.sondas).read_text(encoding="utf-8"))
    avaliar_ = {Path(x).name: carregar_dados(Path(x)) for x in args.avaliar}
    r = avaliar(Path(args.aluno), sondas, carregar_dados(Path(args.treino)), carregar_dados(Path(args.dev)),
                avaliar_, args.dispositivo)
    print(f"Aluno da época {r['epoca']} (perda no dev {r['perda_dev']:.4f}), imitando a {r['camada']}")
    print(f"{'sonda':24s} {'EER dev':>8s} {'AUC dev':>8s} " + " ".join(f"{'AUC ' + n:>16s}" for n in avaliar_))
    for nome, x in r["sondas"].items():
        print(f"{'aluno, sonda ' + nome:24s} {100 * x['eer_dev']:7.2f}% {x['auc_media_dev']:8.3f} "
              + " ".join(f"{x['avaliar'][n]['auc']:16.3f}" for n in avaliar_))
    for nome, x in r["referencias"].items():
        print(f"{'referência ' + nome:24s} {'':>8s} {x['auc_media_dev']:8.3f} "
              + " ".join(f"{(x[n] if x[n] is not None else float('nan')):16.3f}" for n in avaliar_))
    print("\nPor ataque no dev (sonda nova): " + ", ".join(
        f"{c} {v['auc']:.2f}" for c, v in r["sondas"]["nova"]["dev"]["classes"].items() if v["falso"]))
    if args.tsv:
        from scripts.estudo_ssl import ler_codecs, por_codec

        codecs = ler_codecs(Path(args.tsv))
        r["por_codec"] = {}
        for n, d in avaliar_.items():
            if any(Path(str(a)).stem in codecs for a in d["arquivo"][:20]):
                r["por_codec"][n] = por_codec(np.asarray(r["sondas"]["nova"]["scores"][n]), d["arquivo"],
                                              d["rotulo"], codecs)
                print(f"\n{n}, AUC por codec (sonda nova): " + ", ".join(
                    f"{x['nome']} {x['auc']:.2f}" for x in r["por_codec"][n].values()))
    if "criterio" in r:
        c = r["criterio"]
        print(f"\nCritério (ganho de {MARGEM} de AUC sobre o LFCC no dev e na 2019): dev {c['ganho_dev']:+.3f}, "
              f"2019 {c['ganho_2019']:+.3f} -> {'passa' if c['passa'] else 'não passa'}")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Resultado: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
