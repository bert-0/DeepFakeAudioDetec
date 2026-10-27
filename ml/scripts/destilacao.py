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

#: Convolução do XLS-R (kernels e passos): dá o número de quadros do professor sem rodá-lo.
CONV_XLSR = ((10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2))


def quadros_do_professor(n_amostras: int) -> int:
    for k, s in CONV_XLSR:
        n_amostras = (n_amostras - k) // s + 1
    return max(0, n_amostras)


def preparar(origem: Path | dict, destino: Path, rede=None, dev=None, camada: int = CAMADA,
             degradar=None, versoes: int = 0, pca: dict | None = None, semente: int = 0,
             transformar=None) -> dict:
    """Uma amostra por janela de 4 s (as mesmas janelas da parte 6). Duas passadas:
    a primeira só conta as janelas, para gravar direto em disco (memmap). Sem `rede`,
    grava só as entradas do aluno: basta para avaliá-lo, e não roda o XLS-R.

    Piloto 2: com `degradar` e `versoes`, cada janela ganha `versoes` cópias
    degradadas das entradas do aluno (a 0 é a limpa); o professor responde sempre
    sobre a janela limpa. Com `pca`, os alvos vão projetados (menos disco).

    Piloto 3: `transformar(wav, rng)` (a sala simulada) muda o áudio antes de tudo,
    e o professor responde sobre o áudio já transformado: o aluno aprende a imitá-lo
    no canal de sala e microfone, não a ignorar o canal. Mantém o comprimento."""
    from scripts.estudo_ssl import janelas, ler_npz
    from scripts.estudo_ssl import preparar as preparar_audio

    d = origem if isinstance(origem, dict) else ler_npz(origem)
    arquivos = [Path(str(a)) for a in d["arquivo"]]
    contagem = [len(janelas(preparar_audio(a)[0])) for a in arquivos]
    n = sum(contagem)
    destino.mkdir(parents=True, exist_ok=True)
    forma = (n, 1 + versoes, MAX_ENTRADA, 140) if versoes else (n, MAX_ENTRADA, 140)
    ent = np.lib.format.open_memmap(destino / "entradas.npy", "w+", np.float16, forma)
    alv = None
    if rede is not None:
        dim = pca["componentes"].shape[0] if pca else int(rede.config.hidden_size)
        alv = np.lib.format.open_memmap(destino / "alvos.npy", "w+", np.float16, (n, MAX_ALVO, dim))
    elif (destino / "alvos.npy").exists():
        (destino / "alvos.npy").unlink()  # alvos de uma rodada anterior não valem para estas janelas
    janela_arquivo = np.zeros(n, np.int32)
    n_ent, n_alv = np.zeros(n, np.int32), np.zeros(n, np.int32)
    k = 0
    for i, arq in enumerate(arquivos):
        wav, _ = preparar_audio(arq)
        if transformar is not None:
            wav = transformar(wav, np.random.default_rng([semente, i, 3]))
        for x in janelas(wav):
            e = entradas(x)[:MAX_ENTRADA]
            if versoes:
                ent[k, 0, :len(e)], ent[k, 0, len(e):] = e, 0
                for v in range(1, versoes + 1):
                    xd = degradar(x.copy(), np.random.default_rng([semente, k, v]))
                    ev = entradas(xd)[:MAX_ENTRADA]
                    ent[k, v, :len(ev)], ent[k, v, len(ev):] = ev, 0
            else:
                ent[k, :len(e)], ent[k, len(e):] = e, 0
            if alv is not None:
                import torch

                xn = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
                with torch.inference_mode():
                    s = rede(torch.from_numpy(xn).float()[None].to(dev), output_hidden_states=True)
                a = s.hidden_states[camada][0].float().cpu().numpy()[:MAX_ALVO]
                if pca:
                    a = (a - pca["media"]) @ pca["componentes"].T
                alv[k, :len(a)], alv[k, len(a):] = a, 0
                n_alv[k] = len(a)
            else:
                n_alv[k] = min(MAX_ALVO, quadros_do_professor(len(x)))
            janela_arquivo[k], n_ent[k] = i, len(e)
            k += 1
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(arquivos)}", flush=True)
    ent.flush()
    if alv is not None:
        alv.flush()
    extra = {"pca_componentes": pca["componentes"], "pca_media": pca["media"]} if pca else {}
    np.savez(destino / "meta.npz", arquivo=d["arquivo"], classe=d["classe"], rotulo=d["rotulo"],
             janela_arquivo=janela_arquivo, n_entrada=n_ent, n_alvo=n_alv, camada=camada, **extra)
    return {"arquivos": len(arquivos), "janelas": n}


def legiveis(arquivos: list[str]) -> tuple[list[str], list[str]]:
    """(os que o `load_audio` lê, os que ele recusa). No Common Voice, raros MP3 vêm
    com o cabeçalho maior que o áudio, e a checagem de truncamento do projeto os
    recusa (1 de 400 no dev da amostra, 26/10/2026)."""
    from src.preprocess.audio import AudioLoadError, load_audio

    bons, ruins = [], []
    for a in arquivos:
        try:
            load_audio(a, SR)
            bons.append(a)
        except AudioLoadError:
            ruins.append(a)
    return bons, ruins


def carregar_dados(pasta: Path) -> dict:
    """Entradas sempre como (janelas, versões, quadros, 140); a versão 0 é a limpa."""
    meta = dict(np.load(pasta / "meta.npz", allow_pickle=False))
    ent = np.load(pasta / "entradas.npy", mmap_mode="r")
    meta["entradas"] = ent if ent.ndim == 4 else ent.reshape(ent.shape[0], 1, *ent.shape[1:])
    alvos = pasta / "alvos.npy"
    meta["alvos"] = np.load(alvos, mmap_mode="r") if alvos.exists() else None
    meta["pca"] = ({"componentes": meta["pca_componentes"], "media": meta["pca_media"]}
                   if "pca_componentes" in meta else None)
    return meta


class _Juntos:
    """Vários memmaps com o mesmo formato (menos a 1ª dimensão), indexados como um só."""

    def __init__(self, partes: list):
        self.partes = partes
        self.inicios = np.cumsum([0] + [len(x) for x in partes])
        self.shape = (int(self.inicios[-1]), *partes[0].shape[1:])

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, chave):
        idx, resto = (chave[0], chave[1:]) if isinstance(chave, tuple) else (chave, ())
        if isinstance(idx, (int, np.integer)):
            k = int(np.searchsorted(self.inicios, idx, side="right") - 1)
            return self.partes[k][(int(idx) - int(self.inicios[k]), *resto)]
        idx = np.asarray(idx)
        k = np.searchsorted(self.inicios, idx, side="right") - 1
        saida = None
        for parte in np.unique(k):
            sel = np.where(k == parte)[0]
            outros = tuple(x[sel] if isinstance(x, np.ndarray) and x.shape == idx.shape else x for x in resto)
            bloco = np.asarray(self.partes[parte][(idx[sel] - self.inicios[parte], *outros)])
            if saida is None:
                saida = np.empty((len(idx), *bloco.shape[1:]), bloco.dtype)
            saida[sel] = bloco
        return saida


def juntar(conjuntos: list[dict]) -> dict:
    """Piloto 3: várias pastas preparadas como um treino só (mesmas versões e alvos)."""
    if len(conjuntos) == 1:
        return conjuntos[0]
    base = conjuntos[0]
    for c in conjuntos[1:]:
        if c["entradas"].shape[1:] != base["entradas"].shape[1:]:
            raise ValueError(f"versões ou quadros diferentes: {c['entradas'].shape} e {base['entradas'].shape}")
        if (c["alvos"] is None) != (base["alvos"] is None) or (
                c["alvos"] is not None and c["alvos"].shape[1:] != base["alvos"].shape[1:]):
            raise ValueError("alvos com formatos diferentes: prepare todas as pastas com a mesma --pca")
        if int(c["camada"]) != int(base["camada"]):
            raise ValueError("pastas preparadas com camadas diferentes")
    deslocados, n_arquivos = [], 0
    for c in conjuntos:
        deslocados.append(np.asarray(c["janela_arquivo"]) + n_arquivos)
        n_arquivos += len(c["arquivo"])
    juntos = {k: np.concatenate([np.asarray(c[k]) for c in conjuntos])
              for k in ("arquivo", "classe", "rotulo", "n_entrada", "n_alvo")}
    juntos.update(janela_arquivo=np.concatenate(deslocados), camada=base["camada"], pca=base["pca"],
                  entradas=_Juntos([c["entradas"] for c in conjuntos]),
                  alvos=None if base["alvos"] is None else _Juntos([c["alvos"] for c in conjuntos]))
    return juntos


def ajustar_pca(dados: dict, k: int = 256, amostra: int = 1000, semente: int = 0) -> tuple[dict, float]:
    """PCA dos quadros do professor (amostra de janelas): ({componentes (k, D), media (D,)},
    fração da variância que fica)."""
    rng = np.random.default_rng(semente)
    idx = np.sort(rng.permutation(len(dados["n_alvo"]))[:amostra])
    q = np.concatenate([np.asarray(dados["alvos"][i, :dados["n_alvo"][i]], dtype=np.float64) for i in idx])
    media = q.mean(axis=0)
    cov = np.cov(q - media, rowvar=False)
    val, vec = np.linalg.eigh(cov)
    ordem = np.argsort(val)[::-1]
    comp = vec[:, ordem[:k]].T.astype(np.float32)
    return {"componentes": comp, "media": media.astype(np.float32)}, float(val[ordem[:k]].sum() / val.sum())


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


def _lote(dados, idx, media, desvio, dispositivo, rng=None, pca=None, com_alvo=True):
    """Com `rng`, uma versão sorteada por janela (limpa ou degradada); sem, a limpa.
    Com `pca`, alvos de 1.024 dimensões são projetados (o dev do piloto 1)."""
    import torch

    idx = np.sort(idx)
    versoes = dados["entradas"].shape[1]
    v = rng.integers(0, versoes, size=len(idx)) if rng is not None else np.zeros(len(idx), int)
    x = torch.from_numpy(np.asarray(dados["entradas"][idx, v], dtype=np.float32)).to(dispositivo)
    y = None
    if dados["alvos"] is not None and com_alvo:
        y = np.asarray(dados["alvos"][idx], dtype=np.float32)
        if pca is not None and y.shape[-1] != len(media):
            y = (y - pca["media"]) @ pca["componentes"].T
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


def treinar(treino: dict, dev: dict | list[dict], saida: Path, epocas: int = 30, lote: int = 16, lr: float = 5e-4,
            semente: int = 42, dispositivo: str = "cuda", arquitetura: dict | None = None,
            amp: bool = False) -> dict:
    import torch

    # Piloto 3: com vários devs (ASVspoof e Common Voice), o checkpoint sai pela média
    # das perdas, sem rótulo.
    devs = dev if isinstance(dev, list) else [dev]
    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    dev_t = torch.device(dispositivo if torch.cuda.is_available() or dispositivo == "cpu" else "cpu")
    media, desvio = estatisticas_alvo(treino)
    pca = treino.get("pca")
    aluno = criar_aluno(dim_alvo=treino["alvos"].shape[2], **(arquitetura or {})).to(dev_t)
    otim = torch.optim.AdamW(aluno.parameters(), lr=lr, weight_decay=0.01)
    passos = epocas * int(np.ceil(len(treino["n_alvo"]) / lote))
    aquecimento = min(500, max(1, passos // 10))
    agenda = torch.optim.lr_scheduler.LambdaLR(
        otim, lambda p: min(1.0, (p + 1) / aquecimento) * 0.5 * (1 + np.cos(np.pi * min(p, passos) / passos)))
    # Desligada por padrão: na GTX 1650 a convolução de passo 2 em fp16 dá NaN já
    # no primeiro lote (medido em 26/10/2026); em fp32 a perda sai normal.
    usar_amp = amp and dev_t.type == "cuda"
    escala = torch.amp.GradScaler("cuda", enabled=usar_amp)
    n_param = sum(p.numel() for p in aluno.parameters())
    print(f"Aluno: {n_param / 1e6:.2f} milhões de parâmetros; {len(treino['n_alvo'])} janelas de treino "
          f"({treino['entradas'].shape[1]} versões cada), {sum(len(d['n_alvo']) for d in devs)} de dev; {epocas} épocas; "
          f"alvos de {treino['alvos'].shape[2]} dimensões; {dev_t}", flush=True)
    historico, melhor = [], float("inf")
    for epoca in range(1, epocas + 1):
        inicio = time.time()
        aluno.train()
        ordem = rng.permutation(len(treino["n_alvo"]))
        soma, cont = 0.0, 0
        for b in range(0, len(ordem), lote):
            x, y, n_e, n_a = _lote(treino, ordem[b:b + lote], media, desvio, dev_t, rng=rng)
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
        perdas_dev = [avaliar_perda(aluno, d, media, desvio, lote, dev_t, pca=pca) for d in devs]
        perda_dev = float(np.mean(perdas_dev))
        historico.append({"epoca": epoca, "perda_treino": soma / cont, "perda_dev": perda_dev,
                          "perdas_dev": perdas_dev})
        marca = ""
        if perda_dev < melhor:
            melhor = perda_dev
            saida.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"estado": aluno.state_dict(), "config": aluno.config, "media_alvo": media,
                        "desvio_alvo": desvio, "epoca": epoca, "perda_dev": perda_dev,
                        "camada": int(treino["camada"]), "pca": pca}, saida)
            marca = "  <- salvo"
        print(f"época {epoca:3d}: perda treino {soma / cont:.4f}, dev {perda_dev:.4f} "
              f"({time.time() - inicio:.0f} s){marca}", flush=True)
    return {"historico": historico, "melhor_perda_dev": melhor, "parametros": n_param}


def avaliar_perda(aluno, dados, media, desvio, lote, dispositivo, pca=None) -> float:
    """Perda de destilação nas entradas limpas (a escolha do checkpoint, sem rótulo)."""
    import torch

    aluno.eval()
    soma, cont = 0.0, 0
    with torch.inference_mode():
        for b in range(0, len(dados["n_alvo"]), lote):
            x, y, n_e, n_a = _lote(dados, np.arange(b, min(b + lote, len(dados["n_alvo"]))), media, desvio,
                                   dispositivo, pca=pca)
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
            x, _, n_e, _ = _lote(dados, np.arange(b, min(b + lote, len(dados["n_alvo"]))), media, desvio,
                                 dispositivo, com_alvo=False)
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

    # 1) a sonda do professor (parte 6) aplicada às respostas do aluno; com PCA, as
    #    respostas voltam ao espaço de 1.024 dimensões do professor (aproximadas)
    prof = {k: np.asarray(v) if isinstance(v, list) else v
            for k, v in sondas["representacoes"][camada]["sonda"].items()}
    pca = ck.get("pca")
    no_professor = {n: (e @ pca["componentes"] + pca["media"]) if pca else e for n, e in emb.items()}
    # 2) uma sonda nova, treinada nas respostas do aluno (a régua justa, como a do LFCC)
    nova = treinar_sonda(emb["treino"].astype(np.float64), treino["rotulo"])
    for nome_sonda, sonda, espaco in (("professor", prof, no_professor), ("nova", nova, emb)):
        s_dev = aplicar(sonda, espaco["dev"].astype(np.float64))
        e, limiar = eer(s_dev[dev["rotulo"] == 0], s_dev[dev["rotulo"] == 1])
        r = {"eer_dev": e, "dev": medir(s_dev, dev["classe"], dev["rotulo"], limiar)}
        r["auc_media_dev"] = float(np.mean([c["auc"] for c in r["dev"]["classes"].values() if c["falso"]]))
        scores = {n: aplicar(sonda, espaco[n].astype(np.float64)) for n in avaliar_}
        r["avaliar"] = {n: medir(scores[n], d["classe"], d["rotulo"], limiar) for n, d in avaliar_.items()}
        r["scores"] = {"dev": s_dev.tolist(), **{n: s.tolist() for n, s in scores.items()}}
        r["arquivos"] = {n: [str(a) for a in d["arquivo"]] for n, d in {"dev": dev, **avaliar_}.items()}
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
    a.add_argument("--npz", help=".npz da parte 6 (a lista de arquivos)")
    a.add_argument("--protocolo", help="ou a lista pelo protocolo, como no extrair da parte 6")
    a.add_argument("--pasta")
    a.add_argument("--pasta-audio", help="piloto 3: todos os áudios de uma pasta, sem rótulo (Common Voice)")
    a.add_argument("--n-humanos", type=int, default=400)
    a.add_argument("--n-ataque", type=int, default=100)
    a.add_argument("--semente", type=int, default=42)
    a.add_argument("--sem-professor", action="store_true",
                   help="só as entradas do aluno (para avaliá-lo); não roda o XLS-R")
    a.add_argument("--degradacao", help="config com audio.degradacao (piloto 2: as versões degradadas)")
    a.add_argument("--ruido", action="store_true", help="soma ruído de 10 a 30 dB (prob. 0,5) à degradação")
    a.add_argument("--versoes", type=int, default=0, help="cópias degradadas por janela, além da limpa")
    a.add_argument("--pca", help="PCA dos alvos (subcomando pca)")
    a.add_argument("--sala", help="piloto 3: pasta do BUT ReverbDB; cada áudio passa por uma sala sorteada")
    pc = sub.add_parser("pca")
    pc.add_argument("--dados", required=True, help="pasta preparada com alvos de 1.024 dimensões")
    pc.add_argument("--k", type=int, default=256)
    pc.add_argument("--saida", required=True)
    a.add_argument("--saida", required=True)
    a.add_argument("--camada", type=int, default=CAMADA)
    a.add_argument("--dispositivo", default="cuda")
    t = sub.add_parser("treinar")
    t.add_argument("--treino", required=True, action="append", help="pode repetir (piloto 3)")
    t.add_argument("--dev", required=True, action="append", help="pode repetir: média das perdas")
    t.add_argument("--saida", required=True, help="checkpoint do aluno (.pt)")
    t.add_argument("--epocas", type=int, default=30)
    t.add_argument("--lote", type=int, default=16)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--semente", type=int, default=42)
    t.add_argument("--dispositivo", default="cuda")
    t.add_argument("--dim", type=int, default=256)
    t.add_argument("--camadas", type=int, default=4)
    t.add_argument("--amp", action="store_true", help="precisão mista (dá NaN na GTX 1650)")
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
        from scripts.estudo_ssl import carregar_rede, selecionar_protocolo

        if args.pasta_audio:
            arquivos = sorted(str(a) for a in Path(args.pasta_audio).iterdir()
                              if a.suffix.lower() in {".wav", ".flac", ".mp3"})
            if not arquivos:
                p.error(f"nenhum áudio em {args.pasta_audio}")
            arquivos, ruins = legiveis(arquivos)
            if ruins:
                print(f"[AVISO] {len(ruins)} áudio(s) ilegível(is) ou truncado(s), fora do preparo: "
                      + ", ".join(Path(r).name for r in ruins), flush=True)
            origem = {"arquivo": np.array(arquivos), "classe": np.array([Path(args.pasta_audio).name] * len(arquivos)),
                      "rotulo": np.zeros(len(arquivos), dtype=np.int8)}
        elif args.npz:
            origem = Path(args.npz)
        elif args.protocolo:
            lista = selecionar_protocolo(Path(args.protocolo), Path(args.pasta), args.n_humanos,
                                         args.n_ataque, args.semente)
            origem = {"arquivo": np.array([str(a) for a, _, _ in lista]),
                      "classe": np.array([c for _, c, _ in lista]),
                      "rotulo": np.array([r for _, _, r in lista], dtype=np.int8)}
        else:
            p.error("preparar precisa de --pasta-audio, --npz ou de --protocolo e --pasta")
        rede, dev = (None, None) if args.sem_professor else carregar_rede(args.dispositivo)
        degradar = None
        if args.versoes:
            from src.config import load_config
            from src.preprocess.degradacao import Degradacao

            cfg = dict(load_config(args.degradacao)["audio"]["degradacao"])
            cfg["etapas"] = list(cfg["etapas"]) + ([{"tipo": "ruido", "prob": 0.5, "faixa": [10, 30]}]
                                                   if args.ruido else [])
            degradar = Degradacao(cfg, SR)
            print(f"Degradação ({args.versoes} versões por janela): {degradar.descricao()}", flush=True)
        pca = None
        if args.pca:
            z = np.load(args.pca)
            pca = {"componentes": z["componentes"], "media": z["media"]}
        transformar = None
        if args.sala:
            from src.preprocess.sala import Sala

            transformar = Sala(args.sala, SR)
            print(f"Sala simulada: {transformar.descricao()}", flush=True)
        r = preparar(origem, Path(args.saida), rede, dev, camada=args.camada, degradar=degradar,
                     versoes=args.versoes, pca=pca, semente=args.semente, transformar=transformar)
        print(f"Pronto: {r['arquivos']} arquivos, {r['janelas']} janelas em {args.saida}")
        return 0
    if args.cmd == "pca":
        pca, fracao = ajustar_pca(carregar_dados(Path(args.dados)), args.k)
        Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.saida, **pca, fracao=fracao)
        print(f"PCA com {args.k} componentes: {100 * fracao:.1f}% da variância. Salvo em {args.saida}")
        return 0
    if args.cmd == "treinar":
        r = treinar(juntar([carregar_dados(Path(x)) for x in args.treino]),
                    [carregar_dados(Path(x)) for x in args.dev], Path(args.saida),
                    args.epocas, args.lote, args.lr, args.semente, args.dispositivo,
                    {"dim": args.dim, "camadas": args.camadas}, amp=args.amp)
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
