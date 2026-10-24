"""Teste com a voz própria: real contra clone, pelo arquivo e pelo microfone.

Lê as três pastas que o `gravar_frases.py` e a ferramenta de clonagem deixam:
- `real/`: as frases lidas no microfone;
- `clone/`: os clones das mesmas frases, como a ferramenta os entregou;
- `clone_ar/`: os clones tocados pelo alto-falante e gravados pelo microfone.

Cada arquivo passa pelo caminho da interface web (`AnalisadorContinuo` sobre
`FileSource`, média das janelas ponderada pela fala), com os dois modelos
fundidos pela média e o limiar escolhido nos devs pelo `limiar_fusao.py`.
Score igual ou acima do limiar é sintético.

Placar, com o critério combinado antes (RESUMO_TCC §10.2.12):
- caminho arquivo: `real/` contra `clone/`;
- caminho microfone: `real/` contra `clone_ar/`.
A meta é 8 de 10 em cada classe, nos dois caminhos. Arquivo sem janela útil
conta como erro. Os scores de cada modelo e a comparação por frase são só
descritivos.

Uso (de dentro de ml/):
    python scripts/teste_voz_propria.py
    python scripts/teste_voz_propria.py --pasta data/vozes/teste_proprio --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASTA_PADRAO = Path("data/vozes/teste_proprio")
LIMIAR_PADRAO = Path("outputs/limiar_fusao/limiar.json")
EXTENSOES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}
#: (pasta, classe esperada). A pasta `real/` entra nos dois caminhos.
CLASSES = {"real": "humano", "clone": "sintético", "clone_ar": "sintético"}
CAMINHOS = {"arquivo": ("real", "clone"), "microfone": ("real", "clone_ar")}
META = 0.8


@dataclass
class Analise:
    pasta: str
    frase: str
    score: float | None
    por_modelo: list[float | None]
    janelas_uteis: int
    taxa_nativa: int | None
    duracao_s: float
    checagens_falhas: list[str] = field(default_factory=list)

    def sintetico(self, limiar: float) -> bool | None:
        return None if self.score is None else self.score >= limiar

    def acertou(self, limiar: float) -> bool:
        """Sem janela útil não há veredito, e isso conta como erro."""
        veredito = self.sintetico(limiar)
        if veredito is None:
            return False
        return veredito == (CLASSES[self.pasta] == "sintético")


def listar(pasta: Path) -> list[Path]:
    if not pasta.is_dir():
        return []
    return sorted(p for p in pasta.iterdir() if p.suffix.lower() in EXTENSOES)


def analisar(caminho: Path, pasta: str, config: dict, checkpoints: list[str], device) -> Analise:
    """O caminho da interface web, com os dois modelos (web/analise.py)."""
    import librosa

    from src.capture import FileSource
    from src.capture.analyzer import Agregador, AnalisadorContinuo
    from src.capture.checagens import checar
    from src.limiares import taxa_nativa

    analisador = AnalisadorContinuo(config, checkpoints, device)
    agregador = Agregador(janela_s=analisador.janela.tamanho / analisador.sample_rate,
                          passo_s=analisador.janela.passo / analisador.sample_rate)
    with FileSource(caminho, analisador.sample_rate) as fonte:
        for bloco in fonte.blocos():
            for leitura in analisador.processar(bloco):
                agregador.adicionar(leitura)
        duracao = len(fonte._wav) / analisador.sample_rate
    for leitura in analisador.finalizar():
        agregador.adicionar(leitura)

    uteis = agregador.uteis
    pesos = np.array([x.peso for x in uteis], dtype=float)
    por_modelo: list[float | None] = [None] * analisador.n_modelos
    if uteis and pesos.sum() > 0:
        matriz = np.array([x.por_modelo for x in uteis], dtype=float)
        por_modelo = [float(v) for v in np.average(matriz, axis=0, weights=pesos)]
    # Checagens na taxa em que o arquivo foi gravado, como logo depois de gravar.
    wav, taxa = librosa.load(str(caminho), sr=None, mono=True)
    falhas = [c.nome for c in checar(wav, taxa) if not c.ok]
    return Analise(pasta=pasta, frase=caminho.stem, score=agregador.resumo()["score_medio"],
                   por_modelo=por_modelo, janelas_uteis=len(uteis),
                   taxa_nativa=taxa_nativa(caminho), duracao_s=duracao,
                   checagens_falhas=falhas)


def minimo_de_acertos(n: int) -> int:
    return math.ceil(META * n - 1e-9)


def placar(analises: list[Analise], limiar: float) -> dict:
    """Acertos por classe em cada caminho e se a meta foi atingida."""
    por_pasta: dict[str, list[Analise]] = {}
    for a in analises:
        por_pasta.setdefault(a.pasta, []).append(a)
    saida = {}
    for caminho, (humanos, falsos) in CAMINHOS.items():
        r = {}
        for pasta in (humanos, falsos):
            lista = por_pasta.get(pasta, [])
            acertos = sum(a.acertou(limiar) for a in lista)
            r[pasta] = {"acertos": acertos, "n": len(lista),
                        "meta": len(lista) > 0 and acertos >= minimo_de_acertos(len(lista))}
        # Descritivo: o falso acima do real da mesma frase, mesmo que os dois errem.
        reais = {a.frase: a.score for a in por_pasta.get(humanos, []) if a.score is not None}
        pares = [(reais[a.frase], a.score) for a in por_pasta.get(falsos, [])
                 if a.score is not None and a.frase in reais]
        r["falso_acima_do_real"] = {"vezes": sum(f > h for h, f in pares), "pares": len(pares)}
        r["meta"] = r[humanos]["meta"] and r[falsos]["meta"]
        saida[caminho] = r
    return saida


def frases_sem_par(analises: list[Analise]) -> dict[str, list[str]]:
    """Frases que faltam numa pasta e existem em outra (o placar não depende disso)."""
    frases = {p: {a.frase for a in analises if a.pasta == p} for p in CLASSES}
    todas = set().union(*frases.values())
    return {p: sorted(todas - f) for p, f in frases.items() if f and todas - f}


def carregar_limiar(caminho: Path) -> dict:
    """O limiar e os checkpoints do `limiar_fusao.py`, conferindo os pesos."""
    import torch

    from src.scores import checkpoint_fingerprint

    r = json.loads(caminho.read_text(encoding="utf-8"))
    for m in r["modelos"]:
        dados = torch.load(m["checkpoint"], map_location="cpu", weights_only=False)
        if checkpoint_fingerprint(dados["model_state"]) != m["impressao"]:
            raise SystemExit(f"[ERRO] {m['checkpoint']} mudou desde a escolha do limiar: "
                             "rode o limiar_fusao.py de novo")
    return r


def _f(x: float | None) -> str:
    return "  —  " if x is None else f"{x:.3f}"


def relatar(analises: list[Analise], limiar: float, nomes: list[str], r: dict) -> None:
    print(f"{'arquivo':16s} {'kHz':>5s} {'fusão':>6s} " + " ".join(f"{n[:22]:>22s}" for n in nomes)
          + "  resultado")
    for a in analises:
        veredito = a.sintetico(limiar)
        texto = "sem fala" if veredito is None else ("sintético" if veredito else "humano")
        marca = "certo" if a.acertou(limiar) else "ERRO"
        taxa = f"{a.taxa_nativa / 1000:g}" if a.taxa_nativa else "?"
        aviso = f"  (checagens: {', '.join(a.checagens_falhas)})" if a.checagens_falhas else ""
        print(f"{a.pasta + '/' + a.frase:16s} {taxa:>5s} {_f(a.score):>6s} "
              + " ".join(f"{_f(s):>22s}" for s in a.por_modelo)
              + f"  {texto:9s} {marca}{aviso}")
    print(f"\nLimiar da fusão: {limiar:.4f}")
    for caminho, (humanos, falsos) in CAMINHOS.items():
        c = r[caminho]
        h, f = c[humanos], c[falsos]
        print(f"Caminho {caminho:9s}: reais julgados humanos {h['acertos']} de {h['n']}, "
              f"{falsos} julgados sintéticos {f['acertos']} de {f['n']} — meta "
              f"{'atingida' if c['meta'] else 'não atingida'}")
        fa = c["falso_acima_do_real"]
        if fa["pares"]:
            print(f"  (descritivo) {falsos} acima do real da mesma frase: {fa['vezes']} de "
                  f"{fa['pares']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Teste com a voz própria: real contra clone")
    p.add_argument("--pasta", default=str(PASTA_PADRAO))
    p.add_argument("--limiar", default=str(LIMIAR_PADRAO), help="o JSON do limiar_fusao.py")
    p.add_argument("--device", default="cpu")
    p.add_argument("--json", default="outputs/teste_voz_propria.json",
                   help="scores e placar (sem áudio)")
    args = p.parse_args(argv)

    from src.config import load_config, resolve_device

    pasta = Path(args.pasta)
    arquivos = {nome: listar(pasta / nome) for nome in CLASSES}
    if not arquivos["real"] or not (arquivos["clone"] or arquivos["clone_ar"]):
        print(f"[ERRO] faltam gravações em {pasta}: precisa de real/ e de clone/ ou clone_ar/")
        return 1
    escolha = carregar_limiar(Path(args.limiar))
    checkpoints = [m["checkpoint"] for m in escolha["modelos"]]
    config = load_config(escolha["modelos"][0]["config"])
    device = resolve_device(args.device)

    analises = [analisar(arq, nome, config, checkpoints, device)
                for nome, lista in arquivos.items() for arq in lista]
    r = placar(analises, escolha["limiar"])
    relatar(analises, escolha["limiar"], [Path(c).stem for c in checkpoints], r)
    for nome, faltam in frases_sem_par(analises).items():
        print(f"[AVISO] {nome}/ não tem: {', '.join(faltam)}")

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(
        {"limiar": escolha["limiar"], "limiar_origem": str(args.limiar),
         "checkpoints": checkpoints, "placar": r, "arquivos": [asdict(a) for a in analises]},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultado: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
