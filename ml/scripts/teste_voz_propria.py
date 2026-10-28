"""Teste com a voz própria: real contra clone, pelo arquivo e pelo microfone.

Lê as três pastas que o `gravar_frases.py` e a ferramenta de clonagem deixam:
- `real/`: as frases lidas no microfone;
- `clone/`: os clones das mesmas frases, como a ferramenta os entregou;
- `clone_ar/`: os clones tocados pelo alto-falante e gravados pelo microfone.

Cada arquivo passa pelo caminho da interface web (`AnalisadorContinuo` sobre
`FileSource`, média das janelas ponderada pela fala), com os dois modelos
fundidos pela regra e com o limiar que o `limiar_fusao.py` escolheu nos devs
(média dos log-odds desde 24/10/2026). Score igual ou acima do limiar é
sintético.

Placar, com o critério combinado antes (RESUMO_TCC §10.2.12):
- caminho arquivo: `real/` contra `clone/`;
- caminho microfone: `real/` contra `clone_ar/`.
A meta é 8 de 10 em cada classe, nos dois caminhos. Arquivo sem janela útil
conta como erro. Os scores de cada modelo e a comparação por frase são só
descritivos.

Com `--xlsr` (o JSON do `limiar_xlsr.py`), o modelo final (RESUMO_TCC §10.2.43):
o caminho arquivo usa o score do Enviar arquivo e o limiar dele; o caminho
microfone, com `--microfone frase`, o do ao vivo, frase por frase, com o limiar
da frase, ou com `--microfone arquivo`, o arquivo inteiro. Qual vale para o
ensaio é critério: fixar antes de rodar (RESUMO_TCC §10.2.44).

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
#: Margem do microfone embutido na sala, aceita antes de gravar (RESUMO_TCC §10.2.12).
PISO_MINIMO_DB = 15.0


@dataclass
class Analise:
    pasta: str
    frase: str
    score: float | None                # fusão, em [0, 1]
    por_modelo: list[float | None]     # cada modelo, na escala da regra
    janelas_uteis: int
    taxa_nativa: int | None
    duracao_s: float
    checagens_falhas: list[str] = field(default_factory=list)
    logodds: float | None = None

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


def ler_na_taxa_do_modelo(caminho: Path, taxa_modelo: int,
                          taxa_comum: int | None = None) -> np.ndarray:
    """O `FileSource`, o mesmo da interface. Com `taxa_comum`, o clone do XTTS
    (24 kHz) e a gravação (48 kHz) chegam ao modelo pela mesma conversão final,
    e a taxa deixa de ser pista (RESUMO_TCC §10.2.13)."""
    from src.capture import FileSource

    with FileSource(caminho, taxa_modelo, taxa_comum=taxa_comum) as fonte:
        return fonte._wav


def analisar(caminho: Path, pasta: str, config: dict, checkpoints: list[str], device,
             regra: str = "probabilidade", taxa_comum: int | None = None) -> Analise:
    """O caminho da interface web (web/analise.py), com um ou mais modelos."""
    import librosa

    from src.capture.analyzer import Agregador, AnalisadorContinuo
    from src.capture.checagens import checar
    from src.limiares import taxa_nativa

    analisador = AnalisadorContinuo(config, checkpoints, device, regra=regra)
    agregador = Agregador(janela_s=analisador.janela.tamanho / analisador.sample_rate,
                          passo_s=analisador.janela.passo / analisador.sample_rate, regra=regra)
    wav16 = ler_na_taxa_do_modelo(caminho, analisador.sample_rate, taxa_comum)
    # Blocos do tamanho dos do FileSource: o mesmo fluxo do ao vivo.
    for i in range(0, len(wav16), 1600):
        for leitura in analisador.processar(wav16[i:i + 1600]):
            agregador.adicionar(leitura)
    duracao = len(wav16) / analisador.sample_rate
    for leitura in analisador.finalizar():
        agregador.adicionar(leitura)

    uteis = agregador.uteis
    pesos = np.array([x.peso for x in uteis], dtype=float)
    por_modelo: list[float | None] = [None] * analisador.n_modelos
    if uteis and pesos.sum() > 0:
        matriz = np.array([x.logodds_por_modelo if regra == "logodds" else x.por_modelo
                           for x in uteis], dtype=float)
        por_modelo = [float(v) for v in np.average(matriz, axis=0, weights=pesos)]
    # Checagens na taxa em que o arquivo foi gravado, como logo depois de gravar.
    wav, taxa = librosa.load(str(caminho), sr=None, mono=True)
    falhas = [c.nome for c in checar(wav, taxa, piso_minimo=PISO_MINIMO_DB) if not c.ok]
    resumo = agregador.resumo()
    return Analise(pasta=pasta, frase=caminho.stem, score=resumo["score_medio"],
                   por_modelo=por_modelo, janelas_uteis=len(uteis),
                   taxa_nativa=taxa_nativa(caminho), duracao_s=duracao,
                   checagens_falhas=falhas, logodds=resumo["logodds_medio"])


def analisar_xlsr(caminho: Path, pasta: str, modelo, por_frases: bool) -> Analise:
    """O modelo final, pelo caminho do Enviar arquivo ou, com `por_frases`, do ao vivo.
    Sem fala ou sem frase achada, fica sem veredito (conta como erro)."""
    import librosa

    from src.capture.checagens import checar
    from src.limiares import taxa_nativa
    from src.preprocess.audio import load_audio
    from src.xlsr_final import SR, FalaCurta

    bruto = load_audio(caminho, SR)
    try:
        p = modelo.pontuar_por_frases(bruto) if por_frases else modelo.pontuar_wav(bruto)
        score, logodds, n = p.score, p.logodds, len(p.janelas)
    except FalaCurta:
        score = logodds = None
        n = 0
    wav, taxa = librosa.load(str(caminho), sr=None, mono=True)
    falhas = [c.nome for c in checar(wav, taxa, piso_minimo=PISO_MINIMO_DB) if not c.ok]
    return Analise(pasta=pasta, frase=caminho.stem, score=score, por_modelo=[logodds],
                   janelas_uteis=n, taxa_nativa=taxa_nativa(caminho), duracao_s=len(bruto) / SR,
                   checagens_falhas=falhas, logodds=logodds)


def placar_xlsr(arquivos: dict[str, list[Path]], modelo, lim, microfone: str) -> tuple[dict, dict]:
    """Cada caminho com o seu score e o seu limiar. Devolve (placar, análises por caminho)."""
    caminhos = {"arquivo": (lim.arquivo_por_frase, lim.probabilidade_arquivo),
                "microfone": (microfone == "frase",
                              lim.probabilidade_frase if microfone == "frase" else lim.probabilidade)}
    r, analises = {}, {}
    for caminho, (por_frases, limiar) in caminhos.items():
        lista = [analisar_xlsr(arq, nome, modelo, por_frases)
                 for nome in CAMINHOS[caminho] for arq in arquivos[nome]]
        r[caminho] = {**placar(lista, limiar)[caminho], "limiar": limiar,
                      "unidade": "frase" if por_frases else "arquivo"}
        analises[caminho] = lista
    return r, analises


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
        # Caminho sem gravações (ensaio só do arquivo): não medido, nem atingido nem não.
        r["meta"] = (None if not r[falsos]["n"] else r[humanos]["meta"] and r[falsos]["meta"])
        saida[caminho] = r
    return saida


def frases_sem_par(analises: list[Analise]) -> dict[str, list[str]]:
    """Frases que faltam numa pasta e existem em outra (o placar não depende disso)."""
    frases = {p: {a.frase for a in analises if a.pasta == p} for p in CLASSES}
    todas = set().union(*frases.values())
    return {p: sorted(todas - f) for p, f in frases.items() if f and todas - f}


def carregar_limiar(caminho: Path):
    """O limiar, a regra e os checkpoints do `limiar_fusao.py`, conferindo os pesos."""
    from src.limiares import ler_limiar_fusao

    try:
        return ler_limiar_fusao(caminho)
    except ValueError as erro:
        raise SystemExit(f"[ERRO] {erro}") from erro


def _f(x: float | None, formato: str = ".4f") -> str:
    return "—" if x is None else format(x, formato)


def relatar(analises: list[Analise], limiar: float, nomes: list[str], r: dict,
            regra: str = "probabilidade", descricao: str = "") -> None:
    """`limiar` na escala do score, em [0, 1]; cada modelo aparece na escala da regra."""
    escala = "log-odds" if regra == "logodds" else "prob."
    formato = "+.2f" if regra == "logodds" else ".4f"
    print(f"{'arquivo':16s} {'kHz':>5s} {'fusão':>7s} "
          + " ".join(f"{(n[:15] + ' ' + escala):>24s}" for n in nomes) + "  resultado")
    for a in analises:
        veredito = a.sintetico(limiar)
        texto = "sem fala" if veredito is None else ("sintético" if veredito else "humano")
        marca = "certo" if a.acertou(limiar) else "ERRO"
        taxa = f"{a.taxa_nativa / 1000:g}" if a.taxa_nativa else "?"
        aviso = f"  (checagens: {', '.join(a.checagens_falhas)})" if a.checagens_falhas else ""
        print(f"{a.pasta + '/' + a.frase:16s} {taxa:>5s} {_f(a.score):>7s} "
              + " ".join(f"{_f(s, formato):>24s}" for s in a.por_modelo)
              + f"  {texto:9s} {marca}{aviso}")
    print(f"\nLimiar da fusão: {limiar:.4f}" + (f" — {descricao}" if descricao else ""))
    for caminho, (humanos, falsos) in CAMINHOS.items():
        c = r[caminho]
        h, f = c[humanos], c[falsos]
        if c["meta"] is None:
            print(f"Caminho {caminho:9s}: não medido (sem gravações em {falsos}/)")
            continue
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
    p.add_argument("--clones", default="clone", metavar="SUBPASTA",
                   help="subpasta dos clones como arquivo (ex.: clone_chatterbox, para "
                        "comparar outra ferramenta com as mesmas gravações reais)")
    p.add_argument("--clones-ar", default="clone_ar", metavar="SUBPASTA",
                   help="subpasta dos clones pelo ar (ex.: clone_ar_chatterbox)")
    p.add_argument("--taxa-comum", type=int, default=None, metavar="HZ",
                   help="leva todo arquivo a esta taxa antes da do modelo (o ensaio 2 usa "
                        "48000: clone de 24 kHz e gravação de 48 kHz pela mesma conversão)")
    p.add_argument("--xlsr", default=None, metavar="JSON",
                   help="o modelo final, com o JSON do limiar_xlsr.py (no lugar de --limiar)")
    p.add_argument("--microfone", choices=("frase", "arquivo"), default=None,
                   help="com --xlsr: o caminho microfone frase por frase (o do ao vivo) ou "
                        "pelo arquivo inteiro; obrigatório, porque é critério do ensaio")
    args = p.parse_args(argv)

    from src.config import load_config, resolve_device

    pasta = Path(args.pasta)
    subpasta = {"clone": args.clones, "clone_ar": args.clones_ar}
    arquivos = {nome: listar(pasta / subpasta.get(nome, nome)) for nome in CLASSES}
    if not arquivos["real"] or not (arquivos["clone"] or arquivos["clone_ar"]):
        print(f"[ERRO] faltam gravações em {pasta}: precisa de real/ e de clone/ ou clone_ar/")
        return 1
    if args.xlsr:
        if args.microfone is None:
            p.error("com --xlsr, diga o caminho microfone: --microfone frase ou arquivo")
        return main_xlsr(args, arquivos)
    escolha = carregar_limiar(Path(args.limiar))
    checkpoints = list(escolha.checkpoints)
    config = load_config(escolha.config)
    device = resolve_device(args.device)
    limiar = escolha.probabilidade

    analises = [analisar(arq, nome, config, checkpoints, device, escolha.regra, args.taxa_comum)
                for nome, lista in arquivos.items() for arq in lista]
    r = placar(analises, limiar)
    relatar(analises, limiar, [Path(c).stem for c in checkpoints], r, escolha.regra,
            escolha.descricao())
    for nome, faltam in frases_sem_par(analises).items():
        print(f"[AVISO] {nome}/ não tem: {', '.join(faltam)}")

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(
        {"regra": escolha.regra, "limiar": escolha.limiar, "limiar_score": limiar,
         "clones": args.clones, "clones_ar": args.clones_ar,
         "limiar_origem": str(args.limiar), "taxa_comum": args.taxa_comum,
         "checkpoints": checkpoints, "placar": r, "arquivos": [asdict(a) for a in analises]},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultado: {args.json}")
    return 0


def main_xlsr(args, arquivos: dict[str, list[Path]]) -> int:
    from src.xlsr_final import ModeloFinal, ler_limiar

    try:
        lim = ler_limiar(args.xlsr)
    except ValueError as erro:
        raise SystemExit(f"[ERRO] {erro}") from erro
    modelo = ModeloFinal(lim.cabeca, args.device, modelo_hf=lim.fonte_modelo)
    r, analises = placar_xlsr(arquivos, modelo, lim, args.microfone)
    for caminho, lista in analises.items():
        c = r[caminho]
        print(f"Caminho {caminho} ({c['unidade']}; limiar {c['limiar']:.4f}):")
        for a in lista:
            veredito = a.sintetico(c["limiar"])
            texto = "sem fala" if veredito is None else ("sintético" if veredito else "humano")
            aviso = f"  (checagens: {', '.join(a.checagens_falhas)})" if a.checagens_falhas else ""
            print(f"  {a.pasta + '/' + a.frase:16s} {_f(a.logodds, '+.3f'):>8s}  {texto:9s} "
                  f"{'certo' if a.acertou(c['limiar']) else 'ERRO'}{aviso}")
        humanos, falsos = CAMINHOS[caminho]
        if c["meta"] is None:
            print(f"  não medido (sem gravações em {falsos}/)")
        else:
            print(f"  reais julgados humanos {c[humanos]['acertos']} de {c[humanos]['n']}, {falsos} "
                  f"julgados sintéticos {c[falsos]['acertos']} de {c[falsos]['n']} — meta "
                  f"{'atingida' if c['meta'] else 'não atingida'}")
    print(f"Limiar: {lim.descricao()}")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(
        {"modelo": "xlsr_congelado", "limiar_origem": str(args.xlsr), "microfone": args.microfone,
         "clones": args.clones, "clones_ar": args.clones_ar, "placar": r,
         "arquivos": {k: [asdict(a) for a in v] for k, v in analises.items()}},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultado: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
