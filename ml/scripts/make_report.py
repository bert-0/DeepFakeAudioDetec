"""Consolida os resultados de `outputs/` em tabelas e gráficos para o relatório.

Cada experimento deixa em `outputs/` um punhado de JSONs soltos (métricas, EER
por ataque, robustez, histórico por época). Compará-los a olho, copiar número por
número para o texto e refazer isso a cada rodada é onde o erro entra — um EER
desatualizado numa tabela não dá nenhum sinal de que está errado.

Este script lê tudo o que existe, monta as tabelas comparativas em três formatos
(Markdown para conferir, LaTeX para colar no documento, CSV para a planilha) e
gera as figuras de comparação entre os incrementos.

Uso:
    python scripts/make_report.py                       # tudo que houver em outputs/
    python scripts/make_report.py --only v4             # só os experimentos com "v4"
    python scripts/make_report.py --partition dev
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path("outputs")
VAZIO = "—"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gera as tabelas e figuras comparativas do relatório")
    p.add_argument("--outputs", default=str(OUTPUT_DIR),
                   help="pasta com os JSONs dos experimentos (padrão: outputs)")
    p.add_argument("--dest", default=None,
                   help="pasta de destino (padrão: <outputs>/report)")
    p.add_argument("--partition", default="eval", choices=["train", "dev", "eval"],
                   help="partição das métricas reportadas (padrão: eval)")
    p.add_argument("--only", action="append", default=None,
                   help="só experimentos cujo nome contenha este texto; repetível")
    p.add_argument("--include-smoke", action="store_true",
                   help="inclui execuções --smoke (excluídas por padrão)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Coleta
# --------------------------------------------------------------------------- #
def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def discover(outputs: Path, partition: str, only, include_smoke: bool) -> list[dict]:
    """Reúne, por experimento, tudo o que foi encontrado em `outputs`."""
    sufixo = f"_{partition}_metrics.json"
    experimentos = []
    for metrics_path in sorted(outputs.glob(f"*{sufixo}")):
        nome = metrics_path.name[: -len(sufixo)]
        if not include_smoke and nome.endswith("_smoke"):
            continue
        if only and not any(termo in nome for termo in only):
            continue
        experimentos.append({
            "nome": nome,
            "metrics": read_json(metrics_path) or {},
            "per_attack": read_json(outputs / f"{nome}_{partition}_per_attack.json"),
            "robustness": read_json(outputs / f"{nome}_{partition}_robustness.json"),
            "history": read_json(outputs / f"{nome}_history.json"),
        })
    return experimentos


def training_summary(history) -> dict:
    """Nº de épocas, melhor época e tempo total — quando o histórico existir."""
    if not history:
        return {}
    eers = [(h.get("dev_eer"), h.get("epoch")) for h in history
            if isinstance(h.get("dev_eer"), (int, float))]
    resumo = {"epocas": len(history)}
    if eers:
        resumo["melhor_epoca"] = min(eers)[1]
    tempo = sum(h.get("t_epoch", 0.0) for h in history)
    if tempo > 0:
        resumo["tempo_s"] = tempo
    return resumo


# --------------------------------------------------------------------------- #
# Formatação
# --------------------------------------------------------------------------- #
def pct(valor, casas: int = 2) -> str:
    return VAZIO if valor is None else f"{valor * 100:.{casas}f}"


def num(valor, casas: int = 4) -> str:
    return VAZIO if valor is None else f"{valor:.{casas}f}"


def duracao(segundos) -> str:
    if not segundos:
        return VAZIO
    h, resto = divmod(int(segundos), 3600)
    m, _ = divmod(resto, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def as_markdown(cabecalho: list[str], linhas: list[list[str]]) -> str:
    largura = [max(len(cabecalho[i]), *(len(l[i]) for l in linhas)) if linhas
               else len(cabecalho[i]) for i in range(len(cabecalho))]
    def linha(celulas):
        return "| " + " | ".join(c.ljust(largura[i]) for i, c in enumerate(celulas)) + " |"
    partes = [linha(cabecalho), "|" + "|".join("-" * (w + 2) for w in largura) + "|"]
    partes += [linha(l) for l in linhas]
    return "\n".join(partes) + "\n"


def as_latex(cabecalho: list[str], linhas: list[list[str]], legenda: str,
             rotulo: str) -> str:
    """Tabela LaTeX em `tabular` puro, sem depender de pacotes extras."""
    def escapar(texto: str) -> str:
        return (texto.replace("\\", r"\textbackslash{}").replace("_", r"\_")
                .replace("%", r"\%").replace("&", r"\&").replace("—", "--"))

    colunas = "l" + "r" * (len(cabecalho) - 1)
    corpo = "\n".join("    " + " & ".join(escapar(c) for c in l) + r" \\"
                      for l in linhas)
    return (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        f"  \\caption{{{escapar(legenda)}}}\n"
        f"  \\label{{{rotulo}}}\n"
        f"  \\begin{{tabular}}{{{colunas}}}\n"
        "    \\hline\n"
        "    " + " & ".join(escapar(c) for c in cabecalho) + r" \\" + "\n"
        "    \\hline\n"
        f"{corpo}\n"
        "    \\hline\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )


def write_table(dest: Path, base: str, cabecalho, linhas, legenda: str, rotulo: str):
    (dest / f"{base}.md").write_text(as_markdown(cabecalho, linhas), encoding="utf-8")
    (dest / f"{base}.tex").write_text(as_latex(cabecalho, linhas, legenda, rotulo),
                                      encoding="utf-8")
    with open(dest / f"{base}.csv", "w", encoding="utf-8", newline="") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(cabecalho)
        escritor.writerows(linhas)
    return [dest / f"{base}.{ext}" for ext in ("md", "tex", "csv")]


# --------------------------------------------------------------------------- #
# Tabelas
# --------------------------------------------------------------------------- #
def tabela_comparativa(experimentos: list[dict]) -> tuple[list[str], list[list[str]]]:
    cabecalho = ["Experimento", "EER (%)", "Acurácia", "Precisão", "Recall", "F1",
                 "Limiar", "Épocas", "Melhor", "Tempo"]
    linhas = []
    for exp in experimentos:
        m = exp["metrics"]
        t = training_summary(exp["history"])
        linhas.append([
            exp["nome"],
            pct(m.get("eer")),
            num(m.get("accuracy")),
            num(m.get("precision")),
            num(m.get("recall")),
            num(m.get("f1")),
            num(m.get("threshold"), 4) if m.get("threshold") is not None else VAZIO,
            str(t.get("epocas", VAZIO)),
            str(t.get("melhor_epoca", VAZIO)),
            duracao(t.get("tempo_s")),
        ])
    return cabecalho, linhas


def tabela_por_ataque(experimentos: list[dict]):
    """Matriz ataque x experimento — mostra ONDE cada modelo falha.

    O EER global esconde a distribuição: 20% pode ser 20% em todos os ataques ou
    0% em doze e 90% em um. Lado a lado, também revela complementaridade entre
    configurações — foi assim que a fusão de scores foi motivada.
    """
    com_dados = [e for e in experimentos if e["per_attack"]]
    if not com_dados:
        return None, None
    ataques = sorted({a for e in com_dados for a in e["per_attack"]["per_attack"]})
    cabecalho = ["Ataque"] + [e["nome"] for e in com_dados]
    linhas = []
    for ataque in ataques:
        linha = [ataque]
        for exp in com_dados:
            dados = exp["per_attack"]["per_attack"].get(ataque)
            linha.append(pct(dados["eer"]) if dados else VAZIO)
        linhas.append(linha)
    linhas.append(["GLOBAL"] + [pct(e["per_attack"].get("global_eer")) for e in com_dados])
    return cabecalho, linhas


def tabela_robustez(experimentos: list[dict]):
    com_dados = [e for e in experimentos if e["robustness"]]
    if not com_dados:
        return None, None
    condicoes = list(com_dados[0]["robustness"])
    for exp in com_dados[1:]:
        for cond in exp["robustness"]:
            if cond not in condicoes:
                condicoes.append(cond)
    cabecalho = ["Condição"] + [e["nome"] for e in com_dados]
    linhas = []
    for cond in condicoes:
        linha = [cond]
        for exp in com_dados:
            dados = exp["robustness"].get(cond)
            linha.append(pct(dados.get("eer")) if dados else VAZIO)
        linhas.append(linha)
    return cabecalho, linhas


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def figura_curvas(experimentos: list[dict], out_path: Path) -> Path | None:
    """EER de validação por época, com todos os experimentos no mesmo eixo."""
    com_historico = [e for e in experimentos if e["history"]]
    if not com_historico:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for exp in com_historico:
        epocas = [h["epoch"] for h in exp["history"]]
        ax1.plot(epocas, [h.get("dev_eer", float("nan")) * 100 for h in exp["history"]],
                 marker="o", markersize=3, label=exp["nome"])
        ax2.plot(epocas, [h.get("train_loss", float("nan")) for h in exp["history"]],
                 marker="o", markersize=3, label=exp["nome"])
    ax1.set_title("EER de validação por época")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("EER (%)")
    ax2.set_title("Loss de treino por época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("loss")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def figura_por_ataque(experimentos: list[dict], out_path: Path) -> Path | None:
    com_dados = [e for e in experimentos if e["per_attack"]]
    if not com_dados:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ataques = sorted({a for e in com_dados for a in e["per_attack"]["per_attack"]})
    x = np.arange(len(ataques))
    largura = 0.8 / len(com_dados)

    fig, ax = plt.subplots(figsize=(max(8, len(ataques) * 0.8), 4.5))
    for i, exp in enumerate(com_dados):
        alturas = [(exp["per_attack"]["per_attack"].get(a, {}).get("eer") or 0) * 100
                   for a in ataques]
        ax.bar(x + i * largura - 0.4 + largura / 2, alturas, largura, label=exp["nome"])
    ax.set_xticks(x, ataques)
    ax.set_ylabel("EER (%)")
    ax.set_xlabel("Algoritmo de síntese")
    ax.set_title("EER por tipo de ataque")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    outputs = Path(args.outputs)
    if not outputs.is_dir():
        print(f"[ERRO] pasta não encontrada: {outputs}")
        return 1

    experimentos = discover(outputs, args.partition, args.only, args.include_smoke)
    if not experimentos:
        print(f"Nenhum experimento com métricas de '{args.partition}' em {outputs}.")
        print("Rode evaluate.py (ou scripts/run_pipeline.py) antes.")
        return 1

    # Melhor primeiro: é a ordem em que a tabela é lida no texto.
    experimentos.sort(key=lambda e: e["metrics"].get("eer", float("inf")))

    dest = Path(args.dest) if args.dest else outputs / "report"
    dest.mkdir(parents=True, exist_ok=True)
    gerados: list[Path] = []

    cab, linhas = tabela_comparativa(experimentos)
    gerados += write_table(dest, f"comparativo_{args.partition}", cab, linhas,
                           f"Comparação dos incrementos na partição {args.partition}.",
                           f"tab:comparativo-{args.partition}")
    print(as_markdown(cab, linhas))

    cab, linhas = tabela_por_ataque(experimentos)
    if cab:
        gerados += write_table(dest, f"por_ataque_{args.partition}", cab, linhas,
                               "EER por algoritmo de síntese, avaliado contra todos "
                               "os áudios bonafide.",
                               f"tab:por-ataque-{args.partition}")

    cab, linhas = tabela_robustez(experimentos)
    if cab:
        gerados += write_table(dest, f"robustez_{args.partition}", cab, linhas,
                               "EER sob perturbações de ruído e ganho.",
                               f"tab:robustez-{args.partition}")

    for figura in (figura_curvas(experimentos, dest / "curvas_comparadas.png"),
                   figura_por_ataque(experimentos, dest / "eer_por_ataque.png")):
        if figura:
            gerados.append(figura)

    print(f"{len(experimentos)} experimento(s) consolidados em {dest}:")
    for caminho in gerados:
        print(f"  {caminho}")

    faltando = [e["nome"] for e in experimentos if not e["per_attack"]]
    if faltando:
        print("\nSem análise por ataque (rode scripts/per_attack_eval.py): "
              + ", ".join(faltando))
    faltando = [e["nome"] for e in experimentos if not e["robustness"]]
    if faltando:
        print("Sem avaliação de robustez (rode scripts/robustness_eval.py): "
              + ", ".join(faltando))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
