"""Executa a sequência completa de um experimento, de ponta a ponta.

Em vez de digitar treino, avaliação, análise por ataque e robustez um a um para
cada config, este script encadeia tudo e ao final imprime uma tabela comparativa
com os resultados de todos os experimentos pedidos.

Pensado para execução longa e desacompanhada: cada etapa é registrada com o tempo
gasto, uma falha não derruba os experimentos seguintes, e tudo é salvo em log.

Uso:
    # um experimento completo
    python scripts/run_pipeline.py --config configs/fusion_v4.yaml

    # vários em sequência (a tabela final compara todos)
    python scripts/run_pipeline.py \\
        --config configs/fusion_v4.yaml \\
        --config configs/attention_v4.yaml

    # só reavaliar modelos já treinados
    python scripts/run_pipeline.py --config configs/fusion_v4.yaml --skip-train

    # ver o que seria executado, sem executar
    python scripts/run_pipeline.py --config configs/fusion_v4.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ML_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ML_DIR / "outputs"
CHECKPOINT_DIR = ML_DIR / "checkpoints"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Roda treino + avaliação + análises de um ou mais experimentos")
    p.add_argument("--config", action="append", required=True,
                   help="caminho do YAML; repita a opção para vários experimentos")
    p.add_argument("--skip-train", action="store_true",
                   help="pula o treino e usa o checkpoint existente")
    p.add_argument("--skip-check", action="store_true",
                   help="pula a verificação da base (check_data)")
    p.add_argument("--robustness", action="store_true",
                   help="inclui a avaliação de robustez (mais lenta)")
    p.add_argument("--smoke", action="store_true",
                   help="modo rápido com dados sintéticos, para testar o encadeamento")
    p.add_argument("--dry-run", action="store_true",
                   help="apenas mostra os comandos que seriam executados")
    p.add_argument("--no-report", action="store_true",
                   help="não consolida as tabelas do relatório ao final")
    return p.parse_args()


def experiment_name(config_path: str, smoke: bool = False) -> str:
    """Nome dos artefatos do experimento.

    Precisa acompanhar o sufixo `_smoke` que os scripts aplicam, senão o
    pipeline procuraria checkpoints e JSONs com o nome errado.
    """
    nome = yaml.safe_load(open(config_path, encoding="utf-8"))["experiment"]["name"]
    return f"{nome}_smoke" if smoke else nome


def build_steps(config: str, name: str, args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """Monta a lista de (rótulo, comando) de um experimento."""
    py = [sys.executable]
    ckpt = f"checkpoints/{name}.pt"
    smoke = ["--smoke"] if args.smoke else []
    steps: list[tuple[str, list[str]]] = []

    if not args.skip_check and not args.smoke:
        steps.append(("verificação da base",
                      py + ["scripts/check_data.py", "--config", config]))
    if not args.skip_train:
        steps.append(("treino", py + ["train.py", "--config", config] + smoke))
    steps.append(("avaliação (eval)",
                  py + ["evaluate.py", "--config", config, "--checkpoint", ckpt,
                        "--partition", "eval",
                        "--score-file", f"outputs/{name}_eval_scores.txt"] + smoke))
    steps.append(("EER por ataque",
                  py + ["scripts/per_attack_eval.py", "--config", config,
                        "--checkpoint", ckpt] + smoke))
    if args.robustness:
        steps.append(("robustez",
                      py + ["scripts/robustness_eval.py", "--config", config,
                            "--checkpoint", ckpt] + smoke))
    return steps


def child_env() -> dict[str, str]:
    """Ambiente dos subprocessos, com duas variáveis que o Windows exige.

    **`PYTHONUNBUFFERED`** — com o stdout num pipe (e não num terminal), o Python
    filho passa a usar buffer de bloco (~8 KB) e nenhum `print` de `train.py`
    chama `flush`. O `bufsize=1` do `Popen` configura o *pai*, não o filho. Sem
    isto, um treino de horas não imprime nada até terminar: medido aqui, três
    linhas espaçadas de 0,5 s chegaram todas juntas em t=1,51 s.

    **`PYTHONIOENCODING`** — o `encoding` do `Popen` diz apenas como o *pai
    decodifica*. Quem escolhe como o *filho codifica* é o `sys.stdout` dele, que
    no Windows segue o locale (`cp1252`). O pai então lê cp1252 como UTF-8 e cada
    acento vira `U+FFFD` — corrupção permanente, já que o byte original se perde
    ao ser gravado no log. Pior: se a saída do próprio pipeline for redirecionada
    para arquivo, imprimir `U+FFFD` levanta `UnicodeEncodeError`, porque esse
    caractere não existe em cp1252, e o pipeline morre no meio.
    """
    return {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}


def run_step(label: str, cmd: list[str], log_file) -> tuple[bool, float]:
    """Executa um passo, ecoando a saída na tela e no log. Devolve (ok, segundos)."""
    header = f"\n{'=' * 70}\n>>> {label}\n    {' '.join(cmd)}\n{'=' * 70}"
    print(header, flush=True)
    log_file.write(header + "\n")

    inicio = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=ML_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1,
                            env=child_env())
    for linha in proc.stdout:
        print(linha, end="", flush=True)
        log_file.write(linha)
    proc.wait()
    dur = time.perf_counter() - inicio

    status = "OK" if proc.returncode == 0 else f"FALHOU (código {proc.returncode})"
    rodape = f"--- {label}: {status} em {fmt_dur(dur)}\n"
    print(rodape, flush=True)
    log_file.write(rodape)
    log_file.flush()
    return proc.returncode == 0, dur


def fmt_dur(s: float) -> str:
    h, resto = divmod(int(s), 3600)
    m, seg = divmod(resto, 60)
    return f"{h}h{m:02d}m{seg:02d}s" if h else (f"{m}m{seg:02d}s" if m else f"{seg}s")


def collect_results(name: str) -> dict:
    """Lê os JSONs gerados para montar a tabela final."""
    res: dict = {"experimento": name}
    metrics = OUTPUT_DIR / f"{name}_eval_metrics.json"
    if metrics.exists():
        res.update(json.loads(metrics.read_text(encoding="utf-8")))
    per_attack = OUTPUT_DIR / f"{name}_eval_per_attack.json"
    if per_attack.exists():
        dados = json.loads(per_attack.read_text(encoding="utf-8"))
        eers = [v["eer"] for v in dados["per_attack"].values()]
        if eers:
            res["eer_medio_por_ataque"] = sum(eers) / len(eers)
            piores = sorted(dados["per_attack"].items(), key=lambda kv: -kv[1]["eer"])[:3]
            res["piores_ataques"] = ", ".join(f"{a}={v['eer'] * 100:.1f}%" for a, v in piores)
    return res


def print_summary(resultados: list[dict], tempos: dict[str, float]) -> None:
    print("\n" + "=" * 78)
    print("RESUMO FINAL")
    print("=" * 78)
    if not resultados:
        print("Nenhum resultado coletado.")
        return
    print(f"{'experimento':26s} {'EER':>8s} {'F1':>8s} {'méd/ataque':>11s} {'tempo':>10s}")
    print("-" * 78)
    for r in sorted(resultados, key=lambda x: x.get("eer", 9e9)):
        eer = f"{r['eer'] * 100:.2f}%" if "eer" in r else "—"
        f1 = f"{r['f1']:.4f}" if "f1" in r else "—"
        med = (f"{r['eer_medio_por_ataque'] * 100:.2f}%"
               if "eer_medio_por_ataque" in r else "—")
        print(f"{r['experimento'][:26]:26s} {eer:>8s} {f1:>8s} {med:>11s} "
              f"{fmt_dur(tempos.get(r['experimento'], 0)):>10s}")
    print()
    for r in resultados:
        if "piores_ataques" in r:
            print(f"  {r['experimento']}: piores ataques -> {r['piores_ataques']}")


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # Os subprocessos rodam com cwd=ML_DIR, então caminhos relativos ao diretório
    # atual precisam ser resolvidos aqui — senão o pré-check passa e cada etapa
    # morre com FileNotFoundError.
    configs = []
    for cfg in args.config:
        p = Path(cfg)
        if p.exists():
            configs.append(str(p.resolve()))
        elif (ML_DIR / cfg).exists():
            configs.append(cfg)
        else:
            print(f"[ERRO] config não encontrado: {cfg}")
            return 1

    planos = [(cfg, experiment_name(cfg, args.smoke)) for cfg in configs]

    if args.dry_run:
        print("Comandos que seriam executados:\n")
        for cfg, name in planos:
            print(f"# --- {name} ({cfg}) ---")
            for label, cmd in build_steps(cfg, name, args):
                print(f"  # {label}\n  {' '.join(cmd)}")
            print()
        return 0

    print(f"Pipeline: {len(planos)} experimento(s) — "
          f"{', '.join(n for _, n in planos)}")
    print(f"Início: {time.strftime('%d/%m/%Y %H:%M:%S')}\n")

    inicio_total = time.perf_counter()
    resultados, tempos, falhas = [], {}, []

    for cfg, name in planos:
        log_path = OUTPUT_DIR / f"{name}_pipeline.log"
        t0 = time.perf_counter()
        print(f"\n{'#' * 78}\n# EXPERIMENTO: {name}\n# log: {log_path}\n{'#' * 78}")
        with open(log_path, "w", encoding="utf-8") as log_file:
            for label, cmd in build_steps(cfg, name, args):
                ok, _ = run_step(label, cmd, log_file)
                if not ok:
                    falhas.append(f"{name} / {label}")
                    # Sem checkpoint não adianta seguir para as análises.
                    if label == "treino":
                        print(f"[AVISO] treino de {name} falhou — pulando as análises.")
                        break
        tempos[name] = time.perf_counter() - t0
        resultados.append(collect_results(name))

    # As tabelas do relatório saem de uma vez ao final, e não por experimento:
    # a comparação entre incrementos só existe com todos eles já avaliados.
    if not args.no_report:
        cmd = [sys.executable, "scripts/make_report.py"]
        if args.smoke:
            cmd.append("--include-smoke")
        with open(OUTPUT_DIR / "make_report.log", "w", encoding="utf-8") as log_file:
            run_step("tabelas do relatório", cmd, log_file)

    print_summary(resultados, tempos)
    print(f"\nTempo total: {fmt_dur(time.perf_counter() - inicio_total)}")
    if falhas:
        print(f"\n{len(falhas)} etapa(s) falharam:")
        for f in falhas:
            print(f"  - {f}")
        return 1
    print("\nTodas as etapas concluíram com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
