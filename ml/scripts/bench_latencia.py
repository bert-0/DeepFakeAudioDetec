"""Latência ponta a ponta de uma análise (RNF01 da APS).

O RNF01 exige **máx. 30 s por análise** para arquivos de até 60 s. Este script
mede o caminho completo que a aplicação percorreria: ler o arquivo do disco,
pré-processar, extrair features, inferir e agregar — não só a passada do modelo.

Mede a **janela deslizante** (`AnalisadorContinuo`), e não o `infer.py`. O
`infer.py` chama `fix_length`, que corta o sinal em `audio.duration`: um envio
de 60 s teria só os primeiros 4 s analisados, 6,7% do arquivo. Para um sistema
de perícia isso é inaceitável, então a aplicação precisa janelar o arquivo
inteiro e agregar.

Uso:
    python scripts/bench_latencia.py --config configs/fusion_v4.yaml \\
        --checkpoint checkpoints/fusion_lcnn_v4.pt --device cpu
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture.analyzer import AnalisadorContinuo, Agregador  # noqa: E402
from src.capture.sources import FileSource  # noqa: E402
from src.config import load_config, resolve_device  # noqa: E402

#: Limite do RNF01: nenhuma análise pode passar disto.
LIMITE_RNF01_S = 30.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Latência ponta a ponta (RNF01)")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default=None, help="cuda | cpu (padrão: do config)")
    p.add_argument("--duracoes", default="5,15,30,60",
                   help="durações em segundos, separadas por vírgula")
    p.add_argument("--repeticoes", type=int, default=3)
    return p.parse_args()


def gerar_wav(destino: Path, segundos: float, sr: int) -> Path:
    """Fala sintética simples: portadora modulada, com energia de banda larga."""
    import soundfile as sf

    t = np.arange(int(sr * segundos)) / sr
    sinal = 0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    sinal += 0.05 * np.random.default_rng(0).standard_normal(len(t))
    sf.write(destino, sinal.astype(np.float32), sr)
    return destino


def medir(analisador: AnalisadorContinuo, caminho: Path) -> tuple[float, int]:
    """Uma análise completa, do disco ao resultado. Devolve (segundos, janelas)."""
    inicio = time.perf_counter()
    fonte = FileSource(caminho, analisador.sample_rate)
    agregador = Agregador()
    analisador.janela._buffer = np.zeros(0, dtype=np.float32)
    analisador.janela._consumidas = 0
    analisador._n = 0
    for bloco in fonte.blocos():
        for leitura in analisador.processar(bloco):
            agregador.adicionar(leitura)
    agregador.resumo()
    return time.perf_counter() - inicio, len(agregador.leituras)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device or config["train"]["device"])
    analisador = AnalisadorContinuo(config, args.checkpoint, device)

    print(f"Modelo: {analisador.config['model']['name']} | dispositivo: {device}")
    print(f"Janela: {analisador.janela.tamanho / analisador.sample_rate:.0f}s | "
          f"passo: {analisador.janela.passo / analisador.sample_rate:.0f}s")
    print(f"Limite do RNF01: {LIMITE_RNF01_S:.0f}s por análise\n")

    duracoes = [float(x) for x in args.duracoes.split(",")]
    tmp = Path(tempfile.mkdtemp())
    print(f"{'áudio':>8s} {'janelas':>8s} {'latência':>10s} {'x tempo real':>13s} "
          f"{'folga RNF01':>12s}")
    print("-" * 58)
    falhou = False
    for segundos in duracoes:
        caminho = gerar_wav(tmp / f"{segundos:.0f}s.wav", segundos,
                            analisador.sample_rate)
        medir(analisador, caminho)                       # aquece
        tempos = [medir(analisador, caminho)[0] for _ in range(args.repeticoes)]
        t = min(tempos)
        _, janelas = medir(analisador, caminho)
        folga = LIMITE_RNF01_S / t if t else float("inf")
        marca = "" if t <= LIMITE_RNF01_S else "  <-- ACIMA DO LIMITE"
        print(f"{segundos:7.0f}s {janelas:8d} {t:9.3f}s {segundos / t:12.1f}x "
              f"{folga:11.0f}x{marca}")
        falhou |= t > LIMITE_RNF01_S

    print("\n" + "=" * 58)
    if falhou:
        print("[FALHA] alguma duração excedeu o limite do RNF01.")
        return 1
    print("[ OK ] RNF01 atendido em todas as durações testadas.")
    print("       Medido na janela deslizante, que analisa o arquivo INTEIRO.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
