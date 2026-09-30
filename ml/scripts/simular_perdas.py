"""Simula perdas de amostra (descontinuidades) nos recortes de uma sessão.

Pergunta que responde: **um clique basta?** No primeiro controle real, com 58
descontinuidades da captura em 3 minutos, todos os áudios — bonafide inclusive
— foram a score ~1,0. Se remover alguns milissegundos de um áudio limpo
reproduz isso, a causa está provada sem gravar nada, e o achado vale para o
texto: perda de pacote é o cotidiano de uma chamada VoIP.

Uma descontinuidade do WASAPI **remove** amostras (o buffer transbordou e o
que chegou foi descartado), então aqui também se remove: o áudio fica mais
curto e emenda com um salto na forma de onda.

Uso:
    python scripts/simular_perdas.py --pasta outputs/canal_real --origem limpo \\
        --destino perdas_10ms --ms 10 --n 1
    python evaluate.py --config outputs/canal_real/perdas_10ms/config_canal_real.yaml \\
        --checkpoint checkpoints/baseline_lfcc_cnn_v2.pt --partition eval
    python scripts/comparar_sessoes.py outputs/canal_real/limpo/config_canal_real.yaml \\
        outputs/canal_real/perdas_10ms/config_canal_real.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.canal_real import CHECKPOINT_PADRAO, CONFIG_PADRAO  # noqa: E402
from src.config import config_derivado, load_config, salvar_config  # noqa: E402


def remover_trechos(wav: np.ndarray, sr: int, n: int, ms: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Remove `n` trechos de `ms` milissegundos em posições sorteadas.

    As posições evitam os 10% das pontas, onde o `trim` de silêncio apagaria o
    efeito, e não se sobrepõem.
    """
    tam = int(round(ms * sr / 1000))
    if n <= 0 or tam <= 0 or len(wav) < 4 * n * tam:
        return wav.copy()
    lo, hi = int(0.1 * len(wav)), int(0.9 * len(wav)) - tam
    inicios: list[int] = []
    while len(inicios) < n:
        c = int(rng.integers(lo, hi))
        if all(abs(c - i) > tam for i in inicios):
            inicios.append(c)
    manter = np.ones(len(wav), dtype=bool)
    for i in inicios:
        manter[i:i + tam] = False
    return wav[manter]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Simula descontinuidades nos recortes")
    p.add_argument("--pasta", default="outputs/canal_real")
    p.add_argument("--origem", default="limpo", help="sessão de onde vêm os recortes")
    p.add_argument("--destino", required=True, help="nome da sessão simulada")
    p.add_argument("--ms", type=float, default=10.0, help="duração de cada perda")
    p.add_argument("--n", type=int, default=1, help="perdas por áudio")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config-base", default=CONFIG_PADRAO)
    args = p.parse_args(argv)

    pasta = Path(args.pasta)
    origem, destino = pasta / args.origem, pasta / args.destino
    proto_origem = origem / "protocolo_canal_real.txt"
    if not proto_origem.is_file():
        print(f"[ERRO] {proto_origem} não existe — rode o alinhar da sessão '{args.origem}'.")
        return 1

    saida = destino / "capturado"
    saida.mkdir(parents=True, exist_ok=True)
    for antigo in saida.glob("*.flac"):
        antigo.unlink()
    rng = np.random.default_rng(args.seed)
    linhas = [ln for ln in proto_origem.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for linha in linhas:
        audio_id = linha.split()[1]
        wav, sr = sf.read(origem / "capturado" / f"{audio_id}.flac", dtype="float32")
        sf.write(saida / f"{audio_id}.flac", remover_trechos(wav, sr, args.n, args.ms, rng), sr)
    proto = destino / "protocolo_canal_real.txt"
    proto.write_text("\n".join(linhas) + "\n", encoding="utf-8")

    cfg = config_derivado(load_config(args.config_base),
                          f"canal_real_{pasta.name}_{args.destino}", proto, saida)
    config = salvar_config(cfg, destino / "config_canal_real.yaml")
    print(f"{len(linhas)} áudios com {args.n} perda(s) de {args.ms:g} ms cada, em {saida}")
    print(f"\n  python evaluate.py --config {config.as_posix()} "
          f"--checkpoint {CHECKPOINT_PADRAO} --partition eval")
    print(f"  python scripts/comparar_sessoes.py {(origem / 'config_canal_real.yaml').as_posix()} "
          f"{config.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
