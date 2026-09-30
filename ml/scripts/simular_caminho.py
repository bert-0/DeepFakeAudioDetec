"""Aplica aos recortes de uma sessão um efeito do caminho do som, isolado.

O controle do teste ao vivo (tocar -> alto-falante -> loopback -> gravar) levou
todos os áudios a score ~1,0, bonafide inclusive, sem perdas de amostra. Este
script reproduz cada etapa desse caminho **em software**, sobre os recortes da
sessão "limpo", para achar qual delas basta:

- `perdas`       — remove N trechos de X ms (descontinuidade do WASAPI);
- `reamostragem` — ida e volta 16 -> 48 -> 16 kHz com o mesmo `soxr` do
                   `tocar` e da captura. Medido em ruído branco: preserva até
                   7,3 kHz e apaga 7,6-8 kHz (-30 dB em 7,7-7,9 kHz, -103 dB
                   acima de 7,9 kHz). Nenhuma qualidade do soxr evita isso: é
                   o filtro antialiasing de qualquer conversão para 16 kHz;
- `passa_baixa`  — corta acima de `--hz`, para achar a partir de onde o
                   modelo colapsa.

Uso:
    python scripts/simular_caminho.py --destino reamostragem --efeito reamostragem
    python scripts/simular_caminho.py --destino pb7000 --efeito passa_baixa --hz 7000
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

EFEITOS = ("perdas", "reamostragem", "passa_baixa")


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


def ida_e_volta(wav: np.ndarray, sr: int, taxa: int = 48000) -> np.ndarray:
    """O que o tocar + a captura fazem com a taxa: sobe para `taxa` e volta."""
    import soxr

    return soxr.resample(soxr.resample(np.asarray(wav, dtype=np.float32), sr, taxa),
                         taxa, sr).astype(np.float32)


def passa_baixa(wav: np.ndarray, sr: int, hz: float) -> np.ndarray:
    """Butterworth de ordem 10, fase zero (sem atraso que desalinhe o recorte)."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(10, hz, btype="low", fs=sr, output="sos")
    return sosfiltfilt(sos, np.asarray(wav, dtype=np.float64)).astype(np.float32)


def gerar_sessao(pasta: Path, origem: str, destino: str, transformar,
                 config_base: str = CONFIG_PADRAO) -> tuple[int, Path]:
    """Copia os recortes de `origem` para `destino` aplicando `transformar`."""
    orig, dest = pasta / origem, pasta / destino
    proto_origem = orig / "protocolo_canal_real.txt"
    if not proto_origem.is_file():
        raise FileNotFoundError(f"{proto_origem} não existe — rode o alinhar da "
                                f"sessão '{origem}'.")
    saida = dest / "capturado"
    saida.mkdir(parents=True, exist_ok=True)
    for antigo in saida.glob("*.flac"):
        antigo.unlink()
    linhas = [ln for ln in proto_origem.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for linha in linhas:
        audio_id = linha.split()[1]
        wav, sr = sf.read(orig / "capturado" / f"{audio_id}.flac", dtype="float32")
        sf.write(saida / f"{audio_id}.flac", np.clip(transformar(wav, sr), -1, 1), sr)
    proto = dest / "protocolo_canal_real.txt"
    proto.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    cfg = config_derivado(load_config(config_base), f"canal_real_{pasta.name}_{destino}",
                          proto, saida)
    return len(linhas), salvar_config(cfg, dest / "config_canal_real.yaml")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Simula uma etapa do caminho do som")
    p.add_argument("--pasta", default="outputs/canal_real")
    p.add_argument("--origem", default="limpo", help="sessão de onde vêm os recortes")
    p.add_argument("--destino", required=True, help="nome da sessão simulada")
    p.add_argument("--efeito", choices=EFEITOS, default="perdas")
    p.add_argument("--ms", type=float, default=10.0, help="perdas: duração de cada uma")
    p.add_argument("--n", type=int, default=1, help="perdas: quantas por áudio")
    p.add_argument("--hz", type=float, default=7000.0, help="passa_baixa: frequência de corte")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config-base", default=CONFIG_PADRAO)
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    transformar = {
        "perdas": lambda w, sr: remover_trechos(w, sr, args.n, args.ms, rng),
        "reamostragem": ida_e_volta,
        "passa_baixa": lambda w, sr: passa_baixa(w, sr, args.hz),
    }[args.efeito]
    pasta = Path(args.pasta)
    try:
        n, config = gerar_sessao(pasta, args.origem, args.destino, transformar,
                                 args.config_base)
    except FileNotFoundError as erro:
        print(f"[ERRO] {erro}")
        return 1
    print(f"{n} áudios com o efeito '{args.efeito}' em {pasta / args.destino}")
    print(f"\n  python evaluate.py --config {config.as_posix()} "
          f"--checkpoint {CHECKPOINT_PADRAO} --partition eval")
    print(f"  python scripts/comparar_sessoes.py "
          f"{(pasta / args.origem / 'config_canal_real.yaml').as_posix()} {config.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
