"""Refaz gravações da sua voz com vocoders da época do ASVspoof 2019.

Controle pareado: original e falso diferem só no vocoder (Griffin-Lim ou WORLD).
O WORLD precisa de `pip install pyworld "setuptools<81"`. Tudo sai a 48 kHz.

Uso:
    python scripts/gerar_sintetico.py --entrada gravacoes/ --saida outputs/demo_propria
    python monitor.py --config configs/baseline_v2.yaml \\
        --checkpoint checkpoints/baseline_lfcc_cnn_v2.pt --arquivo outputs/demo_propria/pareado.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.montar_demo import GAP_S, montar  # noqa: E402
from src.preprocess import load_audio  # noqa: E402

TAXA_MODELO = 16000
TAXA_SAIDA = 48000
EXTENSOES = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac")


def griffin_lim(wav: np.ndarray, sr: int, n_iter: int = 60) -> np.ndarray:
    """Mel de 80 bandas (como o Tacotron2) e reconstrução por Griffin-Lim."""
    import librosa

    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=1024, hop_length=256,
                                         n_mels=80, fmax=sr // 2)
    saida = librosa.feature.inverse.mel_to_audio(mel, sr=sr, n_fft=1024, hop_length=256,
                                                 n_iter=n_iter, fmax=sr // 2)
    return _mesmo_nivel(saida, wav)


def world(wav: np.ndarray, sr: int, tom: float = 1.0) -> np.ndarray:
    """Análise e síntese pelo WORLD; `tom` != 1 desloca o F0 (conversão simples)."""
    import pyworld

    x = np.asarray(wav, dtype=np.float64)
    f0, sp, ap = pyworld.wav2world(x, sr)
    saida = pyworld.synthesize(f0 * tom, sp, ap, sr)
    return _mesmo_nivel(saida.astype(np.float32), wav)


def _mesmo_nivel(saida: np.ndarray, referencia: np.ndarray) -> np.ndarray:
    """Mesmo pico do original: o volume não pode ser a diferença entre os dois."""
    pico_ref, pico = np.abs(referencia).max(), np.abs(saida).max()
    return (saida * (pico_ref / pico) if pico > 0 else saida).astype(np.float32)


def _para_saida(wav: np.ndarray) -> np.ndarray:
    import soxr

    return soxr.resample(np.asarray(wav, np.float32), TAXA_MODELO, TAXA_SAIDA)


def gravacoes(entrada: Path) -> list[Path]:
    if entrada.is_file():
        return [entrada]
    return sorted(p for p in entrada.iterdir() if p.suffix.lower() in EXTENSOES)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Versões sintéticas da sua voz (vocoders de 2019)")
    p.add_argument("--entrada", required=True, help="arquivo ou pasta com as gravações")
    p.add_argument("--saida", default="outputs/demo_propria")
    p.add_argument("--metodos", nargs="+", choices=("griffinlim", "world"),
                   default=["griffinlim", "world"])
    p.add_argument("--tom", type=float, default=1.0,
                   help="world: fator do F0 (1,0 = mesma voz; 1,3 = conversão simples)")
    args = p.parse_args(argv)

    if "world" in args.metodos:
        try:
            import pyworld  # noqa: F401
        except ImportError as erro:
            print(f"[AVISO] WORLD indisponível ({erro}). Instale com: "
                  'pip install pyworld "setuptools<81". Seguindo só com Griffin-Lim.')
            args.metodos = [m for m in args.metodos if m != "world"]
    arquivos = gravacoes(Path(args.entrada))
    if not arquivos:
        print(f"[ERRO] nenhuma gravação ({', '.join(EXTENSOES)}) em {args.entrada}")
        return 1

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    geradores = {"griffinlim": griffin_lim, "world": lambda w, sr: world(w, sr, args.tom)}
    itens = []
    for arq in arquivos:
        original = load_audio(arq, TAXA_MODELO)
        versoes = [("original", "humano", original)]
        versoes += [(m, f"sintético ({m})", geradores[m](original, TAXA_MODELO))
                    for m in args.metodos]
        for tipo, rotulo, wav in versoes:
            nome = f"{arq.stem}_{tipo}"
            sf.write(saida / f"{nome}.wav", _para_saida(wav), TAXA_SAIDA)
            itens.append(((nome, rotulo), wav))
        print(f"{arq.name}: " + ", ".join(t for t, _, _ in versoes))

    sinal, marcas = montar(itens, TAXA_MODELO)
    sf.write(saida / "pareado.wav", _para_saida(sinal), TAXA_SAIDA)
    linhas = ["DEMONSTRAÇÃO PAREADA — sua voz, original e refeita por vocoders de 2019", "",
              "Mesmo locutor, microfone e texto: a diferença é só o vocoder.", "",
              f"{'início':>7s}  {'arquivo':32s} rótulo"]
    linhas += [f"{t:6.1f}s  {nome:32s} {rotulo}" for t, (nome, rotulo) in marcas]
    (saida / "pareado_cola.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"\nPlaylist: {saida / 'pareado.wav'} ({len(sinal) / TAXA_MODELO:.0f} s, "
          f"{GAP_S:g} s de silêncio entre os áudios) + pareado_cola.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
