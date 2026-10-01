"""Escolha do limiar conforme a origem do áudio.

Cada modelo tem dois limiares: o original (áudio nativo de 16 kHz) e o de
captura, recalibrado por `scripts/calibrar_captura.py` no caminho 16 -> 48 -> 16
kHz. Com o FIR (src/preprocess/reamostragem.py) os dois ficaram próximos
(0,654 e 0,686 no baseline_v2). A regra olha só a taxa do arquivo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Limiares:
    original: float | None
    captura: float | None


@dataclass(frozen=True)
class Escolha:
    limiar: float | None
    origem: str
    aviso: str | None = None


def caminho_captura(checkpoint: str | Path) -> Path:
    """Onde o `calibrar_captura.py` grava a cópia recalibrada."""
    c = Path(checkpoint)
    return c.with_name(f"{c.stem}_captura{c.suffix}")


def limiares_do_checkpoint(caminho: str | Path, dados: dict) -> Limiares:
    """Os dois limiares disponíveis para este modelo.

    A cópia `_captura` ao lado só é aceita se tiver os mesmos pesos.
    """
    calib = dados.get("calibracao")
    if calib:
        return Limiares(calib.get("threshold_original"), dados.get("threshold"))
    original = dados.get("threshold")
    irmao = caminho_captura(caminho)
    if irmao.is_file():
        import torch

        from src.scores import checkpoint_fingerprint

        outro = torch.load(irmao, map_location="cpu", weights_only=False)
        if (outro.get("calibracao") and
                checkpoint_fingerprint(outro["model_state"]) ==
                checkpoint_fingerprint(dados["model_state"])):
            return Limiares(original, outro.get("threshold"))
    return Limiares(original, None)


def taxa_nativa(caminho: str | Path) -> int | None:
    """Taxa de amostragem em que o arquivo foi gravado, sem decodificá-lo."""
    try:
        import soundfile as sf

        return int(sf.info(str(caminho)).samplerate)
    except Exception:
        pass
    try:
        import librosa

        return int(librosa.get_samplerate(str(caminho)))
    except Exception:
        return None


def escolher_limiar(lims: Limiares, taxa_modelo: int, taxa_arquivo: int | None = None,
                    ao_vivo: bool = False) -> Escolha:
    """Usa o limiar de captura se o áudio veio de uma taxa acima da do modelo."""
    if ao_vivo:
        motivo = f"captura ao vivo a 48 kHz, convertida para {taxa_modelo // 1000} kHz"
    elif taxa_arquivo is None:
        motivo = "taxa do arquivo desconhecida — tratado como áudio do mundo real"
    elif taxa_arquivo > taxa_modelo:
        motivo = (f"arquivo gravado a {taxa_arquivo / 1000:g} kHz, convertido para "
                  f"{taxa_modelo // 1000} kHz")
    elif taxa_arquivo == taxa_modelo:
        return Escolha(lims.original,
                       f"original — arquivo nativo de {taxa_modelo // 1000} kHz, como a base de treino")
    else:
        return Escolha(lims.original,
                       f"original — arquivo a {taxa_arquivo / 1000:g} kHz",
                       f"arquivo abaixo de {taxa_modelo // 1000} kHz (ex.: telefonia): "
                       "sem nada acima de metade da taxa dele; nenhum limiar foi medido "
                       "nessa condição.")

    if lims.captura is not None:
        return Escolha(lims.captura, f"recalibrado para conversão de taxa — {motivo}")
    return Escolha(lims.original, f"original — {motivo}",
                   "não há limiar recalibrado para este modelo. Neste tipo de áudio, "
                   "medido no baseline_v2, 71% dos humanos passam do limiar original. "
                   "Rode scripts/calibrar_captura.py.")
