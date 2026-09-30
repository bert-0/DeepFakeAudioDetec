"""Qual limiar vale para qual áudio.

A faixa de 7,6-8 kHz some em qualquer conversão para 16 kHz — na captura ao
vivo (loopback a 48 kHz) e ao abrir um arquivo gravado a 44,1/48 kHz, porque o
`librosa.load` reamostra com o mesmo tipo de filtro (medido: 7,8-8 kHz ~35 dB
abaixo do resto). O `baseline_v2` depende dessa faixa, e sem ela todos os
scores sobem. Por isso o projeto tem dois limiares para o mesmo modelo:

- **original** — calibrado no dev em áudio nativo de 16 kHz, com a banda
  inteira. Vale para o formato da base de treino;
- **captura** — recalibrado no dev passado pela ida e volta 16 -> 48 -> 16 kHz
  (`scripts/calibrar_captura.py`). Vale para quase todo áudio do mundo real.

Medido no eval com captura (RESUMO 10.2.1): no limiar original, 71% dos humanos
passam por sintéticos; no recalibrado, 17%. Em áudio nativo de 16 kHz, o
recalibrado deixa passar 69% dos sintéticos. A escolha tem de seguir o áudio.

**Limite da regra:** ela olha a taxa do arquivo, não o conteúdo. Um arquivo
salvo a 16 kHz que já tinha sido convertido antes (de 48 kHz, por exemplo)
também perdeu o topo da banda e receberia o limiar original.
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

    Se `caminho` é a cópia recalibrada, o original está nos metadados dela. Se
    é o checkpoint do treino, procura a cópia ao lado — e só a aceita se os
    pesos forem os mesmos: um `_captura.pt` de um treino anterior teria o
    limiar de outro modelo.
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
    """A regra: o áudio passou por uma taxa acima da do modelo? Então captura."""
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
