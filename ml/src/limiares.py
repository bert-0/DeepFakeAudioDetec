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


def _calibrado_no_fir(calib: dict) -> bool:
    """O limiar de captura só vale se foi calibrado com o conversor em uso (FIR).
    Uma cópia antiga, do soxr (0,983 no baseline_v2), marcaria quase tudo como humano."""
    return calib.get("caminho") == "fir"


def limiares_do_checkpoint(caminho: str | Path, dados: dict) -> Limiares:
    """Os dois limiares disponíveis para este modelo.

    A cópia `_captura` ao lado só é aceita se tiver os mesmos pesos e tiver sido
    calibrada com o FIR.
    """
    calib = dados.get("calibracao")
    if calib:
        captura = dados.get("threshold") if _calibrado_no_fir(calib) else None
        return Limiares(calib.get("threshold_original"), captura)
    original = dados.get("threshold")
    irmao = caminho_captura(caminho)
    if irmao.is_file():
        import torch

        from src.scores import checkpoint_fingerprint

        outro = torch.load(irmao, map_location="cpu", weights_only=False)
        if (outro.get("calibracao") and _calibrado_no_fir(outro["calibracao"]) and
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
                   "não há limiar recalibrado com o conversor atual (FIR) para este "
                   "modelo; usando o original. Rode scripts/calibrar_captura.py.")


@dataclass(frozen=True)
class LimiarFusao:
    """Limiar de uma fusão de modelos, escolhido nos devs por `scripts/limiar_fusao.py`."""
    config: str
    checkpoints: tuple[str, ...]
    regra: str
    limiar: float        # na escala da regra (log-odds ou probabilidade)
    alvo: float
    condicao: str

    @property
    def probabilidade(self) -> float:
        """O limiar na escala do score das leituras, em [0, 1]."""
        if self.regra == "logodds":
            import math

            return 0.5 * (1.0 + math.tanh(0.5 * self.limiar))
        return self.limiar

    def descricao(self) -> str:
        regra = "média dos log-odds" if self.regra == "logodds" else "média das probabilidades"
        return (f"fusão ({regra}), {self.alvo * 100:.0f}% de alarme falso nos devs, "
                f"{self.condicao}")


def ler_limiar_fusao(caminho: str | Path) -> LimiarFusao:
    """Lê o JSON e confere se os checkpoints ainda têm os pesos da escolha."""
    import json

    import torch

    from src.scores import checkpoint_fingerprint

    r = json.loads(Path(caminho).read_text(encoding="utf-8"))
    for m in r["modelos"]:
        dados = torch.load(m["checkpoint"], map_location="cpu", weights_only=False)
        if checkpoint_fingerprint(dados["model_state"]) != m["impressao"]:
            raise ValueError(f"{m['checkpoint']} mudou desde a escolha do limiar: "
                             "rode o limiar_fusao.py de novo")
    # JSON anterior à opção de regra: a média das probabilidades.
    return LimiarFusao(config=r["modelos"][0]["config"],
                       checkpoints=tuple(m["checkpoint"] for m in r["modelos"]),
                       regra=r.get("regra", "probabilidade"), limiar=float(r["limiar"]),
                       alvo=float(r.get("alvo", float("nan"))),
                       condicao=r.get("condicao", "?"))


#: Limite do logit: score 0 ou 1 exato não vira infinito.
_EPS = 1e-12


def centrado(score: float | None, limiar: float | None) -> float | None:
    """Score na escala da tela, com o limiar sempre em 0,5.

    sigmoide(logit(score) - logit(limiar)): a ordem e a decisão não mudam (acima
    de 0,5 é acima do limiar). Com a fusão nos log-odds o limiar fica em 0,0117,
    e na escala de 0 a 1 a decisão caía no primeiro 1% da barra (RESUMO_TCC
    §10.2.12). Sem limiar, devolve o score como está.
    """
    import math

    if score is None or limiar is None:
        return score

    def logit(p: float) -> float:
        p = min(max(float(p), _EPS), 1.0 - _EPS)
        return math.log(p) - math.log1p(-p)

    return 0.5 * (1.0 + math.tanh(0.5 * (logit(score) - logit(limiar))))
