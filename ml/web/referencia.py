"""Gravar a referência de uma voz pela aba Clonar (pedido do grupo, 09/11/2026).

Só grava vozes que já estão em `vozes_autorizadas.json` com a clonagem autorizada:
pôr uma pessoa nova na lista continua à mão, depois de conferir o termo em papel
(PLANO_INCREMENTO_4). A gravação é a do `scripts/gravar_frases.py` (microfone em
WASAPI compartilhado, na taxa do dispositivo, sem o primeiro segundo), com as
mesmas checagens; o preparo é o do `scripts/preparar_referencia.py`.

Uma tentativa reprovada não guarda nada. A aprovada vira um arquivo novo, que passa
a ser a referência da voz; a anterior fica onde estava.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from web.clonagem import RAIZ_ML, caminho_vozes

#: O texto de referência leva de 60 a 90 s numa leitura calma; 2 min é o teto.
MAX_S = 120.0
#: Depois do "Parar", o microfone segue um pouco: a checagem pede silêncio no fim.
FOLGA_FINAL_S = 1.0
#: A margem da sala que o teste com a voz própria aceita (RESUMO_TCC §10.2.12).
PISO_MINIMO_DB = 15.0
TEXTO = RAIZ_ML / "docs" / "etica" / "texto_referencia_clonagem.txt"


def pasta_vozes() -> Path:
    return Path(os.environ.get("DETECTOR_PASTA_VOZES", RAIZ_ML / "data" / "vozes"))


def texto_referencia() -> str:
    try:
        return TEXTO.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def vozes_para_gravar(caminho: str | Path | None = None) -> list[str]:
    """As vozes com a clonagem autorizada, com ou sem referência gravada."""
    caminho = Path(caminho or caminho_vozes())
    if not caminho.is_file():
        return []
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    return [str(v["codigo"]) for v in dados.get("vozes", []) if v.get("clonagem_autorizada")]


def trocar_referencia(codigo: str, referencia: Path, caminho: str | Path | None = None) -> None:
    """Aponta a voz para a referência nova, sem mexer em mais nada da lista."""
    caminho = Path(caminho or caminho_vozes())
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    voz = next(v for v in dados["vozes"] if str(v["codigo"]) == codigo)
    try:
        voz["referencia"] = referencia.resolve().relative_to(RAIZ_ML).as_posix()
    except ValueError:
        voz["referencia"] = str(referencia.resolve())
    voz["referencia_gravada_em"] = time.strftime("%Y-%m-%d")
    temporario = caminho.with_suffix(".tmp")
    temporario.write_text(json.dumps(dados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporario.replace(caminho)


def concluir(codigo: str, wav: np.ndarray, taxa: int) -> dict:
    """Checa a gravação e, se passar, guarda a gravação e a referência preparada."""
    from scripts.gravar_frases import salvar
    from scripts.preparar_referencia import FALA_MINIMA_S, preparar
    from src.capture.checagens import checar
    from src.capture.frases import medir_frases

    if len(wav) < taxa:
        return {"ok": False, "motivo": "A gravação ficou curta demais.", "checagens": []}
    checagens = checar(wav, taxa, piso_minimo=PISO_MINIMO_DB, uma_frase=False)
    itens = [{"nome": c.nome, "ok": c.ok, "valor": c.valor, "motivo": c.motivo} for c in checagens]
    fala_s = sum(f for _, _, f in medir_frases(wav, taxa))
    reprovada = next((c for c in checagens if not c.ok), None)
    if reprovada:
        return {"ok": False, "motivo": f"Regrave: {reprovada.motivo or reprovada.nome}.", "checagens": itens}
    if fala_s < FALA_MINIMA_S:
        return {"ok": False, "checagens": itens,
                "motivo": f"Regrave: {fala_s:.0f} s de fala; leia o texto inteiro (pelo menos "
                          f"{FALA_MINIMA_S:.0f} s)."}
    import soundfile as sf

    marca = time.strftime("%Y%m%d_%H%M%S")
    salvar(pasta_vozes() / codigo / "referencia" / f"gravacao_{marca}.wav", wav, taxa)
    pronto, _ = preparar(wav, taxa)
    destino = pasta_vozes() / "referencias" / f"{codigo}_{marca}.wav"
    destino.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destino, pronto, taxa, subtype="PCM_16")
    trocar_referencia(codigo, destino)
    return {"ok": True, "voz": codigo, "fala_s": round(fala_s, 1),
            "duracao_s": round(len(pronto) / taxa, 1), "checagens": itens}


class Gravador:
    """Uma gravação por vez, numa thread do servidor; a página consulta `estado()`."""

    def __init__(self):
        self._trava = threading.Lock()
        self._parar = threading.Event()
        self._thread: threading.Thread | None = None
        self._estado: dict = {"gravando": False}

    def gravando(self) -> bool:
        with self._trava:
            return bool(self._estado.get("gravando"))

    def iniciar(self, codigo: str, abrir=None, max_s: float = MAX_S) -> None:
        """`abrir`: a fonte do microfone (nos testes, a falsa)."""
        with self._trava:
            if self._estado.get("gravando"):
                raise RuntimeError("já há uma gravação em andamento")
            self._parar.clear()
            self._estado = {"gravando": True, "voz": codigo, "decorrido_s": 0.0, "max_s": max_s,
                            "resultado": None}
        self._thread = threading.Thread(target=self._gravar, args=(codigo, abrir, max_s),
                                        name="referencia", daemon=True)
        self._thread.start()

    def parar(self) -> None:
        self._parar.set()

    def esperar(self, segundos: float | None = None) -> None:
        if self._thread:
            self._thread.join(segundos)

    def estado(self) -> dict:
        with self._trava:
            return dict(self._estado)

    def _gravar(self, codigo: str, abrir, max_s: float) -> None:
        try:
            wav, taxa = self._capturar(abrir, max_s)
            with self._trava:
                self._estado["conferindo"] = True
            resultado = concluir(codigo, wav, taxa)
        except Exception as erro:
            print(f"[referencia] {erro!r}")
            resultado = {"ok": False, "motivo": "Não foi possível gravar pelo microfone.", "checagens": []}
        with self._trava:
            self._estado.update(gravando=False, resultado=resultado)

    def _capturar(self, abrir, max_s: float) -> tuple[np.ndarray, int]:
        from scripts.gravar_frases import DESCARTE_S
        from src.capture.sources import MicrofoneWasapiSource

        fonte = abrir() if abrir else MicrofoneWasapiSource(None)
        taxa = fonte.taxa_dispositivo
        descarte, limite = int(DESCARTE_S * taxa), int((DESCARTE_S + max_s) * taxa)
        blocos, n, fim = [], 0, None
        try:
            for bloco in fonte.blocos():
                blocos.append(bloco)
                n += len(bloco)
                with self._trava:
                    self._estado["decorrido_s"] = round(max(0.0, (n - descarte) / taxa), 1)
                if fim is None and self._parar.is_set():
                    fim = n + int(FOLGA_FINAL_S * taxa)
                if n >= limite or (fim is not None and n >= fim):
                    break
        finally:
            fonte.fechar()
        if fonte.aviso:
            print(f"[referencia] {fonte.aviso}")
        wav = np.concatenate(blocos)[descarte:] if blocos else np.zeros(0, np.float32)
        return wav.astype(np.float32), taxa
