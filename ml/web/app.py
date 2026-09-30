"""Aplicação web: envio de áudio, resultado por janela e histórico.

Rodar (de dentro de `ml/`):
    pip install -r web/requirements.txt
    uvicorn web.app:app --reload

Variáveis de ambiente opcionais: DETECTOR_CONFIG, DETECTOR_CHECKPOINT,
DETECTOR_DEVICE (padrão cpu), DETECTOR_BANCO (padrão outputs/web/historico.db).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
import json
import time

from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.capture import CaptureError
from web.analise import Detector
from web.banco import Banco

AQUI = Path(__file__).resolve().parent
EXTENSOES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"}
TAMANHO_MAXIMO = 25 * 1024 * 1024   # 25 MB: ~2 min de WAV a 48 kHz estéreo

app = FastAPI(title="Detector de áudio sintético")
app.mount("/static", StaticFiles(directory=AQUI / "static"), name="static")
templates = Jinja2Templates(directory=AQUI / "templates")


def _br(valor, casas: int = 3) -> str:
    """Número no formato brasileiro (vírgula decimal)."""
    if valor is None:
        return "—"
    return f"{valor:.{casas}f}".replace(".", ",")


def _frase(texto: str | None) -> str:
    """Primeira letra maiúscula, sem mexer no resto (o `capitalize` mexe)."""
    return texto[:1].upper() + texto[1:] if texto else ""


def _data(iso: str | None) -> str:
    """'2026-09-30T13:11:54' -> '30/09/2026 13:11'."""
    if not iso:
        return ""
    dia, _, hora = iso.partition("T")
    a, m, d = dia.split("-")
    return f"{d}/{m}/{a} {hora[:5]}".strip()


def _mmss(segundos) -> str:
    if segundos is None:
        return "--:--"
    s = int(round(segundos))
    return f"{s // 60:02d}:{s % 60:02d}"


templates.env.filters["br"] = _br
templates.env.filters["mmss"] = _mmss
templates.env.filters["data"] = _data
templates.env.filters["frase"] = _frase

_estado: dict = {}


def detector() -> Detector:
    if "detector" not in _estado:
        _estado["detector"] = Detector()
    return _estado["detector"]


def banco() -> Banco:
    if "banco" not in _estado:
        _estado["banco"] = Banco(os.environ.get("DETECTOR_BANCO", "outputs/web/historico.db"))
    return _estado["banco"]


@app.get("/", response_class=HTMLResponse)
def inicio(request: Request):
    return templates.TemplateResponse(request, "inicio.html",
                                      {"extensoes": sorted(EXTENSOES), "aba": "enviar"})


async def _analisar_upload(arquivo: UploadFile):
    nome = Path(arquivo.filename or "audio").name
    ext = Path(nome).suffix.lower()
    if ext not in EXTENSOES:
        raise HTTPException(400, f"Formato não aceito ({ext or 'sem extensão'}). "
                                 f"Use: {', '.join(sorted(EXTENSOES))}.")
    conteudo = await arquivo.read()
    if len(conteudo) > TAMANHO_MAXIMO:
        raise HTTPException(413, "Arquivo maior que 25 MB.")
    if not conteudo:
        raise HTTPException(400, "Arquivo vazio.")
    with tempfile.TemporaryDirectory() as pasta:
        caminho = Path(pasta) / f"envio{ext}"
        caminho.write_bytes(conteudo)
        try:
            resultado = detector().analisar(caminho, nome_original=nome)
        except (CaptureError, RuntimeError) as erro:
            raise HTTPException(422, f"Não foi possível ler o áudio: {erro}") from erro
    return resultado, banco().salvar(resultado, detector().modelo)


@app.post("/analisar")
async def analisar(arquivo: UploadFile = File(...)):
    _, analise_id = await _analisar_upload(arquivo)
    return RedirectResponse(f"/analises/{analise_id}", status_code=303)


@app.get("/analises/{analise_id}", response_class=HTMLResponse)
def ver_analise(request: Request, analise_id: int):
    a = banco().buscar(analise_id)
    if a is None:
        raise HTTPException(404, "Análise não encontrada.")
    return templates.TemplateResponse(request, "resultado.html", {"a": a, "aba": "resultados"})


@app.get("/analises/{analise_id}/relatorio.json")
def relatorio(analise_id: int):
    a = banco().buscar(analise_id)
    if a is None:
        raise HTTPException(404, "Análise não encontrada.")
    nome = Path(a["arquivo"]).stem + "_relatorio.json"
    return JSONResponse(a, headers={"Content-Disposition": f'attachment; filename="{nome}"'})


@app.get("/historico", response_class=HTMLResponse)
def historico(request: Request):
    return templates.TemplateResponse(request, "historico.html",
                                      {"analises": banco().listar(), "aba": "resultados"})


@app.get("/sobre", response_class=HTMLResponse)
def sobre(request: Request):
    return templates.TemplateResponse(request, "sobre.html", {"modelo": detector().modelo})


def arquivo_ao_vivo() -> Path:
    return Path(os.environ.get("DETECTOR_AO_VIVO", "outputs/ao_vivo.json"))


@app.get("/ao-vivo", response_class=HTMLResponse)
def ao_vivo(request: Request):
    return templates.TemplateResponse(request, "ao_vivo.html", {
        "aba": "ao_vivo", "arquivo": arquivo_ao_vivo().as_posix()})


#: Sem atualização por mais que isto, a sessão é dada como encerrada (o monitor
#: grava a cada janela, ou seja, a cada 2 s; Ctrl+C grava com ativo=False).
SESSAO_PARADA_S = 10.0


@app.get("/api/ao-vivo")
def api_ao_vivo(ultimas: int = 40):
    """Estado da sessão do monitor, lido do JSON que ele grava a cada janela."""
    caminho = arquivo_ao_vivo()
    if not caminho.is_file():
        return {"existe": False}
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"existe": False}
    idade = time.time() - float(dados.get("atualizado_em") or 0)
    return {"existe": True,
            "ativo": bool(dados.get("ativo")) and idade < SESSAO_PARADA_S,
            "idade_s": round(idade, 1),
            "limiar": dados.get("limiar"),
            "origem_limiar": dados.get("origem_limiar", ""),
            "resumo": dados.get("resumo", {}),
            "leituras": dados.get("leituras", [])[-ultimas:]}


@app.post("/api/analisar")
async def api_analisar(arquivo: UploadFile = File(...)):
    r, analise_id = await _analisar_upload(arquivo)
    return JSONResponse({"id": analise_id, "arquivo": r.arquivo, "duracao_s": r.duracao_s,
                         "score_medio": r.score_medio, "limiar": r.limiar,
                         "origem_limiar": r.origem_limiar,
                         "indicio_de_sintese": r.indicio_de_sintese,
                         "janelas": r.janelas})
