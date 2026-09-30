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


templates.env.filters["br"] = _br
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


def grafico_svg(janelas: list[dict], limiar: float | None,
                largura: int = 640, altura: int = 180) -> str:
    """Score por janela ao longo do tempo, com a linha do limiar. SVG no servidor:
    sem biblioteca de gráficos no navegador."""
    uteis = [j for j in janelas if j.get("util")]
    if not uteis:
        return ""
    margem = 30
    tmax = max(j["t"] for j in uteis) or 1.0
    x = lambda t: margem + (largura - 2 * margem) * t / tmax          # noqa: E731
    y = lambda s: altura - margem - (altura - 2 * margem) * s          # noqa: E731
    pontos = " ".join(f"{x(j['t']):.1f},{y(j['score']):.1f}" for j in uteis)
    partes = [f'<svg viewBox="0 0 {largura} {altura}" class="grafico" role="img" '
              f'aria-label="Score por janela ao longo do tempo">',
              f'<line x1="{margem}" y1="{y(0)}" x2="{largura - margem}" y2="{y(0)}" class="eixo"/>',
              f'<line x1="{margem}" y1="{y(1)}" x2="{margem}" y2="{y(0)}" class="eixo"/>',
              f'<text x="4" y="{y(1) + 4}" class="rotulo">1</text>',
              f'<text x="4" y="{y(0) + 4}" class="rotulo">0</text>',
              f'<text x="{largura - margem}" y="{altura - 8}" class="rotulo" '
              f'text-anchor="end">{tmax:.0f} s</text>',
              f'<text x="{margem}" y="{altura - 8}" class="rotulo">0 s</text>']
    if limiar is not None:
        partes.append(f'<line x1="{margem}" y1="{y(limiar):.1f}" x2="{largura - margem}" '
                      f'y2="{y(limiar):.1f}" class="limiar"/>')
    partes.append(f'<polyline points="{pontos}" class="serie"/>')
    partes += [f'<circle cx="{x(j["t"]):.1f}" cy="{y(j["score"]):.1f}" r="2.5" class="ponto"/>'
               for j in uteis]
    partes.append("</svg>")
    return "".join(partes)


@app.get("/", response_class=HTMLResponse)
def inicio(request: Request):
    return templates.TemplateResponse(request, "inicio.html", {"extensoes": sorted(EXTENSOES)})


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
    return templates.TemplateResponse(request, "resultado.html", {
        "a": a, "grafico": grafico_svg(a["janelas"], a["limiar"])})


@app.get("/historico", response_class=HTMLResponse)
def historico(request: Request):
    return templates.TemplateResponse(request, "historico.html", {"analises": banco().listar()})


@app.get("/sobre", response_class=HTMLResponse)
def sobre(request: Request):
    return templates.TemplateResponse(request, "sobre.html", {"modelo": detector().modelo})


@app.post("/api/analisar")
async def api_analisar(arquivo: UploadFile = File(...)):
    r, analise_id = await _analisar_upload(arquivo)
    return JSONResponse({"id": analise_id, "arquivo": r.arquivo, "duracao_s": r.duracao_s,
                         "score_medio": r.score_medio, "limiar": r.limiar,
                         "origem_limiar": r.origem_limiar,
                         "indicio_de_sintese": r.indicio_de_sintese,
                         "janelas": r.janelas})
