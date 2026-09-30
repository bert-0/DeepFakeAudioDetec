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

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
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
    return templates.TemplateResponse(request, "resultado.html", {"a": a, "aba": "resultado"})


@app.get("/resultado", response_class=HTMLResponse)
def ultimo_resultado(request: Request):
    """Aba Resultado: a análise mais recente."""
    ultima = banco().ultima()
    if ultima is None:
        return templates.TemplateResponse(request, "resultado_vazio.html", {"aba": "resultado"})
    return RedirectResponse(f"/analises/{ultima}", status_code=303)


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
                                      {"analises": banco().listar(500), "aba": "historico"})


@app.post("/historico/excluir")
def historico_excluir(ids: list[int] = Form(default=[])):
    banco().excluir(ids)
    return RedirectResponse("/historico", status_code=303)


@app.post("/historico/limpar")
def historico_limpar():
    banco().limpar()
    return RedirectResponse("/historico", status_code=303)


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


# --------------------------------------------------------------------------- #
# Ao vivo: a página inicia e para o monitor.py
#
# A captura da saída de som fica no monitor (o navegador não tem acesso ao
# áudio do sistema). O servidor abre o monitor como processo filho, ele grava o
# estado num JSON a cada janela, e a página lê esse JSON a cada 2 s. Para parar,
# o servidor cria um arquivo que o monitor vigia (`--parar-com`): no Windows não
# dá para mandar Ctrl+C a outro processo, e matá-lo perderia a gravação.
# --------------------------------------------------------------------------- #
RAIZ_ML = AQUI.parent


def pasta_ao_vivo() -> Path:
    return Path(os.environ.get("DETECTOR_PASTA_AO_VIVO", "outputs/ao_vivo"))


def _sessao() -> dict | None:
    return _estado.get("sessao")


def _rodando(sessao: dict | None) -> bool:
    return bool(sessao) and sessao["proc"].poll() is None


def _iniciar_monitor(json_path: Path, wav_path: Path, parar_path: Path, log_path: Path):
    import subprocess
    import sys

    d = detector()
    cmd = [sys.executable, str(RAIZ_ML / "monitor.py"), "--config", d.config_path,
           "--checkpoint", d.checkpoint, "--json", str(json_path),
           "--gravar", str(wav_path), "--parar-com", str(parar_path)]
    log = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=RAIZ_ML, stdout=log, stderr=subprocess.STDOUT)


@app.get("/ao-vivo", response_class=HTMLResponse)
def ao_vivo(request: Request):
    return templates.TemplateResponse(request, "ao_vivo.html", {"aba": "ao_vivo"})


@app.post("/api/ao-vivo/iniciar")
def api_ao_vivo_iniciar():
    if _rodando(_sessao()):
        raise HTTPException(409, "Já há uma sessão ao vivo em andamento.")
    pasta = pasta_ao_vivo()
    pasta.mkdir(parents=True, exist_ok=True)
    marca = time.strftime("%Y%m%d_%H%M%S")
    caminhos = {k: pasta / f"sessao_{marca}{ext}" for k, ext in
                (("json", ".json"), ("wav", ".wav"), ("parar", ".parar"), ("log", ".log"))}
    proc = _iniciar_monitor(caminhos["json"], caminhos["wav"], caminhos["parar"], caminhos["log"])
    _estado["sessao"] = {**caminhos, "proc": proc, "inicio": time.time(), "analise_id": None}
    return {"ok": True}


def _ler_json_sessao(sessao: dict) -> dict | None:
    try:
        return json.loads(Path(sessao["json"]).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _salvar_sessao_no_historico(sessao: dict) -> int | None:
    """A sessão encerrada vai para Resultados, como uma análise."""
    from web.analise import Resultado

    dados = _ler_json_sessao(sessao)
    if not dados or not dados.get("leituras"):
        return None
    leituras, resumo = dados["leituras"], dados.get("resumo", {})
    r = Resultado(
        arquivo="Sessão ao vivo " + time.strftime("%d/%m/%Y %H:%M",
                                                  time.localtime(sessao["inicio"])),
        duracao_s=leituras[-1]["instante"] + 4.0,
        taxa_nativa=None,
        score_medio=resumo.get("score_medio"),
        score_maximo=resumo.get("score_maximo"),
        limiar=dados.get("limiar"),
        origem_limiar=dados.get("origem_limiar", ""),
        aviso_limiar=None,
        janelas_uteis=resumo.get("janelas_uteis", 0),
        janelas_independentes=resumo.get("janelas_independentes", 0.0),
        janelas=[{"t": round(x["instante"], 2), "score": round(x["score"], 4),
                  "util": x.get("util", not x.get("silencio")), "peso": x.get("peso", 1.0)}
                 for x in leituras])
    return banco().salvar(r, detector().modelo)


@app.post("/api/ao-vivo/parar")
def api_ao_vivo_parar():
    sessao = _sessao()
    if not sessao:
        raise HTTPException(409, "Nenhuma sessão ao vivo em andamento.")
    if _rodando(sessao):
        Path(sessao["parar"]).touch()
        try:
            sessao["proc"].wait(timeout=20)
        except Exception:
            sessao["proc"].terminate()
    if sessao["analise_id"] is None:
        sessao["analise_id"] = _salvar_sessao_no_historico(sessao)
    return {"ok": True, "analise_id": sessao["analise_id"]}


@app.get("/api/ao-vivo")
def api_ao_vivo(ultimas: int = 40):
    """Só a sessão iniciada por esta página — sessões antigas ficam em Resultados."""
    sessao = _sessao()
    if not sessao:
        return {"existe": False, "rodando": False}
    rodando = _rodando(sessao)
    dados = _ler_json_sessao(sessao)
    base = {"rodando": rodando, "analise_id": sessao["analise_id"],
            "decorrido_s": round(time.time() - sessao["inicio"], 1)}
    if dados is None:
        erro = None
        if not rodando:
            try:
                erro = Path(sessao["log"]).read_text(encoding="utf-8")[-600:]
            except OSError:
                erro = "o monitor encerrou sem registrar nada"
        return {**base, "existe": False, "erro": erro}
    return {**base, "existe": True,
            "limiar": dados.get("limiar"),
            "origem_limiar": dados.get("origem_limiar", ""),
            "resumo": dados.get("resumo", {}),
            "leituras": dados.get("leituras", [])[-ultimas:]}
