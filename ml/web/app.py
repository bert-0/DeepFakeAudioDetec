"""Aplicação web: envio de áudio, resultado por janela e histórico.

Rodar (de dentro de `ml/`):
    pip install -r web/requirements.txt
    uvicorn web.app:app --reload

Variáveis de ambiente opcionais: DETECTOR_CONFIG, DETECTOR_CHECKPOINT,
DETECTOR_DEVICE (padrão cpu), DETECTOR_BANCO (padrão outputs/web/historico.db),
DETECTOR_LIMIAR_FUSAO e DETECTOR_XLSR (o modelo final, com HF_HOME no cache do
XLS-R; ver web/analise.py).
"""

from __future__ import annotations

import os
import re
import tempfile
import unicodedata
from pathlib import Path
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
import json
import time
from typing import Literal

from pydantic import BaseModel

from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.capture import CaptureError
from src.limiares import centrado
from src.xlsr_final import NOME as NOME_XLSR, FalaCurta
from web.analise import Detector, nome_dos_modelos, significativos
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


@app.middleware("http")
async def mesma_origem(request: Request, call_next):
    """Recusa POST vindo de outro site aberto no mesmo navegador: sem isso, uma
    página qualquer poderia limpar o histórico ou ligar a captura."""
    if request.method == "POST":
        origem = request.headers.get("origin")
        if origem is not None and urlsplit(origem).netloc != request.headers.get("host"):
            return PlainTextResponse("Pedido de outro site recusado.", status_code=403)
    return await call_next(request)


templates.env.filters["br"] = _br
templates.env.filters["mmss"] = _mmss
templates.env.filters["data"] = _data
templates.env.filters["frase"] = _frase
# Escala da tela: o limiar fica no meio da barra (src/limiares.py).
templates.env.filters["centrado"] = centrado

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
        except FalaCurta as erro:
            print(f"[envio] {nome}: {erro}")
            raise HTTPException(422, "Há pouca fala no áudio para analisar.") from erro
        except (CaptureError, RuntimeError) as erro:
            print(f"[envio] {nome}: {erro!r}")   # o detalhe fica no terminal, não na tela
            raise HTTPException(422, _mensagem_de_leitura(ext)) from erro
    return resultado, banco().salvar(resultado, detector().modelo)


def _mensagem_de_leitura(ext: str) -> str:
    msg = "Não foi possível ler o áudio: o arquivo pode estar corrompido ou incompleto."
    if ext in (".m4a", ".aac"):
        msg += (" Arquivos M4A e AAC precisam do FFmpeg instalado no computador;"
                " sem ele, converta para WAV ou MP3.")
    return msg


@app.post("/analisar")
async def analisar(request: Request, arquivo: UploadFile = File(...)):
    try:
        _, analise_id = await _analisar_upload(arquivo)
    except HTTPException as erro:
        return templates.TemplateResponse(
            request, "inicio.html",
            {"extensoes": sorted(EXTENSOES), "aba": "enviar", "erro": erro.detail},
            status_code=erro.status_code)
    return RedirectResponse(f"/analises/{analise_id}", status_code=303)


@app.get("/analises/{analise_id}", response_class=HTMLResponse)
def ver_analise(request: Request, analise_id: int):
    a = banco().buscar(analise_id)
    if a is None:
        raise HTTPException(404, "Análise não encontrada.")
    return templates.TemplateResponse(request, "resultado.html", {
        "a": a, "aba": "resultado", "audio": _audio_da_analise(a) is not None})


@app.get("/analises/{analise_id}/audio")
def audio_da_analise(analise_id: int):
    """A gravação da sessão ao vivo ou o clone, para ouvir na página."""
    a = banco().buscar(analise_id)
    caminho = _audio_da_analise(a) if a else None
    if caminho is None:
        raise HTTPException(404, "Áudio não encontrado.")
    return FileResponse(caminho, media_type="audio/wav")


def _audio_da_analise(a: dict) -> Path | None:
    """Só arquivos da pasta ao vivo ou de um clone: o caminho vem do banco, mas a
    rota não serve nada fora delas."""
    if not a.get("gravacao"):
        return None
    wav = Path(a["gravacao"]).resolve()
    dentro = wav.parent == pasta_ao_vivo().resolve() or wav.parent.parent == pasta_clones().resolve()
    return wav if dentro and wav.is_file() else None


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
    return JSONResponse(a, headers={"Content-Disposition": _anexo(a["arquivo"], "_relatorio.json")})


def _anexo(arquivo: str, sufixo: str) -> str:
    """Content-Disposition que aceita qualquer nome: `filename*` em UTF-8 para os
    navegadores e `filename` em ASCII de reserva. A "/" das datas vira "-"."""
    base = Path(re.sub(r'[\\/:*?"<>|]', "-", arquivo)).stem or "analise"
    nome = base + sufixo
    ascii_ = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode()
    ascii_ = re.sub(r"[^A-Za-z0-9._() -]", "_", ascii_)
    return f"attachment; filename=\"{ascii_}\"; filename*=UTF-8''{quote(nome)}"


@app.get("/historico", response_class=HTMLResponse)
def historico(request: Request):
    return templates.TemplateResponse(request, "historico.html",
                                      {"analises": banco().listar(500), "aba": "historico"})


@app.post("/historico/excluir")
def historico_excluir(ids: list[int] = Form(default=[])):
    gravacoes = banco().gravacoes(ids)
    banco().excluir(ids)
    for wav in gravacoes:
        _apagar_sessao(Path(wav))
    return RedirectResponse("/historico", status_code=303)


@app.post("/historico/limpar")
def historico_limpar():
    banco().limpar()
    # Varre a pasta inteira: pega também sessões salvas antes de o caminho da
    # gravação ir para o banco.
    for wav in pasta_ao_vivo().glob("sessao_*.wav"):
        _apagar_sessao(wav)
    for pasta in pasta_clones().glob("clone_*"):
        _apagar_sessao(pasta / "frase_01.wav")
    return RedirectResponse("/historico", status_code=303)


@app.get("/sobre", response_class=HTMLResponse)
def sobre(request: Request):
    d = detector()
    if d.xlsr:
        return templates.TemplateResponse(request, "sobre.html", {
            "modelo": d.modelo, "origem": d.limiar_xlsr.descricao(), "ensaio2": False, "xlsr": True})
    origem = d.fusao.descricao() if d.fusao else None
    # O texto do ensaio 2 só vale para o modelo do recorte no limiar do treino.
    ensaio2 = d.modelo == "fusion_lcnn_v4_asv5" and bool(origem) and "treino" in origem
    return templates.TemplateResponse(request, "sobre.html",
                                      {"modelo": d.modelo, "origem": origem, "ensaio2": ensaio2})


@app.post("/api/analisar")
async def api_analisar(arquivo: UploadFile = File(...)):
    r, analise_id = await _analisar_upload(arquivo)
    return JSONResponse({"id": analise_id, "arquivo": r.arquivo, "duracao_s": r.duracao_s,
                         "score_medio": r.score_medio, "limiar": r.limiar,
                         "origem_limiar": r.origem_limiar,
                         "indicio_de_sintese": r.indicio_de_sintese,
                         "janelas": r.janelas})


# --- Clonar ------------------------------------------------------------------
# O clone da demonstração, gerado na hora e analisado como arquivo (web/clonagem.py).
from web.clonagem import (  # noqa: E402
    FERRAMENTAS,
    ErroClonagem,
    clonar,
    ferramentas_disponiveis,
    ler_vozes,
    pasta_clones,
    validar_frase,
)


class PedidoClone(BaseModel):
    voz: str
    ferramenta: str = "xtts"
    frase: str


@app.get("/clonar", response_class=HTMLResponse)
def pagina_clonar(request: Request):
    vozes = ler_vozes()
    if not vozes:
        print("[clonar] nenhuma voz autorizada (data/vozes/vozes_autorizadas.json)")
    return templates.TemplateResponse(request, "clonar.html", {
        "aba": "clonar", "vozes": [v.codigo for v in vozes], "ferramentas": ferramentas_disponiveis()})


@app.post("/api/clonar")
def api_clonar(pedido: PedidoClone):
    """Síncrona de propósito: roda numa thread do servidor, e a página espera."""
    if _rodando(_sessao()):
        raise HTTPException(409, "Pare a análise ao vivo antes de gerar um clone.")
    voz = next((v for v in ler_vozes() if v.codigo == pedido.voz), None)
    if voz is None:
        raise HTTPException(400, "Essa voz não está entre as autorizadas.")
    if pedido.ferramenta not in ferramentas_disponiveis():
        raise HTTPException(400, "Ferramenta de clonagem indisponível neste computador.")
    try:
        frase = validar_frase(pedido.frase)
    except ValueError as erro:
        raise HTTPException(400, str(erro)) from erro
    if _estado.get("clonando"):
        raise HTTPException(409, "Já há um clone sendo gerado.")
    _estado["clonando"] = True
    try:
        wav = clonar(voz, frase, pedido.ferramenta)
    except ErroClonagem as erro:
        print(f"[clonar] {erro}")
        raise HTTPException(500, "Não foi possível gerar o clone.") from erro
    finally:
        _estado["clonando"] = False
    nome = f"Clone {FERRAMENTAS[pedido.ferramenta][0]} de {voz.codigo}: {frase[:60]}"
    try:
        r = detector().analisar(wav, nome_original=nome)
    except FalaCurta as erro:
        print(f"[clonar] {wav}: {erro}")
        raise HTTPException(422, "O clone saiu com pouca fala para analisar.") from erro
    return {"analise_id": banco().salvar(r, detector().modelo, gravacao=str(wav.resolve()))}


# --- Ao vivo -----------------------------------------------------------------
# O navegador não acessa o áudio do sistema: o servidor roda o monitor.py como
# processo filho e a página lê o JSON que ele grava. Para parar, cria o arquivo
# de `--parar-com` (no Windows não há Ctrl+C entre processos; matar perde o WAV).
RAIZ_ML = AQUI.parent


def pasta_ao_vivo() -> Path:
    return Path(os.environ.get("DETECTOR_PASTA_AO_VIVO", "outputs/ao_vivo"))


def _sessao() -> dict | None:
    return _estado.get("sessao")


def _apagar_sessao(wav: Path) -> None:
    """Apaga a gravação e os arquivos da sessão (JSON, log). Só dentro da pasta
    ao vivo, e nunca a sessão que ainda está gravando. Um clone sai com a pasta dele."""
    import shutil

    wav = wav.resolve()
    if wav.parent.parent == pasta_clones().resolve() and wav.parent.name.startswith("clone_"):
        shutil.rmtree(wav.parent, ignore_errors=True)
        return
    if wav.parent != pasta_ao_vivo().resolve():
        return
    atual = _sessao()
    if atual and Path(atual["wav"]).resolve() == wav:
        if _rodando(atual):
            return
        _estado.pop("sessao", None)
    for ext in (".wav", ".json", ".log", ".parar"):
        wav.with_suffix(ext).unlink(missing_ok=True)


def _rodando(sessao: dict | None) -> bool:
    return bool(sessao) and sessao["proc"].poll() is None


FONTES = {"sistema": "som do computador", "microfone": "microfone"}


class PedidoAoVivo(BaseModel):
    fonte: Literal["sistema", "microfone"] = "sistema"
    dispositivo: str | None = None


def _iniciar_monitor(json_path: Path, wav_path: Path, parar_path: Path, log_path: Path,
                     fonte: str = "sistema", dispositivo: str | None = None):
    import subprocess
    import sys

    fusao = os.environ.get("DETECTOR_LIMIAR_FUSAO")
    if os.environ.get("DETECTOR_XLSR"):
        # O modelo final analisa frase por frase, pelo mesmo caminho do arquivo.
        modelo = ["--xlsr", os.environ["DETECTOR_XLSR"]]
    elif fusao:
        # A fusão do teste com voz real: modelos, regra e limiar escolhidos nos devs.
        modelo = ["--limiar-fusao", fusao]
    else:
        d = detector()
        modelo = ["--config", d.config_path, "--checkpoint", d.checkpoint]
    # -u: sem buffer, o registro mostra na página o motivo de uma falha.
    cmd = [sys.executable, "-u", str(RAIZ_ML / "monitor.py"), *modelo, "--json", str(json_path),
           "--gravar", str(wav_path), "--parar-com", str(parar_path), "--fonte", fonte]
    if dispositivo:
        cmd += ["--dispositivo-audio", dispositivo]
    log = open(log_path, "w", encoding="utf-8")
    # Em UTF-8: no código de página do Windows, um acento no registro derrubava a leitura dele.
    return subprocess.Popen(cmd, cwd=RAIZ_ML, stdout=log, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})


@app.get("/ao-vivo", response_class=HTMLResponse)
def ao_vivo(request: Request):
    return templates.TemplateResponse(request, "ao_vivo.html", {
        "aba": "ao_vivo", "por_frase": bool(os.environ.get("DETECTOR_XLSR"))})


@app.get("/api/ao-vivo/dispositivos")
def api_ao_vivo_dispositivos():
    """Saídas e microfones do computador; vazio onde não há captura (ex.: Linux sem
    PulseAudio), e a página fica só com o dispositivo padrão."""
    from src.capture import listar_dispositivos

    try:
        return listar_dispositivos()
    except Exception as erro:
        return {"sistema": [], "microfone": [], "padrao": {}, "erro": str(erro)}


@app.post("/api/ao-vivo/iniciar")
def api_ao_vivo_iniciar(pedido: PedidoAoVivo | None = None):
    pedido = pedido or PedidoAoVivo()
    if _rodando(_sessao()):
        raise HTTPException(409, "Já há uma sessão ao vivo em andamento.")
    pasta = pasta_ao_vivo()
    pasta.mkdir(parents=True, exist_ok=True)
    marca = time.strftime("%Y%m%d_%H%M%S")
    caminhos = {k: pasta / f"sessao_{marca}{ext}" for k, ext in
                (("json", ".json"), ("wav", ".wav"), ("parar", ".parar"), ("log", ".log"))}
    proc = _iniciar_monitor(caminhos["json"], caminhos["wav"], caminhos["parar"], caminhos["log"],
                            fonte=pedido.fonte, dispositivo=pedido.dispositivo or None)
    _estado["sessao"] = {**caminhos, "proc": proc, "inicio": time.time(), "analise_id": None,
                         "fonte": pedido.fonte}
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
        arquivo=f"Sessão ao vivo ({FONTES[sessao.get('fonte', 'sistema')]}) "
                + time.strftime("%d/%m/%Y %H:%M", time.localtime(sessao["inicio"])),
        duracao_s=dados.get("duracao_s") or leituras[-1]["instante"] + 4.0,
        taxa_nativa=None,
        score_medio=resumo.get("score_medio"),
        score_maximo=resumo.get("score_maximo"),
        limiar=dados.get("limiar"),
        origem_limiar=dados.get("origem_limiar", ""),
        aviso_limiar=None,
        janelas_uteis=resumo.get("janelas_uteis", 0),
        janelas_independentes=resumo.get("janelas_independentes", 0.0),
        janelas=[{"t": round(x["instante"], 2), "score": significativos(x["score"]),
                  "util": x.get("util", not x.get("silencio")), "peso": x.get("peso", 1.0)}
                 for x in leituras])
    return banco().salvar(r, _modelo_ao_vivo(), gravacao=str(Path(sessao["wav"]).resolve()))


def _modelo_ao_vivo() -> str:
    if os.environ.get("DETECTOR_XLSR"):
        return NOME_XLSR
    fusao = os.environ.get("DETECTOR_LIMIAR_FUSAO")
    if not fusao:
        return detector().modelo
    try:
        checkpoints = [m["checkpoint"] for m in
                       json.loads(Path(fusao).read_text(encoding="utf-8"))["modelos"]]
    except (OSError, ValueError, KeyError):
        return "fusão"
    return nome_dos_modelos(checkpoints)


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
            "fonte": sessao.get("fonte", "sistema"),
            "decorrido_s": round(time.time() - sessao["inicio"], 1)}
    if dados is None:
        erro = None
        if not rodando:
            try:
                erro = Path(sessao["log"]).read_text(encoding="utf-8", errors="replace")[-600:]
            except OSError:
                erro = "o monitor encerrou sem registrar nada"
        return {**base, "existe": False, "erro": erro}
    limiar, resumo = dados.get("limiar"), dados.get("resumo", {})
    return {**base, "existe": True,
            "unidade": dados.get("unidade", "janela"),
            "limiar": limiar,
            "origem_limiar": dados.get("origem_limiar", ""),
            "resumo": resumo,
            "centrado_medio": centrado(resumo.get("score_medio"), limiar),
            "leituras": [{**x, "centrado": centrado(x.get("score"), limiar)}
                         for x in dados.get("leituras", [])[-ultimas:]]}
