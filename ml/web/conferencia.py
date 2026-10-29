"""Conferência antes da apresentação: o que pode falhar na frente da banca, de uma vez.

Cada item sai como "ok", "atencao" (funciona, mas vale olhar), "falha" (precisa
resolver) ou "lembrete" (não dá para medir daqui: conferir à mão). A página
`/conferir` e o `scripts/conferir.py` usam as mesmas funções.

O microfone é aberto como no ao vivo (WASAPI compartilhado) e escuta 1,5 s da
sala, sem gravar nada; o modelo final é carregado, e a primeira análise da
apresentação já não espera por ele.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

RAIZ_ML = Path(__file__).resolve().parent.parent
#: Abaixo disto, o microfone está mudo ou sem permissão; acima, a sala está barulhenta
#: (o ruído do embutido fica perto de −50 dBFS no modo compartilhado).
MUDO_DBFS, BARULHO_DBFS = -85.0, -35.0
#: Medido em 28/10/2026: o Chatterbox chega a 3,44 GB da GPU; o XLS-R do ao vivo, ~1,2 GB.
GPU_CLONAGEM_GB, GPU_AO_VIVO_GB = 3.45, 1.5
DISCO_MINIMO_GB, DISCO_FOLGA_GB = 2.0, 5.0


@dataclass
class Item:
    nome: str
    estado: str          # ok | atencao | falha | lembrete
    detalhe: str
    acao: str = ""


def modelo_final(caminho_json: str | None, carregar=None) -> list[Item]:
    """O JSON do limiar (com os pesos da cabeça conferidos), a pasta do XLS-R e a carga."""
    if not caminho_json:
        return [Item("Modelo", "atencao", "a interface está com o modelo antigo (baseline_v2)",
                     "Suba a interface com DETECTOR_XLSR apontando para o limiar do modelo final.")]
    from src.xlsr_final import ler_limiar

    try:
        lim = ler_limiar(caminho_json)
    except Exception as erro:
        print(f"[conferir] limiar: {erro!r}")
        return [Item("Modelo", "falha", "o arquivo do limiar não abre ou a cabeça mudou",
                     "Rode o scripts/limiar_xlsr.py de novo.")]
    itens = []
    data = json.loads(Path(caminho_json).read_text(encoding="utf-8")).get("data", "?")
    n = len(lim.locutores)
    frase = "—" if lim.limiar_frase is None else f"{lim.limiar_frase:.3f}".replace(".", ",")
    detalhe = (f"{n} {'voz' if n == 1 else 'vozes'} de calibração ({', '.join(lim.locutores) or 'dev'}), "
               f"em {data}; arquivo {lim.limiar:.3f} · frase {frase}".replace(".", ","))
    if not lim.calibrado_por_frase:
        itens.append(Item("Limiar", "falha", detalhe, "Calibre com vozes reais (limiar_xlsr.py --real)."))
    elif n < 2:
        itens.append(Item("Limiar", "atencao", detalhe,
                          "Limiar de uma voz só é frágil: calibre com mais pessoas."))
    else:
        itens.append(Item("Limiar", "ok", detalhe))
    if lim.calibrado_por_frase and not lim.arquivo_por_frase:
        itens.append(Item("Enviar arquivo", "atencao", "analisa o arquivo inteiro, com o ruído da sala",
                          "O combinado (29/10/2026) é só a fala: refaça o limiar com o scripts/limiar_xlsr.py."))
    pasta = Path(lim.fonte_modelo)
    if pasta.is_dir():
        itens.append(Item("XLS-R no disco", "ok", "a pasta do modelo está no lugar (não depende de internet)"))
    else:
        itens.append(Item("XLS-R no disco", "atencao", "a pasta guardada no limiar sumiu; vale o cache do HF_HOME",
                          "Rode o scripts/limiar_xlsr.py de novo para guardar a pasta."))
    if carregar is not None:
        inicio = time.perf_counter()
        try:
            carregar()
            itens.append(Item("Carga do modelo", "ok",
                              f"carregado em {time.perf_counter() - inicio:.1f} s".replace(".", ",")))
        except Exception as erro:
            print(f"[conferir] carga do modelo: {erro!r}")
            itens.append(Item("Carga do modelo", "falha", "o modelo final não carregou",
                              "Veja o erro no terminal; confira o HF_HOME e o checkpoint."))
    return itens


def gpu() -> Item:
    try:
        import torch

        if not torch.cuda.is_available():
            return Item("GPU", "atencao", "sem GPU: o ao vivo roda na CPU (~2,8 s por frase)",
                        "Confira o driver da NVIDIA.")
        livre, total = (x / 1e9 for x in torch.cuda.mem_get_info())
        nome = torch.cuda.get_device_name(0)
    except Exception as erro:
        print(f"[conferir] GPU: {erro!r}")
        return Item("GPU", "atencao", "não foi possível ler a GPU")
    detalhe = f"{nome}, {livre:.1f} de {total:.1f} GB livres".replace(".", ",")
    if livre < GPU_AO_VIVO_GB:
        return Item("GPU", "falha", detalhe, "Feche programas que usam a GPU (jogos, outro servidor).")
    if livre < GPU_CLONAGEM_GB:
        return Item("GPU", "atencao", detalhe, "O Chatterbox pode não caber: feche o que usa a GPU antes de clonar.")
    return Item("GPU", "ok", detalhe)


#: Com o driver de áudio travado, abrir o microfone não volta (visto em 29/10/2026):
#: a conferência marca a falha em vez de travar junto.
MICROFONE_MAXIMO_S = 12.0


def _com_limite(funcao, tempo_maximo_s: float, nome: str) -> Item:
    """Roda numa thread à parte; se o driver de áudio não voltar a tempo, vira falha."""
    import threading

    resultado: dict = {}

    def rodar():
        resultado["item"] = funcao()

    t = threading.Thread(target=rodar, name=f"conferir-{nome}", daemon=True)
    t.start()
    t.join(tempo_maximo_s)
    if t.is_alive() or "item" not in resultado:
        print(f"[conferir] {nome} não respondeu em {tempo_maximo_s:.0f} s")
        return Item(nome, "falha", f"não respondeu em {tempo_maximo_s:.0f} s",
                    "O áudio do Windows travou: reinicie o computador e confira de novo.")
    return resultado["item"]


def microfone(abrir=None, segundos: float = 1.5, tempo_maximo_s: float = MICROFONE_MAXIMO_S) -> Item:
    """Abre o microfone padrão como o ao vivo e mede o ruído da sala, com tempo limite."""
    return _com_limite(lambda: _microfone(abrir, segundos), tempo_maximo_s, "Microfone")


def _microfone(abrir, segundos: float) -> Item:
    import numpy as np

    if abrir is None:
        from src.capture import abrir_microfone as abrir
    try:
        fonte = abrir(16000)
    except Exception as erro:
        print(f"[conferir] microfone: {erro!r}")
        return Item("Microfone", "falha", "o microfone não abre",
                    "Confira a permissão do microfone no Windows e se ele está conectado.")
    blocos, total = [], int(segundos * 16000)
    try:
        with fonte:
            for b in fonte.blocos():
                blocos.append(np.asarray(b, np.float32))
                if sum(len(x) for x in blocos) >= total:
                    break
    except Exception as erro:
        print(f"[conferir] microfone: {erro!r}")
        return Item("Microfone", "falha", "o microfone abriu, mas não entregou áudio",
                    "Feche outros programas que usam o microfone e confira de novo.")
    wav = np.concatenate(blocos) if blocos else np.zeros(1, np.float32)
    db = float(20 * np.log10(np.sqrt(np.mean(wav.astype(np.float64) ** 2)) + 1e-12))
    nome = getattr(fonte, "nome", None) or "microfone padrão"
    modo = getattr(fonte, "modo", None) or "compartilhado"
    detalhe = f"{nome}, modo {modo}; ruído da sala {db:.0f} dBFS"
    if db < MUDO_DBFS:
        return Item("Microfone", "falha", detalhe, "Está mudo: confira o volume e a permissão do microfone.")
    if modo != "compartilhado":
        return Item("Microfone", "atencao", detalhe, "O exclusivo insere cópias de trechos: use o compartilhado.")
    if db > BARULHO_DBFS:
        return Item("Microfone", "atencao", detalhe,
                    "Sala barulhenta ou alguém falando: confira em silêncio.")
    return Item("Microfone", "ok", detalhe)


def saida_de_som(listar=None, tempo_maximo_s: float = MICROFONE_MAXIMO_S) -> Item:
    return _com_limite(lambda: _saida_de_som(listar), tempo_maximo_s, "Som do computador")


def _saida_de_som(listar=None) -> Item:
    if listar is None:
        from src.capture import listar_dispositivos as listar
    try:
        d = listar()
    except Exception as erro:
        print(f"[conferir] saída: {erro!r}")
        return Item("Som do computador", "atencao", "não foi possível listar as saídas de som")
    padrao = (d.get("padrao") or {}).get("sistema")
    if not padrao:
        return Item("Som do computador", "falha", "nenhuma saída de som padrão", "Conecte ou ative o alto-falante.")
    return Item("Som do computador", "ok", f"saída padrão: {padrao}")


def clonagem() -> list[Item]:
    from web.clonagem import FERRAMENTAS, ferramentas_disponiveis, ler_vozes

    prontas = ferramentas_disponiveis()
    faltam = [nome for f, (nome, _) in FERRAMENTAS.items() if f not in prontas]
    itens = [Item("Ferramentas de clonagem", "ok" if not faltam else ("atencao" if prontas else "falha"),
                  ("prontas: " + ", ".join(prontas.values())) if prontas else "nenhuma instalada",
                  ("Falta o ambiente conda de: " + ", ".join(faltam)) if faltam else "")]
    vozes = ler_vozes()
    if vozes:
        itens.append(Item("Vozes para clonar", "ok", ", ".join(v.codigo for v in vozes)))
    else:
        itens.append(Item("Vozes para clonar", "atencao", "nenhuma voz ligada",
                          "Ligue em data/vozes/vozes_autorizadas.json, depois de conferir o termo."))
    return itens


def disco(pasta: str | Path = "outputs") -> Item:
    pasta = Path(pasta)
    livre = shutil.disk_usage(pasta if pasta.exists() else RAIZ_ML).free / 1e9
    detalhe = f"{livre:.0f} GB livres onde ficam as gravações e os clones"
    if livre < DISCO_MINIMO_GB:
        return Item("Disco", "falha", detalhe, "Libere espaço: as sessões ao vivo e os clones não cabem.")
    if livre < DISCO_FOLGA_GB:
        return Item("Disco", "atencao", detalhe, "Pouco espaço: libere antes de gravar muito.")
    return Item("Disco", "ok", detalhe)


def sessao_ao_vivo(rodando: bool) -> Item:
    if rodando:
        return Item("Ao vivo", "atencao", "há uma análise ao vivo rodando", "Pare antes de clonar ou de conferir o microfone.")
    return Item("Ao vivo", "ok", "nenhuma sessão rodando")


def lembretes() -> list[Item]:
    return [
        Item("Aprimoramentos de áudio", "lembrete", "desligados no microfone e no alto-falante",
             "Configurações › Som › dispositivo › Aprimoramentos de áudio."),
        Item("Fone", "lembrete", "sem fone com microfone conectado",
             "Com fone, o driver troca o microfone embutido sem mudar o nome."),
        Item("Volume do microfone", "lembrete", "o mesmo da frase de teste e da calibração"),
    ]


def conferir_tudo(caminho_xlsr: str | None, carregar=None, rodando: bool = False,
                  abrir_microfone=None, listar=None) -> list[Item]:
    itens = modelo_final(caminho_xlsr, carregar)
    itens.append(gpu())
    itens.append(sessao_ao_vivo(rodando))
    # Com o ao vivo rodando, o microfone está ocupado: não dá para medir sem atrapalhar.
    if not rodando:
        itens.append(microfone(abrir_microfone))
    itens.append(saida_de_som(listar))
    itens.extend(clonagem())
    itens.append(disco(os.environ.get("DETECTOR_PASTA_AO_VIVO", "outputs")))
    itens.extend(lembretes())
    return itens


def resumo(itens: list[Item]) -> str:
    falhas = sum(i.estado == "falha" for i in itens)
    atencoes = sum(i.estado == "atencao" for i in itens)
    if falhas:
        return f"{falhas} {'falha' if falhas == 1 else 'falhas'} para resolver antes de apresentar"
    if atencoes:
        return f"pronto, com {atencoes} {'ponto' if atencoes == 1 else 'pontos'} de atenção"
    return "pronto para apresentar"


def como_dicts(itens: list[Item]) -> list[dict]:
    return [asdict(i) for i in itens]


def situacao(itens: list[Item]) -> str:
    """A pior situação: "falha", "atencao" ou "ok" (os lembretes não contam)."""
    estados = {i.estado for i in itens}
    return "falha" if "falha" in estados else ("atencao" if "atencao" in estados else "ok")
