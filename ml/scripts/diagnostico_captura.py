"""Diagnóstico da captura: pacotes do microfone que chegam repetidos.

Nas gravações reais do projeto, cerca de 1 em cada 3 pacotes de 10 ms é cópia
exata de um pacote anterior: de 11 pacotes antes (110 ms) com os blocos antigos
de 100 ms, de 2 pacotes antes (20 ms) com os pacotes do driver (RESUMO_TCC
§10.2.34). Num sinal de microfone, dois trechos de 480 amostras idênticos bit a
bit não acontecem por acaso: o som novo daquele pacote se perdeu. O teste do tom
de 1.000 Hz (§10.2.16) não via isso, porque 10 ms são 10 ciclos exatos.

As cópias são inseridas, não trocadas: depois de cada uma, o sinal segue do
ponto onde parou (a emenda fica contínua), e a gravação sai ~1,47x mais longa.
Por isso `reparar` recupera o áudio só apagando as cópias.

Subcomandos:
- `arquivos`: conta os pacotes repetidos em gravações já feitas;
- `gravar`: grava trechos curtos do microfone (só ruído ambiente, nenhum som é
  tocado, nada é salvo) em várias configurações (modo exclusivo ou
  compartilhado, 1 ou 2 canais, tamanho do pacote, latência);
- `reparar`: grava cópias das gravações sem os pacotes repetidos (fora do Git,
  como toda voz coletada).

Uso (de dentro de ml/):
    python scripts/diagnostico_captura.py arquivos --pasta data/vozes/ensaio2/real
    python scripts/diagnostico_captura.py gravar --segundos 5
    python scripts/diagnostico_captura.py reparar --pasta data/vozes/ensaio2/real \\
        --destino data/vozes/reparadas/ensaio2_real
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Abaixo disso o pacote é silêncio digital, que pode se repetir sem defeito.
MINIMO = 1e-6


def contar_repetidos(x: np.ndarray, pacote: int = 480, minimo: float = MINIMO) -> dict:
    """Pacotes de `pacote` amostras idênticos a um anterior, e a que distância (em pacotes)."""
    n = len(x) // pacote
    blocos = x[:n * pacote].reshape(n, pacote)
    primeiro: dict[bytes, int] = {}
    distancias: Counter = Counter()
    vivos = 0
    for i in range(n):
        if np.abs(blocos[i]).max() <= minimo:
            continue
        vivos += 1
        chave = blocos[i].tobytes()
        if chave in primeiro:
            distancias[i - primeiro[chave]] += 1
        else:
            primeiro[chave] = i
    repetidos = sum(distancias.values())
    return {"pacotes": vivos, "repetidos": repetidos, "fracao": repetidos / vivos if vivos else 0.0,
            "distancias": dict(distancias.most_common(5))}


def remover_repetidos(x: np.ndarray, pacote: int = 480, minimo: float = MINIMO) -> np.ndarray:
    """O sinal sem os pacotes que repetem um anterior (o resto final, menor que um pacote, fica).

    Com vários canais, a repetição é procurada no primeiro e o quadro sai inteiro."""
    ref = x if x.ndim == 1 else x[:, 0]
    n = len(ref) // pacote
    vistos: set[bytes] = set()
    fica = np.ones(len(x), dtype=bool)
    for i in range(n):
        trecho = ref[i * pacote:(i + 1) * pacote]
        if np.abs(trecho).max() <= minimo:
            continue
        chave = trecho.tobytes()
        if chave in vistos:
            fica[i * pacote:(i + 1) * pacote] = False
        else:
            vistos.add(chave)
    return x[fica]


def gravar_trecho(segundos: float, exclusivo: bool, canais: int, blocksize: int,
                  latencia: str | None, dispositivo=None) -> tuple[np.ndarray, int, dict]:
    """Grava pelo sounddevice como o gravador (callback que copia cada pacote)."""
    import queue

    import sounddevice as sd

    from src.capture.sources import _microfone_wasapi

    indice, nome = _microfone_wasapi(sd, dispositivo)
    fila: queue.Queue = queue.Queue()
    tamanhos: Counter = Counter()
    avisos = {"overflow": 0}

    def callback(dados, n, tempo, status):
        if getattr(status, "input_overflow", False):
            avisos["overflow"] += 1
        tamanhos[int(n)] += 1
        fila.put(dados.copy())

    kw = {} if latencia is None else {"latency": latencia}
    with sd.InputStream(device=indice, channels=canais, dtype="float32", samplerate=48000,
                        blocksize=blocksize, extra_settings=sd.WasapiSettings(exclusive=exclusivo),
                        callback=callback, **kw):
        sd.sleep(int(segundos * 1000))
    blocos = []
    while not fila.empty():
        blocos.append(fila.get())
    x = np.concatenate(blocos)
    return x[:, 0] if x.ndim > 1 else x, 48000, {"dispositivo": nome, "pacotes_do_driver": dict(tamanhos),
                                                 "overflow": avisos["overflow"]}


#: (rótulo, exclusivo, canais, blocksize, latência). A primeira é a do gravador hoje.
CONFIGURACOES = (
    ("exclusivo, 1 canal, pacote do driver (o gravador)", True, 1, 0, None),
    ("exclusivo, 1 canal, blocos de 100 ms (o gravador antigo)", True, 1, 4800, None),
    ("exclusivo, 2 canais, pacote do driver", True, 2, 0, None),
    ("exclusivo, 1 canal, latência alta", True, 1, 0, "high"),
    ("exclusivo, 1 canal, latência baixa", True, 1, 0, "low"),
    ("compartilhado, 1 canal, pacote do driver", False, 1, 0, None),
    ("compartilhado, 2 canais, pacote do driver", False, 2, 0, None),
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Pacotes repetidos na captura do microfone")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arquivos")
    a.add_argument("--pasta", action="append", required=True)
    g = sub.add_parser("gravar")
    g.add_argument("--segundos", type=float, default=5.0)
    g.add_argument("--dispositivo", default=None)
    g.add_argument("--json", default="outputs/diagnostico_captura.json")
    r = sub.add_parser("reparar")
    r.add_argument("--pasta", required=True)
    r.add_argument("--destino", required=True)
    args = p.parse_args(argv)

    if args.cmd == "reparar":
        import soundfile as sf

        destino = Path(args.destino)
        destino.mkdir(parents=True, exist_ok=True)
        arquivos = sorted(Path(args.pasta).glob("*.wav"))
        if not arquivos:
            p.error(f"{args.pasta}: nenhum .wav")
        for arq in arquivos:
            info = sf.info(arq)
            x, sr = sf.read(arq, dtype="float64")
            y = remover_repetidos(x, pacote=sr // 100)
            sf.write(destino / arq.name, y, sr, subtype=info.subtype)
            print(f"  {arq.name}: {len(x) / sr:.2f} s -> {len(y) / sr:.2f} s")
        print(f"Reparadas em {destino} ({len(arquivos)} arquivos)")
        return 0

    if args.cmd == "arquivos":
        import soundfile as sf

        for pasta in args.pasta:
            tot = {"pacotes": 0, "repetidos": 0}
            dist: Counter = Counter()
            for arq in sorted(Path(pasta).glob("*.wav")):
                x, sr = sf.read(arq, dtype="float64")
                r = contar_repetidos(x if x.ndim == 1 else x[:, 0], pacote=sr // 100)
                tot["pacotes"] += r["pacotes"]
                tot["repetidos"] += r["repetidos"]
                dist.update(r["distancias"])
            fr = tot["repetidos"] / tot["pacotes"] if tot["pacotes"] else 0.0
            print(f"{pasta}: {tot['repetidos']}/{tot['pacotes']} pacotes de 10 ms repetidos ({100 * fr:.1f}%); "
                  f"distâncias (pacotes): {dict(dist.most_common(5))}")
        return 0

    resultado = {}
    for rotulo, exclusivo, canais, blocksize, latencia in CONFIGURACOES:
        try:
            x, sr, info = gravar_trecho(args.segundos, exclusivo, canais, blocksize, latencia, args.dispositivo)
        except Exception as erro:
            print(f"  {rotulo:58s} não abriu: {str(erro)[:80]}")
            resultado[rotulo] = {"erro": str(erro)}
            continue
        r = contar_repetidos(x, 480)
        resultado[rotulo] = {**r, **info, "segundos": len(x) / sr}
        print(f"  {rotulo:58s} {r['repetidos']:4d}/{r['pacotes']:4d} repetidos ({100 * r['fracao']:5.1f}%), "
              f"distâncias {r['distancias']}, pacotes do driver {info['pacotes_do_driver']}, "
              f"overflow {info['overflow']}")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(resultado, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nResultado: {args.json} (nenhum áudio foi guardado)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
