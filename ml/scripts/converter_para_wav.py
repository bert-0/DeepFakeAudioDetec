"""Converte para WAV os áudios de um ou mais protocolos, uma única vez.

Existe por causa de um defeito medido nos .flac do ASVspoof 2021: o libsndfile
(o leitor rápido) falhou em 38 de 50 arquivos testados, com
"unknown error in flac decoder" — mesmo na versão mais nova (1.2.2), e com o
cabeçalho abrindo normalmente. Nesses casos o librosa cai no `audioread`, que no
Windows abre um processo do FFmpeg POR ARQUIVO. A avaliação funciona, mas fica
tão lenta que foi interrompida, e repetiria o custo a cada modelo avaliado.

Aqui cada áudio é decodificado uma vez e gravado em WAV PCM 16 bits, que o
libsndfile lê sem problema. Tenta primeiro o leitor rápido; só usa o FFmpeg
quando ele falha. Os áudios são 16 bits na origem, então a conversão é exata:
nenhuma amostra muda.

É retomável: arquivos já convertidos são pulados, e cada WAV é gravado primeiro
num temporário e depois renomeado, para que uma interrupção não deixe um arquivo
pela metade que pareceria pronto.

Uso:
    python scripts/converter_para_wav.py \\
        --protocolo outputs/asvspoof2021/none_eval_n10000.txt \\
                    outputs/asvspoof2021/opus_eval_n10000.txt \\
        --origem data/ASVspoof2021_LA/ASVspoof2021_LA_eval/flac \\
        --destino data/ASVspoof2021_LA/wav
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import parse_protocol_with_systems  # noqa: E402

RAPIDO, ALTERNATIVO, PULADO = "rapido", "alternativo", "pulado"


def converter_um(tarefa: tuple[str, str, str]) -> tuple[str, str]:
    """Converte um arquivo. Devolve (id, status) — status é o caminho usado,
    ou `falhou: <erro>`. Nunca levanta: um arquivo ruim não derruba o lote."""
    audio_id, origem, destino = tarefa
    saida = Path(destino)
    if saida.is_file() and saida.stat().st_size > 0:
        return audio_id, PULADO

    import numpy as np
    import soundfile as sf

    try:
        try:
            wav, sr = sf.read(origem, dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)              # (amostras, canais) -> mono
            caminho = RAPIDO
        except Exception:  # noqa: BLE001 — é exatamente o caso que motiva o script
            import warnings

            import librosa

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wav, sr = librosa.load(origem, sr=None, mono=True)
            caminho = ALTERNATIVO
        wav = np.asarray(wav, dtype=np.float32)
        if wav.size == 0:
            return audio_id, "falhou: áudio vazio após decodificar"

        temporario = saida.with_name(saida.name + ".tmp")
        sf.write(temporario, wav, int(sr), subtype="PCM_16", format="WAV")
        os.replace(temporario, saida)
        return audio_id, caminho
    except Exception as erro:  # noqa: BLE001
        return audio_id, f"falhou: {erro or type(erro).__name__}"


def ids_dos_protocolos(protocolos) -> list[str]:
    """IDs únicos, na ordem em que aparecem. Referência e Opus não se repetem,
    mas o mesmo protocolo pode ser passado duas vezes sem converter em dobro."""
    vistos: dict[str, None] = {}
    for protocolo in protocolos:
        for nome, _, _ in parse_protocol_with_systems(protocolo):
            vistos.setdefault(nome, None)
    return list(vistos)


def converter(ids, origem: Path, destino: Path, ext_origem: str = ".flac",
              workers: int = 4, a_cada: int = 500) -> dict[str, list[str]]:
    destino.mkdir(parents=True, exist_ok=True)
    tarefas = [(i, str(origem / f"{i}{ext_origem}"), str(destino / f"{i}.wav"))
               for i in ids]
    resultado: dict[str, list[str]] = {RAPIDO: [], ALTERNATIVO: [], PULADO: [], "falhou": []}
    erros: dict[str, str] = {}

    inicio = time.perf_counter()
    if workers <= 1:
        iterador = map(converter_um, tarefas)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        iterador = pool.map(converter_um, tarefas, chunksize=16)
    try:
        for n, (audio_id, status) in enumerate(iterador, 1):
            if status.startswith("falhou"):
                resultado["falhou"].append(audio_id)
                erros[audio_id] = status
            else:
                resultado[status].append(audio_id)
            if n % a_cada == 0 or n == len(tarefas):
                gasto = time.perf_counter() - inicio
                resta = gasto / n * (len(tarefas) - n)
                print(f"   {n:,}/{len(tarefas):,}  ({gasto / 60:.1f} min, "
                      f"~{resta / 60:.1f} min restantes)", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()

    resultado["erros"] = [f"{k}: {v}" for k, v in list(erros.items())[:5]]
    return resultado


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Converte os áudios de protocolos para WAV")
    p.add_argument("--protocolo", nargs="+", required=True)
    p.add_argument("--origem", required=True, help="pasta dos áudios originais")
    p.add_argument("--destino", required=True, help="pasta dos WAV")
    p.add_argument("--ext-origem", default=".flac")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)

    ids = ids_dos_protocolos(args.protocolo)
    print(f"{len(ids):,} áudios únicos em {len(args.protocolo)} protocolo(s)")
    print(f"origem:  {args.origem}\ndestino: {args.destino}\n")
    r = converter(ids, Path(args.origem), Path(args.destino),
                  ext_origem=args.ext_origem, workers=args.workers)

    print(f"\n   leitor rápido (libsndfile) ... {len(r[RAPIDO]):,}")
    print(f"   leitor alternativo (FFmpeg) .. {len(r[ALTERNATIVO]):,}")
    print(f"   já convertidos (pulados) ..... {len(r[PULADO]):,}")
    print(f"   falharam ..................... {len(r['falhou']):,}")
    if r["falhou"]:
        print("\n[ERRO] alguns áudios não puderam ser convertidos:")
        for linha in r["erros"]:
            print(f"   {linha}")
        print("Rode de novo: os já convertidos são pulados.")
        return 1
    print(f"\nPronto. Rode a importação de novo com:\n"
          f"   --audio-dir {Path(args.destino).as_posix()} --audio-ext .wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
