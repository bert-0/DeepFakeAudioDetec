"""Clona uma voz com o XTTS v2 (Coqui), no próprio computador.

A voz não sai da máquina. Roda num ambiente conda separado: o XTTS fixa versões
do PyTorch e de outras bibliotecas que podem quebrar o ambiente do detector. Por
isso este script não importa nada do projeto.

Cada linha de --frases vira `fNN.wav` em --saida (24 kHz, a taxa do XTTS).
Arquivos que já existem são pulados. Em --saida fica também `clonagem.json`:
modelo, idioma, semente, referência e o tempo de cada frase, que o registro do
teste usa (RESUMO_TCC §10.2.12).

Uso (de dentro de ml/, no ambiente do XTTS):
    python scripts/clonar_xtts.py \\
        --referencia data/vozes/teste_proprio/referencia/referencia_envio.wav \\
        --frases docs/etica/frases_teste_voz_propria.txt \\
        --saida data/vozes/teste_proprio/clone
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODELO = "tts_models/multilingual/multi-dataset/xtts_v2"


def ler_frases(caminho: str | Path) -> list[str]:
    return [linha.strip() for linha in Path(caminho).read_text(encoding="utf-8").splitlines()
            if linha.strip()]


def carregar(dispositivo: str):
    from TTS.api import TTS

    return TTS(MODELO).to(dispositivo)


def duracao_s(caminho: Path) -> float:
    import soundfile as sf

    info = sf.info(str(caminho))
    return info.frames / info.samplerate


def clonar(tts, frases: list[str], referencia: Path, saida: Path, idioma: str = "pt",
           semente: int = 0) -> list[dict]:
    """Gera uma fala por frase com a voz da referência; devolve o registro de cada uma."""
    import torch

    saida.mkdir(parents=True, exist_ok=True)
    registro = []
    for i, frase in enumerate(frases, start=1):
        destino = saida / f"f{i:02d}.wav"
        if destino.exists():
            print(f"[{i:02d}] já existe, pulando: {destino.name}")
            continue
        # O XTTS sorteia na geração: a semente por frase deixa cada uma reproduzível.
        torch.manual_seed(semente + i)
        inicio = time.perf_counter()
        tts.tts_to_file(text=frase, speaker_wav=str(referencia), language=idioma,
                        file_path=str(destino))
        gasto = time.perf_counter() - inicio
        dur = duracao_s(destino)
        print(f"[{i:02d}] {dur:.1f} s de fala em {gasto:.1f} s "
              f"({gasto / dur:.2f}x o tempo real): {frase[:50]}")
        registro.append({"arquivo": destino.name, "frase": frase, "semente": semente + i,
                         "duracao_s": round(dur, 2), "tempo_s": round(gasto, 2)})
    return registro


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Clona uma voz com o XTTS v2, no próprio computador")
    p.add_argument("--referencia", required=True, help="gravação da voz a clonar")
    p.add_argument("--frases", required=True, help="arquivo com uma frase por linha")
    p.add_argument("--saida", required=True)
    p.add_argument("--idioma", default="pt")
    p.add_argument("--semente", type=int, default=0)
    p.add_argument("--dispositivo", default=None, help="cuda | cpu (padrão: cuda, se houver)")
    args = p.parse_args(argv)

    referencia = Path(args.referencia)
    if not referencia.is_file():
        print(f"[ERRO] referência não encontrada: {referencia}")
        return 1
    import torch

    dispositivo = args.dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Modelo: {MODELO} | dispositivo: {dispositivo}")
    inicio = time.perf_counter()
    tts = carregar(dispositivo)
    print(f"Carregado em {time.perf_counter() - inicio:.1f} s")

    saida = Path(args.saida)
    registro = clonar(tts, ler_frases(args.frases), referencia, saida, args.idioma, args.semente)

    manifesto = saida / "clonagem.json"
    anterior = (json.loads(manifesto.read_text(encoding="utf-8"))
                if manifesto.is_file() else {"frases": []})
    anterior.update(modelo=MODELO, idioma=args.idioma, referencia=referencia.as_posix(),
                    dispositivo=dispositivo)
    anterior["frases"] += registro
    manifesto.write_text(json.dumps(anterior, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRegistro: {manifesto}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
