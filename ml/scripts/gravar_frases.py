"""Grava frases pelo microfone: o teste com a voz própria e, depois, a coleta.

Modos:
- `voz`: mostra cada linha de --frases, grava pelo microfone em WASAPI
  exclusivo, na taxa do dispositivo e em 24 bits, e roda as checagens; Enter
  segue para a próxima, r regrava. Com o texto de referência e `--segundos 90`,
  grava a amostra para a ferramenta de clonagem.
- `pelo-ar`: toca cada arquivo de --clones pelo alto-falante padrão e grava pelo
  microfone ao mesmo tempo: o clone "pelo microfone" da demonstração.

Arquivos que já existem na saída são pulados (dá para parar e continuar).
A saída deve ficar em ml/data/vozes/, fora do Git: é voz de uma pessoa.

Uso (de dentro de ml/), sem fone com microfone conectado. O teste com a voz
própria aceita ruído a 15 dB da fala (`--piso-minimo 15`), a margem do microfone
embutido na sala (RESUMO_TCC §10.2.12):
    python scripts/gravar_frases.py voz --frases docs/etica/frases_teste_voz_propria.txt \\
        --piso-minimo 15 --saida data/vozes/teste_proprio/real
    python scripts/gravar_frases.py voz --frases docs/etica/texto_referencia_clonagem.txt \\
        --segundos 90 --piso-minimo 15 --saida data/vozes/teste_proprio/referencia
    python scripts/gravar_frases.py pelo-ar --clones data/vozes/teste_proprio/clone \\
        --piso-minimo 15 --saida data/vozes/teste_proprio/clone_ar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture.checagens import checar  # noqa: E402
from src.capture.sources import CaptureError, MicrofoneWasapiSource  # noqa: E402

#: O começo é descartado: pega o clique da tecla (RESUMO_TCC §10.2.10).
DESCARTE_S = 0.5
#: Silêncio antes e depois da fala, que a checagem de fala exige (0,5 s no mínimo).
MARGEM_S = 1.0
EXTENSOES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}


def ler_frases(caminho: str | Path) -> list[str]:
    return [linha.strip() for linha in Path(caminho).read_text(encoding="utf-8").splitlines()
            if linha.strip()]


def gravar(segundos: float, dispositivo: str | None = None,
           avisos: tuple[tuple[float, object], ...] = ()) -> tuple[np.ndarray, int, str, str]:
    """Grava `segundos` na taxa do dispositivo, sem conversão. Cada aviso é
    (instante em s, texto ou função): o texto é mostrado e a função chamada
    quando a gravação passa daquele instante."""
    fonte = MicrofoneWasapiSource(None, dispositivo)
    taxa = fonte.taxa_dispositivo
    descarte, total = int(DESCARTE_S * taxa), int(segundos * taxa)
    pendentes = sorted(avisos, key=lambda a: a[0])
    blocos, n = [], 0
    for bloco in fonte.blocos():
        blocos.append(bloco)
        n += len(bloco)
        decorrido = n / taxa - DESCARTE_S
        while pendentes and decorrido >= pendentes[0][0]:
            _, acao = pendentes.pop(0)
            if callable(acao):
                acao()
            else:
                print(f"    >> {acao}")
        if decorrido * taxa >= total:
            break
    fonte.fechar()
    if fonte.aviso:
        print(f"    [AVISO] {fonte.aviso}")
    if fonte.descontinuidades:
        print(f"    [AVISO] o driver avisou {fonte.descontinuidades} perda(s) de amostras: "
              "feche outros programas e regrave")
    nome = fonte.nome + (f", pacotes de {fonte.quadros_por_pacote} amostras"
                         if fonte.quadros_por_pacote else "")
    return np.concatenate(blocos)[descarte:descarte + total], taxa, fonte.modo, nome


def salvar(destino: Path, wav: np.ndarray, taxa: int) -> None:
    import soundfile as sf

    destino.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destino, wav, taxa, subtype="PCM_24")


def relatar_checagens(wav: np.ndarray, taxa: int, piso_minimo: float | None = None) -> bool:
    resultado = checar(wav, taxa, piso_minimo=piso_minimo)
    for c in resultado:
        print(f"    {'ok' if c.ok else 'REGRAVAR':8s} {c.nome:14s} {c.valor}"
              + (f" — {c.motivo}" if c.motivo else ""))
    return all(c.ok for c in resultado)


def _seguir(sem_pausa: bool) -> bool:
    """True para seguir; False para regravar."""
    if sem_pausa:
        return True
    return input("    Enter: próxima | r: regravar > ").strip().lower() != "r"


def modo_voz(frases: list[str], saida: Path, segundos: float, dispositivo: str | None,
             sem_pausa: bool, piso_minimo: float | None = None) -> int:
    for i, frase in enumerate(frases, start=1):
        destino = saida / f"f{i:02d}.wav"
        if destino.exists():
            print(f"[{i:02d}] já existe, pulando: {destino.name}")
            continue
        while True:
            print(f"\n[{i:02d}/{len(frases)}] {frase}")
            if not sem_pausa:
                input("    Enter para gravar; espere o aviso para começar a ler...")
            wav, taxa, modo, nome = gravar(segundos, dispositivo, (
                (MARGEM_S, "leia agora"), (segundos - MARGEM_S, "silêncio até o fim")))
            print(f"    {nome} (WASAPI {modo}, {taxa // 1000} kHz)")
            relatar_checagens(wav, taxa, piso_minimo)
            if _seguir(sem_pausa):
                salvar(destino, wav, taxa)
                break
    return 0


def modo_pelo_ar(clones: Path, saida: Path, dispositivo: str | None, sem_pausa: bool,
                 piso_minimo: float | None = None) -> int:
    import librosa
    import sounddevice as sd

    arquivos = (sorted(p for p in clones.iterdir() if p.suffix.lower() in EXTENSOES)
                if clones.is_dir() else [])
    if not arquivos:
        print(f"[ERRO] nenhum áudio em {clones}. Gere os clones antes, na ferramenta de "
              "clonagem, e salve um arquivo por frase: f01, f02, ... f10.")
        return 1
    for arq in arquivos:
        destino = saida / f"{arq.stem}.wav"
        if destino.exists():
            print(f"[{arq.stem}] já existe, pulando")
            continue
        clone, taxa_clone = librosa.load(str(arq), sr=None, mono=True)
        duracao = len(clone) / taxa_clone
        while True:
            print(f"\n[{arq.stem}] tocando {arq.name} ({duracao:.1f} s) pelo alto-falante")
            wav, taxa, modo, nome = gravar(duracao + 2 * MARGEM_S, dispositivo, (
                (MARGEM_S, lambda: sd.play(clone, taxa_clone)),))
            sd.stop()
            print(f"    {nome} (WASAPI {modo}, {taxa // 1000} kHz)")
            relatar_checagens(wav, taxa, piso_minimo)
            if _seguir(sem_pausa):
                salvar(destino, wav, taxa)
                break
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Grava frases pelo microfone (voz própria e coleta)")
    p.add_argument("modo", choices=("voz", "pelo-ar"))
    p.add_argument("--frases", help="arquivo com uma frase por linha (modo voz)")
    p.add_argument("--clones", help="pasta com os clones a tocar (modo pelo-ar)")
    p.add_argument("--saida", required=True)
    # 12 s: as frases do teste levaram ~7,5 s, começando ~1 s depois do aviso; em
    # 10 s a fala terminava colada no fim (24/10/2026).
    p.add_argument("--segundos", type=float, default=12.0,
                   help="duração de cada gravação no modo voz")
    p.add_argument("--piso-minimo", type=float, default=None, metavar="DB",
                   help="distância mínima entre fala e ruído (padrão 35 dB; o teste com "
                        "a voz própria usa 15, RESUMO_TCC §10.2.12)")
    p.add_argument("--dispositivo", default=None, help="parte do nome do microfone")
    p.add_argument("--sem-pausa", action="store_true", help="não espera Enter (testes)")
    args = p.parse_args(argv)

    saida = Path(args.saida)
    try:
        if args.modo == "voz":
            if not args.frases:
                p.error("o modo voz precisa de --frases")
            return modo_voz(ler_frases(args.frases), saida, args.segundos, args.dispositivo,
                            args.sem_pausa, args.piso_minimo)
        if not args.clones:
            p.error("o modo pelo-ar precisa de --clones")
        return modo_pelo_ar(Path(args.clones), saida, args.dispositivo, args.sem_pausa,
                            args.piso_minimo)
    except CaptureError as erro:
        print(f"[ERRO] {erro}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
