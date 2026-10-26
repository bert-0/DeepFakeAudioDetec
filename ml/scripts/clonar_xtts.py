"""Clona uma voz no próprio computador: XTTS v2 (Coqui) ou Chatterbox Multilingual.

A voz não sai da máquina. Cada ferramenta roda num ambiente conda separado
(`xtts` ou `chatterbox`): elas fixam versões do PyTorch e de outras bibliotecas
que podem quebrar o ambiente do detector. Por isso este script não importa nada
do projeto. `--ferramenta chatterbox` usa a segunda ferramenta do plano
(Resemble AI, MIT, 2025), que marca todo áudio com uma marca d'água inaudível.

Três jeitos de dizer o que o clone fala:
- `--frases arquivo.txt`: cada linha vira `fNN.wav` (os ensaios);
- `--texto "..."` (repita a opção): cada frase vira `frase_NN.wav`;
- `--interativo`: carrega o modelo uma vez e pede frases até `sair`;
  `r` refaz a última com outra semente (o XTTS às vezes corta o fim). É o modo
  da demonstração: a banca escolhe a frase e o clone sai em ~10 s.
A frase vai sem o ponto final, que o XTTS às vezes lê como "ponto"
(RESUMO_TCC §10.2.14); `--com-ponto-final` repete os ensaios 1 e 2.
Arquivos que já existem não são sobrescritos. Em --saida fica `clonagem.json`:
modelo, idioma, semente, referência e o tempo de cada frase (RESUMO_TCC §10.2.12).

Uso (de dentro de ml/, no ambiente do XTTS):
    python scripts/clonar_xtts.py \\
        --referencia data/vozes/teste_proprio/referencia/referencia_envio.wav \\
        --frases docs/etica/frases_teste_voz_propria.txt \\
        --saida data/vozes/teste_proprio/clone
    python scripts/clonar_xtts.py --referencia <gravação> --saida <pasta> --interativo
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODELO = "tts_models/multilingual/multi-dataset/xtts_v2"
MODELOS = {"xtts": MODELO, "chatterbox": "ResembleAI/chatterbox (multilingual, t3 v3)"}
#: Uma janela do detector é de 4 s; frase mais curta é completada por repetição.
PALAVRAS_MINIMAS = 8


def ler_frases(caminho: str | Path) -> list[str]:
    return [linha.strip() for linha in Path(caminho).read_text(encoding="utf-8").splitlines()
            if linha.strip()]


class _Chatterbox:
    """O Chatterbox com a mesma interface do XTTS (`tts_to_file`)."""

    def __init__(self, dispositivo: str):
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self.modelo = ChatterboxMultilingualTTS.from_pretrained(device=dispositivo,
                                                                t3_model="v3")

    def tts_to_file(self, text, speaker_wav, language, file_path):
        import soundfile as sf

        wav = self.modelo.generate(text, language_id=language, audio_prompt_path=speaker_wav)
        sf.write(file_path, wav.squeeze().detach().cpu().numpy(), self.modelo.sr)


def carregar(dispositivo: str, ferramenta: str = "xtts"):
    try:
        if ferramenta == "chatterbox":
            return _Chatterbox(dispositivo)
        from TTS.api import TTS
    except ImportError:
        raise SystemExit(f"[ERRO] o {ferramenta} não está instalado neste ambiente: rode "
                         f"`conda activate {ferramenta}` e o comando de novo.") from None
    return TTS(MODELO).to(dispositivo)


def duracao_s(caminho: Path) -> float:
    import soundfile as sf

    info = sf.info(str(caminho))
    return info.frames / info.samplerate


def sem_ponto(frase: str) -> str:
    """Tira o ponto final: o XTTS às vezes o lê como a palavra "ponto" (clones do
    L02, 25/10/2026). Interrogação e exclamação ficam, pela entonação."""
    texto = frase.rstrip()
    while texto.endswith(".") and not texto.endswith("..."):
        texto = texto[:-1].rstrip()
    return texto


def gerar(tts, frase: str, referencia: Path, destino: Path, idioma: str, semente: int,
          sem_ponto_final: bool = True) -> dict:
    """Uma fala com a voz da referência; devolve o registro dela."""
    import torch

    texto = sem_ponto(frase) if sem_ponto_final else frase

    destino.parent.mkdir(parents=True, exist_ok=True)
    # O XTTS sorteia na geração: a semente deixa cada frase reproduzível.
    torch.manual_seed(semente)
    inicio = time.perf_counter()
    tts.tts_to_file(text=texto, speaker_wav=str(referencia), language=idioma,
                    file_path=str(destino))
    gasto = time.perf_counter() - inicio
    dur = duracao_s(destino)
    print(f"[{destino.stem}] {dur:.1f} s de fala em {gasto:.1f} s "
          f"({gasto / dur:.2f}x o tempo real), semente {semente}: {frase[:50]}")
    return {"arquivo": destino.name, "frase": frase, "texto_enviado": texto, "semente": semente,
            "duracao_s": round(dur, 2), "tempo_s": round(gasto, 2)}


def clonar(tts, frases: list[str], referencia: Path, saida: Path, idioma: str = "pt",
           semente: int = 0, sem_ponto_final: bool = True) -> list[dict]:
    """Uma fala por frase do arquivo (fNN.wav); pula as que já existem."""
    registro = []
    for i, frase in enumerate(frases, start=1):
        destino = saida / f"f{i:02d}.wav"
        if destino.exists():
            print(f"[{i:02d}] já existe, pulando: {destino.name}")
            continue
        registro.append(gerar(tts, frase, referencia, destino, idioma, semente + i,
                              sem_ponto_final))
    return registro


def proximo_livre(saida: Path) -> int:
    usados = [int(p.stem.split("_")[1]) for p in saida.glob("frase_*.wav")
              if p.stem.split("_")[1].isdigit()]
    return max(usados, default=0) + 1


def falar(tts, frase: str, referencia: Path, saida: Path, idioma: str, semente: int,
          sem_ponto_final: bool = True) -> dict:
    """Frase livre: vira frase_NN.wav, sem sobrescrever nenhuma anterior."""
    if len(frase.split()) < PALAVRAS_MINIMAS:
        print(f"    [AVISO] menos de {PALAVRAS_MINIMAS} palavras: a fala não deve encher "
              "uma janela de 4 s do detector")
    n = proximo_livre(saida)
    return gerar(tts, frase, referencia, saida / f"frase_{n:02d}.wav", idioma, semente + n,
                 sem_ponto_final)


def interativo(tts, referencia: Path, saida: Path, idioma: str, semente: int,
               ler=None, sem_ponto_final: bool = True) -> list[dict]:
    """Pede frases até `sair`; `r` refaz a última com outra semente.

    Linha vazia não encerra: um Enter sem querer na frente da banca custaria
    recarregar o modelo (~48 s)."""
    ler = ler or input
    print("\nDigite a frase que o clone vai dizer e tecle Enter. "
          "r: refazer a última com outra semente | sair: encerrar")
    registro, ultima, extra = [], None, 0
    while True:
        try:
            texto = ler("\nFrase > ").strip()
        except EOFError:
            return registro
        if not texto:
            continue
        if texto.lower() == "sair":
            return registro
        if texto.lower() == "r":
            if ultima is None:
                print("    digite uma frase primeiro; r refaz a última gerada")
                continue
            extra += 1000
            texto = ultima
        ultima = texto
        registro.append(falar(tts, texto, referencia, saida, idioma, semente + extra,
                              sem_ponto_final))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Clona uma voz no próprio computador")
    p.add_argument("--referencia", required=True, help="gravação da voz a clonar")
    modo = p.add_mutually_exclusive_group(required=True)
    modo.add_argument("--frases", help="arquivo com uma frase por linha (vira fNN.wav)")
    modo.add_argument("--texto", action="append", help="uma frase livre (repita a opção)")
    modo.add_argument("--interativo", action="store_true",
                      help="pede frases uma a uma, com o modelo já carregado")
    p.add_argument("--saida", required=True)
    p.add_argument("--com-ponto-final", action="store_true",
                   help="manda a frase com o ponto final, como nos ensaios 1 e 2 (o padrão "
                        "tira o ponto: o XTTS às vezes o lê como a palavra \"ponto\")")
    p.add_argument("--idioma", default="pt")
    p.add_argument("--semente", type=int, default=0)
    p.add_argument("--dispositivo", default=None, help="cuda | cpu (padrão: cuda, se houver)")
    p.add_argument("--ferramenta", choices=tuple(MODELOS), default="xtts")
    args = p.parse_args(argv)

    referencia = Path(args.referencia)
    if not referencia.is_file():
        print(f"[ERRO] referência não encontrada: {referencia}")
        return 1
    import torch

    dispositivo = args.dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Modelo: {MODELOS[args.ferramenta]} | dispositivo: {dispositivo}")
    inicio = time.perf_counter()
    tts = carregar(dispositivo, args.ferramenta)
    print(f"Carregado em {time.perf_counter() - inicio:.1f} s")

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    if args.frases:
        registro = clonar(tts, ler_frases(args.frases), referencia, saida, args.idioma,
                          args.semente, not args.com_ponto_final)
    elif args.texto:
        registro = [falar(tts, t, referencia, saida, args.idioma, args.semente,
                          not args.com_ponto_final)
                    for t in args.texto]
    else:
        registro = interativo(tts, referencia, saida, args.idioma, args.semente,
                              sem_ponto_final=not args.com_ponto_final)

    manifesto = saida / "clonagem.json"
    anterior = (json.loads(manifesto.read_text(encoding="utf-8"))
                if manifesto.is_file() else {"frases": []})
    anterior.update(ferramenta=args.ferramenta, modelo=MODELOS[args.ferramenta],
                    idioma=args.idioma, referencia=referencia.as_posix(),
                    dispositivo=dispositivo, sem_ponto_final=not args.com_ponto_final)
    anterior["frases"] += registro
    manifesto.write_text(json.dumps(anterior, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRegistro: {manifesto}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
