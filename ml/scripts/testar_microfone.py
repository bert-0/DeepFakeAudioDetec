"""Teste do microfone da coleta: o caminho processa o áudio?

Grava a mesma cena duas vezes, em WASAPI exclusivo e compartilhado, com um ruído
constante ao fundo (ventilador, ou ruído branco tocando num celular parado):
silêncio, uma frase, silêncio. Depois compara:

- o ruído constante cai durante a gravação: supressão de ruído no caminho. No
  exclusivo ela é do driver ou do hardware, e o Windows não a desliga;
- o compartilhado muda o nível (ganho fixo ou automático), a distância entre
  fala e ruído, o formato do espectro do ruído ou perde 7,6–8 kHz: os
  aprimoramentos ou a conversão de taxa do Windows estão no caminho.

Uso:
    python scripts/testar_microfone.py [--dispositivo "<nome>"] [--segundos 12]
    python scripts/testar_microfone.py --analisar exclusivo.wav compartilhado.wav

As gravações vão para outputs/teste_microfone/, fora do Git, em 24 bits. É teste
de equipamento: nada disso entra no conjunto de dados.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture.checagens import checar, energia_por_faixa  # noqa: E402
from src.capture.sources import CaptureError, MicrofoneWasapiSource  # noqa: E402

#: Abaixo disso é a variação natural entre duas leituras da mesma frase.
LIMIAR_DB = 6.0
#: Desvio do espectro do ruído em relação ao ganho geral; abaixo disso, a
#: diferença entre dois trechos de ruído do mesmo ventilador.
LIMIAR_ESPECTRO_DB = 10.0
#: Ruído constante tão abaixo da fala é o ruído do próprio microfone: ou o
#: ruído de fundo não estava tocando, ou o caminho o remove.
DISTANCIA_SEM_RUIDO_DB = 45.0
SILENCIO_S = 3.0
#: O começo da gravação é descartado: pega o clique do Enter (medido: 0,3 a
#: 0,4 s no microfone do notebook).
DESCARTE_S = 0.5
#: 16 bits perdem resolução no nível baixo do modo exclusivo (pico de −40 dBFS
#: no microfone do notebook).
FORMATO = "PCM_24"
BANDAS_HZ = ((50, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000))
FRASE = ("O detector compara a voz gravada com a voz sintética, frase por frase, "
         "em janelas de quatro segundos.")


def _db(x) -> np.ndarray:
    return 20 * np.log10(np.maximum(x, 1e-12))


def _espectro(wav: np.ndarray, sr: int, de: float, ate: float) -> np.ndarray:
    from scipy.signal import welch

    trecho = wav[int(max(de, 0) * sr):int(ate * sr)]
    f, p = welch(trecho, sr, nperseg=min(4096, max(len(trecho), 8)))
    return np.array([10 * np.log10(p[(f >= a) & (f < b)].mean() + 1e-30) for a, b in BANDAS_HZ])


def medir(wav: np.ndarray, sr: int) -> dict:
    """Níveis da gravação em dBFS, a linha do tempo e a energia por faixa."""
    import librosa

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    hop = 512
    rms = _db(librosa.feature.rms(y=wav, frame_length=2048, hop_length=hop)[0])
    t = np.arange(len(rms)) * hop / sr
    dur = len(wav) / sr

    def mediana(de, ate):
        sel = rms[(t >= de) & (t < ate)]
        return float(np.median(sel)) if sel.size else float("nan")

    return {
        "pico": float(_db(np.abs(wav).max())) if wav.size else float("-inf"),
        "fala": float(np.percentile(rms, 95)),
        # logo no começo, antes de uma supressão de ruído convergir
        "ruido_inicio": mediana(0.25, 0.75),
        "ruido_fim": mediana(dur - 2.0, dur - 0.5),
        # antes do aviso de leitura, que vem aos 3 s
        "espectro_ruido_inicio": _espectro(wav, sr, 1.0, 2.75),
        "espectro_ruido_fim": _espectro(wav, sr, dur - 2.0, dur - 0.5),
        "linha_do_tempo": [mediana(a, a + 0.5) for a in np.arange(0, dur - 0.25, 0.5)],
        "faixas": energia_por_faixa(wav, sr),
        "checagens": checar(wav, sr),
    }


def comparar(exclusivo: dict | None, compartilhado: dict | None) -> list[str]:
    """Indícios de processamento no caminho, um por linha."""
    saida, adaptou = [], []
    for modo, m in (("exclusivo", exclusivo), ("compartilhado", compartilhado)):
        if m is None:
            continue
        queda = m["ruido_fim"] - m["ruido_inicio"]
        if queda <= -LIMIAR_DB:
            onde = "driver ou hardware" if modo == "exclusivo" else "Windows, driver ou hardware"
            saida.append(f"{modo}: o ruído constante caiu {-queda:.0f} dB durante a gravação; "
                         f"supressão de ruído no caminho ({onde})")
        # Uma faixa do ruído que some em relação às outras: supressão adaptativa.
        desvio = m["espectro_ruido_fim"] - m["espectro_ruido_inicio"]
        desvio = desvio - np.median(desvio)
        if desvio.min() <= -LIMIAR_ESPECTRO_DB:
            a, b = BANDAS_HZ[int(np.argmin(desvio))]
            adaptou.append((modo, desvio.min(), a, b))
    if len(adaptou) == 2:
        saida.append("nos dois modos uma faixa do ruído some do começo para o fim: supressão "
                     "no hardware, ou o ruído de fundo não foi constante")
    for modo, d, a, b in adaptou if len(adaptou) == 1 else []:
        onde = ("driver ou hardware, ou o ruído de fundo variou" if modo == "exclusivo"
                else "supressão adaptativa do Windows")
        saida.append(f"{modo}: o ruído perde {-d:.0f} dB em {a}–{b} Hz do começo para o fim ({onde})")
    if exclusivo is None or compartilhado is None:
        return saida

    dist_e = exclusivo["fala"] - exclusivo["ruido_fim"]
    dist_c = compartilhado["fala"] - compartilhado["ruido_fim"]
    if min(dist_e, dist_c) >= DISTANCIA_SEM_RUIDO_DB:
        saida.append(f"o ruído constante ficou a mais de {DISTANCIA_SEM_RUIDO_DB:.0f} dB da fala "
                     "nos dois modos: confira se ele estava tocando; se estava, o hardware o remove")
    if dist_c - dist_e >= LIMIAR_DB:
        saida.append(f"o compartilhado afasta o ruído da fala em mais {dist_c - dist_e:.0f} dB: "
                     "supressão de ruído no Windows")

    dif_fala = compartilhado["fala"] - exclusivo["fala"]
    dif_ruido = compartilhado["ruido_inicio"] - exclusivo["ruido_inicio"]
    if abs(dif_fala) >= LIMIAR_DB:
        if abs(dif_fala - dif_ruido) < LIMIAR_DB:
            saida.append(f"o compartilhado aplica um ganho fixo de {dif_fala:+.0f} dB na fala e no "
                         "ruído: volume ou reforço do microfone no Windows")
        else:
            saida.append(f"no compartilhado a fala muda {dif_fala:+.0f} dB e o ruído {dif_ruido:+.0f} dB: "
                         "ganho automático (AGC) no Windows")

    # Ganho muda todas as faixas por igual; supressão e equalização, não.
    desvio = compartilhado["espectro_ruido_fim"] - exclusivo["espectro_ruido_fim"]
    desvio = desvio - np.median(desvio)
    pior = int(np.argmax(np.abs(desvio)))
    if abs(desvio[pior]) >= LIMIAR_ESPECTRO_DB:
        a, b = BANDAS_HZ[pior]
        saida.append(f"o compartilhado muda o formato do espectro do ruído, além do ganho "
                     f"({desvio[pior]:+.0f} dB em {a}–{b} Hz): supressão de ruído ou equalização")

    alta_e = exclusivo["faixas"].get("7,6–8 kHz", 0.0)
    alta_c = compartilhado["faixas"].get("7,6–8 kHz", 0.0)
    if alta_e > 1e-5 and alta_c < alta_e / 4:
        saida.append("o compartilhado perde a faixa de 7,6–8 kHz: o Windows converte a taxa "
                     "(formato do dispositivo diferente de 48 kHz)")
    if not saida:
        saida.append(f"nenhuma diferença de {LIMIAR_DB:.0f} dB ou mais: os dois modos entregam o "
                     "mesmo áudio e o ruído constante não foi suprimido")
    return saida


def relatar(medidas: dict[str, dict], indicios: list[str]) -> None:
    modos = list(medidas)
    print("\n" + f"{'':22s}" + "".join(f"{m:>16s}" for m in modos))
    for chave, nome in (("pico", "pico"), ("fala", "fala (p95)"),
                        ("ruido_inicio", "ruído no começo"), ("ruido_fim", "ruído no fim")):
        print(f"{nome:22s}" + "".join(f"{medidas[m][chave]:11.1f} dBFS" for m in modos))
    for faixa in medidas[modos[0]]["faixas"]:
        print(f"{'energia ' + faixa:22s}"
              + "".join(f"{100 * medidas[m]['faixas'].get(faixa, 0.0):15.2f}%" for m in modos))
    for i, c in enumerate(medidas[modos[0]]["checagens"]):
        estados = ["ok" if medidas[m]["checagens"][i].ok else "regravar" for m in modos]
        print(f"{'checagem ' + c.nome:22s}" + "".join(f"{e:>16s}" for e in estados))

    print("\nNível por meio segundo (dBFS); a frase deve aparecer entre 3 e 9 s:")
    for m in modos:
        print(f"  {m:14s} " + " ".join(f"{x:4.0f}" for x in medidas[m]["linha_do_tempo"]))
    falhas = [(m, c) for m in modos for c in medidas[m]["checagens"] if not c.ok]
    if falhas:
        print("\nChecagens que falharam (com ruído de fundo proposital, fala e piso "
              "reprovam por construção; valem na gravação sem ruído):")
        for m, c in falhas:
            print(f"  {m} — {c.nome} ({c.valor}): {c.motivo}")
    print("\nIndícios:")
    for linha in indicios:
        print(f"  - {linha}")


def gravar(segundos: float, exclusivo: bool,
           dispositivo: str | None) -> tuple[np.ndarray, str, str, int]:
    """Grava `segundos` na taxa do dispositivo, sem conversão, e devolve (áudio,
    modo, nome, taxa). Os primeiros `DESCARTE_S` segundos não entram."""
    fonte = MicrofoneWasapiSource(None, dispositivo, exclusivo=exclusivo)
    taxa = fonte.taxa_dispositivo
    descarte, total = int(DESCARTE_S * taxa), int(segundos * taxa)
    blocos, avisos = [], [SILENCIO_S, segundos - SILENCIO_S]
    for bloco in fonte.blocos():
        blocos.append(bloco)
        decorrido = sum(len(b) for b in blocos) / taxa - DESCARTE_S
        if avisos and decorrido >= avisos[0]:
            print("    >> leia a frase" if len(avisos) == 2 else "    >> silêncio até o fim")
            avisos.pop(0)
        if decorrido * taxa >= total:
            break
    fonte.fechar()
    return np.concatenate(blocos)[descarte:descarte + total], fonte.modo, fonte.nome, taxa


def salvar(destino: str | Path, wav: np.ndarray, taxa: int) -> None:
    import soundfile as sf

    sf.write(destino, wav, taxa, subtype=FORMATO)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Teste do microfone: exclusivo x compartilhado")
    p.add_argument("--dispositivo", default=None, help="parte do nome do microfone (padrão do Windows)")
    p.add_argument("--segundos", type=float, default=12.0)
    p.add_argument("--pasta", default="outputs/teste_microfone")
    p.add_argument("--analisar", nargs=2, metavar=("EXCLUSIVO", "COMPARTILHADO"), default=None,
                   help="só analisa duas gravações já feitas")
    args = p.parse_args(argv)

    if args.analisar:
        import soundfile as sf

        medidas = {}
        for modo, caminho in zip(("exclusivo", "compartilhado"), args.analisar, strict=True):
            wav, sr = sf.read(caminho, dtype="float32", always_2d=True)
            medidas[modo] = medir(wav.mean(axis=1), sr)
        relatar(medidas, comparar(medidas["exclusivo"], medidas["compartilhado"]))
        return 0

    pasta = Path(args.pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    print("Deixe um ruído constante tocando ao fundo (ventilador ou ruído branco no "
          "celular), no mesmo lugar e volume nas duas gravações.")
    print(f"Frase: \"{FRASE}\"")
    medidas, carimbo = {}, time.strftime("%Y%m%d_%H%M%S")
    for pedido in ("exclusivo", "compartilhado"):
        input(f"\n[{pedido}] Enter para gravar {args.segundos:.0f} s; espere o aviso "
              "para ler a frase...")
        try:
            wav, modo, nome, taxa = gravar(args.segundos, pedido == "exclusivo", args.dispositivo)
        except CaptureError as erro:
            print(f"[ERRO] {erro}")
            return 1
        destino = pasta / f"{carimbo}_{pedido}.wav"
        salvar(destino, wav, taxa)
        print(f"    {nome} (WASAPI {modo}, {taxa // 1000} kHz) -> {destino}")
        if modo != pedido:
            print(f"[AVISO] pedido {pedido}, o driver abriu {modo}: a comparação entre os "
                  "modos não vale")
        medidas[pedido] = medir(wav, taxa)
    relatar(medidas, comparar(medidas["exclusivo"], medidas["compartilhado"]))
    print("\nOuça também os dois arquivos: se o ruído some ou \"respira\" junto com a fala, "
          "há supressão de ruído.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
