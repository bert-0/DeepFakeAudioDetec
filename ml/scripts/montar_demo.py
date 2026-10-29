"""Monta as playlists de demonstração do monitor: áudios reais e áudios falsos.

Gera, em `outputs/demo/`:
    reais.wav   + reais_cola.txt    — só bonafide
    falsos.wav  + falsos_cola.txt   — só spoof
    misto.wav   + misto_cola.txt    — alternados, para o contraste numa passada

A "cola" traz o início, o rótulo, o ataque e o score esperado de cada áudio.
Os áudios padrão são acertos do modelo: servem de ilustração, não de medida.

Uso:
    python scripts/montar_demo.py --config configs/baseline_v2.yaml
    python scripts/canal_real.py tocar --arquivo outputs/demo/falsos.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.data.dataset import parse_protocol_with_systems  # noqa: E402
from src.preprocess import load_audio  # noqa: E402

#: Silêncio entre áudios igual à janela do monitor (4 s), para uma janela não
#: misturar dois áudios.
GAP_S = 4.0
#: Tempo para o monitor mostrar a tabela antes do 1º áudio.
INICIO_S = 3.0

#: (id, score medido ao vivo pela captura; limiar recalibrado 0,9829, baseline_v2).
REAIS_PADRAO = [("LA_E_4943653", 0.049), ("LA_E_6693368", 0.298),
                ("LA_E_5826594", 0.355)]
FALSOS_PADRAO = [("LA_E_1297915", 1.000), ("LA_E_8004902", 1.000),
                 ("LA_E_1395043", 1.000)]


def montar(audios, sr: int, gap_s: float = GAP_S, inicio_s: float = INICIO_S):
    """Concatena com silêncio. Devolve (sinal, [(início em s, item), ...])."""
    partes = [np.zeros(int(inicio_s * sr), np.float32)]
    marcas, cursor = [], len(partes[0])
    gap = np.zeros(int(gap_s * sr), np.float32)
    for item, wav in audios:
        marcas.append((cursor / sr, item))
        partes += [np.asarray(wav, np.float32), gap]
        cursor += len(wav) + len(gap)
    return np.concatenate(partes), marcas


def escrever_cola(marcas, destino: Path, titulo: str, limiar: float | None) -> None:
    linhas = [f"{titulo}", ""]
    if limiar is not None:
        linhas.append(f"Limiar ao vivo: {limiar:.4f} — acima dele, o monitor indica síntese.")
        linhas.append("")
    linhas.append(f"{'início':>7s}  {'id':14s} {'rótulo':9s} {'ataque':6s} {'score esperado':>15s}")
    for t, (aid, rotulo, ataque, esperado) in marcas:
        esp = f"{esperado:.3f}" if esperado is not None else "—"
        linhas.append(f"{t:6.1f}s  {aid:14s} {rotulo:9s} {ataque:6s} {esp:>15s}")
    linhas += ["", "O monitor mostra uma linha a cada 2 s; a janela que começa no",
               "instante do áudio (ou logo antes) é a que o representa."]
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def _parse_ids(texto: str | None, padrao):
    if not texto:
        return padrao
    return [(i.strip(), None) for i in texto.split(",") if i.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Playlists de demonstração do monitor")
    p.add_argument("--config", required=True)
    p.add_argument("--reais", default=None, help="ids bonafide, separados por vírgula")
    p.add_argument("--falsos", default=None, help="ids spoof, separados por vírgula")
    p.add_argument("--saida", default="outputs/demo")
    p.add_argument("--limiar", type=float, default=0.9829,
                   help="limiar mostrado na cola (o recalibrado para captura)")
    args = p.parse_args(argv)

    config = load_config(args.config)
    sr = int(config["audio"]["sample_rate"])
    protocolo = {nome: (rot, sis) for nome, rot, sis in
                 parse_protocol_with_systems(config["data"]["protocols"]["eval"])}
    pasta_audio = Path(config["data"]["audio_dir"]["eval"])
    ext = config["data"].get("file_ext", ".flac")

    def carregar(lista, rotulo_esperado: int):
        itens = []
        for aid, esperado in lista:
            if aid not in protocolo:
                raise SystemExit(f"[ERRO] {aid} não está no protocolo de eval.")
            rot, sis = protocolo[aid]
            if rot != rotulo_esperado:
                nome = "bonafide" if rotulo_esperado == 0 else "spoof"
                raise SystemExit(f"[ERRO] {aid} não é {nome} no protocolo.")
            wav = load_audio(pasta_audio / f"{aid}{ext}", sr)
            itens.append(((aid, "bonafide" if rot == 0 else "spoof", sis, esperado), wav))
        return itens

    reais = carregar(_parse_ids(args.reais, REAIS_PADRAO), 0)
    falsos = carregar(_parse_ids(args.falsos, FALSOS_PADRAO), 1)
    misto = [x for par in zip(reais, falsos) for x in par]

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    for nome, itens, titulo in (("reais", reais, "DEMONSTRAÇÃO — ÁUDIOS REAIS"),
                                ("falsos", falsos, "DEMONSTRAÇÃO — ÁUDIOS FALSOS"),
                                ("misto", misto, "DEMONSTRAÇÃO — REAIS E FALSOS ALTERNADOS")):
        sinal, marcas = montar(itens, sr)
        sf.write(saida / f"{nome}.wav", sinal, sr)
        escrever_cola(marcas, saida / f"{nome}_cola.txt", titulo, args.limiar)
        print(f"{nome:7s}: {saida / f'{nome}.wav'} ({len(sinal) / sr:.0f} s, "
              f"{len(itens)} áudios) + {nome}_cola.txt")

    print(f"""
Ao vivo (dois terminais; aprimoramentos de áudio desligados):
  1) python monitor.py --config {args.config} --checkpoint <checkpoint>
  2) quando aparecer a tabela:
     python scripts/canal_real.py tocar --arquivo {(saida / 'falsos.wav').as_posix()}

Plano B, sem placa de som (mesmo resultado sempre):
  python monitor.py --config {args.config} --checkpoint <checkpoint> --arquivo {(saida / 'falsos.wav').as_posix()}
  (lendo o arquivo, o limiar escolhido é o original: o arquivo é de 16 kHz nativo.)""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
