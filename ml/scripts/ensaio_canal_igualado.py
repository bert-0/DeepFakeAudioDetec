"""Ensaios com o canal igualado: os clones recebem o ruído do nosso microfone.

Nos ensaios, a voz real passou pelo microfone do notebook (ruído, estalos) e o
clone é um arquivo limpo. O aluno 2 aprendeu a ignorar o canal e deixou de
separar os dois (RESUMO_TCC §10.2.31). Aqui o canal dos clones é aproximado do
real: soma-se a eles o ruído das pausas das próprias gravações reais (sem fala,
estalos inclusos), na SNR pedida, e mede-se de novo o aluno 1, o aluno 2 e o
nosso detector contra as mesmas vozes reais. Não é o clone pelo ar (falta o
alto-falante e a sala), é só o ruído. Exploratório, sem critério de aprovação
(RESUMO_TCC §10.2.32). Os áudios com ruído ficam numa pasta temporária, apagada
no fim: voz é dado biométrico.

Uso (de dentro de ml/):
    python scripts/ensaio_canal_igualado.py --real L01_real=data/vozes/ensaio2/real \\
        --real L02_real=data/vozes/L02/real --clone L01_xtts=data/vozes/ensaio2/clone ... \\
        --aluno aluno1=checkpoints/aluno_xlsr_l7.pt --aluno aluno2=checkpoints/aluno2_xlsr_l7.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SR = 16000


def clones_com_canal(clones: dict[str, list[Path]], ruido: np.ndarray, snr_db: float, destino: Path,
                     semente: int = 0) -> dict[str, list[Path]]:
    """Cada clone, lido a 16 kHz pelo FIR, mais o ruído do microfone na SNR pedida."""
    import soundfile as sf

    from scripts import estudo_surpresa as es
    from src.preprocess.audio import load_audio

    saida = {}
    for nome, arquivos in clones.items():
        (destino / nome).mkdir(parents=True, exist_ok=True)
        saida[nome] = []
        for i, arq in enumerate(arquivos):
            wav = load_audio(arq, SR)
            antes, es.SNR_DB = es.SNR_DB, snr_db
            try:
                y = es.aplicar_canal(wav, "microfone_20db", np.random.default_rng([semente, i]), ruido)
            finally:
                es.SNR_DB = antes
            caminho = destino / nome / arq.name
            sf.write(caminho, y, SR)
            saida[nome].append(caminho)
    return saida


def montar_origem(reais: dict[str, list[Path]], clones: dict[str, list[Path]]) -> dict:
    arquivos, classes, rotulos = [], [], []
    for grupo, rotulo in ((reais, 0), (clones, 1)):
        for nome, lista in grupo.items():
            for a in lista:
                arquivos.append(str(a))
                classes.append(nome)
                rotulos.append(rotulo)
    return {"arquivo": np.array(arquivos), "classe": np.array(classes), "rotulo": np.array(rotulos, dtype=np.int8)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Ensaios com o ruído do microfone também nos clones")
    p.add_argument("--real", action="append", required=True, metavar="NOME=PASTA")
    p.add_argument("--clone", action="append", required=True, metavar="NOME=PASTA")
    p.add_argument("--aluno", action="append", required=True, metavar="NOME=CHECKPOINT")
    p.add_argument("--snr", type=float, action="append", help="SNR em dB (padrão: 20; pode repetir)")
    p.add_argument("--treino", default="outputs/destilacao/asv5_train")
    p.add_argument("--dev", default="outputs/destilacao/asv5_dev")
    p.add_argument("--sondas", default="outputs/ssl/sondas.json")
    p.add_argument("--detector", default="outputs/ssl/detector_eval.json", help="scores do dev do detector (limiar)")
    p.add_argument("--json", default="outputs/destilacao/ensaio_canal_igualado.json")
    args = p.parse_args(argv)

    from scripts.destilacao import avaliar, carregar_dados, preparar
    from scripts.estudo_ssl import comparar_detector, ler_npz, pontuar_detector
    from scripts.estudo_surpresa import banco_de_ruido

    def grupos(itens):
        saida = {}
        for item in itens:
            nome, pasta = item.split("=", 1)
            saida[nome] = sorted(Path(pasta).glob("*.wav"))
        return saida

    reais, clones = grupos(args.real), grupos(args.clone)
    ruido = banco_de_ruido([a for lista in reais.values() for a in lista])
    sondas = json.loads(Path(args.sondas).read_text(encoding="utf-8"))
    treino, dev = carregar_dados(Path(args.treino)), carregar_dados(Path(args.dev))
    det_dev = np.asarray(json.loads(Path(args.detector).read_text(encoding="utf-8"))["scores"]["dev"])
    dev_npz = ler_npz(Path("outputs/ssl/asv5_dev.npz"))
    resultado = {}
    pasta = Path(tempfile.mkdtemp(prefix="canal_igualado_"))
    try:
        for snr in args.snr or [20.0]:
            com_canal = clones_com_canal(clones, ruido, snr, pasta / f"snr{snr:g}")
            origem = montar_origem(reais, com_canal)
            nome_conj = f"ensaios_snr{snr:g}"
            preparar(origem, pasta / nome_conj, None, None)
            dados = carregar_dados(pasta / nome_conj)
            r = {}
            for item in args.aluno:
                nome, ck = item.split("=", 1)
                av = avaliar(Path(ck), sondas, treino, dev, {nome_conj: dados})
                r[nome] = av["sondas"]["nova"]["avaliar"][nome_conj]
            s = pontuar_detector([str(a) for a in origem["arquivo"]], "configs/fusion_v4_asv5.yaml",
                                 "checkpoints/fusion_lcnn_v4_asv5.pt")
            cmp = comparar_detector({"dev": det_dev, "e": s}, {"dev": dev_npz, "e": origem})
            r["detector"] = cmp["conjuntos"]["e"]
            resultado[f"{snr:g} dB"] = r
            print(f"\nClones com o ruído do microfone a {snr:g} dB: AUC (certos pelo limiar do dev)")
            for nome, m in r.items():
                print(f"  {nome:9s} AUC {m['auc']:.3f}; " + ", ".join(
                    f"{c} {v['certos']}/{v['n']}" for c, v in m["classes"].items()))
    finally:
        shutil.rmtree(pasta, ignore_errors=True)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({"ruido_s": len(ruido) / SR, "resultado": resultado},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nResultado: {args.json} (áudios com ruído apagados)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
