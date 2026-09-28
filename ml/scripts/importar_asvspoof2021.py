"""Converte o metadado do ASVspoof 2021 no protocolo — e no config — deste projeto.

O ASVspoof 2021 LA transmite os **mesmos ataques A07–A19** do eval de 2019 por
redes reais (VoIP e PSTN) com codecs reais, e traz uma condição de referência
sem codec e sem transmissão. Isso dá o experimento pareado que a simulação da
Seção 5 só consegue aproximar: mesmo ataque, canal real.

Onde ficam os arquivos (a pasta inteira está no .gitignore):

    ml/data/ASVspoof2021_LA/
    ├── <o que sai do Zenodo 4837263>/flac/*.flac      ← áudio, SEM rótulo
    └── keys/LA/CM/trial_metadata.txt                  ← rótulos: LA-keys-full.tar.gz
                                                          de www.asvspoof.org/asvspoof2021/

As chaves NÃO estão no repositório asvspoof-challenge/2021 do GitHub — lá só
há a descrição do formato. E não confunda com o `.trl.txt` que vem junto do
áudio no Zenodo: ele lista os arquivos sem rótulo, e este script o recusa.

Uso:
    # 1. ver quais condições existem no metadado
    python scripts/importar_asvspoof2021.py --metadata <keys>/LA/CM/trial_metadata.txt --listar

    # 2. gerar protocolo + config de uma condição, subamostrada
    python scripts/importar_asvspoof2021.py --metadata <...>/trial_metadata.txt \\
        --codec opus --amostra 10000 \\
        --config-base configs/fusion_v4.yaml \\
        --audio-dir data/ASVspoof2021_LA_eval/flac

    # 3. o passo 2 imprime o comando do evaluate.py, já com o config gerado

**Por que o config é gerado, e não copiado à mão.** Os artefatos do `evaluate.py`
levam o nome `experiment.name` + partição. Copiar `fusion_v4.yaml` e trocar só
os caminhos do `eval` faria a avaliação do 2021 gravar **por cima** dos
resultados do eval de 2019 — métricas, scores reaproveitados e o cache de
features. O config gerado muda o nome e desliga o cache (ver
`src/config.config_derivado`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config_derivado, load_config, salvar_config  # noqa: E402
from src.data.asvspoof2021 import (  # noqa: E402
    MetadadoInvalido,
    condicoes,
    escrever_protocolo,
    fases,
    filtrar,
    ler_metadata,
    subamostrar,
)

SAIDA = Path("outputs/asvspoof2021")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Importa o ASVspoof 2021")
    p.add_argument("--metadata", required=True,
                   help="trial_metadata.txt do eval-package (LA/CM/)")
    p.add_argument("--listar", action="store_true",
                   help="só lista as condições encontradas e sai")
    p.add_argument("--condicao", default=None, help="condição exata `codec/canal`")
    p.add_argument("--codec", default=None, help="filtra só pelo codec")
    p.add_argument("--fase", default=None,
                   help="filtra pela fase do desafio; use `eval`, o conjunto oficial")
    p.add_argument("--amostra", type=int, default=0,
                   help="subamostra estratificada de N trials (0 = todos)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config-base", default=None,
                   help="config do modelo treinado; gera um config derivado seguro")
    p.add_argument("--audio-dir", default=None,
                   help="pasta de áudio do eval do 2021 (exigida com --config-base)")
    p.add_argument("--audio-ext", default=".flac",
                   help="extensão dos áudios: .flac (original) ou .wav (convertidos)")
    p.add_argument("--saida", default=None,
                   help="protocolo de saída (padrão: outputs/asvspoof2021/<rótulo>.txt)")
    return p.parse_args(argv)


def audios_faltando(selecao, pasta: str | Path, ext: str = ".flac") -> list[str]:
    """IDs da seleção cujo arquivo não existe na pasta de áudio.

    Sem esta checagem, um caminho errado ou uma extração incompleta só aparece
    no `evaluate.py`, como `FileNotFoundError` dentro de um worker do DataLoader,
    enterrado num traceback de cem linhas. Conferir 10 mil caminhos custa menos
    de um segundo.
    """
    pasta = Path(pasta)
    return [t.arquivo for t in selecao if not (pasta / f"{t.arquivo}{ext}").is_file()]


def falhas_de_leitura(selecao, pasta: str | Path, n: int = 50,
                      ext: str = ".flac") -> tuple[int, list[tuple[str, str]]]:
    """Decodifica `n` áudios da amostra INTEIROS com o libsndfile.

    Quando ele falha, o librosa cai em silêncio no `audioread`, que no Windows
    abre um processo do FFmpeg POR ARQUIVO: a avaliação continua funcionando,
    mas fica muito lenta e pesada. O aviso do librosa ("PySoundFile failed")
    não diz o motivo, e aparece uma vez por processo — com 4 workers, não dá
    para saber se falharam 4 arquivos ou todos.

    **Decodificar, não só abrir.** A primeira versão usava `sf.info`, que lê só
    o cabeçalho. Rodado nos arquivos reais do 2021, o cabeçalho abriu sem erro
    (FLAC, 16 kHz, 16 bits) — e a falha do FLAC documentada no python-soundfile
    ("unknown error in flac decoder") é justamente cabeçalho bom com decodificação
    quebrada. O mesmo vale para arquivo truncado: o cabeçalho sobrevive.
    """
    import soundfile as sf

    pasta = Path(pasta)
    testados = selecao[:n]
    falhas = []
    for t in testados:
        try:
            sf.read(str(pasta / f"{t.arquivo}{ext}"), dtype="float32")
        except Exception as erro:  # noqa: BLE001 — qualquer falha interessa aqui
            falhas.append((t.arquivo, str(erro) or type(erro).__name__))
    return len(testados), falhas


def rotulo(args) -> str:
    base = args.condicao or args.codec or "todas"
    rot = base.replace("/", "-").replace("-" + "-", "-").rstrip("-")
    if args.fase:
        rot = f"{rot}_{args.fase}"
    return f"{rot}_n{args.amostra}" if args.amostra else rot


def main(argv=None) -> int:
    args = parse_args(argv)
    relatorio: dict = {}
    try:
        trials = ler_metadata(args.metadata, relatorio)
    except (OSError, MetadadoInvalido) as erro:
        print(f"[ERRO] {erro}")
        return 1

    ataques = sorted({t.ataque for t in trials if t.ataque != "-"})
    n_bona = sum(1 for t in trials if t.chave == "bonafide")
    print(f"{len(trials):,} trials ({n_bona:,} bonafide, {len(trials) - n_bona:,} spoof)"
          f" | {len(ataques)} ataques: {', '.join(ataques)}")
    if relatorio.get("ignoradas"):
        print(f"[AVISO] {relatorio['ignoradas']:,} linhas ignoradas. Exemplos:")
        for linha in relatorio["exemplos"]:
            print(f"   {linha}")
    if n_bona == 0 or n_bona == len(trials):
        print("[ERRO] o metadado inteiro tem uma classe só — quase certamente as "
              "linhas da outra\n       classe estão sendo descartadas. Não siga "
              "adiante; mande as primeiras linhas\n       do arquivo.")
        return 1

    if args.listar:
        print(f"\n{'condição (codec/canal)':32s} {'bonafide':>10s} {'spoof':>10s} "
              f"{'total':>10s}")
        print("-" * 66)
        for nome, bona, spoof in condicoes(trials):
            print(f"{nome:32s} {bona:10,} {spoof:10,} {bona + spoof:10,}")
        print(f"\n{'fase do desafio':32s} {'bonafide':>10s} {'spoof':>10s}")
        for fase, (bona, spoof) in fases(trials).items():
            print(f"{fase:32s} {bona:10,} {spoof:10,}")
        print("\nA condição sem codec e sem transmissão reproduz o cenário do eval"
              "\nde 2019 — é o controle pareado. Gere ela e a do Opus com a mesma"
              "\n--seed e o mesmo --amostra.")
        return 0

    if args.config_base and not args.audio_dir:
        print("[ERRO] --config-base exige --audio-dir (a pasta de .flac do 2021).")
        return 1

    selecao = filtrar(trials, condicao=args.condicao, codec=args.codec,
                      fase=args.fase)
    if not args.fase:
        print("[AVISO] sem --fase: a seleção mistura eval, progress e hidden. "
              "Para o conjunto\n        oficial, use --fase eval.")
    if not selecao:
        print("[ERRO] nenhum trial casa com o filtro pedido. Use --listar.")
        return 1
    selecao = subamostrar(selecao, args.amostra, seed=args.seed)

    bona = sum(1 for t in selecao if t.chave == "bonafide")
    if bona == 0 or bona == len(selecao):
        print("[ERRO] a seleção tem uma classe só; não dá para calcular EER.")
        return 1

    tag = rotulo(args)
    protocolo = Path(args.saida) if args.saida else SAIDA / f"{tag}.txt"
    n = escrever_protocolo(selecao, protocolo)
    print(f"\n{n:,} trials em {protocolo} ({bona:,} bonafide, {n - bona:,} spoof, "
          f"{len({t.ataque for t in selecao if t.ataque != '-'})} ataques)")

    if not args.config_base:
        print("\nSem --config-base: nenhum config gerado. NÃO copie o config do"
              "\nmodelo trocando só os caminhos — isso sobrescreve os resultados de"
              "\n2019. Rode de novo com --config-base e --audio-dir.")
        return 0

    faltando = audios_faltando(selecao, args.audio_dir, args.audio_ext)
    if faltando:
        pasta = Path(args.audio_dir)
        existentes = (len(list(pasta.glob(f"*{args.audio_ext}")))
                      if pasta.is_dir() else 0)
        print(f"\n[ERRO] {len(faltando):,} dos {len(selecao):,} áudios da amostra não "
              f"estão em {pasta}")
        print(f"       exemplos: {', '.join(faltando[:3])}")
        if not pasta.is_dir():
            print("       A pasta não existe. Confira o nome real da pasta que o "
                  "Zenodo criou.")
        else:
            print(f"       A pasta tem {existentes:,} arquivos {args.audio_ext}; o eval "
                  f"completo do 2021 tem 181.566.")
            print("       Se tem menos, a extração está incompleta: o áudio vem em "
                  "várias partes\n       no Zenodo e todas precisam ser extraídas "
                  "na mesma pasta.")
        print("       Nenhum config foi gerado.")
        return 1

    testados, falhas = falhas_de_leitura(selecao, args.audio_dir, ext=args.audio_ext)
    if falhas:
        import soundfile as sf

        print(f"\n[AVISO] o libsndfile não conseguiu decodificar {len(falhas)} de "
              f"{testados} áudios testados.")
        print(f"        exemplo: {falhas[0][0]}")
        print(f"        erro real: {falhas[0][1]}")
        print(f"        soundfile {sf.__version__}, libsndfile {sf.__libsndfile_version__}")
        print("        A avaliação ainda funciona (o librosa cai no audioread), mas")
        print("        fica MUITO lenta: no Windows, um processo do FFmpeg por arquivo.")
        print("        Converta a amostra para WAV uma vez e aponte para lá:")
        print(f"          python scripts/converter_para_wav.py --protocolo {protocolo.as_posix()} "
              f"--origem {Path(args.audio_dir).as_posix()} --destino <pasta wav>")
        print("        e rode esta importação de novo com --audio-dir <pasta wav> "
              "--audio-ext .wav")

    base = load_config(args.config_base)
    cfg = config_derivado(base, f"2021_{tag}", protocolo, args.audio_dir,
                          ext=args.audio_ext)
    # Um config por MODELO: o protocolo é o mesmo para os dois modelos, mas o
    # config não — sem o nome do experimento, avaliar o segundo modelo na mesma
    # condição gravaria por cima do config do primeiro.
    destino = salvar_config(
        cfg, protocolo.with_name(f"{protocolo.stem}__{base['experiment']['name']}.yaml"))
    print(f"Config derivado: {destino}")
    print(f"   experimento: {cfg['experiment']['name']}  (os de 2019 ficam intactos)")
    print("\nPróximo passo:")
    print(f"   python evaluate.py --config {destino.as_posix()} "
          "--checkpoint checkpoints/<modelo>.pt")
    print(f"   python scripts/per_attack_eval.py --config {destino.as_posix()} "
          "--checkpoint checkpoints/<modelo>.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
