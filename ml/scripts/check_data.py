"""Verificação da base ASVspoof 2019 LA antes do treino.

Confere, para cada partição (train/dev/eval):
  - se o arquivo de protocolo existe e é legível;
  - a contagem de amostras bonafide vs. spoof (e o balanceamento);
  - se o diretório de áudio existe;
  - se uma amostra dos arquivos .flac referenciados existe no disco e abre;
  - duração média de alguns arquivos.

Uso:
    python scripts/check_data.py --config configs/baseline.yaml
    python scripts/check_data.py --config configs/baseline.yaml --sample 50

Sai com código 0 se tudo estiver OK, ou 1 se encontrar problemas.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Permite rodar a partir de ml/ (importar o pacote src).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.data.dataset import parse_protocol  # noqa: E402

OK = "[ OK ]"
WARN = "[AVISO]"
ERR = "[ERRO]"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verifica a base ASVspoof 2019 LA")
    p.add_argument("--config", required=True, help="caminho do YAML de configuração")
    p.add_argument("--sample", type=int, default=20,
                   help="quantos arquivos .flac amostrar para teste de leitura por partição")
    p.add_argument("--file-ext", default=".flac", help="extensão dos áudios (padrão: .flac)")
    p.add_argument("--deep", action="store_true",
                   help="decodifica TODOS os áudios em vez de só ler o cabeçalho de "
                        "uma amostra. Leva minutos, mas é o único jeito de achar "
                        "arquivos truncados por download interrompido.")
    return p.parse_args()


# Contagens oficiais do ASVspoof 2019 LA. Servem para detectar um protocolo
# truncado: o parser ignora linhas malformadas em silêncio, então um download
# interrompido produziria uma partição menor sem nenhum sinal.
CONTAGENS_OFICIAIS = {"train": 25380, "dev": 24844, "eval": 71237}


def check_partition(
    name: str, protocol: str, audio_dir: str, sample: int, file_ext: str,
    deep: bool = False
) -> bool:
    """Verifica uma partição. Devolve True se estiver tudo OK."""
    print(f"\n=== Partição: {name} ===")
    ok = True

    proto_path = Path(protocol)
    if not proto_path.is_file():
        print(f"{ERR} protocolo não encontrado: {proto_path}")
        return False
    print(f"{OK} protocolo: {proto_path}")

    items = parse_protocol(proto_path)
    if not items:
        print(f"{ERR} protocolo sem entradas válidas (formato esperado: "
              f"'SPEAKER FILE - SYSTEM_ID KEY').")
        return False

    counts = Counter(label for _, label in items)
    n_bonafide, n_spoof = counts.get(0, 0), counts.get(1, 0)
    total = n_bonafide + n_spoof
    print(f"{OK} {total} amostras  ->  bonafide={n_bonafide}  spoof={n_spoof}")
    esperado = CONTAGENS_OFICIAIS.get(name)
    if esperado is not None and total != esperado:
        print(f"{WARN} o ASVspoof 2019 LA tem {esperado} amostras em '{name}', "
              f"mas o protocolo lido tem {total} ({total - esperado:+d}). "
              "Protocolo possivelmente truncado — linhas malformadas são "
              "descartadas em silêncio pelo parser.")
        ok = False
    if n_bonafide == 0 or n_spoof == 0:
        print(f"{WARN} a partição tem apenas uma classe — confira o protocolo.")
    else:
        ratio = n_spoof / n_bonafide
        print(f"       proporção spoof:bonafide = {ratio:.1f}:1")

    audio_path = Path(audio_dir)
    if not audio_path.is_dir():
        print(f"{ERR} diretório de áudio não encontrado: {audio_path}")
        return False
    print(f"{OK} diretório de áudio: {audio_path}")

    # Teste de existência/leitura em uma amostra dos arquivos.
    ok = _check_sample(items, audio_path, sample, file_ext, deep) and ok
    return ok


def _check_sample(items, audio_path: Path, sample: int, file_ext: str,
                  deep: bool = False) -> bool:
    """Verifica existência e legibilidade de uma amostra (ou de tudo, com `deep`).

    **`deep` importa mais do que parece.** Sem ele a checagem usa `sf.info()`,
    que lê apenas o cabeçalho STREAMINFO — e o cabeçalho de um FLAC continua
    íntegro mesmo que o arquivo tenha sido cortado no meio do download. Medido:
    um FLAC com 10% dos bytes reporta "frames=64000, dur=4.00s" e passa como
    `[ OK ]`, mas `sf.read()` estoura com `flac decoder lost sync`. Ou seja, a
    verificação que existe para evitar surpresas não pega o caso mais provável.

    Com `deep`, cada arquivo é **decodificado** de verdade e todos são
    percorridos. Leva minutos; um treino perdido leva horas.
    """
    import soundfile as sf

    if deep:
        sampled = items
    else:
        step = max(1, len(items) // sample) if sample else 1
        sampled = items[::step][:sample] if sample else items
    missing, unreadable, durations = [], [], []

    # Em modo raso são ~20 arquivos e a varredura é instantânea; em modo deep
    # são dezenas de milhares e demora minutos. Sem este sinal de vida, quem
    # roda não distingue "decodificando" de "travado".
    passo_aviso = 5000 if deep else 0

    for i, (file_name, _) in enumerate(sampled):
        if passo_aviso and i and i % passo_aviso == 0:
            print(f"       ... {i}/{len(sampled)} decodificados "
                  f"({len(unreadable)} com falha)", flush=True)
        fpath = audio_path / f"{file_name}{file_ext}"
        if not fpath.is_file():
            missing.append(file_name)
            continue
        try:
            if deep:
                dados, taxa = sf.read(str(fpath), dtype="float32")
                if dados.size == 0:
                    raise ValueError("arquivo sem amostras")
                durations.append(len(dados) / taxa)
            else:
                info = sf.info(str(fpath))
                durations.append(info.frames / info.samplerate)
        except Exception as erro:  # noqa: BLE001 - reporta qualquer falha de leitura
            unreadable.append((file_name, str(erro) or type(erro).__name__))

    modo = "decodificados" if deep else "verificados (só cabeçalho)"
    if missing:
        print(f"{ERR} {len(missing)}/{len(sampled)} arquivos NÃO existem "
              f"(ex.: {missing[0]}{file_ext}).")
    if unreadable:
        print(f"{ERR} {len(unreadable)}/{len(sampled)} arquivos não puderam ser lidos:")
        for nome, motivo in unreadable[:10]:
            print(f"        {nome}{file_ext}: {motivo}")
        if len(unreadable) > 10:
            print(f"        ... e mais {len(unreadable) - 10}")
    if not missing and not unreadable:
        avg = sum(durations) / len(durations) if durations else 0.0
        print(f"{OK} {len(sampled)} arquivos {modo}: todos íntegros "
              f"(duração média ~{avg:.1f}s)")
    return not missing and not unreadable


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    print(f"Config: {args.config}")
    print(f"Base (data.root): {config['data']['root']}")
    if not args.deep:
        print(f"{WARN} modo rápido: só o cabeçalho de {args.sample} arquivos por\n       partição. Um FLAC truncado passa como íntegro. Use --deep antes\n       de um treino longo.")

    all_ok = True
    for part in ("train", "dev", "eval"):
        all_ok &= check_partition(
            part,
            config["data"]["protocols"][part],
            config["data"]["audio_dir"][part],
            args.sample,
            args.file_ext,
            args.deep,
        )

    print("\n" + "=" * 40)
    if all_ok:
        print(f"{OK} Tudo certo — a base parece pronta para o treino.")
        return 0
    print(f"{ERR} Foram encontrados problemas. Confira ml/data/README.md.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
