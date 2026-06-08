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
    return p.parse_args()


def check_partition(
    name: str, protocol: str, audio_dir: str, sample: int, file_ext: str
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
    ok = _check_sample(items, audio_path, sample, file_ext) and ok
    return ok


def _check_sample(items, audio_path: Path, sample: int, file_ext: str) -> bool:
    import soundfile as sf

    step = max(1, len(items) // sample) if sample else 1
    sampled = items[::step][:sample] if sample else items
    missing, unreadable, durations = [], [], []

    for file_name, _ in sampled:
        fpath = audio_path / f"{file_name}{file_ext}"
        if not fpath.is_file():
            missing.append(file_name)
            continue
        try:
            info = sf.info(str(fpath))
            durations.append(info.frames / info.samplerate)
        except Exception:  # noqa: BLE001 - reporta qualquer falha de leitura
            unreadable.append(file_name)

    if missing:
        print(f"{ERR} {len(missing)}/{len(sampled)} arquivos amostrados NÃO existem "
              f"(ex.: {missing[0]}{file_ext}).")
    if unreadable:
        print(f"{ERR} {len(unreadable)}/{len(sampled)} arquivos não puderam ser lidos "
              f"(ex.: {unreadable[0]}{file_ext}).")
    if not missing and not unreadable:
        avg = sum(durations) / len(durations) if durations else 0.0
        print(f"{OK} amostra de {len(sampled)} arquivos: todos existem e abrem "
              f"(duração média ~{avg:.1f}s)")
    return not missing and not unreadable


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    print(f"Config: {args.config}")
    print(f"Base (data.root): {config['data']['root']}")

    all_ok = True
    for part in ("train", "dev", "eval"):
        all_ok &= check_partition(
            part,
            config["data"]["protocols"][part],
            config["data"]["audio_dir"][part],
            args.sample,
            args.file_ext,
        )

    print("\n" + "=" * 40)
    if all_ok:
        print(f"{OK} Tudo certo — a base parece pronta para o treino.")
        return 0
    print(f"{ERR} Foram encontrados problemas. Confira ml/data/README.md.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
