"""Descobre o rótulo verdadeiro de um áudio, procurando nos três protocolos.

Existe porque procurar o ID "na mão" falha em silêncio: um `findstr` que não
acha nada é indistinguível de protocolo errado, pasta errada, ID de outra
partição ou formato diferente do esperado. Aqui, quando não acha, o script diz
**onde procurou, quantas linhas tinha cada protocolo e com que cara são os IDs
de verdade** — que é o que permite ver o erro num relance.

Uso:
    python scripts/rotulo.py --config configs/fusion_v4.yaml LA_E_A9898607
    python scripts/rotulo.py --config configs/fusion_v4.yaml data/LA/.../LA_E_1000147.flac
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.data.dataset import parse_protocol_with_systems  # noqa: E402

ROTULO = {0: "bonafide", 1: "spoof"}


def identificador(alvo: str) -> str:
    """Aceita tanto o ID quanto o caminho do arquivo.

    A barra invertida é normalizada à mão: num caminho do Windows colado
    dentro de um shell Linux, `Path` não a trata como separador e o `stem`
    devolveria a linha inteira. Como o caminho quase sempre vem copiado do
    PowerShell, o caso importa.
    """
    return Path(alvo.replace("\\", "/")).stem


def main() -> int:
    p = argparse.ArgumentParser(description="Rótulo verdadeiro de um áudio")
    p.add_argument("alvo", help="ID (LA_E_1000147) ou caminho do .flac")
    p.add_argument("--config", required=True)
    args = p.parse_args()

    config = load_config(args.config)
    alvo = identificador(args.alvo)
    print(f"Procurando: {alvo}\n")

    diagnostico = []
    for particao, caminho in config["data"]["protocols"].items():
        caminho = Path(caminho)
        if not caminho.is_file():
            diagnostico.append((particao, caminho, None, []))
            continue
        registros = parse_protocol_with_systems(caminho)
        for nome, label, sistema in registros:
            if nome == alvo:
                ataque = f" | ataque {sistema}" if sistema != "-" else ""
                print(f"  ENCONTRADO em '{particao}'")
                print(f"  Rótulo verdadeiro: {ROTULO[label].upper()}{ataque}")
                print(f"\n  No monitor, score BAIXO = voz humana, ALTO = sintético.")
                return 0
        diagnostico.append((particao, caminho, len(registros),
                            [n for n, _, _ in registros[:2]]))

    print("  NÃO ENCONTRADO em nenhum protocolo.\n")
    print(f"  {'partição':>8s}  {'linhas':>8s}  exemplos de ID / problema")
    print("  " + "-" * 68)
    for particao, caminho, n, exemplos in diagnostico:
        if n is None:
            print(f"  {particao:>8s}  {'—':>8s}  ARQUIVO NÃO EXISTE: {caminho}")
        else:
            print(f"  {particao:>8s}  {n:8,}  {', '.join(exemplos)}")
    print("\n  Compare o formato: se os IDs acima não se parecem com o que você")
    print("  procurou, o áudio provavelmente vem de outra base ou foi renomeado.")
    print("  Se alguma partição diz ARQUIVO NÃO EXISTE, ajuste `data.protocols`")
    print("  no config ou rode a partir da pasta `ml`.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
