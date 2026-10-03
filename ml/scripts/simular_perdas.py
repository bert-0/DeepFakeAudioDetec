"""Atalho para `simular_caminho.py --efeito perdas` (mantido por compatibilidade).

Resultado registrado: uma perda de 10 ms por áudio quase não muda o score
(sessão "limpo", 40 áudios: EER 22,50% -> 17,50%, dentro do ruído da amostra).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.simular_caminho import main as _main, remover_trechos  # noqa: E402,F401


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--efeito" not in argv:
        argv += ["--efeito", "perdas"]
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
