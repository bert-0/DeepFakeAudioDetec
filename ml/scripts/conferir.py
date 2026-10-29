"""Conferência antes da apresentação, pelo terminal: a mesma da página /conferir.

Confere o modelo final e o limiar, a GPU, o microfone (escuta 1,5 s da sala, sem
gravar), a saída de som, a clonagem e o disco, e lembra o que não dá para medir.
Sai com código 1 se houver falha.

Uso (de dentro de ml/):
    python scripts/conferir.py --xlsr outputs/limiar_xlsr/limiar.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MARCA = {"ok": "ok      ", "atencao": "ATENÇÃO ", "falha": "FALHA   ", "lembrete": "conferir"}


def main(argv=None) -> int:
    from web.conferencia import conferir_tudo, resumo, situacao

    p = argparse.ArgumentParser(description="Conferência antes da apresentação")
    p.add_argument("--xlsr", default=os.environ.get("DETECTOR_XLSR"), help="o JSON do limiar_xlsr.py")
    p.add_argument("--dispositivo", default="cpu", help="onde carregar o modelo para conferir")
    p.add_argument("--sem-modelo", action="store_true", help="não carrega o XLS-R (mais rápido)")
    args = p.parse_args(argv)

    carregar = None
    if args.xlsr and not args.sem_modelo:
        def carregar():
            from src.xlsr_final import ModeloFinal, ler_limiar

            lim = ler_limiar(args.xlsr)
            return ModeloFinal(lim.cabeca, args.dispositivo, modelo_hf=lim.fonte_modelo)
    itens = conferir_tudo(args.xlsr, carregar=carregar)
    for i in itens:
        print(f"[{MARCA[i.estado]}] {i.nome}: {i.detalhe}")
        if i.acao:
            print(f"             {i.acao}")
    print(f"\n{resumo(itens).capitalize()}.")
    return 1 if situacao(itens) == "falha" else 0


if __name__ == "__main__":
    codigo = main()
    sys.stdout.flush()
    # Uma checagem presa no driver de áudio (thread em segundo plano) não segura a saída.
    os._exit(codigo)
