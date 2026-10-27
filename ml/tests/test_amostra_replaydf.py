"""Amostra do ReplayDF (scripts/montar_amostra_replaydf.py) com montagens falsas."""

from scripts.montar_amostra_replaydf import main, selecionar

CAB = "original_file|recorded_file|label|architecture|language|mic|speaker|uid\n"


def _montagem(raiz, uid, n_reais=5, por_ferramenta=4):
    pasta = raiz / "wav" / uid
    pasta.mkdir(parents=True)
    linhas = [CAB]
    for i in range(n_reais):
        linhas.append(f"o{i}.wav|wav/{uid}/benign/en/r{i}.wav|bona-fide|vits|en|Mic A|Caixa B|{uid}\n")
    for a in ("bark", "vits", "xtts_v2.0"):
        for i in range(por_ferramenta):
            linhas.append(f"f.wav|wav/{uid}/spoof/{a}/en/{a}{i}.wav|spoof|{a}|en|Mic A|Caixa B|{uid}\n")
    (pasta / "meta.csv").write_text("".join(linhas), encoding="utf-8")


def test_selecao_equilibrada_por_montagem_e_ferramenta(tmp_path):
    from scripts.montar_amostra_replaydf import ler_meta

    for uid in ("aaa", "bbb"):
        _montagem(tmp_path, uid)
    sel = selecionar(ler_meta(tmp_path), por_montagem=3)
    for uid in ("aaa", "bbb"):
        da = [r for r in sel if r["uid"] == uid]
        assert sum(r["label"] == "bona-fide" for r in da) == 3
        assert sorted(r["architecture"] for r in da if r["label"] == "spoof") == ["bark", "vits", "xtts_v2.0"]
    assert sel == selecionar(ler_meta(tmp_path), por_montagem=3)


def test_lista_no_formato_do_extrair(tmp_path):
    from scripts.estudo_ssl import ler_lista

    _montagem(tmp_path / "base", "aaa", n_reais=2, por_ferramenta=1)
    lista = tmp_path / "l.tsv"
    assert main(["--pasta", str(tmp_path / "base"), "--por-montagem", "2", "--lista", str(lista)]) == 0
    itens = ler_lista(lista)
    assert len(itens) == 4 and sum(r for _, _, r in itens) == 2
    assert {c for _, c, r in itens if r == 0} == {"bonafide"}
    assert all(str(a).endswith(".wav") and "/wav/aaa/" in str(a).replace("\\", "/") for a, _, _ in itens)
