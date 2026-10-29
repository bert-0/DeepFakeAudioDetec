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


def test_divide_montagens_sem_cruzar_treino_e_regua(tmp_path, monkeypatch):
    import csv

    from scripts.montar_amostra_replaydf import dividir_montagens

    partes = dividir_montagens([f"u{i}" for i in range(10)] * 3, 5, 2)
    assert [len(partes[k]) for k in ("treino", "dev", "regua")] == [5, 2, 3]
    assert not (set(partes["treino"]) & set(partes["regua"])) and not (set(partes["dev"]) & set(partes["regua"]))

    for i in range(6):
        _montagem(tmp_path / "base", f"m{i}", n_reais=3, por_ferramenta=2)
    monkeypatch.chdir(tmp_path)
    assert main(["--pasta", str(tmp_path / "base"), "--por-montagem", "2", "--lista", "data/recortes/amostra.tsv"]) == 0
    assert main(["--pasta", str(tmp_path / "base"), "--por-montagem", "3", "--dividir", "3", "1",
                 "--regua-existente", "data/recortes/amostra.tsv"]) == 0
    ler = lambda n: list(csv.DictReader(open(tmp_path / "data/recortes" / n, encoding="utf-8"), delimiter="	"))  # noqa: E731
    montagens = {r["montagem"]: r["parte"] for r in ler("replaydf_montagens_s42.tsv")}
    treino, regua = ler("replaydf_treino_3_s42.tsv"), ler("replaydf_regua_s42.tsv")
    assert {montagens[r["montagem"]] for r in treino} == {"treino"}
    assert {montagens[r["montagem"]] for r in regua} == {"regua"} and len(regua) == 2 * 4
    assert not {r["arquivo"] for r in treino} & {r["arquivo"] for r in regua}
