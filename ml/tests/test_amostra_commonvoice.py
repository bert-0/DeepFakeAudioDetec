"""Amostra do Common Voice (scripts/montar_amostra_commonvoice.py), com tabelas e tar sintéticos."""

import io
import tarfile

from scripts.montar_amostra_commonvoice import PORTUGAL, extrair, main, selecionar


def _tabelas(n_pessoas=30, clipes=5):
    validados, dur = [], {}
    for p in range(n_pessoas):
        for c in range(clipes):
            nome = f"cv_{p:03d}_{c}.mp3"
            validados.append({"client_id": f"pessoa{p}", "path": nome,
                              "variant": PORTUGAL if p == 0 else "Portuguese (Brasil)"})
            dur[nome] = 3000 if c == 0 else 6000
    return validados, dur


def test_selecao_sem_portugal_curtos_e_com_teto_por_pessoa():
    validados, dur = _tabelas()
    sel = selecionar(validados, dur, n_treino=40, n_dev=6, min_s=5.0, por_locutor=3)
    todos = sel["treino"] + sel["dev"]
    assert len(sel["dev"]) == 6 and len(sel["treino"]) == 40
    assert all(s >= 5.0 for _, s in todos)
    assert not any(c.startswith("cv_000_") for c, _ in todos), "a variante de Portugal fica de fora"
    pessoas = [c.split("_")[1] for c, _ in todos]
    assert max(pessoas.count(p) for p in set(pessoas)) <= 3


def test_dev_e_treino_com_pessoas_diferentes_e_mesma_semente_mesma_selecao():
    validados, dur = _tabelas()
    a = selecionar(validados, dur, 40, 6)
    pessoas = lambda parte: {c.split("_")[1] for c, _ in a[parte]}  # noqa: E731
    assert not pessoas("dev") & pessoas("treino")
    assert a == selecionar(validados, dur, 40, 6)
    assert a != selecionar(validados, dur, 40, 6, semente=7)


def _tar(caminho, nomes):
    with tarfile.open(caminho, "w:gz") as tar:
        for nome in nomes:
            dados = nome.encode()
            info = tarfile.TarInfo(nome)
            info.size = len(dados)
            tar.addfile(info, io.BytesIO(dados))


def test_extrai_so_os_escolhidos_e_so_pelo_nome(tmp_path):
    _tar(tmp_path / "cv.tar.gz", ["corpus/pt/clips/a.mp3", "corpus/pt/clips/b.mp3", "../../fora.mp3"])
    n = extrair(tmp_path / "cv.tar.gz", {"a.mp3": "treino", "fora.mp3": "dev"}, tmp_path / "saida")
    assert n == 2
    assert (tmp_path / "saida/treino/a.mp3").read_bytes() == b"corpus/pt/clips/a.mp3"
    assert (tmp_path / "saida/dev/fora.mp3").exists(), "o caminho do tar nunca é usado"
    assert not (tmp_path.parent / "fora.mp3").exists()
    assert not (tmp_path / "saida/treino/b.mp3").exists()


def test_linha_de_comando_grava_a_lista_sem_a_pessoa(tmp_path):
    validados, dur = _tabelas(n_pessoas=12)
    (tmp_path / "tab").mkdir()
    with open(tmp_path / "tab/validated.tsv", "w", encoding="utf-8") as f:
        f.write("client_id\tpath\tvariant\n")
        f.writelines(f"{r['client_id']}\t{r['path']}\t{r['variant']}\n" for r in validados)
    with open(tmp_path / "tab/clip_durations.tsv", "w", encoding="utf-8") as f:
        f.write("clip\tduration[ms]\n")
        f.writelines(f"{k}\t{v}\n" for k, v in dur.items())
    _tar(tmp_path / "cv.tar.gz", [f"pt/clips/{r['path']}" for r in validados])
    lista = tmp_path / "lista.tsv"
    assert main(["--tar", str(tmp_path / "cv.tar.gz"), "--tabelas", str(tmp_path / "tab"),
                 "--saida", str(tmp_path / "amostra"), "--treino", "10", "--dev", "3",
                 "--lista", str(lista)]) == 0
    linhas = lista.read_text(encoding="utf-8").splitlines()
    assert linhas[0] == "clipe\tparte\tsegundos" and len(linhas) == 14
    assert "pessoa" not in lista.read_text(encoding="utf-8")
    assert len(list((tmp_path / "amostra/treino").glob("*.mp3"))) == 10
