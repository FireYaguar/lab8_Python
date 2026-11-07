"""Semantic search on synthetic word vectors (no model download needed)."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from fastapi.testclient import TestClient

from findex.corpus import Document
from findex.index import build_index
from findex.search import search
from findex.semantic import SemanticIndex, WordVectors, build_semantic, chunks_of, rrf
from findex.store import save
from findex.web.app import create_app
from findex.web.settings import Settings

# 3 topics: animals / money / code; synonyms share a direction
TOPICS = {
    "animals": ["cat", "feline", "kitten", "dog"],
    "money": ["stock", "market", "shares"],
    "code": ["python", "program", "script", "code"],
}
TEXTS = {
    "a.txt": "the cat sat with a kitten",
    "b.txt": "the stock market and shares fell",
    "c.txt": "write a python program",
}


@pytest.fixture
def wv() -> WordVectors:
    rng = np.random.default_rng(0)
    words: list[str] = []
    rows: list[int] = []
    vecs: list[npt.NDArray[np.float32]] = []
    for i, (_, ws) in enumerate(TOPICS.items()):
        base = np.zeros(8, np.float32)
        base[i] = 1.0
        for w in ws:
            words.append(w)
            rows.append(len(vecs))
            vecs.append(np.asarray(base + 0.05 * rng.standard_normal(8), np.float32))
    for w in ["the", "sat", "with", "a", "and", "fell", "write"]:
        words.append(w)
        rows.append(len(vecs))
        vecs.append(np.asarray(0.1 * rng.standard_normal(8), np.float32))
    return WordVectors(np.array(words), np.array(rows, np.int32), np.array(vecs, np.float32))


@pytest.fixture
def setup(wv: WordVectors, tmp_path: Path) -> tuple[object, SemanticIndex]:
    for name, text in TEXTS.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    docs = [Document(i, str(tmp_path / n), n, t) for i, (n, t) in enumerate(TEXTS.items())]
    ix = build_index(iter(docs))
    return ix, build_semantic(ix, wv)


def test_semantic_finds_synonyms_without_shared_words(setup: tuple[object, SemanticIndex]) -> None:
    ix, sem = setup
    assert sem.rank("feline", 1)[0][0] == 0  # "feline" is not in doc 0, "cat" is
    assert sem.rank("code script", 1)[0][0] == 2
    res = search(ix, "feline", k=3, mode="semantic", semantic=sem)  # type: ignore[arg-type]
    assert res[0].doc_id == 0 and res[0].snippet  # best-chunk snippet even without a term hit
    assert search(ix, "feline", mode="keyword") == []  # type: ignore[arg-type]
    assert search(ix, "feline", k=3, mode="hybrid", semantic=sem)[0].doc_id == 0  # type: ignore[arg-type]
    np.testing.assert_allclose(np.linalg.norm(sem.E, axis=1), 1.0, rtol=1e-5)


def test_save_load_round_trip(setup: tuple[object, SemanticIndex], tmp_path: Path) -> None:
    _, sem = setup
    sem.save(tmp_path / "e.npz")
    back = SemanticIndex.load(tmp_path / "e.npz")
    assert back.rank("stock", 3)[0][0] == sem.rank("stock", 3)[0][0] == 1
    np.testing.assert_allclose(back.E, sem.E, atol=1e-3)  # stored as float16


def test_rrf() -> None:
    fused = rrf([[1, 2, 3], [3, 1]], k=60)
    assert [d for d, _ in fused] == [1, 3, 2]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_chunks_cover_tokens() -> None:
    text = " ".join(f"w{i}" for i in range(450))
    chunks = chunks_of(text, size=200)
    assert [len(t) for t, _, _ in chunks] == [200, 200, 50]
    assert text[chunks[1][1] : chunks[1][2]].split()[0] == "w200"


def test_web_semantic_mode(setup: tuple[object, SemanticIndex], tmp_path: Path) -> None:
    ix, sem = setup
    save(ix, tmp_path / "ix.bin")  # type: ignore[arg-type]
    sem.save(tmp_path / "e.npz")
    app = create_app(Settings(index_path=tmp_path / "ix.bin", embeddings_path=tmp_path / "e.npz"))
    with TestClient(app) as c:
        r = c.get("/search", params={"q": "feline", "mode": "semantic"}).json()
        assert r["mode"] == "semantic" and r["results"][0]["title"] == "a.txt"
        assert "hybrid" in c.get("/", params={"q": "feline", "mode": "hybrid"}).text
