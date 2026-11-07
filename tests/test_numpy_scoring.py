"""Lab 8: the NumPy scorer must give exactly the Lab 3 ranking."""

import pickle
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from findex.corpus import Document, iter_documents
from findex.index import Index, build_index
from findex.scoring import BM25, Scorer, TfIdf
from findex.search import search
from findex.store import load

QUERIES = ["python", "loop OR java", "event loop", "NOT python", "java OR async OR the", "filler"]


@pytest.mark.parametrize("scorer", [BM25(), TfIdf(), BM25(k1=1.2, b=0.0)])
@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize("k", [1, 2, 10])
def test_numpy_ranking_equals_lab3(ix: Index, scorer: Scorer, query: str, k: int) -> None:
    fast = search(ix, query, scorer, k=k, engine="numpy")
    slow = search(ix, query, scorer, k=k, engine="python")
    assert [r.doc_id for r in fast] == [r.doc_id for r in slow]
    assert [r.score for r in fast] == pytest.approx([r.score for r in slow], rel=1e-12)


words = st.sampled_from(["a", "b", "c", "d", "e"])
corpora = st.lists(st.lists(words, min_size=1, max_size=40).map(" ".join), min_size=1, max_size=30)


@settings(max_examples=60, deadline=None)
@given(corpora, st.sampled_from(["a", "a OR b", "a b", "NOT c", "c OR d OR e"]), st.integers(1, 8))
def test_numpy_equals_python_property(texts: list[str], query: str, k: int) -> None:
    """Many ties on purpose (tiny vocabulary): argpartition + lexsort must keep Lab 3 order."""
    ix = build_index(Document(i, f"d{i}", f"T{i}", t) for i, t in enumerate(texts))
    for scorer in (BM25(), TfIdf()):
        fast = [(r.doc_id, r.score) for r in search(ix, query, scorer, k=k, engine="numpy")]
        slow = [(r.doc_id, r.score) for r in search(ix, query, scorer, k=k, engine="python")]
        assert fast == slow


def test_arrays_layout_and_lazy_postings(ix: Index) -> None:
    ids, tfs = ix.term_arrays("java")
    assert (
        ids.dtype.name == tfs.dtype.name == "int32" and ids.tolist() == [2] and tfs.tolist() == [2]
    )
    assert ix.doc_len_array.dtype.name == "int32" and ix.doc_len_array.tolist() == [6, 4, 6, 53]
    assert ix.term_arrays("nope")[0].size == 0 and ix.df("nope") == 0
    assert ix.num_postings == sum(len(p) for p in ix.postings.values())


def test_pickle_stores_arrays_not_posting_objects(ix: Index) -> None:
    blob = pickle.dumps(ix)
    assert b"Posting" not in blob  # buffers + dicts only
    assert pickle.loads(blob) == ix


@pytest.mark.slow
def test_full_corpus_rankings_match(tmp_path: Path) -> None:
    data = Path("data")
    if not data.is_dir():
        pytest.skip("corpus not downloaded")
    ix = build_index(iter_documents(data), positions=True)
    from findex.store import save

    save(ix, tmp_path / "ix.bin")
    ix = load(tmp_path / "ix.bin")
    for q in ["the", "winsound", "python async queue", '"event loop" NOT thread', "json OR pickle"]:
        for scorer in (BM25(), TfIdf()):
            fast = search(ix, q, scorer, k=20, engine="numpy")
            slow = search(ix, q, scorer, k=20, engine="python")
            assert [r.doc_id for r in fast] == [r.doc_id for r in slow], q
