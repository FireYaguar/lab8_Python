import pytest

from findex.index import Index, Posting
from findex.query import And, Node, Not, Or, Phrase, QueryError, Term, parse
from findex.scoring import BM25, Scorer, TfIdf
from findex.search import SearchResult, search, snippet
from findex.timing import timed


@pytest.mark.parametrize(
    ("text", "tree"),
    [
        ("a OR b c", Or(Term("a"), And(Term("b"), Term("c")))),
        ("a AND b", And(Term("a"), Term("b"))),
        ("a NOT b", And(Term("a"), Not(Term("b")))),
        ('"Event Loop"', Phrase(("event", "loop"))),
        ("(a OR b) c", And(Or(Term("a"), Term("b")), Term("c"))),
        ("well-known", And(Term("well"), Term("known"))),
        (
            'python AND (async OR await) NOT java "event loop"',
            And(
                And(And(Term("python"), Or(Term("async"), Term("await"))), Not(Term("java"))),
                Phrase(("event", "loop")),
            ),
        ),
    ],
)
def test_parse_trees(text: str, tree: Node) -> None:
    assert parse(text) == tree


def test_operators_build_same_tree() -> None:
    assert parse("a OR b NOT c") == Term("a") | (Term("b") & ~Term("c"))


@pytest.mark.parametrize("bad", ["(a OR b", "a OR", '"open', "...", ")"])
def test_parse_errors(bad: str) -> None:
    with pytest.raises(QueryError):
        parse(bad)


def test_phrase_with_unknown_word_matches_nothing(ix: Index) -> None:
    # regression: a pasted command with a quoted URL used to raise KeyError -> HTTP 500
    assert parse('"event zzzunknown"').evaluate(ix) == []
    assert parse('python "https://example.onrender.com/search?q=x"').evaluate(ix) == []


def test_phrase_needs_adjacent_positions(ix: Index) -> None:
    assert parse('"event loop"').evaluate(ix) == [0]  # doc 1 has "loop event"
    assert parse("event loop").evaluate(ix) == [0, 1]
    assert parse("python AND (async OR await) NOT java").evaluate(ix) == [3]


@pytest.mark.parametrize("scorer", [TfIdf(), BM25()])
def test_scorers_are_interchangeable(ix: Index, scorer: Scorer) -> None:
    res = search(ix, "python OR java", scorer, k=2)
    assert len(res) == 2 and res[0].score >= res[1].score
    assert res[0].doc_id == 2  # java: rarer term and tf=2


def _two_docs(short: int, long: int) -> Index:
    return Index({"t": [Posting(0, 1), Posting(1, 1)], "u": [Posting(0, 1)]}, {0: short, 1: long})


def test_sanity_rare_term_beats_common() -> None:
    ix = _two_docs(10, 10)  # "t" in both docs (common), "u" in one (rare)
    bm = BM25()
    assert bm("u", Posting(0, 1), ix) > bm("t", Posting(1, 1), ix)


def test_sanity_tf_saturates() -> None:
    ix, bm = _two_docs(10, 10), BM25()
    gain20 = bm("t", Posting(0, 20), ix) - bm("t", Posting(0, 19), ix)
    gain2 = bm("t", Posting(0, 2), ix) - bm("t", Posting(0, 1), ix)
    assert gain20 < gain2 / 20
    tf = TfIdf()  # TF-IDF is linear: the 20th occurrence adds as much as the 1st
    assert tf("t", Posting(0, 20), ix) == pytest.approx(20 * tf("t", Posting(0, 1), ix))


def test_sanity_short_doc_beats_long() -> None:
    ix, bm = _two_docs(10, 1000), BM25()
    assert bm("t", Posting(0, 1), ix) > bm("t", Posting(1, 1), ix)
    flat = BM25(b=0)  # b = 0: no length normalization
    assert flat("t", Posting(0, 1), ix) == flat("t", Posting(1, 1), ix)


def test_search_results_sorted_and_cached(ix: Index, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("DEBUG")
    r1 = search(ix, "python")
    r2 = search(ix, "python", "bm25")
    assert r1 == r2 and sorted(r1, reverse=True) == r1
    assert ix.match.cache_info().hits == 1
    assert "cache hit" in caplog.text and "search took" in caplog.text
    assert SearchResult(2.0, 0, "", "") > SearchResult(1.0, 9, "", "")


def test_timed_keeps_metadata() -> None:
    @timed
    def f() -> int:
        """Doc."""
        return 1

    assert (f.__name__, f.__doc__, f()) == ("f", "Doc.", 1)


def test_snippet_highlights() -> None:
    text = "x " * 100 + "the Event loop is here" + " y" * 100
    s = snippet(text, {"event", "loop"})
    assert "**Event** **loop**" in s and s.startswith("…") and s.endswith("…")
    assert len(s) < 200
