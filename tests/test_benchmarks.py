"""pytest-benchmark on search and build_index (run in CI with --benchmark-enable).

Absolute timings differ between machines, so the hard regression guard is RELATIVE:
the NumPy scorer must stay several times faster than the Lab 3 loop on a long postings list.
"""

import random
import timeit

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from findex.corpus import Document
from findex.index import Index, build_index
from findex.query import parse
from findex.scoring import BM25
from findex.search import score_numpy, score_python, search
from tests.conftest import TINY_DOCS


@pytest.fixture(scope="module")
def big() -> Index:
    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(300)] + ["common"] * 30
    docs = [
        Document(
            i, f"d{i}", f"T{i}", " ".join(rng.choice(vocab) for _ in range(rng.randint(20, 300)))
        )
        for i in range(3000)
    ]
    return build_index(iter(docs))


def test_bench_search(benchmark: BenchmarkFixture, big: Index) -> None:
    def run() -> None:
        big.match.cache_clear()
        search(big, "common OR w1 OR w2", k=10)

    benchmark(run)


def test_bench_build_index(benchmark: BenchmarkFixture) -> None:
    benchmark(lambda: build_index(iter(TINY_DOCS * 20), positions=True))


def test_numpy_scorer_stays_faster_than_lab3(big: Index) -> None:
    ids = big.match("common")
    terms = sorted(parse("common").terms())
    bm = BM25()
    py = min(timeit.repeat(lambda: score_python(big, ids, terms, bm, 10), number=5, repeat=3))
    npy = min(timeit.repeat(lambda: score_numpy(big, ids, terms, bm, 10), number=5, repeat=3))
    assert len(ids) > 2000
    assert npy * 3 < py, f"NumPy scorer regressed: {npy:.4f}s vs Lab 3 {py:.4f}s"
