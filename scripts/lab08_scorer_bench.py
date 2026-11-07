"""Lab 8 table: scorer only (match is cached, snippets excluded), Lab 3 loop vs NumPy."""

from __future__ import annotations

import sys
import timeit
from pathlib import Path

from findex.query import parse
from findex.scoring import BM25
from findex.search import score_numpy, score_python
from findex.store import load

QUERIES = [
    ("common term", "the"),
    ("rare term", "winsound"),
    ("3-term query (OR)", "python OR async OR queue"),
    ("3-term query (AND)", "python async queue"),
]

if __name__ == "__main__":
    ix = load(Path(sys.argv[1] if len(sys.argv) > 1 else "index.bin"))
    bm = BM25()
    print(
        "| Query | Docs matched | Postings scored | Lab 3 scorer (ms) | NumPy scorer (ms) | Speedup |"
    )
    print("|---|---|---|---|---|---|")
    for label, q in QUERIES:
        ids = ix.match(q)
        terms = sorted(parse(q).terms())
        n_post = sum(ix.df(t) for t in terms)
        py = (
            min(
                timeit.repeat(
                    lambda ids=ids, terms=terms: score_python(ix, ids, terms, bm, 10),
                    number=20,
                    repeat=5,
                )
            )
            / 20
        )
        npy = (
            min(
                timeit.repeat(
                    lambda ids=ids, terms=terms: score_numpy(ix, ids, terms, bm, 10),
                    number=20,
                    repeat=5,
                )
            )
            / 20
        )
        assert [d for d, _ in score_python(ix, ids, terms, bm, 10)] == [
            d for d, _ in score_numpy(ix, ids, terms, bm, 10)
        ]
        print(
            f"| {label}: `{q}` | {len(ids)} | {n_post} | {py * 1000:.3f} | {npy * 1000:.3f} "
            f"| {py / npy:.1f}× |"
        )
