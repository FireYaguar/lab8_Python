"""Before/after numbers for the README: index load, search end-to-end, snippet share, build."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

from findex.corpus import list_sources
from findex.parallel import build_parallel
from findex.search import search
from findex.store import load


def med(fn, n: int) -> float:  # type: ignore[no-untyped-def]
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


if __name__ == "__main__":
    path = Path(sys.argv[1])
    print(f"load index.bin        {med(lambda: load(path), 3):8.1f} ms")
    ix = load(path)

    def one() -> None:
        ix.match.cache_clear()
        search(ix, "python async queue", "bm25", k=10)

    print(f"search 3 terms (full) {med(one, 30):8.2f} ms")
    paths = list_sources(Path("data"))
    print(f"build (serial, pos)   {med(lambda: build_parallel(paths, positions=True), 3):8.0f} ms")
