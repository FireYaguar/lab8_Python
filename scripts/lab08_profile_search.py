"""Profile the search hot path in-process (index already loaded): 50 x a 3-term query."""

from __future__ import annotations

import sys
from pathlib import Path

from findex.search import search
from findex.store import load

QUERY = "python async queue"

if __name__ == "__main__":
    ix = load(Path(sys.argv[1] if len(sys.argv) > 1 else "index.bin"))
    ix.match.cache_clear()
    for _ in range(50):
        ix.match.cache_clear()  # measure the whole path, not the lru_cache hit
        search(ix, QUERY, "bm25", k=10)
