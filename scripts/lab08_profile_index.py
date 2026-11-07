"""Profile the index build in-process (serial): the whole corpus, positions on."""

from __future__ import annotations

import sys
from pathlib import Path

from findex.corpus import list_sources
from findex.parallel import build_parallel

if __name__ == "__main__":
    build_parallel(
        list_sources(Path(sys.argv[1] if len(sys.argv) > 1 else "data")),
        executor="serial",
        positions=True,
    )
