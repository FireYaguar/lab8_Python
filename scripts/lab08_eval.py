"""Lab 8: precision@5 for keyword (BM25) / semantic / hybrid (RRF).

    uv run python scripts/lab08_eval.py index.bin embeddings.npz
10 labeled queries from Lab 3 + 5 PARAPHRASES (labeled before running) that avoid the
key words of the relevant documents (module names, title words).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lab03_eval import QRELS, rel  # the Lab 3 labels, unchanged

from findex.query import QueryError
from findex.search import Mode, search
from findex.semantic import SemanticIndex
from findex.store import load

PARAPHRASES: dict[str, set[str]] = {
    "run many jobs at once on all cores": {
        "library/multiprocessing.rst",
        "library/concurrent.futures.rst",
        "library/threading.rst",
        "howto/free-threading-python.rst",
        "library/concurrent.interpreters.rst",
    },
    "turn objects into bytes and restore them later": {
        "library/pickle.rst",
        "library/marshal.rst",
        "library/copyreg.rst",
        "library/shelve.rst",
        "library/json.rst",
    },
    "spot textual patterns with wildcards": {
        "library/re.rst",
        "howto/regex.rst",
        "library/fnmatch.rst",
        "library/glob.rst",
    },
    "isolated sandbox with its own installed packages": {
        "library/venv.rst",
        "tutorial/venv.rst",
        "installing/index.rst",
        "library/ensurepip.rst",
    },
    "wait for many network operations without stalling": {
        "library/asyncio.rst",
        "library/asyncio-eventloop.rst",
        "library/asyncio-task.rst",
        "howto/a-conceptual-overview-of-asyncio.rst",
        "library/selectors.rst",
        "library/asyncio-stream.rst",
    },
}
MODES: list[Mode] = ["keyword", "semantic", "hybrid"]


def p_at_5(ix: object, sem: SemanticIndex, q: str, relevant: set[str], mode: Mode) -> float:
    try:
        top = [rel(r.path) for r in search(ix, q, "bm25", k=5, mode=mode, semantic=sem)]  # type: ignore[arg-type]
    except QueryError:
        top = []
    return sum(p in relevant for p in top) / 5


if __name__ == "__main__":
    ix = load(Path(sys.argv[1] if len(sys.argv) > 1 else "index.bin"))
    sem = SemanticIndex.load(Path(sys.argv[2] if len(sys.argv) > 2 else "embeddings.npz"))
    print("| Query | kind | max | keyword (BM25) | semantic | hybrid (RRF) |")
    print("|---|---|---|---|---|---|")
    sums: dict[str, list[float]] = {"orig": [0, 0, 0, 0], "para": [0, 0, 0, 0]}
    for kind, qrels in (("orig", QRELS), ("para", PARAPHRASES)):
        for q, relevant in qrels.items():
            row = [p_at_5(ix, sem, q, relevant, m) for m in MODES]
            mx = min(len(relevant), 5) / 5
            for i, v in enumerate([mx, *row]):
                sums[kind][i] += v
            label = "original" if kind == "orig" else "paraphrase"
            print(f"| `{q}` | {label} | {mx:.1f} | " + " | ".join(f"{v:.1f}" for v in row) + " |")
    for kind, n in (("orig", len(QRELS)), ("para", len(PARAPHRASES))):
        s = sums[kind]
        name = "10 original" if kind == "orig" else "5 paraphrases"
        print(
            f"| **mean, {name}** | | {s[0] / n:.2f} | "
            + " | ".join(f"**{v / n:.2f}**" for v in s[1:])
            + " |"
        )
    tot = [sums["orig"][i] + sums["para"][i] for i in range(4)]
    print(
        f"| **mean, all 15** | | {tot[0] / 15:.2f} | "
        + " | ".join(f"**{v / 15:.2f}**" for v in tot[1:])
        + " |"
    )
