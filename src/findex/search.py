"""Ranked search.

1. parse the query into a tree and evaluate it -> matching doc ids (lru_cache per index)
2. score the matches with the query's positive terms (Scorer protocol)
3. top-k with heapq.nlargest, then a snippet with highlighted terms for each result
"""

from __future__ import annotations

import heapq
import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from findex.index import Index
from findex.query import QueryError, parse
from findex.scoring import SCORERS, Scorer, ScorerName, VectorScorer
from findex.semantic import rrf
from findex.timing import timed
from findex.tokenize import fix_apostrophes, tokenize

if TYPE_CHECKING:
    from findex.semantic import SemanticIndex

log = logging.getLogger(__name__)
SNIPPET_RADIUS = 80


@dataclass(frozen=True, slots=True, order=True)
class SearchResult:
    score: float  # first field: sorted(results) orders by score
    doc_id: int
    title: str
    path: str
    snippet: str = ""

    def to_dict(self) -> dict[str, object]:
        return {f: getattr(self, f) for f in ("doc_id", "score", "title", "path", "snippet")}


def match(index: Index, query: str) -> tuple[int, ...]:
    """Doc ids matching the Boolean query. Cached per index with lru_cache."""
    before = index.match.cache_info().hits
    ids = index.match(query)
    if index.match.cache_info().hits > before:
        log.debug("cache hit for %r", query)
    return ids


type Engine = Literal["numpy", "python"]


def score_python(
    index: Index, ids: Sequence[int], terms: Sequence[str], scorer: Scorer, k: int
) -> list[tuple[int, float]]:
    """Lab 3 scorer: a Python loop over every posting, then heapq.nlargest."""
    wanted = set(ids)
    scores: dict[int, float] = dict.fromkeys(ids, 0.0)  # "NOT x" matches still appear
    for term in terms:
        for posting in index.postings.get(term, ()):
            if posting.doc_id in wanted:
                scores[posting.doc_id] += scorer(term, posting, index)
    # nlargest is stable: equal scores keep ascending doc_id order
    return heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])  # O(n log k)


def score_numpy(
    index: Index, ids: Sequence[int], terms: Sequence[str], scorer: VectorScorer, k: int
) -> list[tuple[int, float]]:
    """Lab 8 scorer: per term ONE array expression over its postings, a scatter-add into a
    dense score vector, then top-k with np.argpartition (O(n), not a sort)."""
    scores = np.zeros(len(index.doc_len_array), np.float64)
    for term in terms:
        doc_ids, tfs = index.term_arrays(term)  # views into the CSR arrays, no copy
        if len(doc_ids):
            scores[doc_ids] += scorer.weights(term, doc_ids, tfs, index)  # ids unique per term
    matched = np.asarray(ids, np.int64)
    cand = scores[matched]
    if len(cand) > k:
        top = np.argpartition(cand, -k)[-k:]  # the k largest, unordered
        keep = np.flatnonzero(cand >= cand[top].min())  # + everything tied with the k-th
    else:
        keep = np.arange(len(cand))
    order = keep[np.lexsort((matched[keep], -cand[keep]))][:k]  # score desc, then doc_id asc
    return [(int(matched[i]), float(cand[i])) for i in order]


type Mode = Literal["keyword", "semantic", "hybrid"]
HYBRID_DEPTH = 100  # how deep each ranking goes into RRF


def rank_keyword(
    index: Index, query: str, scorer: Scorer, k: int, engine: Engine = "numpy"
) -> list[tuple[int, float]]:
    """Boolean match (cached) + BM25/TF-IDF ranking -> [(doc_id, score)], no snippets."""
    ids = match(index, query)
    if not ids:
        return []
    terms = sorted(parse(query).terms())  # fixed order -> identical float sums in both engines
    if engine == "numpy" and isinstance(scorer, VectorScorer):
        return score_numpy(index, ids, terms, scorer, k)
    return score_python(index, ids, terms, scorer, k)


@timed
def search(
    index: Index,
    query: str,
    scorer: Scorer | ScorerName = "bm25",
    k: int = 10,
    engine: Engine = "numpy",
    mode: Mode = "keyword",
    semantic: SemanticIndex | None = None,
) -> list[SearchResult]:
    """Top-k documents for a query.

    mode="keyword"  Boolean query language + BM25/TF-IDF (Labs 2-3, NumPy scorer of Lab 8)
    mode="semantic" cosine similarity of embeddings, best chunk per document (Lab 8)
    mode="hybrid"   both rankings fused with Reciprocal Rank Fusion
    engine="python" is the Lab 3 scoring loop, kept for the equality test and the benchmark.
    """
    if isinstance(scorer, str):
        scorer = SCORERS[scorer]
    if mode != "keyword" and semantic is None:
        raise QueryError("semantic index not loaded (build it with `findex embed`)")

    keyword: list[tuple[int, float]] = []
    if mode in ("keyword", "hybrid"):
        try:
            keyword = rank_keyword(
                index, query, scorer, k if mode == "keyword" else HYBRID_DEPTH, engine
            )
        except QueryError:
            if mode == "keyword":
                raise
            # hybrid: a query that is not valid query syntax still works semantically
    best_chunk: dict[int, int] = {}
    if mode == "keyword":
        top = keyword
    else:
        assert semantic is not None
        sem = semantic.rank(query, k if mode == "semantic" else HYBRID_DEPTH)
        best_chunk = {d: c for d, _, c in sem}
        if mode == "semantic":
            top = [(d, s) for d, s, _ in sem]
        else:
            top = rrf([[d for d, _ in keyword], [d for d, _, _ in sem]])[:k]

    try:
        terms = parse(query).terms() if mode == "keyword" else set(tokenize(query))
    except QueryError:
        terms = set(tokenize(query))
    # Highlight only informative terms: words found in more than half of all documents
    # ("into", "the") would produce thousands of regex hits per page for no information.
    informative = {t for t in terms if index.df(t) <= index.num_docs // 2}
    terms = informative or terms
    results: list[SearchResult] = []
    for doc_id, score in top:
        meta = index.doc_meta[doc_id]
        text = read_text(meta.path)
        snip = snippet(text, terms)
        if not snip and doc_id in best_chunk and semantic is not None:
            s, e = semantic.chunk_span[best_chunk[doc_id]].tolist()
            nfc = unicodedata.normalize("NFC", text)
            snip = re.sub(r"\s+", " ", nfc[s : min(e, s + 2 * SNIPPET_RADIUS)]).strip() + "…"
        results.append(SearchResult(score, doc_id, meta.title, meta.path, snip))
    return results


def read_text(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""


def _term_pattern(terms: frozenset[str]) -> re.Pattern[str]:
    alt = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    # same token boundaries as TOKEN_RE: not inside a word, not across an inner apostrophe
    return re.compile(rf"(?<!\w)(?<!\w')(?:{alt})(?!\w)(?!'\w)", re.IGNORECASE)


_patterns: dict[frozenset[str], re.Pattern[str]] = {}


def snippet(text: str, terms: set[str], radius: int = SNIPPET_RADIUS) -> str:
    """A 2·radius window (±radius around the best occurrence) with the most distinct query
    terms; query terms are highlighted as **term**. Sliding window: O(number of hits).

    Lab 8: hits are found by ONE case-insensitive regex over the text (a C loop) instead of
    tokenizing the whole document token by token in Python — the #1 hotspot of search."""
    if not text or not terms:
        return ""
    nfc = fix_apostrophes(unicodedata.normalize("NFC", text))
    key = frozenset(terms)
    pattern = _patterns.get(key)
    if pattern is None:
        pattern = _patterns.setdefault(key, _term_pattern(key))
    hits = [
        (m.start(), m.end(), t)
        for m in pattern.finditer(nfc)
        if (t := m.group().casefold()) in terms
    ]
    if not hits:
        return ""
    window: Counter[str] = Counter()
    best_i, best_j, best_n, j = 0, 0, 0, 0
    for i, (s, _, _) in enumerate(hits):
        while j < len(hits) and hits[j][1] <= s + 2 * radius:
            window[hits[j][2]] += 1
            j += 1
        if len(window) > best_n:
            best_i, best_j, best_n = i, j, len(window)
        window[hits[i][2]] -= 1
        if not window[hits[i][2]]:
            del window[hits[i][2]]
    center = (hits[best_i][0] + hits[best_j - 1][1]) // 2
    lo, hi = max(0, center - radius), min(len(nfc), center + radius)
    out: list[str] = []
    cur = lo
    for s, e, _ in hits[best_i:best_j]:
        if s >= cur and e <= hi:
            out += [nfc[cur:s], "**", nfc[s:e], "**"]
            cur = e
    out.append(nfc[cur:hi])
    body = re.sub(r"\s+", " ", "".join(out)).strip()
    return ("…" if lo > 0 else "") + body + ("…" if hi < len(nfc) else "")
