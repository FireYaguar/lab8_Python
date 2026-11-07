"""Scorers. Any object with a matching __call__ satisfies the Scorer protocol —
no common base class (duck typing; Lab 4 checks it with pyright)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from findex.index import Index, Posting

type F64 = npt.NDArray[np.float64]
type I32 = npt.NDArray[np.int32]


class Scorer(Protocol):
    def __call__(self, term: str, posting: Posting, index: Index) -> float: ...


@runtime_checkable
class VectorScorer(Scorer, Protocol):
    """Lab 8: the same formula for ALL postings of a term at once (array expression)."""

    def weights(self, term: str, doc_ids: I32, tfs: I32, index: Index) -> F64: ...


@dataclass(frozen=True, slots=True)
class TfIdf:
    """tf · log(N / df): linear in tf, no length normalization."""

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return posting.tf * math.log(index.num_docs / index.df(term))

    def weights(self, term: str, doc_ids: I32, tfs: I32, index: Index) -> F64:
        return np.multiply(tfs, math.log(index.num_docs / index.df(term)), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class BM25:
    """Okapi BM25. k1: how fast tf saturates. b: how much long documents are penalized
    (b = 0 -> no length normalization, b = 1 -> full)."""

    k1: float = 1.5
    b: float = 0.75

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        n, df = index.num_docs, index.df(term)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        norm = self.k1 * (
            1 - self.b + self.b * index.doc_length(posting.doc_id) / index.avg_doc_length
        )
        return idf * posting.tf * (self.k1 + 1) / (posting.tf + norm)

    def weights(self, term: str, doc_ids: I32, tfs: I32, index: Index) -> F64:
        # identical formula and operation order as __call__ -> identical floats
        n, df = index.num_docs, index.df(term)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        dl = index.doc_len_array[doc_ids]  # fancy indexing: lengths of all n docs at once
        norm = self.k1 * (1 - self.b + self.b * dl / index.avg_doc_length)
        return idf * tfs * (self.k1 + 1) / (tfs + norm)


type ScorerName = Literal["bm25", "tfidf"]
SCORERS: dict[ScorerName, Scorer] = {"bm25": BM25(), "tfidf": TfIdf()}
