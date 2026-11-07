"""Postings as NumPy arrays (Lab 8), CSR layout — one contiguous block for the whole index:

    terms      ["a", "b", ...]                 term i
    offsets    int64[T+1]   postings of term i are [offsets[i], offsets[i+1])
    doc_ids    int32[P]     sorted inside each term
    tfs        int32[P]
    pos_offsets int64[P+1], positions int32[...]   (only with --positions)

Pickling this is a handful of buffers instead of 331k Python objects, and scoring is
array expressions over slices (views, no copies). Posting objects are built lazily, per
term, only where code really needs them (phrase queries, Mapping access).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from findex.index import Posting

type I32 = npt.NDArray[np.int32]
type I64 = npt.NDArray[np.int64]
type RawPosting = tuple[int, int, tuple[int, ...]]  # (doc_id, tf, positions)


@dataclass(slots=True, eq=False)
class PostingArrays:
    terms: list[str]
    offsets: I64
    doc_ids: I32
    tfs: I32
    pos_offsets: I64 | None = None
    positions: I32 | None = None

    @classmethod
    def from_raw(cls, raw: Mapping[str, list[RawPosting]], has_positions: bool) -> PostingArrays:
        terms = list(raw)
        lengths = np.fromiter((len(raw[t]) for t in terms), np.int64, count=len(terms))
        offsets = np.zeros(len(terms) + 1, np.int64)
        np.cumsum(lengths, out=offsets[1:])
        total = int(offsets[-1])
        flat = list(chain.from_iterable(raw[t] for t in terms))
        doc_ids = np.fromiter((p[0] for p in flat), np.int32, count=total)
        tfs = np.fromiter((p[1] for p in flat), np.int32, count=total)
        if not has_positions:
            return cls(terms, offsets, doc_ids, tfs)
        plen = np.fromiter((len(p[2]) for p in flat), np.int64, count=total)
        pos_offsets = np.zeros(total + 1, np.int64)
        np.cumsum(plen, out=pos_offsets[1:])
        positions = np.fromiter(
            chain.from_iterable(p[2] for p in flat), np.int32, count=int(pos_offsets[-1])
        )
        return cls(terms, offsets, doc_ids, tfs, pos_offsets, positions)

    @classmethod
    def from_postings(
        cls, postings: Mapping[str, list[Posting]], has_positions: bool
    ) -> PostingArrays:
        raw = {t: [(p.doc_id, p.tf, p.positions) for p in pl] for t, pl in postings.items()}
        return cls.from_raw(raw, has_positions)

    def state(self) -> dict[str, Any]:
        return {
            "terms": self.terms,
            "offsets": self.offsets,
            "doc_ids": self.doc_ids,
            "tfs": self.tfs,
            "pos_offsets": self.pos_offsets,
            "positions": self.positions,
        }


class LazyPostings(Mapping[str, "list[Posting]"]):
    """term -> list[Posting], materialized on first access and cached."""

    def __init__(self, arrays: PostingArrays, cache: dict[str, list[Posting]] | None = None):
        self.arrays = arrays
        self.term_index: dict[str, int] = {t: i for i, t in enumerate(arrays.terms)}
        self._cache: dict[str, list[Posting]] = cache if cache is not None else {}

    def span(self, term: str) -> tuple[int, int] | None:
        i = self.term_index.get(term)
        if i is None:
            return None
        return int(self.arrays.offsets[i]), int(self.arrays.offsets[i + 1])

    def __getitem__(self, term: str) -> list[Posting]:
        cached = self._cache.get(term)
        if cached is not None:
            return cached
        span = self.span(term)
        if span is None:
            raise KeyError(term)
        from findex.index import Posting

        s, e = span
        a = self.arrays
        ids, tfs = a.doc_ids[s:e].tolist(), a.tfs[s:e].tolist()
        if a.pos_offsets is not None and a.positions is not None:
            po, pos = a.pos_offsets, a.positions
            plist = [
                Posting(d, tf, tuple(pos[po[j] : po[j + 1]].tolist()))
                for j, (d, tf) in zip(range(s, e), zip(ids, tfs))
            ]
        else:
            plist = [Posting(d, tf) for d, tf in zip(ids, tfs)]
        self._cache[term] = plist
        return plist

    def __contains__(self, term: object) -> bool:
        return term in self.term_index

    def __iter__(self) -> Iterator[str]:
        return iter(self.arrays.terms)

    def __len__(self) -> int:
        return len(self.arrays.terms)
