"""Semantic search (Lab 8): documents as vectors of meaning, compared by cosine similarity.

Embedding model: pre-trained 300-d word vectors (spaCy en_core_web_md, exported to
models/wordvec.npz by scripts/export_wordvec.py). A text's embedding = IDF-weighted mean of
its word vectors, L2-normalized. (A sentence-transformer such as all-MiniLM-L6-v2 is the
better model; it could not be downloaded in the build environment — see README.)

Documents are split into chunks of CHUNK_TOKENS tokens; a document's score is its best
chunk's score. E (n_chunks x d, float32, unit rows) @ q gives all similarities at once.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from findex.index import Index
from findex.timing import timed
from findex.tokenize import iter_spans, tokenize

CHUNK_TOKENS = 200
MIN_CHUNKS_FOR_SIF = 50  # the common direction needs many texts to be estimated
type F32 = npt.NDArray[np.float32]
type I32 = npt.NDArray[np.int32]


def _pack_words(words: npt.NDArray[np.str_] | list[str]) -> npt.NDArray[np.uint8]:
    """Many short strings as ONE utf-8 buffer ("\\n"-joined): a fixed-width '<U30' array
    would cost 120 bytes per word."""
    return np.frombuffer("\n".join(list(words)).encode("utf-8"), np.uint8)


def _unpack_words(buf: npt.NDArray[np.generic]) -> npt.NDArray[np.str_]:
    if buf.dtype.kind == "U":  # older format
        return buf.astype(np.str_)
    return np.array(buf.tobytes().decode("utf-8").split("\n"))


@dataclass(slots=True, eq=False)
class WordVectors:
    words: npt.NDArray[np.str_]
    rows: I32
    vectors: F32  # one row per distinct vector; several words may share a row

    @classmethod
    def load(cls, path: Path) -> WordVectors:
        z = np.load(path)
        return cls(_unpack_words(z["words"]), z["rows"], z["vectors"].astype(np.float32))

    def lookup(self) -> dict[str, int]:
        return dict(zip(self.words.tolist(), self.rows.tolist()))


class Embedder:
    """IDF-weighted mean of word vectors; IDF comes from the keyword index (rare words count
    more, "the" almost nothing). Unknown words are skipped."""

    def __init__(
        self,
        wv: WordVectors,
        idf: dict[str, float],
        default_idf: float,
        mean: F32 | None = None,
        pc: F32 | None = None,
    ) -> None:
        self.wv, self.row, self.idf, self.default_idf = wv, wv.lookup(), idf, default_idf
        d = wv.vectors.shape[1]
        # SIF post-processing: averaged word vectors share one big "common" direction that
        # makes every text look alike (cosine ~0.8 for anything). Remove the corpus mean and
        # the first principal component; the same transform is applied to queries.
        self.mean = mean if mean is not None else np.zeros(d, np.float32)
        self.pc = pc if pc is not None else np.zeros(d, np.float32)

    def raw(self, tokens: list[str]) -> F32:
        """IDF-weighted mean of word vectors, before post-processing."""
        counts = Counter(t for t in tokens if t in self.row)
        out = np.zeros(self.wv.vectors.shape[1], np.float32)
        if not counts:
            return out
        rows = np.fromiter((self.row[t] for t in counts), np.int64, count=len(counts))
        weights = np.fromiter(
            (n * self.idf.get(t, self.default_idf) for t, n in counts.items()),
            np.float32,
            count=len(counts),
        )
        out = weights @ self.wv.vectors[rows]  # (n,) @ (n, d) -> (d,): one BLAS call
        return out / max(float(weights.sum()), 1e-9)

    def finish(self, v: F32) -> F32:
        if not v.any():
            return v
        v = (v - self.mean).astype(np.float32)
        v = (v - (v @ self.pc) * self.pc).astype(np.float32)
        norm = float(np.linalg.norm(v))
        return (v / norm).astype(np.float32) if norm > 0 else v

    def embed_tokens(self, tokens: list[str]) -> F32:
        return self.finish(self.raw(tokens))

    def embed(self, text: str) -> F32:
        return self.embed_tokens(list(tokenize(text)))


@dataclass(slots=True, eq=False)
class SemanticIndex:
    E: F32  # (n_chunks, d) unit vectors
    chunk_doc: I32  # chunk -> doc_id
    chunk_span: I32  # (n_chunks, 2) char offsets in the NFC text of the document
    embedder: Embedder

    def save(self, path: Path) -> None:
        emb = self.embedder
        wv = emb.wv
        np.savez(
            path,
            E=self.E.astype(np.float16),  # half the size; cosine needs no more precision
            chunk_doc=self.chunk_doc,
            chunk_span=self.chunk_span,
            idf_words=_pack_words(list(emb.idf)),
            idf=np.array(list(emb.idf.values()), np.float32),
            default_idf=np.float32(emb.default_idf),
            mean=emb.mean,
            pc=emb.pc,
            words=_pack_words(wv.words),
            rows=wv.rows,
            vectors=wv.vectors.astype(np.float16),
        )

    @classmethod
    @timed
    def load(cls, path: Path) -> SemanticIndex:
        z = np.load(path)
        wv = WordVectors(_unpack_words(z["words"]), z["rows"], z["vectors"].astype(np.float32))
        idf = dict(zip(_unpack_words(z["idf_words"]).tolist(), z["idf"].tolist()))
        emb = Embedder(wv, idf, float(z["default_idf"]), z["mean"], z["pc"])
        return cls(z["E"].astype(np.float32), z["chunk_doc"], z["chunk_span"], emb)

    def rank(self, query: str, k: int) -> list[tuple[int, float, int]]:
        """Top-k documents as (doc_id, cosine of best chunk, best chunk index)."""
        q = self.embedder.embed(query)
        if not q.any():
            return []
        sims = self.E @ q  # all chunk similarities in one matrix-vector product
        # chunks are stored grouped by document (chunk_doc is sorted), so the best chunk per
        # document is a segmented max: np.maximum.reduceat over the group starts
        docs, starts = np.unique(self.chunk_doc, return_index=True)
        best = np.maximum.reduceat(sims, starts)
        k = min(k, len(docs))
        top = np.argpartition(best, -k)[-k:]
        top = top[np.lexsort((docs[top], -best[top]))]
        ends = np.append(starts[1:], len(sims))
        out: list[tuple[int, float, int]] = []
        for i in top.tolist():
            s, e = int(starts[i]), int(ends[i])
            out.append((int(docs[i]), float(best[i]), s + int(np.argmax(sims[s:e]))))
        return out


def chunks_of(text: str, size: int = CHUNK_TOKENS) -> list[tuple[list[str], int, int]]:
    """(tokens, start, end) windows of `size` tokens; offsets into the NFC text."""
    spans = list(iter_spans(text))
    return [
        (
            [t for t, _, _ in spans[i : i + size]],
            spans[i][1],
            spans[min(i + size, len(spans)) - 1][2],
        )
        for i in range(0, len(spans), size)
    ]


@timed
def build_semantic(
    index: Index, wv: WordVectors, read_text: Callable[[str], str] | None = None
) -> SemanticIndex:
    """Chunk every document of `index`, embed the chunks, stack them into E."""
    from findex.search import read_text as default_read

    reader = default_read if read_text is None else read_text
    n = index.num_docs
    idf = {t: math.log(n / index.df(t)) for t in index}
    embedder = Embedder(wv, idf, default_idf=math.log(n))
    rows: list[F32] = []
    chunk_doc: list[int] = []
    spans: list[tuple[int, int]] = []
    for doc_id, meta in sorted(index.doc_meta.items()):
        text = reader(meta.path)
        for tokens, start, end in chunks_of(text):
            vec = embedder.raw(tokens)
            if vec.any():
                rows.append(vec)
                chunk_doc.append(doc_id)
                spans.append((start, end))
    raw = (
        np.vstack(rows).astype(np.float32)
        if rows
        else np.zeros((0, wv.vectors.shape[1]), np.float32)
    )
    if len(raw) >= MIN_CHUNKS_FOR_SIF:  # fit SIF on the corpus, then apply it to every chunk
        embedder.mean = raw.mean(axis=0)
        embedder.pc = np.linalg.svd(raw - embedder.mean, full_matrices=False)[2][0].astype(
            np.float32
        )
    E = np.vstack([embedder.finish(v) for v in raw]) if len(raw) else raw
    return SemanticIndex(
        E, np.array(chunk_doc, np.int32), np.array(spans, np.int32).reshape(-1, 2), embedder
    )


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank). No score scaling."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
