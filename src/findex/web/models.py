"""Pydantic models = the HTTP boundary. Inside, the Lab 2-3 dataclasses stay as they are."""

from __future__ import annotations

from pydantic import BaseModel, Field

from findex.index import DocMeta
from findex.search import SearchResult


class Hit(BaseModel):
    doc_id: int
    title: str
    score: float
    snippet: str = Field(description="±80 chars around the best match; terms marked as **term**")
    url: str = Field(description="link to this document in the API")

    @classmethod
    def from_result(cls, r: SearchResult) -> Hit:
        return cls(
            doc_id=r.doc_id,
            title=r.title,
            score=round(r.score, 4),
            snippet=r.snippet,
            url=f"/docs/{r.doc_id}",
        )


class SearchResponse(BaseModel):
    query: str
    scorer: str
    mode: str
    total: int = Field(description="all documents matching the Boolean query")
    page: int
    k: int
    pages: int
    took_ms: float
    results: list[Hit]


class DocumentOut(BaseModel):
    doc_id: int
    title: str
    path: str
    length: int = Field(description="tokens")
    text: str

    @classmethod
    def from_meta(cls, meta: DocMeta, length: int, text: str) -> DocumentOut:
        return cls(doc_id=meta.doc_id, title=meta.title, path=meta.path, length=length, text=text)


class StatsOut(BaseModel):
    documents: int
    terms: int
    postings: int
    avg_doc_length: float
    positions: bool
    index_path: str
    index_bytes: int
    uptime_s: float


class Health(BaseModel):
    status: str
    documents: int = 0
