"""FastAPI app.

    findex serve --index index.bin [--port 8000] [--workers N]
    uvicorn findex.web.app:app          # INDEX_PATH from the environment

Routes: GET /search, /docs/{doc_id}, /stats, /health (JSON) and GET /, /doc/{doc_id} (HTML).
OpenAPI UI: /docs (generated from the type hints and Pydantic models).
"""

from __future__ import annotations

import logging
import math
import re
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup

from findex import __version__
from findex.index import Index
from findex.query import QueryError
from findex.search import match, read_text, search
from findex.semantic import SemanticIndex
from findex.store import open_index
from findex.web.models import DocumentOut, Health, Hit, SearchResponse, StatsOut
from findex.web.settings import Settings

log = logging.getLogger("findex.web")
HERE = Path(__file__).parent
templates = Jinja2Templates(directory=HERE / "templates")


def highlight(snippet: str) -> Markup:
    """Escape the snippet, then turn **term** into <mark>term</mark>."""
    return Markup(re.sub(r"\*\*(.+?)\*\*", r"<mark>\1</mark>", escape(snippet)))


templates.env.filters["highlight"] = highlight


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Startup: load the index ONCE (a missing file fails the start, not the first query).
    Shutdown: open_index's finally releases it."""
    settings: Settings = app.state.settings
    logging.getLogger("findex").setLevel(settings.log_level)
    t0 = time.perf_counter()
    with open_index(settings.index_path) as index:
        app.state.index = index
        app.state.semantic = None
        if settings.embeddings_path is not None:
            app.state.semantic = SemanticIndex.load(settings.embeddings_path)
            log.info(
                "loaded %d chunk embeddings from %s",
                len(app.state.semantic.E),
                settings.embeddings_path,
            )
        app.state.started = time.time()
        log.info(
            "loaded %r from %s in %.2f s", index, settings.index_path, time.perf_counter() - t0
        )
        yield
        app.state.index = None
        app.state.semantic = None


# ---------- dependencies ----------
def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_index(request: Request) -> Index:
    index: Index | None = getattr(request.app.state, "index", None)
    if index is None:
        raise HTTPException(503, "index not loaded")
    return index


IndexDep = Annotated[Index, Depends(get_index)]
Q = Annotated[str, Query(min_length=1, max_length=200, description='e.g. "event loop" NOT thread')]
K = Annotated[int, Query(ge=1, le=50, description="results per page")]
PageNo = Annotated[int, Query(ge=1, le=100)]
ScorerParam = Literal["bm25", "tfidf"]
ModeParam = Literal["keyword", "semantic", "hybrid"]


def get_semantic(request: Request) -> SemanticIndex | None:
    return getattr(request.app.state, "semantic", None)


SemanticDep = Annotated[SemanticIndex | None, Depends(get_semantic)]


def run_search(
    index: Index,
    q: str,
    k: int,
    scorer: ScorerParam,
    page: int,
    mode: ModeParam = "keyword",
    semantic: SemanticIndex | None = None,
) -> SearchResponse:
    """Plain (blocking, CPU-bound) function shared by the JSON and HTML routes."""
    t0 = time.perf_counter()
    try:
        hits_all = search(index, q, scorer, k=k * page, mode=mode, semantic=semantic)
        total = len(match(index, q)) if mode == "keyword" else len(hits_all)
        hits = hits_all[k * (page - 1) :]
    except QueryError as e:
        raise HTTPException(400, f"bad query: {e}") from None
    return SearchResponse(
        query=q,
        scorer=scorer,
        mode=mode,
        total=total,
        page=page,
        k=k,
        pages=max(1, math.ceil(total / k)),
        took_ms=round((time.perf_counter() - t0) * 1000, 2),
        results=[Hit.from_result(r) for r in hits],
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="findex",
        version=__version__,
        lifespan=lifespan,
        redoc_url=None,  # Swagger UI stays at /docs; /docs/{doc_id} is a different path
        description="Full-text search over the Python documentation (BM25 / TF-IDF).",
    )
    app.state.settings = settings or Settings()
    app.state.index = None
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        rid = getattr(request.state, "request_id", "-")
        log.exception("request %s %s %s failed", rid, request.method, request.url.path)
        return JSONResponse(
            {"detail": "internal server error", "request_id": rid},
            status_code=500,
            headers={"X-Request-ID": rid},
        )

    # ---------- JSON API ----------
    # `def`, not `async def`: scoring is CPU work with no await. FastAPI runs def-routes in
    # a thread pool, so the event loop stays free (see README, Lab 7 measurements).
    @app.get("/search", response_model=SearchResponse)
    def search_route(
        index: IndexDep,
        semantic: SemanticDep,
        q: Q,
        k: K = 10,
        scorer: ScorerParam = "bm25",
        page: PageNo = 1,
        mode: ModeParam = "keyword",
    ) -> SearchResponse:
        return run_search(index, q, k, scorer, page, mode, semantic)

    @app.get("/docs/{doc_id}", response_model=DocumentOut)
    def document(index: IndexDep, doc_id: int) -> DocumentOut:
        meta = index.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(404, f"document {doc_id} not found")
        return DocumentOut.from_meta(meta, index.doc_length(doc_id), read_text(meta.path))

    @app.get("/stats", response_model=StatsOut)
    def stats(index: IndexDep, settings: Annotated[Settings, Depends(get_settings)]) -> StatsOut:
        path = settings.index_path
        return StatsOut(
            documents=index.num_docs,
            terms=len(index),
            postings=index.num_postings,
            avg_doc_length=round(index.avg_doc_length, 1),
            positions=index.has_positions,
            index_path=str(path),
            index_bytes=path.stat().st_size if path.is_file() else 0,
            uptime_s=round(time.time() - getattr(app.state, "started", time.time()), 1),
        )

    # async def: a few attribute reads, never blocks — so it runs right on the event loop and
    # does not queue behind CPU-heavy /search calls in the thread pool.
    @app.get("/health", response_model=Health, responses={503: {"model": Health}})
    async def health(request: Request) -> Health | JSONResponse:
        index: Index | None = request.app.state.index
        if index is None:
            return JSONResponse({"status": "unavailable", "documents": 0}, status_code=503)
        return Health(status="ok", documents=index.num_docs)

    # ---------- HTML UI ----------
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(
        request: Request,
        index: IndexDep,
        semantic: SemanticDep,
        q: Annotated[str, Query(max_length=200)] = "",
        k: K = 10,
        scorer: ScorerParam = "bm25",
        page: PageNo = 1,
        mode: ModeParam = "keyword",
    ) -> HTMLResponse:
        result: SearchResponse | None = None
        error = ""
        if q.strip():
            try:
                result = run_search(index, q, k, scorer, page, mode, semantic)
            except HTTPException as e:
                error = str(e.detail)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "q": q,
                "k": k,
                "scorer": scorer,
                "mode": mode,
                "has_semantic": semantic is not None,
                "result": result,
                "error": error,
                "docs": index.num_docs,
            },
        )

    @app.get("/doc/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
    def doc_page(request: Request, index: IndexDep, doc_id: int, q: str = "") -> HTMLResponse:
        meta = index.doc_meta.get(doc_id)
        if meta is None:
            raise HTTPException(404, f"document {doc_id} not found")
        text = read_text(meta.path)
        return templates.TemplateResponse(
            request,
            "doc.html",
            {"meta": meta, "text": text, "q": q, "length": index.doc_length(doc_id)},
        )

    return app


app = create_app()
