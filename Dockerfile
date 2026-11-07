# syntax=docker/dockerfile:1
# Two stages (uv Docker guide): build the venv + index with uv, run in a slim image as non-root.

# ---------- builder ----------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# 1) dependencies only, exactly as pinned in uv.lock (cached layer while the code changes)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
# 2) the project itself, installed as a package (not editable)
COPY src ./src
RUN uv sync --locked --no-dev --no-editable
# 3) corpus + index + embeddings baked into the image (simple, fine for a portfolio demo)
COPY scripts/get_corpus.sh ./scripts/
COPY models/wordvec.npz ./models/
RUN ./scripts/get_corpus.sh \
    && .venv/bin/findex index data/ --out index.bin --positions --executor serial \
    && .venv/bin/findex embed index.bin --out embeddings.npz --model models/wordvec.npz

# ---------- runtime ----------
FROM python:3.13-slim-bookworm
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/data /app/data
COPY --from=builder --chown=app:app /app/index.bin /app/index.bin
COPY --from=builder --chown=app:app /app/embeddings.npz /app/embeddings.npz
ENV PATH="/app/.venv/bin:$PATH" \
    INDEX_PATH=/app/index.bin \
    EMBEDDINGS_PATH=/app/embeddings.npz \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health')"
# $PORT is set by Render/Fly/Railway; WEB_CONCURRENCY = number of uvicorn workers
CMD ["sh", "-c", "exec findex serve --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-1}"]
