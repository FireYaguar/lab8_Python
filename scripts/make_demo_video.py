"""Record docs/demo.mp4 (~2.5 min, no voice): the real UI driven by Playwright + system Chrome,
captions on screen, flame graphs and result tables as slides. Frames are screenshots with
explicit durations, assembled by the system ffmpeg (concat demuxer).

    findex serve --index index.bin --embeddings embeddings.npz --port 8000 &
    uv run --no-project --with playwright python scripts/make_demo_video.py http://127.0.0.1:8000
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parent.parent
W, H = 1280, 720
FRAMES: list[tuple[Path, float]] = []
TMP = Path(tempfile.mkdtemp())


def snap(page: Page, ms: int) -> None:
    path = TMP / f"{len(FRAMES):04d}.png"
    page.screenshot(path=str(path))
    FRAMES.append((path, ms / 1000))


CAPTION_JS = """
(text) => {
  let el = document.getElementById('__cap');
  if (!el) {
    el = document.createElement('div'); el.id = '__cap';
    el.style.cssText = 'position:fixed;left:50%;bottom:28px;transform:translateX(-50%);'
      + 'max-width:1100px;background:rgba(20,24,33,.92);color:#fff;font:500 22px/1.4 system-ui,'
      + 'sans-serif;padding:14px 26px;border-radius:12px;z-index:99999;text-align:center;'
      + 'box-shadow:0 6px 24px rgba(0,0,0,.25)';
    document.body.appendChild(el);
  }
  el.textContent = text;
}
"""

SLIDE_CSS = """
body{margin:0;height:100vh;display:flex;flex-direction:column;justify-content:center;
 padding:0 90px;box-sizing:border-box;font:20px/1.5 system-ui,sans-serif;color:#1d2330;
 background:#fafaf8} h1{font-size:44px;margin:0 0 6px} h1 span{color:#2f5bd3}
 h2{font-size:30px;margin:0 0 18px} p{color:#555;margin:6px 0}
 table{border-collapse:collapse;font-size:19px;margin-top:10px} td,th{padding:7px 16px;
 border-bottom:1px solid #e4e4df;text-align:left} th{color:#69707d;font-weight:600}
 b{color:#2f5bd3} .big{font-size:56px;font-weight:700;color:#2f5bd3}
"""


PACE = 1.4  # reading time multiplier for captions


def caption(page: Page, text: str, ms: int) -> None:
    page.evaluate(CAPTION_JS, text)
    snap(page, int(ms * PACE))


def slide(page: Page, html: str, text: str, ms: int) -> None:
    page.set_content(f"<html><head><style>{SLIDE_CSS}</style></head><body>{html}</body></html>")
    caption(page, text, ms)


def show_svg(page: Page, svg: Path) -> None:
    """Flame graphs are SVG documents (no HTML body for the caption): wrap them in a page."""
    html = TMP / f"{svg.stem}.html"
    html.write_text(
        f"<html><body style='margin:0;background:#fff'><img src='{svg.as_uri()}' "
        "style='width:100%'></body></html>"
    )
    page.goto(html.as_uri())


def type_slow(page: Page, selector: str, text: str) -> None:
    page.locator(selector).fill("")
    for i, ch in enumerate(text):
        page.locator(selector).type(ch)
        if i % 2 == 1 or i == len(text) - 1:
            snap(page, 110)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": W, "height": H})

        slide(
            page,
            "<h1>find<span>ex</span></h1><p>A full-text + semantic search engine over the "
            "Python documentation, built from scratch in 8 labs.</p><p>560 documents · 15 MB · "
            "1.8M tokens · BM25 / TF-IDF · embeddings · FastAPI</p>",
            "findex — the search engine from the Python course (Labs 1–8)",
            5000,
        )

        # 1. keyword search with the query language
        page.goto(BASE + "/")
        caption(
            page,
            "The web UI (Lab 7): one search box, BM25 or TF-IDF, keyword / hybrid / semantic",
            3500,
        )
        type_slow(page, "input[name=q]", '"event loop" NOT thread')
        caption(
            page,
            "Query language (Lab 3): an exact phrase in quotes, NOT, AND, OR, parentheses",
            2500,
        )
        page.locator("button").click()
        page.wait_for_load_state()
        caption(
            page,
            "Ranked by BM25 — snippets ±80 characters around the best match, terms highlighted",
            5000,
        )
        page.mouse.wheel(0, 500)
        page.wait_for_timeout(300)
        caption(page, "The score next to each title; pagination at the bottom", 3500)
        page.mouse.wheel(0, 1600)
        page.wait_for_timeout(600)
        page.locator("text=next →").click()
        page.wait_for_load_state()
        caption(page, "Page 2 of the results", 2500)
        page.locator("a.title").first.click()
        page.wait_for_load_state()
        caption(page, "The document view: the full source text of the page", 3500)

        # 2. paraphrase: keyword vs hybrid vs semantic
        q = "turn objects into bytes and restore them later"
        page.goto(BASE + "/")
        type_slow(page, "input[name=q]", q)
        caption(
            page,
            "A paraphrase: none of the key words of the pickle docs appear in this query",
            3000,
        )
        page.locator("button").click()
        page.wait_for_load_state()
        caption(
            page,
            "Keyword mode: implicit AND finds long pages that merely contain these words",
            4500,
        )
        page.goto(BASE + "/?q=" + q.replace(" ", "+") + "&mode=semantic")
        caption(
            page,
            "Semantic mode (Lab 8): chunk embeddings, cosine similarity — pickle is found",
            5000,
        )
        page.goto(BASE + "/?q=" + q.replace(" ", "+") + "&mode=hybrid")
        caption(page, "Hybrid: BM25 and semantic rankings fused with Reciprocal Rank Fusion", 4500)

        # 3. API
        page.goto(BASE + "/openapi.json")
        caption(
            page,
            "The JSON API: an OpenAPI schema generated from type hints + Pydantic (Swagger UI at /docs)",
            5000,
        )
        page.goto(BASE + "/search?q=%22event%20loop%22&k=2")
        caption(page, "GET /search returns JSON: total, pages, took_ms and the results", 4000)
        page.goto(BASE + "/health")
        caption(
            page, "/health is the platform health check (503 while the index is not loaded)", 3000
        )

        # 4. performance
        show_svg(page, ROOT / "docs/flame_search_before.svg")
        caption(
            page,
            "Lab 8, measure first: py-spy flame graph of search BEFORE — snippets were 99%",
            5500,
        )
        show_svg(page, ROOT / "docs/flame_search_after.svg")
        caption(page, "AFTER: one regex per snippet instead of re-tokenizing whole documents", 4500)
        slide(
            page,
            "<h2>Before → after (Lab 8)</h2><table><tr><th></th><th>before</th><th>after</th>"
            "<th></th></tr><tr><td>load index</td><td>890 ms</td><td>18 ms</td><td><b>49×</b></td></tr>"
            "<tr><td>search, 3 terms</td><td>76 ms</td><td>25 ms</td><td><b>3×</b></td></tr>"
            "<tr><td>build index</td><td>2.73 s</td><td>1.71 s</td><td><b>1.6×</b></td></tr>"
            "<tr><td>throughput, 4 workers</td><td>118 req/s</td><td>232 req/s</td><td><b>2×</b>"
            "</td></tr></table><p>Postings as NumPy int32 arrays (CSR) instead of 331k Python objects.</p>",
            "Where the time really went: unpickling objects and snippets — not the scorer",
            6500,
        )
        slide(
            page,
            "<h2>NumPy BM25 vs the Lab 3 loop</h2><table><tr><th>query</th><th>postings</th>"
            "<th>Lab 3</th><th>NumPy</th><th></th></tr><tr><td>the</td><td>552</td><td>0.650 ms</td>"
            "<td>0.051 ms</td><td><b>12.7×</b></td></tr><tr><td>python OR async OR queue</td><td>568</td>"
            "<td>0.585 ms</td><td>0.079 ms</td><td><b>7.4×</b></td></tr><tr><td>winsound</td><td>5</td>"
            "<td>0.007 ms</td><td>0.016 ms</td><td>0.4×</td></tr></table>"
            "<p>Identical rankings (tested, incl. ties). Short lists: call overhead wins.</p>",
            "Vectorized scoring: big win on long postings lists, a loss on a 5-element list",
            6500,
        )
        slide(
            page,
            "<h2>precision@5</h2><table><tr><th></th><th>keyword</th><th>semantic</th>"
            "<th>hybrid</th></tr><tr><td>10 original queries</td><td><b>0.42</b></td><td>0.32</td>"
            "<td>0.38</td></tr><tr><td>5 paraphrases</td><td>0.04</td><td><b>0.16</b></td><td>0.12</td>"
            "</tr></table><p>Embeddings: pre-trained word vectors (spaCy md) — a transformer would do better.</p>",
            "Keyword wins on exact terms, semantic on paraphrases — limitations stated in the README",
            6500,
        )
        slide(
            page,
            "<h2>One bug that taught me something</h2><p>Lab 5, Python 3.13t (no GIL): 4 threads "
            "doing <code>shared[token] += 1</code> on one Counter lost <b>12–15 thousand</b> updates. "
            "Under the GIL: zero.</p><p>The GIL protects the interpreter, not my invariants. Fix: a "
            "private Counter per thread, merged at the end — exactly how build_partial / merge work.</p>",
            "The GIL was hiding a race condition — free-threaded Python exposed it",
            7500,
        )
        slide(
            page,
            "<h1>find<span>ex</span> v1.0.0</h1><p>generators · inverted index · BM25 · typed CLI · "
            "processes vs GIL · asyncio crawler · FastAPI · NumPy · embeddings</p>"
            "<p>Code, measurements and limitations: see the README.</p>",
            "Thanks for watching",
            4000,
        )

        browser.close()
    listing = TMP / "frames.txt"
    lines = [f"file '{f}'\nduration {d:.3f}" for f, d in FRAMES]
    listing.write_text("\n".join([*lines, f"file '{FRAMES[-1][0]}'"]) + "\n")
    out = ROOT / "docs/demo.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing)]
    cmd += ["-vf", "fps=25,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28"]
    subprocess.run([*cmd, "-movflags", "+faststart", "-an", str(out)], check=True)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"-> {out} ({out.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
