"""Record docs/demo.gif: a real search in the web UI (Playwright + system Chrome, Pillow).

findex serve --index index.bin --embeddings embeddings.npz --port 8000 &
uv run --with playwright --with pillow python scripts/make_demo_gif.py http://127.0.0.1:8000
"""

from __future__ import annotations

import io
import sys

from PIL import Image
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
frames: list[tuple[Image.Image, int]] = []


def shot(page, ms: int) -> None:  # type: ignore[no-untyped-def]
    img = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
    frames.append((img.resize((img.width * 3 // 4, img.height * 3 // 4)), ms))


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 960, "height": 640})
    page.goto(BASE + "/")
    shot(page, 900)
    box = page.locator("input[name=q]")
    for i, ch in enumerate('"event loop" NOT thread'):
        box.type(ch)
        if i % 3 == 2:
            shot(page, 120)
    shot(page, 500)
    page.locator("button").click()
    page.wait_for_load_state()
    shot(page, 2600)
    page.mouse.wheel(0, 420)
    page.wait_for_timeout(200)
    shot(page, 1800)
    # semantic/hybrid: a paraphrase with none of the document's key words
    page.goto(BASE + "/")
    page.select_option("select[name=mode]", "hybrid")
    for i, ch in enumerate("turn objects into bytes and restore them later"):
        page.locator("input[name=q]").type(ch)
        if i % 5 == 4:
            shot(page, 100)
    page.locator("button").click()
    page.wait_for_load_state()
    shot(page, 2800)
    page.locator("a.title").first.click()
    page.wait_for_load_state()
    shot(page, 2200)
    browser.close()

imgs = [f for f, _ in frames]
imgs[0].save(
    "docs/demo.gif",
    save_all=True,
    append_images=imgs[1:],
    optimize=True,
    duration=[ms for _, ms in frames],
    loop=0,
)
print(f"{len(frames)} frames -> docs/demo.gif")
