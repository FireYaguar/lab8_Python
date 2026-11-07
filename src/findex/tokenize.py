"""Streaming tokenizer.

Policy (documented in README too):
- Unicode NFC normalization first, then str.casefold() (not lower()).
- A token is a run of word characters (\\w: letters, digits, underscore, any script).
- Apostrophes INSIDE a word are kept: "don't", "п'ять" are one token.
  Typographic apostrophes (’ U+2019, ʼ U+02BC) are mapped to ASCII ' first.
- Hyphens split words: "well-known" -> "well", "known".
- Digits are kept as tokens: "Python 3.12" -> "python", "3", "12".
- Tokens made only of underscores (rst/markdown decoration "____") are dropped;
  identifiers like "__init__" are kept.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

TOKEN_RE = re.compile(r"\w+(?:'\w+)*")


def fix_apostrophes(text: str) -> str:
    """’ (U+2019) and ʼ (U+02BC) -> ASCII '. Two str.replace (C loops): str.translate with a
    dict table walks the text much slower (Lab 8 profile)."""
    return text.replace("\u2019", "'").replace("\u02bc", "'")


def normalize(text: str) -> str:
    return fix_apostrophes(unicodedata.normalize("NFC", text)).casefold()


def iter_spans(text: str) -> Iterator[tuple[str, int, int]]:
    """Yield (token, start, end); offsets point into unicodedata.normalize("NFC", text).

    The regex runs on the NFC text *before* casefold (casefold can change string length,
    e.g. "ß" -> "ss"), so the offsets are usable for snippets in Lab 3.
    """
    nfc = fix_apostrophes(unicodedata.normalize("NFC", text))  # 1:1 char mapping
    for m in TOKEN_RE.finditer(nfc):  # finditer: lazy, no list of matches
        tok = m.group().casefold()
        if tok.strip("_"):
            yield tok, m.start(), m.end()


def tokenize(text: str) -> Iterator[str]:
    for tok, _, _ in iter_spans(text):
        yield tok
