"""Export the pre-trained word vectors of spaCy's en_core_web_md into models/wordvec.npz.

    uv run --with spacy --with <en_core_web_md wheel> python scripts/export_wordvec.py models/wordvec.npz

Keeps lowercase word keys only (our tokenizer casefolds), shares rows with spaCy's table and
stores it as float16: words.npy (str), rows.npy (int32 word -> row), vectors.npy (float16).
At runtime findex needs only NumPy, not spaCy.
"""

from __future__ import annotations

import sys

import numpy as np
import spacy  # pyright: ignore  # export-time only dependency


def main() -> None:
    out = sys.argv[1]
    nlp = spacy.load(
        "en_core_web_md",
        exclude=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler", "senter", "tok2vec"],
    )
    vectors = nlp.vocab.vectors
    words: list[str] = []
    rows: list[int] = []
    for key, row in vectors.key2row.items():
        word = nlp.vocab.strings[key]
        if word == word.lower() and any(c.isalpha() for c in word) and len(word) < 30:
            words.append(word)
            rows.append(row)
    np.savez_compressed(
        out,
        words=np.frombuffer("\n".join(words).encode("utf-8"), np.uint8),  # one buffer, not <U30
        rows=np.array(rows, dtype=np.int32),
        vectors=vectors.data.astype(np.float16),
    )
    print(
        f"{len(words):,} words, {vectors.data.shape[0]:,} x {vectors.data.shape[1]} vectors -> {out}"
    )


if __name__ == "__main__":
    main()
