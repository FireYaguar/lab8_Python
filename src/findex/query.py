"""Query language: a recursive-descent parser that builds a tree of nodes.

    python AND (async OR await) NOT java "event loop"

Grammar (implicit AND between adjacent items; AND binds tighter than OR):
    or_expr  := and_expr ("OR" and_expr)*
    and_expr := not_expr (["AND"] not_expr)*
    not_expr := "NOT" not_expr | atom
    atom     := WORD | "quoted phrase" | "(" or_expr ")"

Nodes overload & | ~ so trees can be built in code: Term("a") & ~Term("b").
evaluate(index) returns the sorted list of matching doc ids.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from findex.boolean import merge_and, merge_not, merge_or
from findex.tokenize import tokenize

if TYPE_CHECKING:
    from findex.index import Index


class QueryError(ValueError):
    pass


class Node(ABC):
    __slots__ = ()

    def __and__(self, other: Node) -> Node:
        return And(self, other)

    def __or__(self, other: Node) -> Node:
        return Or(self, other)

    def __invert__(self) -> Node:
        return Not(self)

    @abstractmethod
    def evaluate(self, index: Index) -> list[int]: ...

    @abstractmethod
    def terms(self) -> set[str]:
        """Positive terms (not under NOT) — the ones that contribute to the score."""


@dataclass(frozen=True, slots=True)
class Term(Node):
    term: str

    def evaluate(self, index: Index) -> list[int]:
        return index.doc_ids(self.term)

    def terms(self) -> set[str]:
        return {self.term}


@dataclass(frozen=True, slots=True)
class Phrase(Node):
    words: tuple[str, ...]

    def evaluate(self, index: Index) -> list[int]:
        if not index.has_positions:
            raise QueryError("phrase queries need an index built with --positions")
        if any(w not in index for w in self.words):
            return []  # a word of the phrase is not in the corpus at all: no document matches
        candidates = index.doc_ids(self.words[0])
        for w in self.words[1:]:
            candidates = merge_and(candidates, index.doc_ids(w))
        wanted = set(candidates)
        # doc_id -> positions, per word, only for candidate documents
        pos = [
            {p.doc_id: p.positions for p in index.postings[w] if p.doc_id in wanted}
            for w in self.words
        ]
        out: list[int] = []
        for d in candidates:
            later = [set(pos[i][d]) for i in range(1, len(self.words))]
            if any(all(p + i + 1 in s for i, s in enumerate(later)) for p in pos[0][d]):
                out.append(d)
        return out

    def terms(self) -> set[str]:
        return set(self.words)


@dataclass(frozen=True, slots=True)
class And(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> list[int]:
        if isinstance(self.right, Not):  # a AND NOT b: no need to build the complement
            return merge_not(self.left.evaluate(index), self.right.child.evaluate(index))
        if isinstance(self.left, Not):
            return merge_not(self.right.evaluate(index), self.left.child.evaluate(index))
        return merge_and(self.left.evaluate(index), self.right.evaluate(index))

    def terms(self) -> set[str]:
        return self.left.terms() | self.right.terms()


@dataclass(frozen=True, slots=True)
class Or(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> list[int]:
        return merge_or(self.left.evaluate(index), self.right.evaluate(index))

    def terms(self) -> set[str]:
        return self.left.terms() | self.right.terms()


@dataclass(frozen=True, slots=True)
class Not(Node):
    child: Node

    def evaluate(self, index: Index) -> list[int]:
        return merge_not(index.all_doc_ids, self.child.evaluate(index))

    def terms(self) -> set[str]:
        return set()


_LEX = re.compile(r'\s*(?:(\()|(\))|"([^"]*)"|([^\s()"]+))')


def _lex(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pos = 0
    text = text.rstrip()
    while pos < len(text):
        m = _LEX.match(text, pos)
        if m is None:
            raise QueryError(f"unbalanced quote at position {pos}")
        lpar, rpar, phrase, word = m.groups()
        if lpar:
            out.append(("(", lpar))
        elif rpar:
            out.append((")", rpar))
        elif phrase is not None:
            out.append(("PHRASE", phrase))
        elif word in ("AND", "OR", "NOT"):
            out.append((word, word))
        else:
            out.append(("WORD", word))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, text: str) -> None:
        self.toks = _lex(text)
        self.i = 0

    def peek(self) -> str | None:
        return self.toks[self.i][0] if self.i < len(self.toks) else None

    def take(self) -> tuple[str, str]:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def or_expr(self) -> Node:
        node = self.and_expr()
        while self.peek() == "OR":
            self.take()
            node = Or(node, self.and_expr())
        return node

    def and_expr(self) -> Node:
        node = self.not_expr()
        while self.peek() in ("AND", "NOT", "WORD", "PHRASE", "("):
            if self.peek() == "AND":
                self.take()
            node = And(node, self.not_expr())
        return node

    def not_expr(self) -> Node:
        if self.peek() == "NOT":
            self.take()
            return Not(self.not_expr())
        return self.atom()

    def atom(self) -> Node:
        kind = self.peek()
        if kind is None:
            raise QueryError("unexpected end of query")
        _, value = self.take()
        if kind == "(":
            node = self.or_expr()
            if self.peek() != ")":
                raise QueryError("missing ')'")
            self.take()
            return node
        if kind in ("WORD", "PHRASE"):
            words = tuple(tokenize(value))  # same normalization as the corpus
            if not words:
                raise QueryError(f"no searchable words in {value!r}")
            if len(words) == 1:
                return Term(words[0])
            if kind == "PHRASE":
                return Phrase(words)
            node: Node = Term(words[0])  # unquoted "well-known" -> well AND known
            for w in words[1:]:
                node = And(node, Term(w))
            return node
        raise QueryError(f"unexpected {value!r}")


def parse(text: str) -> Node:
    p = _Parser(text)
    node = p.or_expr()
    if p.peek() is not None:
        raise QueryError(f"unexpected {p.toks[p.i][1]!r}")
    return node
