"""Exact original-source code regions from the existing CommonMark parser.

Only syntax boundaries are used. Normalized/token content is never delivery text.
The small source map tracks normalization and block indentation removal, not a
second Markdown grammar. No renderer, network access, or parser fallback is used.
"""
from __future__ import annotations

from collections.abc import Iterator
import re
from typing import cast

from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from markdown_it.rules_core import normalize
from markdown_it.rules_inline.backticks import backtick


class _MappedText(str):
    """Parser text with raw character offsets; virtual tab spaces have no offset."""

    offsets: tuple[int | None, ...]

    def __new__(cls, value: str, offsets: tuple[int | None, ...]):
        if len(value) != len(offsets):
            raise ValueError("Markdown source map length mismatch")
        instance = super().__new__(cls, value)
        instance.offsets = offsets
        return instance

    def __getitem__(self, key):
        value = super().__getitem__(key)
        return _MappedText(value, self.offsets[key]) if isinstance(key, slice) else value

    def strip(self, chars=None):
        start = len(self) - len(super().lstrip(chars))
        stop = len(super().rstrip(chars))
        return self[start:max(start, stop)]


class _MappedBlock(StateBlock):
    def getLines(self, begin: int, end: int, indent: int, keepLastLF: bool) -> str:
        # Upstream getLines emits one source suffix per line, possibly prefixed
        # with spaces from a partially expanded tab. Anchor at each known line
        # END, not a substring search: identical examples cannot share offsets.
        content = super().getLines(begin, end, indent, keepLastLF)
        offsets: list[int | None] = []
        cursor = 0
        for line in range(begin, end):
            newline = content.find("\n", cursor)
            stop = newline + 1 if newline >= 0 else len(content)
            emitted = content[cursor:stop]
            last = self.eMarks[line] + (line + 1 < end or keepLastLF)
            source = cast(_MappedText, self.src[self.bMarks[line]:last])
            common = 0
            while common < min(len(emitted), len(source)) and emitted[-common - 1] == source[-common - 1]:
                common += 1
            virtual = len(emitted) - common
            if emitted[:virtual] != " " * virtual:
                raise ValueError("Unsupported Markdown line source transformation")
            offsets.extend([None] * virtual)
            if common:
                offsets.extend(source.offsets[-common:])
            cursor = stop
        return _MappedText(content, tuple(offsets))


def markdown_code_regions(text: str) -> list[tuple[int, int]]:
    """Return sorted, disjoint half-open offsets into the EXACT original text."""
    regions: list[tuple[int, int]] = []
    parser = MarkdownIt("commonmark")

    def mapped_normalize(state) -> None:
        raw = state.src
        normalize(state)
        offsets: list[int | None] = []
        cursor = 0
        for match in re.finditer(r"\r\n?|\x00", raw):
            offsets.extend(range(cursor, match.start()))
            offsets.append(match.start())
            cursor = match.end()
        offsets.extend(range(cursor, len(raw)))
        state.src = _MappedText(state.src, tuple(offsets))

    def mapped_block(state) -> None:
        block = _MappedBlock(state.src, state.md, state.env, state.tokens)
        # Container rules temporarily rewrite bMarks. Save root line boundaries
        # before tokenizing; token.map still refers to those original line IDs.
        line_starts = tuple(block.bMarks)
        state.md.block.tokenize(block, block.line, block.lineMax)
        for token in state.tokens:
            if token.type in {"fence", "code_block"} and token.map:
                start, stop = (line_starts[line] for line in token.map)
                raw_start = state.src.offsets[start] if start < len(state.src) else len(text)
                raw_stop = state.src.offsets[stop] if stop < len(state.src) else len(text)
                if raw_start is None or raw_stop is None:
                    raise ValueError("Markdown block boundary has no original-source offset")
                regions.append((raw_start, raw_stop))

    def capture_backtick(state, silent: bool) -> bool:
        start = state.pos
        count = len(state.tokens)
        accepted = backtick(state, silent)
        if not silent and len(state.tokens) > count and state.tokens[-1].type == "code_inline":
            if not isinstance(state.src, _MappedText):
                raise ValueError("Missing Markdown inline source map")
            raw_start = state.src.offsets[start]
            raw_last = state.src.offsets[state.pos - 1]
            if raw_start is None or raw_last is None:
                raise ValueError("Markdown delimiter has no original-source offset")
            regions.append((raw_start, raw_last + 1))
        return accepted

    parser.core.ruler.at("normalize", mapped_normalize)
    parser.core.ruler.at("block", mapped_block)
    parser.inline.ruler.at("backticks", capture_backtick)
    parser.parse(text)
    merged: list[tuple[int, int]] = []
    for start, stop in sorted(set(regions)):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def markdown_code_parts(text: str) -> Iterator[tuple[bool, str]]:
    """Yield literal code and transformable prose from one authoritative map."""
    end = 0
    for start, stop in markdown_code_regions(text):
        if start > end:
            yield False, text[end:start]
        yield True, text[start:stop]
        end = stop
    if end < len(text):
        yield False, text[end:]
