"""Hybrid named-entity recognition.

Two complementary strategies:

* **High-precision regex extractors** for structured entities whose surface form
  is well defined: EMAIL, PHONE, MONEY, DATE, URL, PERCENT.
* **A lightweight gazetteer + capitalisation heuristic** for PERSON and ORG,
  which have no reliable regex form. This is deliberately modest — it uses a
  known-name dictionary plus an organisation-suffix rule and title-case runs.

All extractors return :class:`Span` objects carrying ``(label, text, start,
end)`` character offsets, so callers can map entities back onto the source text.
Overlapping spans are resolved by a priority order (structured labels win over
the fuzzy PERSON/ORG heuristics).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .data import COMPANIES, FIRST_NAMES, LAST_NAMES

__all__ = ["Span", "extract_entities", "regex_entities"]


@dataclass(frozen=True)
class Span:
    """A single detected entity."""

    label: str
    text: str
    start: int
    end: int

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "text": self.text,
            "start": self.start,
            "end": self.end,
        }


# --- Regex patterns -------------------------------------------------------

# Order matters only for readability; overlap is resolved later by priority.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "URL",
        re.compile(r"\bhttps?://[^\s<>\")]+", re.IGNORECASE),
    ),
    (
        "MONEY",
        re.compile(r"[$€£]\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b"),
    ),
    (
        "PERCENT",
        re.compile(r"\b\d{1,3}(?:\.\d+)?\s?%"),
    ),
    (
        "PHONE",
        re.compile(
            r"""(?<![\w.])          # not preceded by word char/dot
            (?:\+?\d{1,3}[\s.-]?)?  # optional country code
            (?:\(\d{3}\)|\d{3})     # area code, optional parens
            [\s.-]                  # required separator
            \d{3}                   # prefix
            [\s.-]                  # required separator
            \d{4}                   # line number
            \b""",
            re.VERBOSE,
        ),
    ),
    (
        "DATE",
        re.compile(
            r"""\b(?:
                \d{1,2}[/-]\d{1,2}[/-]\d{2,4}       # 12/05/2023 or 12-05-23
              | \d{4}-\d{2}-\d{2}                    # 2023-05-12
              | (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*
                \.?\s+\d{1,2},?\s+\d{4}              # Jan 5, 2023
            )\b""",
            re.VERBOSE | re.IGNORECASE,
        ),
    ),
]

# Priority: earlier = wins when spans overlap.
_PRIORITY = ["EMAIL", "URL", "DATE", "MONEY", "PERCENT", "PHONE", "PERSON", "ORG"]

_KNOWN_FIRST = {n.lower() for n in FIRST_NAMES}
_KNOWN_LAST = {n.lower() for n in LAST_NAMES}
_KNOWN_ORGS = {c.lower() for c in COMPANIES}
_ORG_SUFFIXES = {"inc", "llc", "corp", "ltd", "company", "co", "industries",
                 "enterprises", "systems", "group", "gmbh"}

_TITLECASE_RUN = re.compile(r"\b[A-Z][a-zA-Z&.]*(?:\s+[A-Z][a-zA-Z&.]*)*")


def regex_entities(text: str) -> list[Span]:
    """Return spans found purely by the high-precision regex extractors."""
    spans: list[Span] = []
    for label, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            matched = m.group().strip()
            # Recompute exact bounds after stripping surrounding whitespace.
            start = m.start() + (m.group().find(matched) if matched else 0)
            spans.append(Span(label, matched, start, start + len(matched)))
    return spans


def _gazetteer_entities(text: str) -> list[Span]:
    """Detect PERSON and ORG using dictionaries + light heuristics."""
    spans: list[Span] = []

    # ORG: known companies (multi-word, matched literally) first.
    for org in sorted(COMPANIES, key=len, reverse=True):
        for m in re.finditer(re.escape(org), text):
            spans.append(Span("ORG", m.group(), m.start(), m.end()))

    for m in _TITLECASE_RUN.finditer(text):
        run = m.group()
        tokens = run.split()
        lowered = [t.lower().strip(".,&") for t in tokens]

        # ORG by suffix (e.g. "Foo Industries", "Bar LLC").
        if len(tokens) >= 2 and lowered[-1] in _ORG_SUFFIXES:
            if not _overlaps(spans, m.start(), m.end()):
                spans.append(Span("ORG", run, m.start(), m.end()))
            continue

        # PERSON: slide a bigram window; accept a name-like pair where a known
        # first name is followed by a capitalised token, or a known last name is
        # preceded by one. Offsets are computed within the run.
        for j in range(len(tokens) - 1):
            first, second = lowered[j], lowered[j + 1]
            if first in _KNOWN_FIRST or second in _KNOWN_LAST:
                # Reject known org-suffix pairs.
                if second in _ORG_SUFFIXES:
                    continue
                phrase = f"{tokens[j]} {tokens[j + 1]}"
                offset = run.find(phrase)
                if offset < 0:
                    continue
                p_start = m.start() + offset
                p_end = p_start + len(phrase)
                if not _overlaps(spans, p_start, p_end):
                    spans.append(Span("PERSON", phrase, p_start, p_end))
                break
    return spans


def _overlaps(spans: list[Span], start: int, end: int) -> bool:
    return any(not (end <= s.start or start >= s.end) for s in spans)


def _resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Keep the highest-priority, then longest, span among overlaps."""
    def rank(s: Span) -> tuple[int, int]:
        pr = _PRIORITY.index(s.label) if s.label in _PRIORITY else len(_PRIORITY)
        return (pr, -(s.end - s.start))

    ordered = sorted(spans, key=rank)
    kept: list[Span] = []
    for s in ordered:
        if not _overlaps(kept, s.start, s.end):
            kept.append(s)
    return sorted(kept, key=lambda s: s.start)


def extract_entities(text: str, include_ml: bool = True) -> list[Span]:
    """Extract all entities from ``text``.

    Parameters
    ----------
    text:
        The document (or section) text.
    include_ml:
        When ``True`` (default) also run the PERSON/ORG gazetteer heuristics.
        When ``False`` only the high-precision regex extractors run.
    """
    spans = regex_entities(text)
    if include_ml:
        spans.extend(_gazetteer_entities(text))
    return _resolve_overlaps(spans)
