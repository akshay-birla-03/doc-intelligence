"""Document segmentation into logical sections/blocks.

A document is split on blank-line boundaries into blocks. Within that, lines
that look like *headers* (short, ALL-CAPS or Title-Case, no trailing period)
start a new section and become that section's title. This is a heuristic
segmenter — it makes no ML calls and is fully deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["Section", "segment"]

_HEADER_MAX_WORDS = 6


@dataclass
class Section:
    """A contiguous logical block of a document."""

    title: str | None
    lines: list[str] = field(default_factory=list)
    start_line: int = 0

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "text": self.text,
            "start_line": self.start_line,
        }


def is_header(line: str) -> bool:
    """Heuristically decide whether ``line`` is a section header."""
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped.split()) > _HEADER_MAX_WORDS:
        return False
    if stripped.endswith((".", ",", ";", ":")):
        # A trailing colon is a key/label, not a standalone header.
        return False
    # Reject lines containing key-value / digit-heavy content.
    if any(ch.isdigit() for ch in stripped):
        return False
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False

    # ALL-CAPS header, e.g. "EXPERIENCE" or "LINE ITEMS".
    if stripped == stripped.upper() and len(letters) >= 2:
        return True

    # Title-Case header: every alphabetic word capitalised, e.g. "Work History".
    words = stripped.split()
    if 1 <= len(words) <= _HEADER_MAX_WORDS and all(
        w[0].isupper() for w in words if w[0].isalpha()
    ):
        # Require it be short and not sentence-like punctuation.
        return not re.search(r"[.!?]", stripped)
    return False


def segment(text: str) -> list[Section]:
    """Split ``text`` into a list of :class:`Section` objects."""
    lines = text.splitlines()
    sections: list[Section] = []
    current = Section(title=None, start_line=0)

    for i, raw in enumerate(lines):
        line = raw.rstrip()
        if not line.strip():
            # Blank line: close current section if it has content.
            if current.lines or current.title:
                sections.append(current)
                current = Section(title=None, start_line=i + 1)
            continue

        if is_header(line):
            # A header starts a fresh section.
            if current.lines or current.title:
                sections.append(current)
            current = Section(title=line.strip(), start_line=i)
        else:
            current.lines.append(line)

    if current.lines or current.title:
        sections.append(current)

    return [s for s in sections if s.lines or s.title]
