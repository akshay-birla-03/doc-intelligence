"""Command-line interface: parse a document and print structured JSON."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .pipeline import DocumentPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docintel",
        description="Parse a business document into structured JSON.",
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Path to a document. Reads from stdin when omitted.",
    )
    parser.add_argument(
        "--indent", type=int, default=2, help="JSON indent (default: 2)."
    )
    parser.add_argument(
        "--version", action="version", version=f"docintel {__version__}"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = sys.stdin.read()

    if not text.strip():
        print("error: empty input", file=sys.stderr)
        return 1

    result = DocumentPipeline().process(text)
    json.dump(result, sys.stdout, indent=args.indent, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
