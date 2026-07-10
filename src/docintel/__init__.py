"""doc-intelligence: an offline NLP toolkit for intelligent document processing.

Public API
----------
- :func:`process` / :class:`DocumentPipeline` -- end-to-end parsing.
- :func:`extract_entities`, :class:`Span` -- hybrid NER.
- :func:`segment`, :class:`Section` -- section segmentation.
- :class:`SectionClassifier` -- TF-IDF + LogisticRegression classifier.
- :func:`extract_fields` -- key-value extraction.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .classify import LABELS, SectionClassifier
from .extract import extract_fields, extract_key_values
from .ner import Span, extract_entities, regex_entities
from .pipeline import DocumentPipeline, process
from .segment import Section, segment

__all__ = [
    "__version__",
    "process",
    "DocumentPipeline",
    "extract_entities",
    "regex_entities",
    "Span",
    "segment",
    "Section",
    "SectionClassifier",
    "LABELS",
    "extract_fields",
    "extract_key_values",
]
