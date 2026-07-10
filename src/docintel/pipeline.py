"""End-to-end document processing pipeline.

``DocumentPipeline`` ties together the four stages:

    segment  ->  classify  ->  NER  ->  key-value extract

and returns a single structured dict describing the document.
"""

from __future__ import annotations

from functools import lru_cache

from .classify import SectionClassifier
from .extract import extract_fields
from .ner import extract_entities
from .segment import segment

__all__ = ["DocumentPipeline", "process"]


class DocumentPipeline:
    """Orchestrates segmentation, classification, NER and field extraction."""

    def __init__(self, classifier: SectionClassifier | None = None) -> None:
        self.classifier = classifier or SectionClassifier.train_default()

    def process(self, text: str) -> dict:
        """Process ``text`` into a structured document representation."""
        sections_out = []
        section_texts = []
        classify_inputs = []

        for sec in segment(text):
            # Classify using the title when present, else the body.
            probe = sec.title if sec.title else sec.text
            classify_inputs.append(probe)
            section_texts.append(sec)

        labels = (
            self.classifier.predict(classify_inputs) if classify_inputs else []
        )

        for sec, label in zip(section_texts, labels):
            spans = extract_entities(sec.text) if sec.text else []
            sections_out.append(
                {
                    "title": sec.title,
                    "label": label,
                    "text": sec.text,
                    "start_line": sec.start_line,
                    "entities": [s.as_dict() for s in spans],
                }
            )

        fields = extract_fields(text)
        doc_type = self._infer_doc_type(sections_out, fields)

        return {
            "doc_type": doc_type,
            "n_sections": len(sections_out),
            "sections": sections_out,
            "fields": fields,
        }

    @staticmethod
    def _infer_doc_type(sections: list[dict], fields: dict) -> str:
        labels = {s["label"] for s in sections}
        text_blob = " ".join(s["text"] for s in sections).lower()
        if "invoice" in text_blob or "invoice_number" in fields or "LINE_ITEM" in labels:
            return "invoice"
        if "EXPERIENCE" in labels or "EDUCATION" in labels:
            return "resume"
        return "unknown"


@lru_cache(maxsize=1)
def _default_pipeline() -> DocumentPipeline:
    return DocumentPipeline()


def process(text: str) -> dict:
    """Convenience wrapper using a cached default pipeline."""
    return _default_pipeline().process(text)
