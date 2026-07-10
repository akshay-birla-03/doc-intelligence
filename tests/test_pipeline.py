"""End-to-end pipeline tests on synthetic invoices and resumes."""

import pytest

from docintel.data import make_invoice, make_resume
from docintel.pipeline import DocumentPipeline, process


@pytest.fixture(scope="module")
def pipeline():
    return DocumentPipeline()


def test_invoice_end_to_end(pipeline):
    doc = make_invoice(seed=42)
    result = pipeline.process(doc.text)

    assert result["doc_type"] == "invoice"
    assert result["n_sections"] >= 3
    fields = result["fields"]
    # Ground-truth key fields recovered.
    assert fields["invoice_number"] == doc.fields["invoice_number"]
    assert fields["total"] == doc.fields["total"]
    assert fields["email"] == doc.fields["email"]
    # Money and email entities detected somewhere.
    assert "MONEY" in fields["entities"]
    assert "EMAIL" in fields["entities"]


def test_resume_end_to_end(pipeline):
    doc = make_resume(seed=7)
    result = pipeline.process(doc.text)

    assert result["doc_type"] == "resume"
    labels = {s["label"] for s in result["sections"]}
    assert "HEADER" in labels
    fields = result["fields"]
    assert fields["email"] == doc.fields["email"]
    # The candidate's name should be detected as a PERSON entity.
    assert doc.fields["name"] in fields["entities"].get("PERSON", [])


def test_sections_carry_entities(pipeline):
    doc = make_invoice(seed=1)
    result = pipeline.process(doc.text)
    assert any(sec["entities"] for sec in result["sections"])


def test_process_convenience_matches_pipeline():
    text = make_invoice(seed=3).text
    a = process(text)
    b = DocumentPipeline().process(text)
    assert a["doc_type"] == b["doc_type"]
    assert a["fields"]["invoice_number"] == b["fields"]["invoice_number"]


def test_result_is_json_serialisable(pipeline):
    import json

    result = pipeline.process(make_resume(seed=5).text)
    json.dumps(result)  # must not raise


def test_invoice_fields_stable_across_seeds(pipeline):
    for seed in (10, 20, 30):
        doc = make_invoice(seed=seed)
        fields = pipeline.process(doc.text)["fields"]
        assert fields["invoice_number"] == doc.fields["invoice_number"]
