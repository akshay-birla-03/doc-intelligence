"""API tests via FastAPI's TestClient."""

from fastapi.testclient import TestClient

from docintel import __version__
from docintel.api import app
from docintel.data import make_invoice

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__


def test_parse_invoice():
    doc = make_invoice(seed=99)
    resp = client.post("/parse", json={"text": doc.text})
    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_type"] == "invoice"
    assert body["fields"]["invoice_number"] == doc.fields["invoice_number"]
    assert body["n_sections"] == len(body["sections"])


def test_parse_missing_text_returns_422():
    resp = client.post("/parse", json={})
    assert resp.status_code == 422


def test_parse_empty_text_ok():
    resp = client.post("/parse", json={"text": ""})
    assert resp.status_code == 200
    assert resp.json()["n_sections"] == 0
