"""Tests for key-value field extraction."""

from docintel.extract import extract_fields, extract_key_values


def test_key_value_basic():
    kv = extract_key_values("Invoice Number: INV-123\nTotal: $45.00")
    assert kv["invoice_number"] == "INV-123"
    assert kv["total"] == "$45.00"


def test_key_alias_normalisation():
    kv = extract_key_values("Amount Due: $99.00")
    assert kv["total"] == "$99.00"


def test_first_occurrence_wins():
    kv = extract_key_values("Email: a@x.com\nEmail: b@y.com")
    assert kv["email"] == "a@x.com"


def test_non_kv_lines_ignored():
    kv = extract_key_values("Just a normal sentence with no colon structure")
    assert kv == {}


def test_extract_fields_includes_entities():
    fields = extract_fields("Email: bob@acme.com\nTotal: $10.00")
    assert "entities" in fields
    assert "bob@acme.com" in fields["entities"]["EMAIL"]


def test_ner_backoff_fills_email_without_label():
    text = "Reach out to sales@vendor.com regarding your order."
    fields = extract_fields(text)
    assert fields.get("email") == "sales@vendor.com"


def test_total_backoff_picks_largest_money():
    text = "Item A $10.00\nItem B $250.00\nItem C $30.00"
    fields = extract_fields(text)
    assert fields["total"] == "$250.00"


def test_explicit_total_not_overridden_by_backoff():
    text = "Subtotal: $10.00\nTotal: $12.00\nHuge line $9999.00"
    fields = extract_fields(text)
    assert fields["total"] == "$12.00"
