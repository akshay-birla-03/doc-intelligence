"""Tests for hybrid NER precision on crafted strings."""

from docintel.ner import Span, extract_entities, regex_entities


def _labels(text, **kw):
    return {(s.label, s.text) for s in extract_entities(text, **kw)}


def test_email_extraction():
    got = _labels("Reach me at john.doe@example.co.uk please.")
    assert ("EMAIL", "john.doe@example.co.uk") in got


def test_url_extraction():
    got = _labels("See https://example.com/path?q=1 for details.")
    assert ("URL", "https://example.com/path?q=1") in got


def test_money_extraction_with_commas():
    got = _labels("The total is $1,234.56 due now.")
    assert ("MONEY", "$1,234.56") in got


def test_percent_extraction():
    got = _labels("Tax applied at 8.5% this quarter.")
    assert ("PERCENT", "8.5%") in got


def test_phone_extraction_paren_format():
    got = _labels("Call (415) 555-1234 anytime.")
    assert ("PHONE", "(415) 555-1234") in got


def test_phone_extraction_dashed_with_country_code():
    spans = extract_entities("Dial +1-979-660-8737 today.")
    phones = [s.text for s in spans if s.label == "PHONE"]
    assert phones == ["+1-979-660-8737"]


def test_date_slash_format():
    got = _labels("Invoice Date: 09/02/2021 confirmed.")
    assert ("DATE", "09/02/2021") in got


def test_date_month_name_format():
    got = _labels("Signed on Jan 5, 2023 by the parties.")
    assert ("DATE", "Jan 5, 2023") in got


def test_person_from_known_names():
    got = _labels("Prepared by Alice Smith for review.")
    assert ("PERSON", "Alice Smith") in got


def test_org_by_suffix():
    got = _labels("Payment to Contoso Industries is overdue.")
    assert ("ORG", "Contoso Industries") in got


def test_org_known_company():
    got = _labels("Billed by Acme Corp last week.")
    assert ("ORG", "Acme Corp") in got


def test_spans_have_correct_offsets():
    text = "Email: bob@acme.com"
    spans = extract_entities(text)
    email = next(s for s in spans if s.label == "EMAIL")
    assert text[email.start : email.end] == email.text


def test_no_false_money_from_plain_number():
    # A bare number without a currency symbol must not be MONEY.
    got = {s.label for s in extract_entities("There are 42 widgets in stock.")}
    assert "MONEY" not in got


def test_include_ml_false_disables_person_org():
    text = "Alice Smith joined Acme Corp."
    labels = {s.label for s in extract_entities(text, include_ml=False)}
    assert "PERSON" not in labels and "ORG" not in labels


def test_regex_entities_only_returns_structured():
    text = "Alice Smith paid $10.00 to bob@x.com"
    labels = {s.label for s in regex_entities(text)}
    assert labels <= {"EMAIL", "URL", "MONEY", "PERCENT", "PHONE", "DATE"}
    assert "MONEY" in labels and "EMAIL" in labels


def test_overlap_resolution_prefers_structured():
    # An email contains an @; ensure it is emitted as a single EMAIL span and
    # no fragment leaks as another label.
    spans = extract_entities("Contact billing@acme.com now.")
    emails = [s for s in spans if s.label == "EMAIL"]
    assert len(emails) == 1
    # No span should overlap the email span.
    e = emails[0]
    others = [s for s in spans if s is not e]
    assert all(s.end <= e.start or s.start >= e.end for s in others)


def test_span_as_dict():
    s = Span("EMAIL", "a@b.com", 0, 7)
    assert s.as_dict() == {"label": "EMAIL", "text": "a@b.com", "start": 0, "end": 7}
