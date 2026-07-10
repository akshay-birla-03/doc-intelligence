"""Tests for the heuristic segmenter."""

from docintel.segment import is_header, segment


def test_allcaps_header_detected():
    assert is_header("EXPERIENCE")
    assert is_header("LINE ITEMS")


def test_titlecase_header_detected():
    assert is_header("Work History")
    assert is_header("Professional Summary")


def test_sentence_is_not_header():
    assert not is_header("This is a normal sentence that runs long.")


def test_key_value_line_is_not_header():
    assert not is_header("Invoice Number: INV-123")


def test_line_with_digits_is_not_header():
    assert not is_header("Total 4500")


def test_blank_splits_blocks():
    text = "Block one line.\n\nBlock two line."
    secs = segment(text)
    assert len(secs) == 2
    assert secs[0].text == "Block one line."
    assert secs[1].text == "Block two line."


def test_header_starts_new_section_with_title():
    text = "SUMMARY\nSome summary text here.\nMore text."
    secs = segment(text)
    assert secs[0].title == "SUMMARY"
    assert "Some summary text here." in secs[0].text


def test_multiple_headers_produce_multiple_sections():
    text = (
        "EXPERIENCE\n"
        "Engineer at Acme Corp\n"
        "\n"
        "EDUCATION\n"
        "B.S. in Computer Science\n"
    )
    secs = segment(text)
    titles = [s.title for s in secs]
    assert "EXPERIENCE" in titles
    assert "EDUCATION" in titles


def test_start_line_tracked():
    text = "line0\nline1\n\nHEADER\nbody"
    secs = segment(text)
    # The header section should reference the header's line index (3).
    header_secs = [s for s in secs if s.title == "HEADER"]
    assert header_secs and header_secs[0].start_line == 3


def test_section_as_dict():
    secs = segment("HEADER\nbody line")
    d = secs[0].as_dict()
    assert set(d) == {"title", "text", "start_line"}
