"""Key-value field extraction combining explicit patterns with NER.

Two mechanisms:

* **Labelled key-value lines** of the form ``Key: value`` are parsed directly,
  with the key normalised to a snake_case field name.
* **NER back-off** fills common fields (email, phone, total/amount, dates) from
  detected entities when an explicit label is missing.
"""

from __future__ import annotations

import re

from .ner import extract_entities

__all__ = ["extract_fields", "extract_key_values"]

_KV_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /()#%.-]{1,40}?)\s*:\s*(.+?)\s*$")

# Map noisy raw keys onto canonical field names.
_KEY_ALIASES = {
    "invoice_number": "invoice_number",
    "invoice_no": "invoice_number",
    "invoice": "invoice_number",
    "invoice_date": "invoice_date",
    "date": "date",
    "email": "email",
    "e_mail": "email",
    "phone": "phone",
    "telephone": "phone",
    "total": "total",
    "amount_due": "total",
    "balance": "total",
    "subtotal": "subtotal",
    "tax": "tax",
    "bill_to": "bill_to",
    "from": "from",
    "name": "name",
}


def _normalise_key(raw: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
    key = re.sub(r"_\d+_$", "", key)  # drop trailing "(8%)" style noise
    return _KEY_ALIASES.get(key, key)


def extract_key_values(text: str) -> dict[str, str]:
    """Parse explicit ``Key: value`` lines into a dict of canonical fields."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        m = _KV_LINE.match(line)
        if not m:
            continue
        key = _normalise_key(m.group(1))
        value = m.group(2).strip()
        if not key or not value:
            continue
        # First occurrence wins for stable output.
        fields.setdefault(key, value)
    return fields


def extract_fields(text: str) -> dict:
    """Extract a structured field dict from ``text``.

    Combines explicit key-value lines with NER-derived back-off values and a
    per-label list of all detected entities.
    """
    kv = extract_key_values(text)
    spans = extract_entities(text)

    by_label: dict[str, list[str]] = {}
    for s in spans:
        by_label.setdefault(s.label, []).append(s.text)

    fields: dict = dict(kv)

    # NER back-off for common fields.
    if "email" not in fields and by_label.get("EMAIL"):
        fields["email"] = by_label["EMAIL"][0]
    if "phone" not in fields and by_label.get("PHONE"):
        fields["phone"] = by_label["PHONE"][0]
    if "total" not in fields and by_label.get("MONEY"):
        # Heuristic: the largest money amount is usually the total.
        fields["total"] = max(by_label["MONEY"], key=_money_value)

    fields["entities"] = {label: vals for label, vals in sorted(by_label.items())}
    return fields


def _money_value(s: str) -> float:
    digits = re.sub(r"[^\d.]", "", s)
    try:
        return float(digits)
    except ValueError:
        return 0.0
