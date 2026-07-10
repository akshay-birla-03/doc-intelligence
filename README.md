# doc-intelligence

An offline NLP toolkit for **intelligent document processing**. It parses
semi-structured business documents (invoices, resumes, contracts) into
structured data: it segments a document into logical sections, classifies each
section, extracts key-value fields, and runs a **rule + ML hybrid** named-entity
recognizer for dates, money amounts, emails, phone numbers, URLs, percentages,
organizations and person names.

Everything runs **fully offline**. No model or dataset downloads: the classifier
is trained in-process on synthetic labelled snippets generated in code.

## Architecture

```
                       ┌──────────────────────────────────────────────┐
   raw document text   │              DocumentPipeline                 │
  ───────────────────► │                                              │
                       │   segment ──► classify ──► NER ──► extract    │
                       │   (heuristic) (TF-IDF+LR) (rule+ML) (KV+NER)  │
                       └───────┬───────────┬─────────┬────────┬────────┘
                               │           │         │        │
                        Section[]     label per   Span[]   fields{}
                        (blocks &      section    (typed    (canonical
                         headers)    (7 classes)  offsets)   key-values)
                               │           │         │        │
                               └───────────┴────┬────┴────────┘
                                                ▼
                                   structured JSON document
                              { doc_type, sections[], fields{} }
```

Stage by stage:

| Stage | Module | Approach |
|-------|--------|----------|
| Segment | `segment.py` | Blank-line blocks + header heuristics (ALL-CAPS / Title-Case short lines without trailing period). |
| Classify | `classify.py` | scikit-learn `TfidfVectorizer` (word 1–2 grams) + `LogisticRegression`. Labels: `HEADER, CONTACT, EXPERIENCE, EDUCATION, LINE_ITEM, TOTAL, OTHER`. |
| NER | `ner.py` | High-precision regex for `EMAIL, PHONE, MONEY, DATE, URL, PERCENT`, plus a gazetteer + capitalisation heuristic for `PERSON` / `ORG`. Overlaps resolved by label priority. |
| Extract | `extract.py` | Explicit `Key: value` line parsing with key normalisation, plus NER back-off for email/phone/total. |

## Why this is non-trivial

- **Hybrid NER.** Structured entities (email, money, dates) are matched with
  tuned regex for high precision; fuzzy entities (person, org) that have no
  reliable surface form use a dictionary + capitalisation heuristic. Overlapping
  detections are resolved by a priority order so an email is never fragmented
  into stray sub-spans.
- **Leakage-safe classifier evaluation.** The reported accuracy comes from a
  held-out split (`train_test_split`, stratified) — the model is never scored on
  snippets it trained on. See `tests/test_classify.py::test_heldout_accuracy_bar`.
- **Typed character offsets.** Every entity carries `(label, text, start, end)`
  so detections map back onto the source text exactly (verified in tests).

## Measured numbers

From the test run in this repository (`python -m pytest`):

- **54 tests pass** in ~8s.
- **Section classifier held-out accuracy: 1.00** on a stratified 75/25 split of
  the synthetic corpus (720 snippets). The synthetic label classes are lexically
  distinct, so this reflects an easy-but-honest synthetic benchmark rather than a
  claim about messy real-world documents; the test asserts a conservative
  `>= 0.85` bar.
- On synthetic invoices, the pipeline recovers `invoice_number`, `total` and
  `email` exactly across multiple seeds (end-to-end tests).

## Install

```bash
pip install -e ".[dev]"
```

## Usage

### Python

```python
from docintel import process

result = process(open("invoice.txt").read())
print(result["doc_type"])          # "invoice"
print(result["fields"]["total"])   # "$51389.50"
```

Example structured output (synthetic invoice, seed 42):

```
doc_type: invoice | n_sections: 5
fields: {'from': 'Soylent Corp', 'bill_to': 'Globex Inc',
         'invoice_number': 'INV-1409', 'invoice_date': '24/05/2019',
         'email': 'billing@soylent.com', 'phone': '+1-428-342-2679',
         'subtotal': '$47582.87', 'tax_8': '$3806.63', 'total': '$51389.50'}
```

Lower-level building blocks:

```python
from docintel import extract_entities, segment, SectionClassifier, extract_fields

extract_entities("Pay $1,234.56 to bob@acme.com by 12/05/2023")
# [Span('MONEY','$1,234.56',...), Span('EMAIL','bob@acme.com',...), Span('DATE',...)]

clf = SectionClassifier.train_default()
clf.predict_one("Total: $500.00")   # "TOTAL"
```

### CLI

```bash
docintel invoice.txt              # parse a file
cat invoice.txt | docintel        # or read from stdin
docintel invoice.txt --indent 0   # compact JSON
```

### API

```bash
uvicorn docintel.api:app --port 8000
# or: make serve
```

- `GET /health` → `{"status": "ok", "version": "0.1.0"}`
- `POST /parse` with `{"text": "..."}` → structured document JSON.

```bash
curl -s localhost:8000/parse -H 'content-type: application/json' \
     -d '{"text": "Invoice Number: INV-1\nTotal: $9.99"}'
```

## Testing

```bash
make test    # python -m pytest
make lint    # ruff check src tests
```

The suite (54 tests) covers NER precision on crafted strings, segmentation
heuristics, the leakage-safe classifier accuracy bar, key-value extraction,
end-to-end invoice/resume parsing, the CLI, and the API via `TestClient`.

## Limitations (honest)

- The classifier and NER are validated on **synthetic** data generated in
  `data.py`. Real-world documents are noisier; the reported accuracy does not
  transfer directly.
- PERSON/ORG recognition is a gazetteer + heuristic, not a trained sequence
  model — it recognises known names and organisation suffixes, and will miss
  novel entities.
- The segmenter is line/heuristic based and assumes reasonably clean text
  (it does not do OCR or layout analysis).

## License

MIT — see [LICENSE](LICENSE).
