"""Synthetic data generators for documents and classifier training snippets.

Everything here is generated in-process with a seeded RNG so the toolkit runs
fully offline with no dataset downloads. The synthetic corpus is intentionally
varied (multiple templates, random names/amounts/dates) so a TF-IDF classifier
trained on it learns real lexical signal rather than memorising a single string.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Isabel", "Jack", "Karen", "Leo", "Maria", "Nathan", "Olivia", "Peter",
    "Quinn", "Rachel", "Samuel", "Tina", "Uma", "Victor", "Wendy", "Xavier",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
]
COMPANIES = [
    "Acme Corp", "Globex Inc", "Initech LLC", "Umbrella Corp", "Stark Industries",
    "Wayne Enterprises", "Wonka Industries", "Cyberdyne Systems", "Hooli Inc",
    "Pied Piper LLC", "Soylent Corp", "Massive Dynamic", "Vandelay Industries",
]
CITIES = ["New York", "London", "Berlin", "Austin", "Seattle", "Boston", "Denver"]
JOB_TITLES = [
    "Software Engineer", "Data Scientist", "Product Manager", "DevOps Engineer",
    "Machine Learning Engineer", "Backend Developer", "Solutions Architect",
]
SKILLS = [
    "Python", "SQL", "Docker", "Kubernetes", "TensorFlow", "PyTorch", "AWS",
    "scikit-learn", "FastAPI", "PostgreSQL", "Spark", "Airflow", "React",
]
ITEMS = [
    "Consulting services", "Software license", "Cloud hosting", "Support plan",
    "Hardware unit", "Training session", "Maintenance fee", "API credits",
]
UNIVERSITIES = [
    "MIT", "Stanford University", "Carnegie Mellon University",
    "University of Cambridge", "ETH Zurich", "University of Toronto",
]


@dataclass
class GeneratedDoc:
    """A synthetic document together with lightweight ground-truth metadata."""

    text: str
    kind: str  # "invoice" or "resume"
    fields: dict = field(default_factory=dict)


def _rng(seed: int | None) -> random.Random:
    return random.Random(seed)


def _money(rng: random.Random) -> float:
    return round(rng.uniform(10, 9999), 2)


def _date(rng: random.Random) -> str:
    return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(2018, 2024)}"


def make_invoice(seed: int | None = None) -> GeneratedDoc:
    """Generate a synthetic invoice document."""
    rng = _rng(seed)
    vendor = rng.choice(COMPANIES)
    customer = rng.choice(COMPANIES)
    inv_no = f"INV-{rng.randint(1000, 9999)}"
    date = _date(rng)
    email = f"billing@{vendor.split()[0].lower()}.com"
    phone = f"+1-{rng.randint(200, 999)}-{rng.randint(200, 999)}-{rng.randint(1000, 9999)}"

    lines = []
    subtotal = 0.0
    n_items = rng.randint(2, 4)
    for _ in range(n_items):
        item = rng.choice(ITEMS)
        qty = rng.randint(1, 9)
        price = _money(rng)
        amount = round(qty * price, 2)
        subtotal += amount
        lines.append(f"{item} x{qty} @ ${price:.2f} = ${amount:.2f}")
    subtotal = round(subtotal, 2)
    tax = round(subtotal * 0.08, 2)
    total = round(subtotal + tax, 2)

    text = "\n".join(
        [
            "INVOICE",
            "",
            f"From: {vendor}",
            f"Bill To: {customer}",
            f"Invoice Number: {inv_no}",
            f"Invoice Date: {date}",
            f"Email: {email}",
            f"Phone: {phone}",
            "",
            "LINE ITEMS",
            *lines,
            "",
            f"Subtotal: ${subtotal:.2f}",
            f"Tax (8%): ${tax:.2f}",
            f"Total: ${total:.2f}",
        ]
    )
    fields = {
        "invoice_number": inv_no,
        "invoice_date": date,
        "email": email,
        "phone": phone,
        "total": f"${total:.2f}",
        "vendor": vendor,
    }
    return GeneratedDoc(text=text, kind="invoice", fields=fields)


def make_resume(seed: int | None = None) -> GeneratedDoc:
    """Generate a synthetic resume document."""
    rng = _rng(seed)
    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    email = f"{name.split()[0].lower()}.{name.split()[1].lower()}@example.com"
    phone = f"({rng.randint(200, 999)}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}"
    city = rng.choice(CITIES)
    title = rng.choice(JOB_TITLES)
    skills = ", ".join(rng.sample(SKILLS, k=5))

    exp_lines = []
    for _ in range(rng.randint(2, 3)):
        company = rng.choice(COMPANIES)
        role = rng.choice(JOB_TITLES)
        start = rng.randint(2015, 2020)
        end = start + rng.randint(1, 4)
        exp_lines.append(f"{role} at {company} ({start} - {end})")
        exp_lines.append(
            f"Built and maintained production systems using {rng.choice(SKILLS)}."
        )

    edu = f"{rng.choice(['B.S.', 'M.S.', 'Ph.D.'])} in Computer Science, {rng.choice(UNIVERSITIES)}"

    text = "\n".join(
        [
            name,
            f"{title} | {city}",
            f"Email: {email}",
            f"Phone: {phone}",
            "",
            "SUMMARY",
            f"Experienced {title.lower()} with a track record of delivering software.",
            "",
            "EXPERIENCE",
            *exp_lines,
            "",
            "EDUCATION",
            edu,
            "",
            "SKILLS",
            skills,
        ]
    )
    fields = {
        "name": name,
        "email": email,
        "phone": phone,
        "title": title,
    }
    return GeneratedDoc(text=text, kind="resume", fields=fields)


# --- Labelled snippets for the section classifier -------------------------

# Each generator returns a single-line/short snippet representative of a label.
# Labels: HEADER, CONTACT, EXPERIENCE, EDUCATION, LINE_ITEM, TOTAL, OTHER

_HEADERS = [
    "INVOICE", "RESUME", "EXPERIENCE", "EDUCATION", "SKILLS", "SUMMARY",
    "LINE ITEMS", "CONTACT INFORMATION", "WORK HISTORY", "PROJECTS",
    "CERTIFICATIONS", "BILLING DETAILS", "PROFESSIONAL SUMMARY",
]


def _contact(rng: random.Random) -> str:
    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    kind = rng.randint(0, 3)
    if kind == 0:
        return f"Email: {name.split()[0].lower()}@example.com"
    if kind == 1:
        return f"Phone: ({rng.randint(200, 999)}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}"
    if kind == 2:
        return f"{rng.randint(1, 999)} Main Street, {rng.choice(CITIES)}"
    return f"Contact {name} at {name.split()[0].lower()}@mail.com"


def _experience(rng: random.Random) -> str:
    role = rng.choice(JOB_TITLES)
    company = rng.choice(COMPANIES)
    start = rng.randint(2010, 2020)
    return f"{role} at {company} ({start} - {start + rng.randint(1, 5)})"


def _education(rng: random.Random) -> str:
    deg = rng.choice(["B.S.", "M.S.", "Ph.D.", "Bachelor", "Master"])
    return f"{deg} in Computer Science, {rng.choice(UNIVERSITIES)}, {rng.randint(2008, 2022)}"


def _line_item(rng: random.Random) -> str:
    item = rng.choice(ITEMS)
    qty = rng.randint(1, 9)
    price = _money(rng)
    return f"{item} x{qty} @ ${price:.2f} = ${round(qty * price, 2):.2f}"


def _total(rng: random.Random) -> str:
    kind = rng.choice(["Subtotal", "Tax", "Total", "Amount Due", "Balance"])
    return f"{kind}: ${_money(rng):.2f}"


def _other(rng: random.Random) -> str:
    templates = [
        "Please remit payment within 30 days of the invoice date.",
        "Thank you for your business and continued partnership.",
        "All prices are quoted in US dollars unless otherwise noted.",
        f"Delivered production systems using {rng.choice(SKILLS)} and {rng.choice(SKILLS)}.",
        "This document is confidential and intended for the recipient only.",
        "References are available upon request.",
    ]
    return rng.choice(templates)


_LABEL_GENERATORS = {
    "CONTACT": _contact,
    "EXPERIENCE": _experience,
    "EDUCATION": _education,
    "LINE_ITEM": _line_item,
    "TOTAL": _total,
    "OTHER": _other,
}


def make_training_data(
    n_per_label: int = 120, seed: int = 7
) -> tuple[list[str], list[str]]:
    """Generate labelled (text, label) snippets for the section classifier.

    Returns two parallel lists: ``texts`` and ``labels``.
    """
    rng = _rng(seed)
    texts: list[str] = []
    labels: list[str] = []

    for _ in range(n_per_label):
        texts.append(rng.choice(_HEADERS))
        labels.append("HEADER")

    for label, gen in _LABEL_GENERATORS.items():
        for _ in range(n_per_label):
            texts.append(gen(rng))
            labels.append(label)

    # Shuffle deterministically.
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    texts = [texts[i] for i in idx]
    labels = [labels[i] for i in idx]
    return texts, labels
