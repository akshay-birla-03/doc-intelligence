"""Tests for the section classifier, including a leakage-safe accuracy bar."""

import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from docintel.classify import LABELS, SectionClassifier
from docintel.data import make_training_data


@pytest.fixture(scope="module")
def trained():
    return SectionClassifier.train_default()


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        SectionClassifier().predict(["anything"])


def test_heldout_accuracy_bar():
    # Train and evaluate on disjoint splits: no snippet is seen in both.
    X, y = make_training_data()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    clf = SectionClassifier().fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    assert acc >= 0.85, f"held-out accuracy too low: {acc}"


def test_predicts_expected_labels(trained):
    assert trained.predict_one("EXPERIENCE") == "HEADER"
    assert trained.predict_one("Email: jane@example.com") == "CONTACT"
    assert trained.predict_one("Total: $500.00") == "TOTAL"


def test_all_predictions_in_label_set(trained):
    preds = trained.predict(
        ["SKILLS", "Phone: (212) 555-9000", "B.S. in Computer Science, MIT, 2015"]
    )
    assert all(p in LABELS for p in preds)


def test_persistence_roundtrip(trained, tmp_path):
    path = tmp_path / "model.joblib"
    trained.save(path)
    loaded = SectionClassifier.load(path)
    assert loaded.fitted
    sample = ["EDUCATION", "Total: $12.00"]
    assert loaded.predict(sample) == trained.predict(sample)


def test_predict_proba_shape(trained):
    proba = trained.predict_proba(["Total: $9.99"])
    assert proba.shape[1] == len(set(LABELS))
