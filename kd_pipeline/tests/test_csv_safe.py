"""T4 · src/csv_safe.py — CSV formula injection prevention.

Addresses security finding S4: model output `=HYPERLINK(...)` or `=IMPORTXML(...)`
would execute when the error_samples.csv is opened in Excel/LibreOffice.
Also addresses log injection via CR/LF in field values.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.csv_safe import FORMULA_TRIGGERS, escape_cell, sanitize_row  # noqa: E402


# ---------------- escape_cell ----------------


@pytest.mark.parametrize(
    "dangerous",
    [
        "=1+1",
        "=HYPERLINK(\"http://evil\",\"click\")",
        "=IMPORTXML(\"http://evil/xml\",\"//\")",
        "+cmd|calc",
        "-1+1",
        "@SUM(1,1)",
    ],
)
def test_escape_cell_prefixes_formula_triggers(dangerous):
    out = escape_cell(dangerous)
    assert out.startswith("'")
    assert out[1:] == dangerous


def test_escape_cell_strips_cr_lf():
    out = escape_cell("line1\r\nline2")
    assert "\r" not in out
    assert "\n" not in out


def test_escape_cell_preserves_benign_values():
    assert escape_cell("normal text") == "normal text"
    assert escape_cell("2B wrong answer") == "2B wrong answer"
    assert escape_cell("123") == "123"


def test_escape_cell_empty_and_none():
    assert escape_cell("") == ""
    assert escape_cell(None) == ""


def test_escape_cell_with_tab_prefix():
    # Tab prefix (\t) can also cause issues; we strip it
    out = escape_cell("\tinjected")
    assert "\t" not in out


def test_escape_cell_numeric_not_triggered():
    # "0" is fine, "-0" looks like minus so IS a trigger
    assert escape_cell("0") == "0"
    assert escape_cell("-0").startswith("'")


def test_escape_cell_unicode_preserved():
    assert escape_cell("左边") == "左边"


def test_formula_triggers_has_expected_chars():
    assert "=" in FORMULA_TRIGGERS
    assert "+" in FORMULA_TRIGGERS
    assert "-" in FORMULA_TRIGGERS
    assert "@" in FORMULA_TRIGGERS


# ---------------- sanitize_row ----------------


def test_sanitize_row_applies_to_all_values():
    row = {
        "sample_id": "17",
        "distilled_answer": "=1+1",
        "teacher_answer": "normal",
        "question": "what is this?",
    }
    out = sanitize_row(row)
    assert out["distilled_answer"].startswith("'")
    assert out["teacher_answer"] == "normal"
    assert out["sample_id"] == "17"


def test_sanitize_row_preserves_keys():
    row = {"a": "1", "b": "2"}
    out = sanitize_row(row)
    assert set(out.keys()) == {"a", "b"}


def test_sanitize_row_handles_non_string_values():
    row = {"score": 0.87, "label": "L2"}
    out = sanitize_row(row)
    assert out["score"] == 0.87  # non-str passed through
    assert out["label"] == "L2"


def test_sanitize_row_empty():
    assert sanitize_row({}) == {}


# ---------------- round-trip with csv module ----------------


def test_escape_cell_roundtrips_through_csv(tmp_path: Path):
    dangerous = "=HYPERLINK(\"http://evil\",\"click\")"
    out = tmp_path / "r.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["val"])
        w.writerow([escape_cell(dangerous)])
    rows = list(csv.reader(out.open(encoding="utf-8")))
    # Value is stored as '=HYPERLINK(...)' — Excel would treat the leading ' as text marker
    assert rows[1][0].startswith("'")
    assert "HYPERLINK" in rows[1][0]


def test_no_false_positive_on_email_like():
    # An answer might contain "see foo@bar.com" — the @ is not at position 0
    assert escape_cell("see foo@bar.com") == "see foo@bar.com"
