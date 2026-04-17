"""T8 refactored · scripts/sample_error_cases.py

Uses src.lmms_eval_io (schema probing) + src.csv_safe (formula injection prevention).
Addresses C4 (no mutation of pools) and S4 (CSV formula injection).
"""

from __future__ import annotations

import copy
import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from sample_error_cases import (  # noqa: E402
    join_by_id,
    stratified_sample,
    write_csv,
)


def _s(
    sid: str,
    q: str,
    correct: bool,
    ans: str = "x",
    img: str = "img.jpg",
) -> dict:
    return {
        "doc_id": sid,
        "doc": {"question": q, "image": img},
        "filtered_resps": [ans],
        "cv_bench_acc,none": 1.0 if correct else 0.0,
    }


# ---------------- join_by_id ----------------


def test_join_by_id_matches_common_ids():
    d = [_s("a", "q1", False), _s("b", "q2", True)]
    t = [_s("a", "q1", True), _s("b", "q2", True)]
    paired = join_by_id(d, t)
    assert len(paired) == 2
    assert {p["sid"] for p in paired} == {"a", "b"}


def test_join_by_id_drops_unmatched():
    d = [_s("a", "q1", False), _s("b", "q2", False)]
    t = [_s("a", "q1", True)]
    paired = join_by_id(d, t)
    assert len(paired) == 1
    assert paired[0]["sid"] == "a"


def test_join_by_id_handles_empty_inputs():
    assert join_by_id([], []) == []
    assert join_by_id([_s("a", "q", True)], []) == []


# ---------------- stratified_sample ----------------


def test_stratified_sample_targets_wrong_right_only():
    d = [_s(f"l3_{i}", "reach the mug", False) for i in range(5)]
    t = [_s(f"l3_{i}", "reach the mug", True) for i in range(5)]
    d += [_s(f"l2_{i}", "is A left of B?", False) for i in range(5)]
    t += [_s(f"l2_{i}", "is A left of B?", True) for i in range(5)]
    d += [_s(f"ok_{i}", "reach the mug", True) for i in range(3)]
    t += [_s(f"ok_{i}", "reach the mug", True) for i in range(3)]

    paired = join_by_id(d, t)
    chosen, summary = stratified_sample(paired, n_total=6, seed=0)
    assert len(chosen) == 6
    assert summary["chosen_by_level"]["L3"] == 3
    assert summary["chosen_by_level"]["L2"] == 3
    assert all(c["level"] in ("L2", "L3") for c in chosen)


def test_stratified_sample_respects_right_right_exclusion():
    d = [_s(f"x_{i}", "reach the mug", True) for i in range(20)]
    t = [_s(f"x_{i}", "reach the mug", True) for i in range(20)]
    paired = join_by_id(d, t)
    chosen, _ = stratified_sample(paired, n_total=10, seed=0)
    assert len(chosen) == 0


def test_stratified_sample_falls_back_to_l1():
    d = [_s(f"l1_{i}", "how many?", False) for i in range(10)]
    t = [_s(f"l1_{i}", "how many?", True) for i in range(10)]
    paired = join_by_id(d, t)
    chosen, summary = stratified_sample(paired, n_total=5, seed=0)
    assert len(chosen) == 5
    assert summary["chosen_by_level"].get("L1", 0) == 5


def test_stratified_sample_deterministic():
    d = [_s(f"l3_{i}", "reach the mug", False) for i in range(50)]
    t = [_s(f"l3_{i}", "reach the mug", True) for i in range(50)]
    paired = join_by_id(d, t)
    c1, _ = stratified_sample(paired, n_total=10, seed=7)
    c2, _ = stratified_sample(paired, n_total=10, seed=7)
    assert [c["pair"]["sid"] for c in c1] == [c["pair"]["sid"] for c in c2]


def test_stratified_sample_does_not_mutate_paired():
    """C4 fix: stratified_sample must not mutate input list/dicts or shared pools."""
    d = [_s(f"l3_{i}", "reach the mug", False) for i in range(10)]
    t = [_s(f"l3_{i}", "reach the mug", True) for i in range(10)]
    paired = join_by_id(d, t)
    snapshot = copy.deepcopy(paired)
    stratified_sample(paired, n_total=5, seed=0)
    assert paired == snapshot


def test_stratified_sample_idempotent_on_double_call():
    """C4 fix: calling stratified_sample twice on same paired with same seed → same output."""
    d = [_s(f"l3_{i}", "reach the mug", False) for i in range(30)]
    t = [_s(f"l3_{i}", "reach the mug", True) for i in range(30)]
    paired = join_by_id(d, t)
    c1, _ = stratified_sample(paired, n_total=10, seed=3)
    c2, _ = stratified_sample(paired, n_total=10, seed=3)
    assert [c["pair"]["sid"] for c in c1] == [c["pair"]["sid"] for c in c2]


def test_stratified_sample_l3_quota_respected():
    """Verify L3 gets exactly floor(n/2), L2 gets the rest."""
    d = [_s(f"l3_{i}", "reach the mug", False) for i in range(20)]
    t = [_s(f"l3_{i}", "reach the mug", True) for i in range(20)]
    d += [_s(f"l2_{i}", "is A left of B?", False) for i in range(20)]
    t += [_s(f"l2_{i}", "is A left of B?", True) for i in range(20)]
    paired = join_by_id(d, t)
    _, summary = stratified_sample(paired, n_total=10, seed=0)
    assert summary["chosen_by_level"]["L3"] == 5
    assert summary["chosen_by_level"]["L2"] == 5


# ---------------- write_csv ----------------


def test_write_csv_has_expected_columns(tmp_path: Path):
    pair = {
        "distilled": _s("a", "q1", False, ans="wrong"),
        "teacher": _s("a", "q1", True, ans="right"),
        "sid": "a",
    }
    chosen = [{"pair": pair, "question": "q1", "level": "L2"}]
    out = tmp_path / "e.csv"
    write_csv(chosen, out)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0]["sample_id"] == "a"
    assert rows[0]["level"] == "L2"
    assert rows[0]["distilled_answer"] == "wrong"
    assert rows[0]["teacher_answer"] == "right"
    assert rows[0]["error_category"] == ""


def test_write_csv_sanitizes_formula_injection(tmp_path: Path):
    """S4 fix: distilled_answer starting with = gets prefixed with '."""
    pair = {
        "distilled": _s("a", "q1", False, ans="=HYPERLINK(\"http://evil\",\"click\")"),
        "teacher": _s("a", "q1", True, ans="Yes"),
        "sid": "a",
    }
    chosen = [{"pair": pair, "question": "q1", "level": "L2"}]
    out = tmp_path / "e.csv"
    write_csv(chosen, out)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0]["distilled_answer"].startswith("'")
    assert "HYPERLINK" in rows[0]["distilled_answer"]


def test_write_csv_sanitizes_teacher_answer_too(tmp_path: Path):
    pair = {
        "distilled": _s("a", "q1", False, ans="wrong"),
        "teacher": _s("a", "q1", True, ans="+cmd|calc"),
        "sid": "a",
    }
    chosen = [{"pair": pair, "question": "q1", "level": "L2"}]
    out = tmp_path / "e.csv"
    write_csv(chosen, out)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0]["teacher_answer"].startswith("'")


def test_write_csv_benign_values_unchanged(tmp_path: Path):
    pair = {
        "distilled": _s("a", "q1", False, ans="2 cats"),
        "teacher": _s("a", "q1", True, ans="3 cats"),
        "sid": "a",
    }
    chosen = [{"pair": pair, "question": "q1", "level": "L1"}]
    out = tmp_path / "e.csv"
    write_csv(chosen, out)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0]["distilled_answer"] == "2 cats"
    assert rows[0]["teacher_answer"] == "3 cats"
    assert not rows[0]["distilled_answer"].startswith("'")


def test_write_csv_crlf_in_question_stripped(tmp_path: Path):
    pair = {
        "distilled": _s("a", "line1\r\nline2", False, ans="x"),
        "teacher": _s("a", "line1\r\nline2", True, ans="y"),
        "sid": "a",
    }
    chosen = [{"pair": pair, "question": "line1\r\nline2", "level": "L1"}]
    out = tmp_path / "e.csv"
    write_csv(chosen, out)
    raw = out.read_text(encoding="utf-8")
    # Header line + one data line + trailing newline = 2 or 3 lines max (no embedded CR)
    assert "\r\n" not in raw or raw.count("\n") <= 3


# ---------------- lmms-eval schema variants ----------------


def test_stratified_sample_with_lmms_eval_resps_form():
    """Test with the raw lmms-eval resps shape (list-of-list)."""
    d = []
    t = []
    for i in range(6):
        d.append(
            {
                "doc_id": i,
                "doc": {"question": "reach the red cup"},
                "resps": [["wrong"]],
                "cv_bench_acc,none": 0.0,
            }
        )
        t.append(
            {
                "doc_id": i,
                "doc": {"question": "reach the red cup"},
                "resps": [["right"]],
                "cv_bench_acc,none": 1.0,
            }
        )
    paired = join_by_id(d, t)
    chosen, _ = stratified_sample(paired, n_total=4, seed=0)
    assert len(chosen) == 4
    assert all(c["level"] == "L3" for c in chosen)
