"""T7 refactored · scripts/select_logit_subset.py

Addresses H3 (missing-id counter) and ensures no shared-pool mutation.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from select_logit_subset import bucket_key, select  # noqa: E402


def _row(sid: str | None, source: str, conf: int | None, q: str = "") -> dict:
    d = {"source": source, "confidence": conf, "question": q}
    if sid is not None:
        d["id"] = sid
    return d


# ---------------- bucket_key ----------------


@pytest.mark.parametrize(
    "row, expected",
    [
        (_row("a", "coco_spatial", 5), "B1_spatial_conf5"),
        (_row("a", "coco_spatial", 4), "B2_spatial_conf4"),
        (_row("a", "llava_general", 5, "Is the dog to the left of the cat?"), "B3_general_spatial_question_conf5"),
        (_row("a", "llava_general", 5, "What color is the cat?"), "B5_other"),
        (_row("a", "coco_spatial", 3), "B4_spatial_other"),
    ],
)
def test_bucket_key_cases(row, expected):
    assert bucket_key(row) == expected


def test_bucket_key_l3_question_routes_to_general_spatial():
    r = _row("a", "llava_general", 5, "Reach the mug in front of the plate")
    assert bucket_key(r) == "B3_general_spatial_question_conf5"


# ---------------- select ----------------


def test_select_fills_priority_order():
    rows = (
        [_row(f"s5_{i}", "coco_spatial", 5) for i in range(3)]
        + [_row(f"s4_{i}", "coco_spatial", 4) for i in range(3)]
        + [_row(f"g_{i}", "llava_general", 5, "Is A to the left of B?") for i in range(3)]
    )
    ids, manifest = select(rows, n=5, seed=0)
    assert len(ids) == 5
    assert manifest["chosen_by_bucket"]["B1_spatial_conf5"] == 3
    assert manifest["chosen_by_bucket"]["B2_spatial_conf4"] == 2
    assert "B3_general_spatial_question_conf5" not in manifest["chosen_by_bucket"]


def test_select_caps_at_request():
    rows = [_row(f"s_{i}", "coco_spatial", 5) for i in range(20)]
    ids, manifest = select(rows, n=7, seed=0)
    assert len(ids) == 7
    assert manifest["selected"] == 7


def test_select_insufficient_returns_all_available():
    rows = [_row(f"s_{i}", "coco_spatial", 5) for i in range(3)]
    ids, _ = select(rows, n=100, seed=0)
    assert len(ids) == 3


def test_select_deterministic_with_seed():
    rows = [_row(f"s_{i}", "coco_spatial", 5) for i in range(50)]
    ids1, _ = select(rows, n=10, seed=7)
    ids2, _ = select(rows, n=10, seed=7)
    assert ids1 == ids2


def test_select_skips_rows_missing_id():
    """H3 fix: rows without id are counted, not silently dropped."""
    rows = [
        _row("good", "coco_spatial", 5),
        _row(None, "coco_spatial", 5),  # missing id
        _row("", "coco_spatial", 5),    # empty id
    ]
    ids, manifest = select(rows, n=5, seed=0)
    assert "good" in ids
    assert manifest.get("skipped_no_id", 0) >= 2


def test_select_does_not_mutate_input():
    """C4-analog: select must not reorder or mutate the input list of dicts."""
    rows = [_row(f"s_{i}", "coco_spatial", 5) for i in range(20)]
    snapshot = copy.deepcopy(rows)
    select(rows, n=5, seed=0)
    assert rows == snapshot


def test_select_idempotent_on_repeated_calls():
    """Calling select twice on the same input with same seed yields same output."""
    rows = [_row(f"s_{i}", "coco_spatial", 5) for i in range(20)]
    ids1, _ = select(rows, n=7, seed=42)
    ids2, _ = select(rows, n=7, seed=42)
    assert ids1 == ids2


def test_select_manifest_contains_all_bucket_totals():
    rows = [_row(f"s_{i}", "coco_spatial", 5) for i in range(5)]
    rows += [_row(f"g_{i}", "llava_general", 5) for i in range(3)]
    _, manifest = select(rows, n=3, seed=0)
    assert "B1_spatial_conf5" in manifest["total_by_bucket"]
    assert manifest["total_by_bucket"]["B1_spatial_conf5"] == 5


def test_select_zero_n_returns_empty():
    rows = [_row(f"s_{i}", "coco_spatial", 5) for i in range(5)]
    ids, manifest = select(rows, n=0, seed=0)
    assert ids == []
    assert manifest["selected"] == 0
