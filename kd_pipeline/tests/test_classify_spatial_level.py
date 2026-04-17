"""T5 refactored · scripts/classify_spatial_level.py

Now a thin CLI wrapper over src.spatial_vocab + src.lmms_eval_io.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from classify_spatial_level import level_stats, recovery  # noqa: E402
from src.lmms_eval_io import load_samples  # noqa: E402
from src.spatial_vocab import classify_level  # noqa: E402


# ---------------- classify_level via re-export ----------------


@pytest.mark.parametrize(
    "q, expected",
    [
        ("How many apples are on the table?", "L1"),
        ("What color is the cat?", "L1"),
        ("Is the dog to the left of the cat?", "L2"),
        ("Which object is closer to the camera?", "L2"),
        ("What is directly above the lamp?", "L2"),
        ("From the camera's perspective, which object would you reach first?", "L3"),
        ("If you walked forward from this viewpoint, what would you encounter?", "L3"),
        ("Navigate to the red chair; what blocks your path?", "L3"),
    ],
)
def test_classify_level_rules(q, expected):
    assert classify_level(q) == expected


def test_classify_level_l3_over_l2_when_both_present():
    assert classify_level("Reach for the object to the left of the table") == "L3"


def test_classify_level_empty_is_l1():
    assert classify_level("") == "L1"


def test_classify_level_substring_safety_l2():
    assert classify_level("What is behindsight meaning?") == "L1"


def test_classify_level_substring_safety_l3_critical():
    """C1 fix: bare-substring matching for L3 no longer triggers false positives."""
    assert classify_level("I see a sidewalk in the image.") != "L3"
    assert classify_level("What research does this poster show?") != "L3"
    assert classify_level("Is this location unapproachable?") != "L3"


# ---------------- load_samples ----------------


def test_load_samples_jsonl(tmp_path: Path):
    p = tmp_path / "s.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"question": "how many", "correct": True}),
                json.dumps({"question": "left or right?", "correct": False}),
            ]
        ),
        encoding="utf-8",
    )
    assert len(load_samples(p)) == 2


def test_load_samples_lmms_eval_cv_bench_shape(tmp_path: Path):
    """Real lmms-eval sample shape with ,none suffix and nested doc."""
    p = tmp_path / "s.jsonl"
    p.write_text(
        json.dumps(
            {
                "doc_id": 0,
                "doc": {"question": "Is A left of B?"},
                "cv_bench_acc,none": 1.0,
                "filtered_resps": ["Yes"],
            }
        ),
        encoding="utf-8",
    )
    out = load_samples(p)
    assert len(out) == 1


# ---------------- level_stats ----------------


def test_level_stats_distribution():
    samples = [
        {"question": "how many cats?", "correct": True},  # L1
        {"question": "how many dogs?", "correct": False},  # L1
        {"question": "is A to the left of B?", "correct": True},  # L2
        {"question": "reach the mug", "correct": False},  # L3
    ]
    st = level_stats(samples)
    assert st["L1"]["n"] == 2 and st["L1"]["correct"] == 1 and st["L1"]["acc"] == 0.5
    assert st["L2"]["n"] == 1 and st["L2"]["acc"] == 1.0
    assert st["L3"]["n"] == 1 and st["L3"]["acc"] == 0.0


def test_level_stats_uses_lmms_eval_schema():
    """Samples in lmms-eval shape (doc.question + cv_bench_acc,none) parsed correctly."""
    samples = [
        {
            "doc_id": 0,
            "doc": {"question": "Reach the cup"},
            "cv_bench_acc,none": 1.0,
        },
        {
            "doc_id": 1,
            "doc": {"question": "How many cats?"},
            "cv_bench_acc,none": 0.0,
        },
    ]
    st = level_stats(samples)
    assert st["L3"]["n"] == 1 and st["L3"]["correct"] == 1
    assert st["L1"]["n"] == 1 and st["L1"]["correct"] == 0


def test_level_stats_handles_unknown_correctness():
    samples = [{"question": "how many cats?"}]  # no correct key
    st = level_stats(samples)
    assert st["L1"]["n"] == 1
    assert st["_meta"]["unknown_correctness"] == 1


def test_level_stats_all_empty():
    st = level_stats([])
    assert st["L1"]["n"] == 0
    assert st["_meta"]["total"] == 0


# ---------------- recovery ----------------


def test_recovery_basic():
    assert recovery(0.75, 0.5, 1.0) == pytest.approx(50.0)


def test_recovery_zero_gap_returns_nan():
    import math
    assert math.isnan(recovery(0.5, 0.5, 0.5))


def test_recovery_negative_when_below_baseline():
    assert recovery(0.4, 0.5, 1.0) == pytest.approx(-20.0)


def test_recovery_over_100_when_above_teacher():
    assert recovery(1.1, 0.5, 1.0) == pytest.approx(120.0)
