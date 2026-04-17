"""T12 · end-to-end integration test of D's full pipeline on synthetic fixtures.

Covers:
  M1 filter_cot_quality   (from synthetic CoT responses)
  M2 classify_spatial_level (from synthetic lmms-eval samples)
  M3 select_logit_subset  (from synthetic teacher_responses)
  M4 build_teacher_cache  (from synthetic teacher_responses)
  M6 sample_error_cases   (from synthetic distilled+teacher samples)

Demo UI (M5) is visually verified; its helpers covered in test_demo_app.py.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(ROOT))


# ---------------- fixtures ----------------


@pytest.fixture
def teacher_jsonl(tmp_path: Path) -> Path:
    """20-sample synthetic teacher_responses.jsonl with mixed sources/types/confidences."""
    rows = []
    # 10 spatial samples: 7 conf=5, 2 conf=4, 1 conf=3
    for i in range(7):
        rows.append(
            {
                "id": f"coco_spatial_{i:04d}",
                "source": "coco_spatial",
                "type": "relational" if i % 2 else "metric",
                "image": f"/ocean/datasets/coco/train2014/COCO_{i:08d}.jpg",
                "question": f"Is object A to the left of B in scene {i}?",
                "response": f"<answer>Left</answer><confidence>5</confidence>",
                "confidence": 5,
            }
        )
    for i in range(2):
        rows.append(
            {
                "id": f"coco_spatial_{i+7:04d}",
                "source": "coco_spatial",
                "type": "egocentric",
                "image": f"/ocean/datasets/coco/train2014/COCO_{i+7:08d}.jpg",
                "question": f"Reach the mug in scene {i}",
                "response": f"<answer>Yes</answer><confidence>4</confidence>",
                "confidence": 4,
            }
        )
    rows.append(
        {
            "id": "coco_spatial_0009",
            "source": "coco_spatial",
            "type": "relational",
            "image": "/ocean/datasets/coco/train2014/COCO_9.jpg",
            "question": "Is A closer?",
            "response": "<answer>Yes</answer><confidence>3</confidence>",
            "confidence": 3,
        }
    )
    # 10 general samples: all conf=5, varied questions
    for i in range(10):
        q = "How many cats?" if i < 4 else (
            "Is A to the left of B?" if i < 7 else "What color is the cat?"
        )
        rows.append(
            {
                "id": f"llava_general_{i:04d}",
                "source": "llava_general",
                "type": "general",
                "image": f"/ocean/datasets/coco/train2014/COCO_{i+100:08d}.jpg",
                "question": q,
                "response": f"<answer>Answer {i}</answer><confidence>5</confidence>",
                "confidence": 5,
            }
        )

    p = tmp_path / "teacher.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


@pytest.fixture
def teacher_cot_jsonl(tmp_path: Path) -> Path:
    """Synthetic CoT-enabled teacher output with <think> tags (as if A ran thinking=True)."""
    good_cot = (
        "<think>The cup is to the left of the plate. Because the cup sits closer "
        "to the camera than the plate, therefore its depth is shallower. "
        "Since the plate is behind the cup, the distance between them is small. "
        "Hence the answer is cup.</think>"
        "<answer>The cup</answer><confidence>5</confidence>"
    )
    bad_short = "<think>left.</think><answer>A</answer><confidence>5</confidence>"
    bad_conf = good_cot.replace("<confidence>5</confidence>", "<confidence>3</confidence>")

    rows = (
        [
            {"id": f"good_{i}", "source": "coco_spatial", "type": "relational",
             "image": "img.jpg", "question": "q", "response": good_cot, "confidence": 5}
            for i in range(5)
        ]
        + [
            {"id": f"short_{i}", "source": "coco_spatial", "type": "relational",
             "image": "img.jpg", "question": "q", "response": bad_short, "confidence": 5}
            for i in range(3)
        ]
        + [
            {"id": f"conf_{i}", "source": "coco_spatial", "type": "relational",
             "image": "img.jpg", "question": "q", "response": bad_conf, "confidence": 3}
            for i in range(2)
        ]
    )
    p = tmp_path / "teacher_cot.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


@pytest.fixture
def lmms_samples(tmp_path: Path):
    """Return (distilled, teacher) per-sample JSONL paths in lmms-eval v0.7 shape."""
    def mk(ok_ratio: float, stem: str) -> Path:
        rows = []
        for i in range(30):
            q = (
                "Reach the mug" if i < 10 else
                "Is A to the left of B?" if i < 20 else
                "How many cats?"
            )
            ok = 1.0 if (i % 100) / 100 < ok_ratio else 0.0
            rows.append(
                {
                    "doc_id": i,
                    "doc": {"question": q, "image": f"img_{i}.jpg"},
                    "filtered_resps": ["Yes" if ok else "No"],
                    "cv_bench_acc,none": ok,
                }
            )
        p = tmp_path / f"{stem}.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return p

    # Distilled: ~40% correct; Teacher: 100% correct → will generate 18 error cases
    distilled_rows = []
    teacher_rows = []
    for i in range(30):
        q = (
            "Reach the mug" if i < 10 else
            "Is A to the left of B?" if i < 20 else
            "How many cats?"
        )
        d_ok = 1.0 if i % 3 == 0 else 0.0  # 10 right, 20 wrong
        distilled_rows.append(
            {
                "doc_id": i,
                "doc": {"question": q, "image": f"img_{i}.jpg"},
                "filtered_resps": ["Yes" if d_ok else "Wrong"],
                "cv_bench_acc,none": d_ok,
            }
        )
        teacher_rows.append(
            {
                "doc_id": i,
                "doc": {"question": q, "image": f"img_{i}.jpg"},
                "filtered_resps": ["Yes"],
                "cv_bench_acc,none": 1.0,  # teacher always right
            }
        )

    d_p = tmp_path / "distilled.jsonl"
    t_p = tmp_path / "teacher_eval.jsonl"
    d_p.write_text("\n".join(json.dumps(r) for r in distilled_rows), encoding="utf-8")
    t_p.write_text("\n".join(json.dumps(r) for r in teacher_rows), encoding="utf-8")
    return d_p, t_p


# ---------------- e2e tests ----------------


def test_e2e_m3_logit_subset(tmp_path: Path, teacher_jsonl: Path):
    from select_logit_subset import _main
    out_ids = tmp_path / "ids.txt"
    code = _main(
        [
            "--input", str(teacher_jsonl),
            "--n", "7",
            "--out_ids", str(out_ids),
            "--seed", "0",
        ]
    )
    assert code == 0
    ids = out_ids.read_text(encoding="utf-8").splitlines()
    assert len(ids) == 7
    # All chosen IDs should be from coco_spatial (priority B1)
    assert all(sid.startswith("coco_spatial_") for sid in ids)


def test_e2e_m4_build_teacher_cache(tmp_path: Path, teacher_jsonl: Path):
    from build_teacher_cache import _main
    out_json = tmp_path / "cache.json"
    code = _main(
        [
            "--input", str(teacher_jsonl),
            "--out", str(out_json),
            "--per_category", "3",
            "--image_root", "./local/coco",
        ]
    )
    assert code == 0
    cache = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(cache) > 0
    # All image paths rewritten
    assert all("/ocean/" not in e["image"] for e in cache)
    assert any("./local/coco" in e["image"] or "local/coco" in e["image"] for e in cache)


def test_e2e_m1_filter_cot(tmp_path: Path, teacher_cot_jsonl: Path):
    from filter_cot_quality import _main
    out_pass = tmp_path / "pass.jsonl"
    out_fail = tmp_path / "fail.jsonl"
    report = tmp_path / "report.json"
    code = _main(
        [
            "--input", str(teacher_cot_jsonl),
            "--out_pass", str(out_pass),
            "--out_fail", str(out_fail),
            "--report", str(report),
        ]
    )
    assert code == 0
    report_data = json.loads(report.read_text(encoding="utf-8"))
    # 5 good_* should pass (they contain all required elements)
    assert report_data["n_pass"] == 5
    assert report_data["n_in"] == 10


def test_e2e_m2_classify_spatial_level(tmp_path: Path, lmms_samples):
    from classify_spatial_level import _main
    distilled, teacher = lmms_samples
    out_csv = tmp_path / "per_level.csv"
    code = _main(
        [
            "--samples_jsonl", str(distilled),
            "--baseline_samples", str(distilled),
            "--teacher_samples", str(teacher),
            "--out_csv", str(out_csv),
        ]
    )
    assert code == 0
    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    levels = {r["level"] for r in rows}
    assert levels == {"L1", "L2", "L3"}


def test_e2e_m6_sample_error_cases(tmp_path: Path, lmms_samples):
    from sample_error_cases import _main
    distilled, teacher = lmms_samples
    out_csv = tmp_path / "errors.csv"
    code = _main(
        [
            "--distilled_samples", str(distilled),
            "--teacher_samples", str(teacher),
            "--n", "10",
            "--out_csv", str(out_csv),
        ]
    )
    assert code == 0
    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert len(rows) == 10
    # All rows should have populated required columns
    for r in rows:
        assert r["sample_id"]
        assert r["level"] in ("L1", "L2", "L3")
        assert r["question"]
        assert r["error_category"] == ""  # blank for human to fill


def test_e2e_pipeline_all_outputs_shaped_correctly(
    tmp_path: Path, teacher_jsonl: Path, lmms_samples
):
    """Full chain: teacher data → logit subset, cache; eval outputs → per-level, errors."""
    from build_teacher_cache import _main as build_cache_main
    from classify_spatial_level import _main as classify_main
    from sample_error_cases import _main as sample_err_main
    from select_logit_subset import _main as select_main

    out_ids = tmp_path / "ids.txt"
    out_cache = tmp_path / "cache.json"
    out_level = tmp_path / "level.csv"
    out_errors = tmp_path / "errors.csv"

    assert select_main(
        ["--input", str(teacher_jsonl), "--n", "5", "--out_ids", str(out_ids)]
    ) == 0
    assert build_cache_main(
        [
            "--input", str(teacher_jsonl),
            "--out", str(out_cache),
            "--per_category", "2",
        ]
    ) == 0
    distilled, teacher = lmms_samples
    assert classify_main(
        [
            "--samples_jsonl", str(distilled),
            "--baseline_samples", str(distilled),
            "--teacher_samples", str(teacher),
            "--out_csv", str(out_level),
        ]
    ) == 0
    assert sample_err_main(
        [
            "--distilled_samples", str(distilled),
            "--teacher_samples", str(teacher),
            "--n", "5",
            "--out_csv", str(out_errors),
        ]
    ) == 0

    # Validate shapes
    assert len(out_ids.read_text(encoding="utf-8").splitlines()) == 5
    assert len(json.loads(out_cache.read_text(encoding="utf-8"))) > 0
    assert len(list(csv.DictReader(out_level.open(encoding="utf-8")))) == 3  # L1/L2/L3
    err_rows = list(csv.DictReader(out_errors.open(encoding="utf-8")))
    assert len(err_rows) == 5
