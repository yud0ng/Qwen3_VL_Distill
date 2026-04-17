"""T2 · src/lmms_eval_io.py — schema probing for lmms-eval outputs.

Verified schema (per Explore review):
  - doc_id, doc, target, resps (list[list[str]]), filtered_resps, arguments
  - CV-Bench correctness: cv_bench_acc binary 1.0/0.0
  - Metric keys often carry ",none" suffix
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lmms_eval_io import (  # noqa: E402
    extract_answer_text,
    extract_correctness,
    extract_image_path,
    extract_question,
    extract_sample_id,
    load_samples,
)


# ---------------- load_samples ----------------


def test_load_samples_jsonl(tmp_path: Path):
    p = tmp_path / "s.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"doc_id": 0, "doc": {"question": "q1"}}),
                json.dumps({"doc_id": 1, "doc": {"question": "q2"}}),
            ]
        ),
        encoding="utf-8",
    )
    out = load_samples(p)
    assert len(out) == 2


def test_load_samples_json_list(tmp_path: Path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"doc_id": 0}]), encoding="utf-8")
    assert load_samples(p) == [{"doc_id": 0}]


def test_load_samples_json_wrapped_samples_key(tmp_path: Path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"samples": [{"doc_id": 0}]}), encoding="utf-8")
    assert load_samples(p) == [{"doc_id": 0}]


def test_load_samples_json_wrapped_logs_key(tmp_path: Path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"logs": [{"doc_id": 0}]}), encoding="utf-8")
    assert load_samples(p) == [{"doc_id": 0}]


def test_load_samples_empty_file(tmp_path: Path):
    p = tmp_path / "s.json"
    p.write_text("", encoding="utf-8")
    assert load_samples(p) == []


def test_load_samples_missing_file(tmp_path: Path):
    assert load_samples(tmp_path / "nope.json") == []


# ---------------- extract_sample_id ----------------


def test_extract_sample_id_from_doc_id():
    assert extract_sample_id({"doc_id": 42}) == "42"


def test_extract_sample_id_from_id():
    assert extract_sample_id({"id": "abc"}) == "abc"


def test_extract_sample_id_from_doc_nested():
    assert extract_sample_id({"doc": {"id": "nested"}}) == "nested"


def test_extract_sample_id_from_doc_id_field_nested():
    assert extract_sample_id({"doc": {"doc_id": 7}}) == "7"


def test_extract_sample_id_fallback_hash_on_question():
    sid = extract_sample_id({"doc": {"question": "what?"}})
    assert sid.startswith("hash_")


def test_extract_sample_id_stable_across_calls():
    s = {"doc": {"question": "same question"}}
    assert extract_sample_id(s) == extract_sample_id(s)


# ---------------- extract_question ----------------


def test_extract_question_from_doc_nested():
    assert extract_question({"doc": {"question": "q1"}}) == "q1"


def test_extract_question_from_top_level():
    assert extract_question({"question": "q2"}) == "q2"


def test_extract_question_from_prompt_fallback():
    assert extract_question({"prompt": "p"}) == "p"


def test_extract_question_missing_returns_empty():
    assert extract_question({}) == ""


# ---------------- extract_correctness ----------------


def test_extract_correctness_cv_bench_acc_none_suffix():
    assert extract_correctness({"cv_bench_acc,none": 1.0}) is True
    assert extract_correctness({"cv_bench_acc,none": 0.0}) is False


def test_extract_correctness_cv_bench_acc_no_suffix():
    assert extract_correctness({"cv_bench_acc": 1}) is True


def test_extract_correctness_correct_bool():
    assert extract_correctness({"correct": True}) is True
    assert extract_correctness({"correct": False}) is False


def test_extract_correctness_acc_float():
    assert extract_correctness({"acc": 1.0}) is True
    assert extract_correctness({"acc": 0.0}) is False


def test_extract_correctness_missing_returns_none():
    assert extract_correctness({}) is None


def test_extract_correctness_nested_metrics():
    assert extract_correctness({"metrics": {"cv_bench_acc": 1.0}}) is True


# ---------------- extract_answer_text ----------------


def test_extract_answer_text_from_filtered_resps():
    assert extract_answer_text({"filtered_resps": ["answer text"]}) == "answer text"


def test_extract_answer_text_from_resps_nested():
    assert extract_answer_text({"resps": [["nested answer"]]}) == "nested answer"


def test_extract_answer_text_from_response_string():
    assert extract_answer_text({"response": "plain"}) == "plain"


def test_extract_answer_text_missing_returns_empty():
    assert extract_answer_text({}) == ""


# ---------------- extract_image_path ----------------


def test_extract_image_path_from_doc():
    assert extract_image_path({"doc": {"image": "/a.jpg"}}) == "/a.jpg"


def test_extract_image_path_from_image_path_field():
    assert extract_image_path({"image_path": "/b.png"}) == "/b.png"


def test_extract_image_path_missing_returns_empty():
    assert extract_image_path({}) == ""


# ---------------- realistic lmms-eval sample ----------------


def test_realistic_lmms_eval_cv_bench_sample():
    """Minimal sample matching verified lmms-eval v0.7 cv_bench output shape."""
    sample = {
        "doc_id": 17,
        "doc": {
            "idx": 17,
            "question": "Is the cup to the left of the plate?",
            "image": "train2014/example.jpg",
            "source": "cv_bench",
        },
        "target": "Yes",
        "arguments": {"gen_kwargs": {"temperature": 0.0}},
        "resps": [["The cup is to the left of the plate."]],
        "filtered_resps": ["Yes"],
        "cv_bench_acc,none": 1.0,
    }
    assert extract_sample_id(sample) == "17"
    assert extract_question(sample) == "Is the cup to the left of the plate?"
    assert extract_correctness(sample) is True
    assert extract_answer_text(sample) == "Yes"
    assert extract_image_path(sample) == "train2014/example.jpg"
