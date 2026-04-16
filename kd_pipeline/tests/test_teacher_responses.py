import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.teacher_responses import (
    extract_trace_and_answer,
    normalize_teacher_text,
    row_teacher_responses,
)


def test_answer_only_extracts_inner():
    r = "Some reasoning.\n\n<answer>Gray and white</answer><confidence>5</confidence>"
    assert normalize_teacher_text(r, target="answer_only") == "Gray and white"


def test_min_confidence_skip():
    obj = {
        "id": "x",
        "question": "q?",
        "response": "<answer>a</answer><confidence>5</confidence>",
        "image": "/tmp/none",
        "confidence": 2,
    }
    u, a, img, m = row_teacher_responses(obj, min_confidence=4)
    assert m["skip"] is True


def test_extract_trace_before_answer_tag():
    r = "Reasoning line.\n\n<answer>Yes</answer><confidence>5</confidence>"
    tr, ans = extract_trace_and_answer(r)
    assert "Reasoning" in tr
    assert ans == "Yes"


def test_row_ok():
    obj = {
        "id": "x",
        "question": "What?",
        "response": "Blah.\n\n<answer>Yes</answer><confidence>5</confidence>",
        "image": "/tmp/x.jpg",
        "confidence": 5,
        "source": "llava_general",
        "type": "general",
    }
    u, a, img, m = row_teacher_responses(obj, min_confidence=0)
    assert u == "What?"
    assert a == "Yes"
    assert img == "/tmp/x.jpg"
    assert m["skip"] is False
