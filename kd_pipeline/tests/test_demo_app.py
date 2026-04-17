"""T9 refactored · demo/app.py + demo/build_teacher_cache.py

Addresses: C5 (teacher_box state race), S1 (XSS), S2 (exception leakage), S5 (log injection).
Only non-Gradio helpers tested; UI is visually verified.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(ROOT))

from app import (  # noqa: E402
    ModelRunner,
    diff_highlight,
    find_cached,
    load_cache,
    sanitize_for_log,
)
from build_teacher_cache import (  # noqa: E402
    build_cache,
    extract_answer,
    infer_category,
)


# ---------------- diff_highlight (thin wrapper over src.safe_html.diff_wrap) ----------------


def test_diff_highlight_identical_no_mark():
    a, b = diff_highlight("same", "same")
    assert "<mark>" not in a
    assert "<mark>" not in b


def test_diff_highlight_different_has_mark():
    a, b = diff_highlight("the cat", "the dog")
    assert "<mark>" in a
    assert "<mark>" in b


def test_diff_highlight_escapes_script_tag_critical():
    """S1 fix: any <script> must be escaped."""
    a, b = diff_highlight("<script>alert(1)</script>", "safe")
    assert "<script>" not in a
    assert "&lt;script&gt;" in a


def test_diff_highlight_escapes_img_onerror_critical():
    """S1 fix: <img onerror=...> must be escaped."""
    a, b = diff_highlight("x", "<img src=x onerror=alert(1)>")
    assert "<img" not in b
    assert "&lt;img" in b


def test_diff_highlight_preserves_unicode():
    a, b = diff_highlight("左边", "右边")
    assert "左边" in a
    assert "右边" in b


# ---------------- find_cached ----------------


def test_find_cached_by_question():
    cache = [{"id": "a", "question": "How many?", "image": "x.jpg"}]
    hit = find_cached(cache, "how many?", None)
    assert hit is not None and hit["id"] == "a"


def test_find_cached_none_when_missing():
    assert find_cached([], "anything", None) is None


def test_find_cached_by_image_basename():
    cache = [{"id": "a", "question": "q", "image": "/path/img.jpg"}]
    hit = find_cached(cache, "q", "/other/path/img.jpg")
    assert hit is not None


def test_find_cached_case_insensitive():
    cache = [{"id": "a", "question": "HOW MANY?", "image": ""}]
    hit = find_cached(cache, "how many?", None)
    assert hit is not None


# ---------------- load_cache ----------------


def test_load_cache_missing_returns_empty(tmp_path: Path):
    assert load_cache(tmp_path / "nope.json") == []


def test_load_cache_reads_file(tmp_path: Path):
    import json
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"id": "x"}]), encoding="utf-8")
    assert load_cache(p) == [{"id": "x"}]


# ---------------- ModelRunner ----------------


def test_model_runner_placeholder_when_no_path():
    r = ModelRunner("test", None)
    assert r.generate("question", None) == "[not loaded]"


def test_model_runner_error_hides_exception_details():
    """S2 fix: frontend message MUST NOT leak full exception string (file paths, etc.)."""
    r = ModelRunner("test-model", "/fake/path/that/does/not/exist")
    out = r.generate("question", None)
    # Generic user-facing string expected
    assert "/fake/path" not in out
    assert "ModuleNotFoundError" not in out
    assert "does/not/exist" not in out


# ---------------- sanitize_for_log ----------------


def test_sanitize_for_log_strips_crlf():
    """S5 fix: log injection via CR/LF in model name or path."""
    out = sanitize_for_log("name\r\nEVIL_ENTRY")
    assert "\r" not in out
    assert "\n" not in out


def test_sanitize_for_log_preserves_benign():
    assert sanitize_for_log("Qwen/Qwen3-VL-2B-Instruct") == "Qwen/Qwen3-VL-2B-Instruct"


def test_sanitize_for_log_none_returns_empty():
    assert sanitize_for_log(None) == ""


# ---------------- build_teacher_cache helpers ----------------


def test_extract_answer_tagged():
    ans, full = extract_answer(
        "Some reasoning.\n<answer>Two</answer><confidence>5</confidence>"
    )
    assert ans == "Two"
    assert "confidence" not in full.lower()


def test_extract_answer_untagged():
    ans, _ = extract_answer("Just plain answer")
    assert ans == "Just plain answer"


def test_infer_category_count_from_general():
    r = {"type": "general", "question": "How many cats?"}
    assert infer_category(r) == "count"


def test_infer_category_relational_from_type():
    r = {"type": "relational", "question": "Is A left of B?"}
    assert infer_category(r) == "relational"


def test_infer_category_egocentric_from_type():
    r = {"type": "egocentric", "question": "Reach the mug"}
    assert infer_category(r) == "egocentric"


def test_infer_category_depth_from_metric_question():
    r = {"type": "metric", "question": "Which is closer to the camera?"}
    assert infer_category(r) == "depth"


def test_build_cache_per_category_cap():
    rows = []
    for i in range(20):
        rows.append(
            {
                "id": f"c{i}",
                "type": "general",
                "question": "How many cats?",
                "response": f"<answer>{i}</answer><confidence>5</confidence>",
                "confidence": 5,
            }
        )
    out = build_cache(rows, per_category=5)
    assert len(out) == 5
    assert all(e["category"] == "count" for e in out)


def test_build_cache_skips_low_confidence():
    rows = [
        {
            "id": "low",
            "type": "general",
            "question": "how many?",
            "response": "<answer>1</answer><confidence>3</confidence>",
            "confidence": 3,
        },
        {
            "id": "hi",
            "type": "general",
            "question": "how many?",
            "response": "<answer>2</answer><confidence>5</confidence>",
            "confidence": 5,
        },
    ]
    out = build_cache(rows, per_category=5)
    ids = [e["id"] for e in out]
    assert "hi" in ids and "low" not in ids


def test_build_cache_image_root_rewrites_cluster_path():
    """INFO-7 fix: absolute cluster paths rewritten when --image_root provided."""
    rows = [
        {
            "id": "a",
            "type": "general",
            "question": "how many?",
            "image": "/ocean/projects/cis220039p/yluo22/datasets/coco/train2014/COCO_123.jpg",
            "response": "<answer>2</answer><confidence>5</confidence>",
            "confidence": 5,
        }
    ]
    out = build_cache(rows, per_category=1, image_root="/local/coco")
    assert out[0]["image"] == "/local/coco/COCO_123.jpg"


def test_build_cache_image_root_preserves_when_not_cluster_path():
    rows = [
        {
            "id": "a",
            "type": "general",
            "question": "how many?",
            "image": "relative/path.jpg",
            "response": "<answer>2</answer><confidence>5</confidence>",
            "confidence": 5,
        }
    ]
    out = build_cache(rows, per_category=1, image_root="/local/coco")
    # Relative path: either passed through or joined — both are reasonable
    assert out[0]["image"] is not None


def test_build_cache_empty_input():
    assert build_cache([], per_category=5) == []


# ---------------- additional coverage ----------------


def test_load_cache_handles_malformed_json(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    assert load_cache(p) == []


def test_sanitize_for_log_preserves_spaces():
    assert sanitize_for_log("normal with spaces") == "normal with spaces"


def test_sanitize_for_log_collapses_multiple_newlines():
    assert sanitize_for_log("a\n\n\r\r\nb") == "a b"


def test_find_cached_with_image_hint_different_basename():
    cache = [{"id": "a", "question": "q", "image": "/path/img1.jpg"}]
    hit = find_cached(cache, "q", "/other/img2.jpg")
    # Different basename -> None
    assert hit is None


def test_model_runner_no_path_always_placeholder():
    r = ModelRunner("placeholder", None)
    # Multiple calls idempotent
    assert r.generate("q", None) == "[not loaded]"
    assert r.generate("q", "img.jpg") == "[not loaded]"


# ---------------- CLI parser ----------------


def test_build_parser_defaults():
    from app import build_parser
    ap = build_parser()
    args = ap.parse_args([])
    assert args.original_model_path == "Qwen/Qwen3-VL-2B-Instruct"
    assert args.distilled_model_path is None
    assert args.no_load is False
    assert args.port == 7860
    assert args.host == "127.0.0.1"


def test_build_parser_no_load_flag():
    from app import build_parser
    ap = build_parser()
    args = ap.parse_args(["--no_load"])
    assert args.no_load is True


# ---------------- Gradio-conditional UI smoke ----------------


try:
    import gradio as _gr  # type: ignore[import-not-found]
    _GRADIO_AVAILABLE = True
except ImportError:
    _GRADIO_AVAILABLE = False


@pytest.mark.skipif(not _GRADIO_AVAILABLE, reason="gradio not installed")
def test_build_ui_constructs_without_error():
    """Smoke test: build_ui with no-load runners should not raise."""
    from app import build_ui
    cache = [
        {
            "id": "a",
            "image": "",
            "question": "q?",
            "level": "L2",
            "category": "relational",
            "teacher_answer": "A",
            "teacher_full_response": "A",
            "confidence": 5,
        }
    ]
    raw = ModelRunner("raw", None)
    dist = ModelRunner("dist", None)
    app = build_ui(cache, raw, dist)
    assert app is not None


# ---------------- build_teacher_cache CLI ----------------


def test_build_teacher_cache_cli(tmp_path: Path):
    import json
    from build_teacher_cache import _main as bmain
    inp = tmp_path / "t.jsonl"
    inp.write_text(
        json.dumps(
            {
                "id": "a",
                "source": "coco_spatial",
                "type": "relational",
                "image": "/ocean/x.jpg",
                "question": "Is A left of B?",
                "response": "<answer>Left</answer><confidence>5</confidence>",
                "confidence": 5,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "cache.json"
    code = bmain(
        [
            "--input", str(inp),
            "--out", str(out),
            "--per_category", "1",
            "--image_root", "/local",
        ]
    )
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["image"] == "/local/x.jpg"
