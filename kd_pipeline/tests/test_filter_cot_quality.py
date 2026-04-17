"""T6 refactored · scripts/filter_cot_quality.py

Reuses src.spatial_vocab and src.teacher_responses.
Addresses review: C3 (token→words rename), H4 (deepcopy), M3 (file handle opens in try).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from filter_cot_quality import (  # noqa: E402
    FilterThresholds,
    count_pivots,
    count_spatial_keywords,
    count_words_approx,
    evaluate_sample,
    extract_think,
    run,
)


GOOD_THINK = (
    "<think>The cup is to the left of the plate, and the plate is above the napkin. "
    "Because the cup sits closer to the camera than the plate, the cup's depth must "
    "therefore be shallower. Since the plate is behind the cup, the distance between "
    "them is small.</think>"
)

SHORT_THINK = "<think>Cup is left.</think>"

NO_KW_THINK = (
    "<think>I examine the image carefully and consider different factors about the "
    "scene in detail, therefore arriving at a conclusion about the object properties "
    "through logical deduction applied to the visible evidence in the scene.</think>"
)

NO_PIVOT_THINK = (
    "<think>The cup is to the left of the plate. The plate is above the napkin. "
    "The cup sits closer to the camera than the plate. The depth is shallow. The "
    "distance between the cup and the plate is small. The napkin is below both.</think>"
)


def _sample(response: str, conf: int | None = 5, qid: str = "t") -> dict:
    return {"id": qid, "response": response, "confidence": conf, "question": "q"}


# ---------------- extract / count helpers ----------------


def test_extract_think_basic():
    assert extract_think("<think>hello</think>answer") == "hello"


def test_extract_think_missing_returns_empty():
    assert extract_think("no tag here") == ""


def test_extract_think_multiline():
    txt = "<think>line1\nline2</think>"
    assert "line1" in extract_think(txt)
    assert "line2" in extract_think(txt)


def test_count_words_approx_basic():
    """C3 fix: renamed from count_tokens_approx — counts whitespace-delimited words."""
    assert count_words_approx("one two three") == 3
    assert count_words_approx("") == 0


def test_count_words_approx_documented_not_bpe():
    """Explicit: this is a word count, not a tokenizer-aware BPE count.
    Real BPE token count is typically 1.3-1.5x higher for English reasoning text.
    Teammates MUST set thresholds accordingly."""
    text = "The cup is to the left of the plate."  # 9 words, ~11 BPE tokens
    assert count_words_approx(text) == 9


def test_count_spatial_keywords_case_insensitive_word_boundary():
    assert count_spatial_keywords("left and LEFT") == 2
    assert count_spatial_keywords("behindsight foreground") == 0


def test_count_pivots_positive():
    assert count_pivots("I think, therefore I am") == 1
    assert count_pivots("A is first, then B follows") >= 1
    assert count_pivots("no pivots here at all") == 0


# ---------------- evaluate_sample ----------------


def test_evaluate_good_sample_passes():
    ok, info = evaluate_sample(_sample(GOOD_THINK, conf=5), FilterThresholds())
    assert ok, info


def test_evaluate_low_confidence_fails():
    ok, info = evaluate_sample(_sample(GOOD_THINK, conf=3), FilterThresholds())
    assert not ok
    assert any("confidence" in r for r in info["reasons"])


def test_evaluate_short_trace_fails():
    ok, info = evaluate_sample(_sample(SHORT_THINK, conf=5), FilterThresholds())
    assert not ok
    assert any("trace_words" in r or "trace_tokens" in r for r in info["reasons"])


def test_evaluate_no_keywords_fails():
    ok, info = evaluate_sample(_sample(NO_KW_THINK, conf=5), FilterThresholds())
    assert not ok
    assert any("spatial_keywords" in r for r in info["reasons"])


def test_evaluate_no_pivot_fails():
    ok, info = evaluate_sample(_sample(NO_PIVOT_THINK, conf=5), FilterThresholds())
    assert not ok
    assert any("pivots" in r for r in info["reasons"])


def test_evaluate_no_think_fails():
    ok, info = evaluate_sample(
        {"id": "x", "response": "answer only", "confidence": 5},
        FilterThresholds(),
    )
    assert not ok
    assert "no_think_tag" in info["reasons"]


def test_evaluate_preserves_input_dict_unchanged():
    """H4 fix: evaluate_sample must not mutate the input dict."""
    original = _sample(GOOD_THINK, conf=5)
    snapshot = copy.deepcopy(original)
    evaluate_sample(original, FilterThresholds())
    assert original == snapshot


# ---------------- run end-to-end ----------------


def test_run_end_to_end(tmp_path: Path):
    samples = [
        _sample(GOOD_THINK, conf=5, qid="pass_1"),
        _sample(GOOD_THINK, conf=5, qid="pass_2"),
        _sample(SHORT_THINK, conf=5, qid="fail_short"),
        _sample(NO_PIVOT_THINK, conf=5, qid="fail_no_pivot"),
        _sample(GOOD_THINK, conf=3, qid="fail_low_conf"),
        {"id": "fail_no_think", "response": "just an answer", "confidence": 5},
    ]
    input_path = tmp_path / "cot.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(s) for s in samples), encoding="utf-8"
    )
    out_pass = tmp_path / "pass.jsonl"
    out_fail = tmp_path / "fail.jsonl"
    report_path = tmp_path / "report.json"

    report = run(input_path, out_pass, out_fail, report_path, FilterThresholds())

    assert report["n_in"] == 6
    assert report["n_pass"] == 2
    pass_lines = out_pass.read_text(encoding="utf-8").splitlines()
    pass_ids = [json.loads(l)["id"] for l in pass_lines]
    assert set(pass_ids) == {"pass_1", "pass_2"}

    fail_lines = out_fail.read_text(encoding="utf-8").splitlines()
    assert len(fail_lines) == 4
    for line in fail_lines:
        obj = json.loads(line)
        assert "_filter" in obj, "Every fail row must include _filter metadata"
        assert "reasons" in obj["_filter"]
        assert "metrics" in obj["_filter"]

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["n_pass"] == 2


def test_run_thresholds_relax(tmp_path: Path):
    samples = [_sample(SHORT_THINK, conf=5, qid="relaxed")]
    input_path = tmp_path / "cot.jsonl"
    input_path.write_text("\n".join(json.dumps(s) for s in samples), encoding="utf-8")
    out_pass = tmp_path / "pass.jsonl"
    th = FilterThresholds(
        min_confidence=1,
        min_trace_words=2,
        min_spatial_keywords=1,
        min_pivots=0,
    )
    report = run(input_path, out_pass, None, None, th)
    assert report["n_pass"] == 1


def test_run_empty_input_does_not_crash(tmp_path: Path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    out_pass = tmp_path / "pass.jsonl"
    report = run(p, out_pass, None, None, FilterThresholds())
    assert report["n_in"] == 0
    assert report["n_pass"] == 0
    assert report["pass_rate"] == 0.0


def test_run_writes_pass_rate_correctly(tmp_path: Path):
    samples = [
        _sample(GOOD_THINK, conf=5, qid="a"),
        _sample(GOOD_THINK, conf=5, qid="b"),
        _sample(SHORT_THINK, conf=5, qid="c"),
        _sample(SHORT_THINK, conf=5, qid="d"),
    ]
    p = tmp_path / "in.jsonl"
    p.write_text("\n".join(json.dumps(s) for s in samples), encoding="utf-8")
    out_pass = tmp_path / "pass.jsonl"
    report = run(p, out_pass, None, None, FilterThresholds())
    assert report["n_in"] == 4
    assert report["n_pass"] == 2
    assert report["pass_rate"] == pytest.approx(0.5)


def test_run_fail_file_contains_filter_metadata_structure(tmp_path: Path):
    samples = [_sample(SHORT_THINK, conf=5, qid="short")]
    p = tmp_path / "in.jsonl"
    p.write_text(json.dumps(samples[0]), encoding="utf-8")
    out_pass = tmp_path / "p.jsonl"
    out_fail = tmp_path / "f.jsonl"
    run(p, out_pass, out_fail, None, FilterThresholds())
    fail = json.loads(out_fail.read_text(encoding="utf-8").splitlines()[0])
    assert "_filter" in fail
    assert "metrics" in fail["_filter"]
    metrics = fail["_filter"]["metrics"]
    assert "confidence" in metrics
    assert "trace_words" in metrics or "trace_tokens" in metrics
    assert "spatial_keywords" in metrics
    assert "pivots" in metrics


def test_backward_compat_min_trace_tokens_cli_alias():
    """User may still pass --min_trace_tokens — should map to min_trace_words."""
    from filter_cot_quality import build_parser
    ap = build_parser()
    args = ap.parse_args(
        [
            "--input", "in.jsonl", "--out_pass", "out.jsonl",
            "--min_trace_tokens", "15",
        ]
    )
    # Either field name acceptable depending on how the alias is wired
    assert getattr(args, "min_trace_words", None) == 15 or \
           getattr(args, "min_trace_tokens", None) == 15
