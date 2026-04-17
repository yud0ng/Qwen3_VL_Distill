"""T3 · src/safe_html.py — XSS-safe HTML rendering for Gradio demo.

Addresses security finding S1: diff_highlight into gr.HTML without escaping
allowed <script> and <img onerror> execution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.safe_html import diff_wrap, escape  # noqa: E402


# ---------------- escape ----------------


@pytest.mark.parametrize(
    "raw, expected_contains",
    [
        ("<script>alert(1)</script>", "&lt;script&gt;"),
        ("<img src=x onerror=alert(1)>", "&lt;img"),
        ("a & b", "a &amp; b"),
        ("'quoted'", "&#x27;"),
        ('"double"', "&quot;"),
    ],
)
def test_escape_blocks_dangerous_chars(raw, expected_contains):
    out = escape(raw)
    assert expected_contains in out
    assert "<script>" not in out
    assert "<img " not in out


def test_escape_empty_input():
    assert escape("") == ""


def test_escape_none_returns_empty():
    assert escape(None) == ""


def test_escape_benign_text_preserved():
    assert escape("normal answer") == "normal answer"


def test_escape_idempotent_not_required_but_safe():
    # double-escape is ok (still safe)
    once = escape("<b>")
    twice = escape(once)
    assert "<b>" not in twice
    assert "&lt;b&gt;" in once


# ---------------- diff_wrap ----------------


def test_diff_wrap_identical_no_mark():
    a, b = diff_wrap("same text", "same text")
    assert "<mark>" not in a
    assert "<mark>" not in b


def test_diff_wrap_different_has_mark():
    a, b = diff_wrap("the cat left", "the dog right")
    assert "<mark>" in a
    assert "<mark>" in b


def test_diff_wrap_escapes_script_tag_in_input():
    """XSS fix: any <script> must be escaped BEFORE being placed in HTML."""
    a, b = diff_wrap("<script>alert(1)</script>", "safe answer")
    assert "<script>" not in a
    assert "&lt;script&gt;" in a
    # Same for b side
    a2, b2 = diff_wrap("safe", "<img src=x onerror=alert(1)>")
    assert "<img" not in b2
    assert "&lt;img" in b2


def test_diff_wrap_escapes_angle_brackets_in_shared_text():
    """Even if the input is identical on both sides, content is still escaped."""
    a, b = diff_wrap("<b>hi</b>", "<b>hi</b>")
    assert "<b>" not in a
    assert "&lt;b&gt;" in a


def test_diff_wrap_escapes_html_entities_in_mark():
    """Changed content wrapped in <mark> must also be escaped inside."""
    a, b = diff_wrap("<evil>", "<other>")
    assert "<evil>" not in a
    assert "<other>" not in b
    # mark tag itself is present as real HTML
    assert "<mark>" in a
    assert "&lt;evil&gt;" in a


def test_diff_wrap_empty_inputs():
    a, b = diff_wrap("", "")
    assert a == ""
    assert b == ""


def test_diff_wrap_none_inputs():
    a, b = diff_wrap(None, None)
    assert a == ""
    assert b == ""


def test_diff_wrap_one_side_empty():
    a, b = diff_wrap("hello", "")
    assert a == "hello" or "<mark>" in a
    assert b == ""


def test_diff_wrap_unicode_preserved():
    a, b = diff_wrap("左边 or 右边", "左边 or 上面")
    # Unicode chars should not be entity-escaped
    assert "左" in a
    assert "左" in b


def test_diff_wrap_no_xss_even_for_cached_question_with_script():
    """Simulate a preset question that someone tampered with to contain script."""
    injected = "Is the <script>stealCookies()</script> dog on the left?"
    a, b = diff_wrap(injected, injected)
    assert "<script>" not in a
    assert "<script>" not in b
    assert "&lt;script&gt;" in a


def test_diff_wrap_control_chars_stripped_or_escaped():
    """Control chars like \\r\\n should not break HTML rendering."""
    a, b = diff_wrap("line1\r\nline2", "line1 line2")
    # at minimum no raw CR that would break HTML; newlines may survive
    # but we should never emit literal \r
    assert "\r" not in a
