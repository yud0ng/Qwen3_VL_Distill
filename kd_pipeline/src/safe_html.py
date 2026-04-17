"""Safe HTML rendering for Gradio demo.

Threat model (see reviews.md S1):
  gr.HTML renders its string input without escaping. If we embed untrusted
  content (model output, user question, cached preset) directly, attacker
  payloads like <script>, <img onerror>, <svg onload> execute in the browser.

Mitigation:
  - ALL content routed through Gradio HTML components must go through
    `escape()` or `diff_wrap()` from this module.
  - Only the <mark>...</mark> wrapper itself is raw HTML; content inside is
    always HTML-escaped.
  - Control characters \\r are stripped; \\n preserved as \\n (browsers render
    whitespace).
"""

from __future__ import annotations

import difflib
import html as _html


def escape(s: str | None) -> str:
    """HTML-escape a string. None -> ''. quote=True also escapes ' and \"."""
    if not s:
        return ""
    s = s.replace("\r", "")
    return _html.escape(s, quote=True)


def diff_wrap(a: str | None, b: str | None) -> tuple[str, str]:
    """Escape both sides and wrap changed word-chunks in <mark>.

    Returns (a_html, b_html). Both are safe to insert into gr.HTML.
    """
    if not a and not b:
        return "", ""
    if not a:
        return "", escape(b)
    if not b:
        return escape(a), ""

    a = a.replace("\r", "")
    b = b.replace("\r", "")
    a_tokens = a.split()
    b_tokens = b.split()

    sm = difflib.SequenceMatcher(a=a_tokens, b=b_tokens)
    a_out: list[str] = []
    b_out: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        a_chunk = " ".join(a_tokens[i1:i2])
        b_chunk = " ".join(b_tokens[j1:j2])
        a_safe = _html.escape(a_chunk, quote=True)
        b_safe = _html.escape(b_chunk, quote=True)
        if tag == "equal":
            if a_safe:
                a_out.append(a_safe)
            if b_safe:
                b_out.append(b_safe)
        else:
            if a_safe:
                a_out.append(f"<mark>{a_safe}</mark>")
            if b_safe:
                b_out.append(f"<mark>{b_safe}</mark>")
    return " ".join(a_out), " ".join(b_out)
