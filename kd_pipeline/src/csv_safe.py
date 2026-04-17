"""CSV formula-injection prevention.

Excel, LibreOffice Calc, and Google Sheets interpret cells starting with
`=`, `+`, `-`, `@`, or leading tab as formulas. Model-generated answers may
legitimately contain such prefixes ("=1 + 1"), but also may contain malicious
payloads (`=HYPERLINK("http://evil","click")`, `=IMPORTXML(...)`).

Mitigation: prefix any triggering value with a single quote so the spreadsheet
treats it as text. Strip CR/LF to prevent log / CSV row injection.
"""

from __future__ import annotations

from typing import Any

FORMULA_TRIGGERS: tuple[str, ...] = ("=", "+", "-", "@", "\t")


def escape_cell(value: Any) -> Any:
    """Make a cell safe for CSV / spreadsheet rendering.

    - None → ""
    - Non-string → returned unchanged (csv.DictWriter handles str() conversion)
    - String with formula trigger at position 0 → prefixed with "'"
    - CR / LF / leading \\t stripped
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        return value
    v = value.replace("\r", "").replace("\n", " ").replace("\t", " ")
    v = v.lstrip(" ")
    if v and v[0] in FORMULA_TRIGGERS:
        return "'" + v
    return v


def sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Apply escape_cell to every value in a dict row."""
    return {k: escape_cell(v) for k, v in row.items()}
