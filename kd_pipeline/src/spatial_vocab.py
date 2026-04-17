"""Spatial vocabulary — single source of truth for L1/L2/L3 classification
and CoT quality keyword / pivot counting.

Invariants:
  - All keywords are lowercase.
  - L2_WORDS and L3_WORDS are disjoint.
  - Single-word keywords are matched with \\b word boundaries (case-insensitive).
  - Multi-word L3 phrases are matched as literal substrings (case-insensitive).

Importers MUST NOT redefine or extend these tuples at runtime.
"""

from __future__ import annotations

import re
from typing import Literal

# ---- L3: egocentric / navigation ----

L3_PHRASES: tuple[str, ...] = (
    "from your perspective",
    "from the camera",
    "walked forward",
    "would you encounter",
    "block your path",
    "your viewpoint",
    "from this viewpoint",
)

L3_WORDS: tuple[str, ...] = (
    "reach",
    "walk",
    "approach",
    "navigate",
)

# ---- L2: relational ----

L2_WORDS: tuple[str, ...] = (
    "left",
    "right",
    "above",
    "below",
    "closer",
    "farther",
    "behind",
    "front",
    "depth",
    "near",
    "far",
    "distance",
    "between",
)

# ---- CoT quality filter ----

FILTER_KEYWORDS: tuple[str, ...] = L2_WORDS

PIVOT_PATTERNS: tuple[str, ...] = (
    r"\btherefore\b",
    r"\bbecause\b",
    r"\bsince\b",
    r"\bthus\b",
    r"\bso that\b",
    r"\bhence\b",
    r"\bfirst\b.*\bthen\b",
    r"\bif\b.*\bthen\b",
    r"\bas a result\b",
    r"\bconsequently\b",
)


# ---- matchers ----


def _text(s: str | None) -> str:
    return (s or "").lower()


def match_l3(text: str | None) -> bool:
    t = _text(text)
    if not t:
        return False
    for phrase in L3_PHRASES:
        if phrase in t:
            return True
    for word in L3_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", t):
            return True
    return False


def match_l2(text: str | None) -> bool:
    t = _text(text)
    if not t:
        return False
    for word in L2_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", t):
            return True
    return False


def classify_level(question: str | None) -> Literal["L1", "L2", "L3"]:
    if match_l3(question):
        return "L3"
    if match_l2(question):
        return "L2"
    return "L1"


def count_spatial_keywords(text: str | None) -> int:
    t = _text(text)
    if not t:
        return 0
    total = 0
    for kw in FILTER_KEYWORDS:
        total += len(re.findall(rf"\b{re.escape(kw)}\b", t))
    return total


def count_pivot_patterns(text: str | None) -> int:
    t = _text(text)
    if not t:
        return 0
    return sum(1 for pat in PIVOT_PATTERNS if re.search(pat, t))
