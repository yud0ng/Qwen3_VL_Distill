"""Dependency-free quality checks shared by CoT generation and conversion."""

import re


_SPATIAL_KEYWORDS = frozenset(
    {
        "left",
        "right",
        "above",
        "below",
        "distance",
        "depth",
        "near",
        "far",
        "between",
        "closer",
        "farther",
        "behind",
        "in front",
        "beside",
        "under",
        "over",
    }
)

_PIVOT_RE = re.compile(
    r"\b(therefore|because|since)\b|first\b.{1,150}?\bthen\b",
    re.IGNORECASE | re.DOTALL,
)


def _strip_tags(thinking: str) -> str:
    return re.sub(r"</?think>", "", thinking)


def _spatial_density(thinking: str) -> float:
    inner = _strip_tags(thinking).lower()
    words = inner.split()
    if not words:
        return 0.0
    hits = sum(
        1 for word in words if any(keyword in word for keyword in _SPATIAL_KEYWORDS)
    )
    hits += sum(
        inner.count(keyword) for keyword in _SPATIAL_KEYWORDS if " " in keyword
    )
    return hits / len(words)


def is_cot_quality(
    thinking: str,
    min_tokens: int,
    min_density: float,
) -> tuple[bool, dict]:
    """Return the overall decision and per-filter diagnostic statistics."""
    token_count = len(_strip_tags(thinking).split())
    density = _spatial_density(thinking)
    has_pivot = bool(_PIVOT_RE.search(thinking))
    has_length = token_count >= min_tokens
    has_density = density >= min_density

    return has_length and has_density and has_pivot, {
        "think_len": token_count,
        "density": round(density, 4),
        "has_pivot": has_pivot,
        "fail_len": not has_length,
        "fail_density": not has_density,
        "fail_pivot": not has_pivot,
    }
