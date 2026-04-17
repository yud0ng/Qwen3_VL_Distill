"""lmms-eval v0.7 per-sample output schema probing.

Per verified schema (lmms-eval evaluator.py):
  - doc_id           : int
  - doc              : dict (raw dataset row)
  - target           : str
  - resps            : list[list[str]] (outer = samples, inner = replicate)
  - filtered_resps   : list[str]
  - arguments        : dict
  - <metric>,none    : float (e.g. cv_bench_acc,none = 1.0/0.0)

All extractors are defensive: unknown keys return "" / None; they do not raise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_samples(path: Path) -> list[dict]:
    """Load lmms-eval sample output. Autodetects JSON / JSONL / wrapped.

    Supported shapes:
      [sample, sample, ...]                    (JSON list)
      {"samples": [...], ...}                  (JSON with wrapper key)
      {"logs": [...], ...}                     (alt wrapper)
      {"doc_id": 0, ...}\n{"doc_id": 1, ...}   (JSONL)
    """
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return [
            json.loads(line) for line in text.splitlines() if line.strip()
        ]
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("samples", "logs", "results", "per_sample"):
            v = obj.get(key)
            if isinstance(v, list):
                return v
        return [obj]
    return []


def _get_nested(sample: dict, *paths: str) -> Any:
    """Walk a single key path ('a.b.c') in the sample; return None if absent."""
    for path in paths:
        cur: Any = sample
        for part in path.split("."):
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(part)
            if cur is None:
                break
        if cur is not None:
            return cur
    return None


def extract_sample_id(sample: dict) -> str:
    """Stable sample id. doc_id is primary (lmms-eval provides it)."""
    for key in ("doc_id", "id", "sample_id", "idx"):
        v = sample.get(key)
        if v is not None:
            return str(v)
    nested = _get_nested(sample, "doc.doc_id", "doc.id", "doc.sample_id")
    if nested is not None:
        return str(nested)
    q = extract_question(sample)
    return f"hash_{abs(hash(q))}"


def extract_question(sample: dict) -> str:
    for key in ("question", "prompt", "input", "text"):
        v = sample.get(key)
        if isinstance(v, str):
            return v
    nested = _get_nested(sample, "doc.question", "doc.prompt", "doc.text")
    if isinstance(nested, str):
        return nested
    return ""


def _any_metric_key(sample: dict, prefixes: tuple[str, ...]) -> Any:
    """Match keys with or without lmms-eval's ',none' suffix (filter name)."""
    for k, v in sample.items():
        k_lower = k.lower()
        for p in prefixes:
            if k_lower == p or k_lower == f"{p},none" or k_lower.startswith(f"{p},"):
                return v
    return None


def extract_correctness(sample: dict) -> bool | None:
    """True / False / None. None = unknown (no recognized correctness key)."""
    for k in ("correct", "is_correct"):
        v = sample.get(k)
        if isinstance(v, bool):
            return v

    v = _any_metric_key(
        sample,
        (
            "cv_bench_acc",
            "mmstar_acc",
            "mme_acc",
            "acc",
            "exact_match",
            "score",
        ),
    )
    if isinstance(v, (int, float)):
        return bool(round(float(v)))

    metrics = sample.get("metrics")
    if isinstance(metrics, dict):
        inner = _any_metric_key(metrics, ("cv_bench_acc", "acc", "exact_match"))
        if isinstance(inner, (int, float)):
            return bool(round(float(inner)))

    return None


def extract_answer_text(sample: dict) -> str:
    """Model's final answer string."""
    v = sample.get("filtered_resps")
    if isinstance(v, list) and v:
        first = v[0]
        if isinstance(first, str):
            return first
        if isinstance(first, list) and first and isinstance(first[0], str):
            return first[0]

    v = sample.get("resps")
    if isinstance(v, list) and v:
        first = v[0]
        if isinstance(first, str):
            return first
        if isinstance(first, list) and first and isinstance(first[0], str):
            return first[0]

    for k in ("response", "prediction", "pred", "output"):
        v = sample.get(k)
        if isinstance(v, str):
            return v
    return ""


def extract_image_path(sample: dict) -> str:
    for key in ("image", "image_path", "img_path"):
        v = sample.get(key)
        if isinstance(v, str):
            return v
    nested = _get_nested(sample, "doc.image", "doc.image_path")
    if isinstance(nested, str):
        return nested
    return ""
