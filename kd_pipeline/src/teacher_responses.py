"""Parse `teacher_responses.jsonl` (M1 teacher generation) for SFT / distillation."""

from __future__ import annotations

import re
from typing import Any


def _extract_answer_tag(text: str) -> str | None:
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def normalize_teacher_text(response: str, *, target: str = "answer_only") -> str:
    """
    target:
      - answer_only: only text inside <answer>...</answer>, else full stripped response
      - full: whole response with trailing <answer>...</answer><confidence> stripped to avoid duplicate tags in loss
    """
    if target == "answer_only":
        inner = _extract_answer_tag(response)
        if inner:
            return inner
        return _strip_confidence_and_answer_tags(response)

    if target == "full":
        return _strip_confidence_and_answer_tags(response)

    raise ValueError(f"unknown target: {target}")


def _strip_confidence_and_answer_tags(response: str) -> str:
    """Remove trailing <answer>...</answer><confidence> blocks; trim."""
    s = response
    s = re.sub(
        r"<answer>.*?</answer>\s*<confidence>\s*\d\s*</confidence>\s*$",
        "",
        s,
        flags=re.DOTALL | re.IGNORECASE,
    )
    s = re.sub(r"<confidence>\s*\d\s*</confidence>\s*$", "", s, flags=re.IGNORECASE)
    return s.strip()


def source_to_data_source(source: str | None, type_: str | None) -> str:
    """Map teacher file fields to DATA_FORMAT data_source tags."""
    if not source:
        return "unknown"
    s = source.lower()
    if "llava" in s or "general" in (type_ or "").lower():
        return "llava_instruct"
    if "cv" in s or "bench" in s or "spatial" in s:
        return "cv_bench"
    return source


def row_teacher_responses(
    obj: dict[str, Any],
    *,
    target: str = "answer_only",
    min_confidence: int = 0,
) -> tuple[str, str, str | None, dict[str, Any]]:
    """
    Returns (user_text, assistant_text, image_path, meta).

    Expected keys: id, question, response, image, optional confidence / source / type.
    """
    q = (obj.get("question") or "").strip()
    raw = obj.get("response") or ""
    img = obj.get("image")
    if isinstance(img, str):
        image_path = img.strip() or None
    else:
        image_path = None

    conf = obj.get("confidence")
    if conf is not None and int(conf) < min_confidence:
        return "", "", None, {"skip": True, "reason": "low_confidence", "id": obj.get("id")}

    asst = normalize_teacher_text(str(raw), target=target)
    if not q or not asst:
        return "", "", None, {"skip": True, "reason": "empty_q_or_a", "id": obj.get("id")}

    meta = {
        "skip": False,
        "id": obj.get("id"),
        "data_source": source_to_data_source(obj.get("source"), obj.get("type")),
        "confidence": conf,
    }
    return q, asst, image_path, meta
