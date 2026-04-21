"""Parse `teacher_responses.jsonl` (M1 teacher generation) for SFT / distillation."""

from __future__ import annotations

import re
from typing import Any


def _extract_answer_tag(text: str) -> str | None:
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _extract_redacted_thinking(text: str) -> str | None:
    m = re.search(
        r"<redacted_thinking>(.*?)</redacted_thinking>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return None


def extract_trace_and_answer(raw: str) -> tuple[str, str]:
    """
    用于 Variant B / BC：拆出 trace（推理链）与 answer（最终答案）。
    优先级：<redacted_thinking>；否则「<answer> 之前的正文」为 trace，<answer> 内为 answer；
    若无 <answer>，trace 为空，answer 为 normalize 后的全文（与 answer_only 一致）。
    """
    inner = _extract_answer_tag(raw)
    think = _extract_redacted_thinking(raw)
    if think is not None and inner is not None:
        return think, inner
    if think is not None:
        return think, inner or _strip_confidence_and_answer_tags(raw).strip()

    if inner is not None:
        pre = re.split(r"<answer>", raw, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        trace = pre if pre else ""
        return trace, inner

    a = normalize_teacher_text(str(raw), target="answer_only")
    return "", a


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

    # 与 gen_all.py extract_batch_logits_and_hidden：有 response 走 normalize；无则与教师 forward 相同字符串
    if (raw or "").strip():
        asst = normalize_teacher_text(str(raw), target=target)
    else:
        asst = re.sub(r"</?think>", "", str(obj.get("thinking") or "")).strip()
    if not q or not asst:
        return "", "", None, {"skip": True, "reason": "empty_q_or_a", "id": obj.get("id")}

    meta = {
        "skip": False,
        "id": obj.get("id"),
        "data_source": source_to_data_source(obj.get("source"), obj.get("type")),
        "confidence": conf,
    }
    return q, asst, image_path, meta
