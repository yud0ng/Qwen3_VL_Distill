#!/usr/bin/env python3
"""
CoT（Variant B）质量门控：对含 <think>...</think> 的 teacher_responses_cot.jsonl 做过滤。

后端：src/spatial_vocab.py（统一空间关键词 + pivot）

过滤条件（技术方案 §3.3，全部满足才保留）：
  1) confidence >= 4
  2) 推理链**词数**（非 BPE token）>= 30  — 见 C3 修复注释
  3) 空间关键词计数 >= 2（\\b 边界匹配，区分 "behind" 与 "behindsight"）
  4) 至少 1 个 pivot 词（therefore / because / since / first...then / ...）

用法：
  python scripts/filter_cot_quality.py \\
      --input ../teacher_responses_cot.jsonl \\
      --out_pass data/clean_train_cot.jsonl \\
      --out_fail data/cot_rejected.jsonl \\
      --report data/cot_filter_report.json

注意：`count_words_approx` 是按空白分词的**词数**，不是 BPE token 数；
对英文推理文本，BPE 通常比词数多约 30-50%。threshold 由 CLI 传入时请按词数设计。
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.spatial_vocab import (  # noqa: E402
    count_pivot_patterns,
    count_spatial_keywords as _count_spatial_keywords,
)

logger = logging.getLogger("filter_cot_quality")

MIN_CONF_DEFAULT = 4
MIN_TRACE_WORDS_DEFAULT = 30
MIN_SPATIAL_KEYWORDS_DEFAULT = 2
MIN_PIVOTS_DEFAULT = 1


@dataclass(frozen=True)
class FilterThresholds:
    min_confidence: int = MIN_CONF_DEFAULT
    min_trace_words: int = MIN_TRACE_WORDS_DEFAULT
    min_spatial_keywords: int = MIN_SPATIAL_KEYWORDS_DEFAULT
    min_pivots: int = MIN_PIVOTS_DEFAULT


def extract_think(response: str) -> str:
    m = re.search(
        r"<think>(.*?)</think>", response or "", flags=re.DOTALL | re.IGNORECASE
    )
    return m.group(1).strip() if m else ""


def count_words_approx(text: str) -> int:
    """Whitespace-delimited word count.

    NOTE: this is a proxy for token count. Real BPE token count is ~1.3-1.5x
    for English reasoning text. Thresholds must be set relative to word count,
    not BPE token count.
    """
    return len(re.findall(r"\S+", text or ""))


def count_spatial_keywords(text: str) -> int:
    return _count_spatial_keywords(text)


def count_pivots(text: str) -> int:
    return count_pivot_patterns(text)


def evaluate_sample(obj: dict, th: FilterThresholds) -> tuple[bool, dict]:
    """Evaluate a single sample. Does not mutate ``obj``."""
    reasons: list[str] = []
    conf = obj.get("confidence")
    conf_i = int(conf) if isinstance(conf, (int, float)) else None
    if conf_i is None or conf_i < th.min_confidence:
        reasons.append(f"confidence<{th.min_confidence}")

    trace = extract_think(obj.get("response") or "")
    if not trace:
        reasons.append("no_think_tag")
        metrics = {
            "confidence": conf_i,
            "trace_words": 0,
            "spatial_keywords": 0,
            "pivots": 0,
            "has_think": False,
        }
        return False, {"reasons": reasons, "metrics": metrics}

    n_words = count_words_approx(trace)
    n_kw = count_spatial_keywords(trace)
    n_piv = count_pivots(trace)

    if n_words < th.min_trace_words:
        reasons.append(f"trace_words<{th.min_trace_words}")
    if n_kw < th.min_spatial_keywords:
        reasons.append(f"spatial_keywords<{th.min_spatial_keywords}")
    if n_piv < th.min_pivots:
        reasons.append(f"pivots<{th.min_pivots}")

    metrics = {
        "confidence": conf_i,
        "trace_words": n_words,
        "spatial_keywords": n_kw,
        "pivots": n_piv,
        "has_think": True,
    }
    return (not reasons), {"reasons": reasons, "metrics": metrics}


def run(
    input_path: Path,
    out_pass: Path,
    out_fail: Path | None,
    report_path: Path | None,
    th: FilterThresholds,
) -> dict:
    out_pass.parent.mkdir(parents=True, exist_ok=True)
    if out_fail:
        out_fail.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_pass = 0
    reason_counter: Counter[str] = Counter()

    with contextlib.ExitStack() as stack:
        fp = stack.enter_context(out_pass.open("w", encoding="utf-8"))
        ff = (
            stack.enter_context(out_fail.open("w", encoding="utf-8"))
            if out_fail
            else None
        )
        try:
            fin = stack.enter_context(input_path.open(encoding="utf-8"))
        except FileNotFoundError:
            logger.error("input file not found: %s", input_path)
            fin = []  # type: ignore[assignment]

        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            n_in += 1
            ok, info = evaluate_sample(obj, th)
            if ok:
                fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_pass += 1
            else:
                for r in info["reasons"]:
                    reason_counter[r] += 1
                if ff:
                    obj_out = copy.deepcopy(obj)
                    obj_out["_filter"] = info
                    ff.write(json.dumps(obj_out, ensure_ascii=False) + "\n")

    report = {
        "input": str(input_path),
        "out_pass": str(out_pass),
        "out_fail": str(out_fail) if out_fail else None,
        "thresholds": th.__dict__,
        "n_in": n_in,
        "n_pass": n_pass,
        "pass_rate": (n_pass / n_in) if n_in else 0.0,
        "reject_reasons": dict(reason_counter.most_common()),
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Filter teacher CoT JSONL by confidence / words / keywords / pivots.",
    )
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out_pass", type=Path, required=True)
    ap.add_argument("--out_fail", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--min_confidence", type=int, default=MIN_CONF_DEFAULT)
    # primary name; --min_trace_tokens kept for backward compat
    ap.add_argument(
        "--min_trace_words",
        "--min_trace_tokens",
        dest="min_trace_words",
        type=int,
        default=MIN_TRACE_WORDS_DEFAULT,
    )
    ap.add_argument(
        "--min_spatial_keywords", type=int, default=MIN_SPATIAL_KEYWORDS_DEFAULT
    )
    ap.add_argument("--min_pivots", type=int, default=MIN_PIVOTS_DEFAULT)
    return ap


def _main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = build_parser()
    args = ap.parse_args(argv)

    th = FilterThresholds(
        min_confidence=args.min_confidence,
        min_trace_words=args.min_trace_words,
        min_spatial_keywords=args.min_spatial_keywords,
        min_pivots=args.min_pivots,
    )
    report = run(args.input, args.out_pass, args.out_fail, args.report, th)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["n_in"] == 0:
        return 1
    if report["n_pass"] == 0:
        return 2
    return 0


def main() -> None:
    sys.exit(_main())


if __name__ == "__main__":
    main()
