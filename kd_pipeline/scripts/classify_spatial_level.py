#!/usr/bin/env python3
"""
空间推理难度分层分类（L1/L2/L3）与分层 Recovery% 统计。

后端：
  - src/spatial_vocab.py   — 分层规则与词表（单一真理源）
  - src/lmms_eval_io.py    — lmms-eval 样本 schema 解析

分层规则（技术方案 §5.2）：
  L3  自我中心推断：机器人/相机视角、可达性、导航
  L2  空间关系：左右/上下/远近/深度
  L1  基础感知：存在/数量/颜色（默认兜底）

用法：
  # 单题分类
  python scripts/classify_spatial_level.py --question "Is A left of B?"

  # lmms-eval 样本级产物分层准确率（需 --log_samples 产出 per-sample JSONL）
  python scripts/classify_spatial_level.py \\
      --samples_jsonl runs/.../samples_cv_bench.jsonl \\
      --out_csv runs/.../cv_bench_by_level.csv

  # 分层 Recovery%（对比基线 + 教师）
  python scripts/classify_spatial_level.py \\
      --samples_jsonl logs/variant_A/samples_cv_bench.jsonl \\
      --baseline_samples logs/2b_baseline/samples_cv_bench.jsonl \\
      --teacher_samples logs/32b/samples_cv_bench.jsonl \\
      --out_csv runs/variant_A_by_level.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lmms_eval_io import (  # noqa: E402
    extract_correctness,
    extract_question,
    load_samples,
)
from src.spatial_vocab import classify_level  # noqa: E402

logger = logging.getLogger("classify_spatial_level")


def level_stats(samples: list[dict]) -> dict[str, dict[str, float]]:
    """Compute per-level accuracy. Returns mapping with keys L1/L2/L3 + _meta."""
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    unknown = 0
    for s in samples:
        q = extract_question(s)
        lvl = classify_level(q)
        totals[lvl] += 1
        c = extract_correctness(s)
        if c is None:
            unknown += 1
            continue
        if c:
            correct[lvl] += 1
    out: dict[str, dict[str, float]] = {}
    for lvl in ("L1", "L2", "L3"):
        n = totals[lvl]
        out[lvl] = {
            "n": n,
            "correct": correct[lvl],
            "acc": (correct[lvl] / n) if n else float("nan"),
        }
    out["_meta"] = {"total": sum(totals.values()), "unknown_correctness": unknown}
    return out


def recovery(distilled: float, baseline: float, teacher: float) -> float:
    gap = teacher - baseline
    if gap == 0:
        return float("nan")
    return (distilled - baseline) / gap * 100.0


def _print_and_write(
    rows: list[dict], header_fields: list[str], out_csv: Path | None
) -> None:
    col_w = 12
    print(" | ".join(f"{h:>{col_w}}" for h in header_fields))
    for r in rows:
        parts = []
        for h in header_fields:
            v = r[h]
            if isinstance(v, float):
                parts.append(f"{v:>{col_w}.4f}")
            else:
                parts.append(f"{str(v):>{col_w}}")
        print(" | ".join(parts))

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header_fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        logger.info("wrote %s", out_csv)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="L1/L2/L3 spatial classifier and per-level Recovery%.")
    ap.add_argument(
        "--samples_jsonl",
        "--eval_json",
        dest="samples_jsonl",
        type=Path,
        help="lmms-eval 样本级 JSON/JSONL 产物",
    )
    ap.add_argument("--baseline_samples", dest="baseline_samples", type=Path)
    ap.add_argument("--teacher_samples", dest="teacher_samples", type=Path)
    ap.add_argument("--baseline_json", dest="baseline_samples", type=Path, help="(deprecated alias)")
    ap.add_argument("--teacher_json", dest="teacher_samples", type=Path, help="(deprecated alias)")
    ap.add_argument("--out_csv", type=Path, default=None)
    ap.add_argument("--question", type=str, default=None, help="单题快速分类")
    return ap


def _main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.question is not None:
        print(classify_level(args.question))
        return 0

    if not args.samples_jsonl:
        ap.error("必须提供 --samples_jsonl 或 --question")

    eval_samples = load_samples(args.samples_jsonl)
    if not eval_samples:
        logger.error("no samples loaded from %s", args.samples_jsonl)
        return 1

    eval_stats = level_stats(eval_samples)

    rows: list[dict[str, object]] = []
    has_recovery = bool(args.baseline_samples and args.teacher_samples)
    bs = ts = None
    if has_recovery:
        bs = level_stats(load_samples(args.baseline_samples))
        ts = level_stats(load_samples(args.teacher_samples))

    for lvl in ("L1", "L2", "L3"):
        row: dict[str, object] = {
            "level": lvl,
            "n": eval_stats[lvl]["n"],
            "acc_eval": eval_stats[lvl]["acc"],
        }
        if has_recovery:
            row["acc_baseline"] = bs[lvl]["acc"]
            row["acc_teacher"] = ts[lvl]["acc"]
            row["recovery_pct"] = recovery(
                eval_stats[lvl]["acc"],
                bs[lvl]["acc"],
                ts[lvl]["acc"],
            )
        rows.append(row)

    header_fields = list(rows[0].keys())
    _print_and_write(rows, header_fields, args.out_csv)
    print(f"\n_meta: {eval_stats['_meta']}")
    return 0


def main() -> None:
    sys.exit(_main())


if __name__ == "__main__":
    main()
