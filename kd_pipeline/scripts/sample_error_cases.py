#!/usr/bin/env python3
"""
误差分析抽样：从「2B 蒸馏答错 & 32B 答对」样本中按 L2/L3 分层随机抽 50 条。

后端：
  - src/lmms_eval_io.py — 样本 schema 解析
  - src/csv_safe.py     — CSV formula injection 防御
  - src/spatial_vocab.py — L1/L2/L3 分层

输入：lmms-eval 的样本级 JSON/JSONL（须用 --log_samples 产出）
  (a) --distilled_samples + --teacher_samples，按 id 自动对齐
  (b) --paired_json 已 join

输出：
  - error_samples.csv：sample_id / level / image / question / distilled_answer /
                       teacher_answer / error_category（待人工填）
  - error_summary.json：按 level 的总数、抽样比例

用法：
  python scripts/sample_error_cases.py \\
      --distilled_samples runs/variant_BC/samples_cv_bench.jsonl \\
      --teacher_samples logs/32b/samples_cv_bench.jsonl \\
      --out_csv runs/variant_BC/error_samples.csv \\
      --n 50
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.csv_safe import sanitize_row  # noqa: E402
from src.lmms_eval_io import (  # noqa: E402
    extract_answer_text,
    extract_correctness,
    extract_image_path,
    extract_question,
    extract_sample_id,
    load_samples,
)
from src.spatial_vocab import classify_level  # noqa: E402

logger = logging.getLogger("sample_error_cases")


def join_by_id(distilled: list[dict], teacher: list[dict]) -> list[dict]:
    by_id_t = {extract_sample_id(s): s for s in teacher}
    paired: list[dict] = []
    unmatched = 0
    for d in distilled:
        sid = extract_sample_id(d)
        t = by_id_t.get(sid)
        if t is None:
            unmatched += 1
            continue
        paired.append({"distilled": d, "teacher": t, "sid": sid})
    if unmatched:
        logger.warning(
            "%d/%d distilled samples had no teacher match",
            unmatched,
            len(distilled),
        )
    return paired


def stratified_sample(
    paired: list[dict],
    n_total: int,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """Stratified random sampling of wrong-right error cases.

    Does not mutate ``paired`` or any nested pools. Deterministic per ``seed``.
    """
    rng = random.Random(seed)
    by_level: dict[str, list[dict]] = defaultdict(list)
    error_count: Counter[str] = Counter()
    total_by_level: Counter[str] = Counter()
    for p in paired:
        d = p["distilled"]
        t = p["teacher"]
        q = extract_question(d) or extract_question(t)
        lvl = classify_level(q)
        total_by_level[lvl] += 1
        d_ok = extract_correctness(d)
        t_ok = extract_correctness(t)
        if d_ok is False and t_ok is True:
            error_count[lvl] += 1
            by_level[lvl].append({"pair": p, "question": q, "level": lvl})

    per_level_quota = {"L3": n_total // 2, "L2": n_total - n_total // 2}
    chosen: list[dict] = []
    chosen_by_level: Counter[str] = Counter()

    # Copy + shuffle pools once so ordering is deterministic and non-mutating.
    pools = {lvl: list(by_level.get(lvl, [])) for lvl in ("L1", "L2", "L3")}
    for lvl in pools:
        rng.shuffle(pools[lvl])

    # Pass 1: initial quota for L3 and L2.
    for lvl in ("L3", "L2"):
        quota = per_level_quota.get(lvl, 0)
        quota = min(quota, len(pools[lvl]), n_total - len(chosen))
        if quota <= 0:
            continue
        take = pools[lvl][:quota]
        pools[lvl] = pools[lvl][quota:]
        chosen.extend(take)
        chosen_by_level[lvl] += quota

    # Pass 2: redistribute remaining demand across L3 → L2 → L1 in priority order.
    for lvl in ("L3", "L2", "L1"):
        if len(chosen) >= n_total:
            break
        remaining = n_total - len(chosen)
        take = pools[lvl][:remaining]
        pools[lvl] = pools[lvl][remaining:]
        if take:
            chosen.extend(take)
            chosen_by_level[lvl] += len(take)

    summary = {
        "requested": n_total,
        "selected": len(chosen),
        "error_count_by_level": dict(error_count),
        "total_by_level": dict(total_by_level),
        "chosen_by_level": dict(chosen_by_level),
    }
    return chosen, summary


CSV_COLUMNS = (
    "sample_id",
    "level",
    "image",
    "question",
    "distilled_answer",
    "teacher_answer",
    "error_category",
)


def write_csv(chosen: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        for c in chosen:
            p = c["pair"]
            d = p["distilled"]
            t = p["teacher"]
            raw_row = {
                "sample_id": p["sid"],
                "level": c["level"],
                "image": extract_image_path(d) or extract_image_path(t),
                "question": c["question"],
                "distilled_answer": extract_answer_text(d),
                "teacher_answer": extract_answer_text(t),
                "error_category": "",
            }
            w.writerow(sanitize_row(raw_row))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Sample 2B-wrong/32B-right error cases.")
    ap.add_argument("--distilled_samples", dest="distilled_samples", type=Path, help="蒸馏模型样本级评测输出")
    ap.add_argument("--teacher_samples", dest="teacher_samples", type=Path, help="32B 教师样本级评测输出")
    ap.add_argument("--distilled_json", dest="distilled_samples", type=Path, help="(deprecated alias)")
    ap.add_argument("--teacher_json", dest="teacher_samples", type=Path, help="(deprecated alias)")
    ap.add_argument("--paired_json", type=Path, default=None)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--out_summary", type=Path, default=None)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def _main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.paired_json:
        paired = load_samples(args.paired_json)
    else:
        if not (args.distilled_samples and args.teacher_samples):
            ap.error("需 --distilled_samples + --teacher_samples，或 --paired_json")
        distilled = load_samples(args.distilled_samples)
        teacher = load_samples(args.teacher_samples)
        paired = join_by_id(distilled, teacher)

    if not paired:
        logger.error("no paired samples after join")
        return 1

    chosen, summary = stratified_sample(paired, args.n, args.seed)
    if not chosen:
        logger.warning("no error cases found (distilled=wrong AND teacher=right)")
        write_csv([], args.out_csv)
        return 2

    write_csv(chosen, args.out_csv)

    if args.out_summary is None:
        args.out_summary = args.out_csv.with_suffix(".summary.json")
    args.out_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {len(chosen)} rows -> {args.out_csv}")
    return 0


def main() -> None:
    sys.exit(_main())


if __name__ == "__main__":
    main()
