#!/usr/bin/env python3
"""
从 teacher_responses.jsonl 精选 demo 用 32B 缓存：Count / Depth / Spatial Relation /
Egocentric / Metric 五类覆盖，附 L1/L2/L3 自动分层。

后端：src/teacher_responses.normalize_teacher_text（答案提取复用）
      src/spatial_vocab.classify_level

输出格式（demo/teacher_cache.json）：
[
  {
    "id": "...",
    "image": "/abs/path/or/relative/path",
    "question": "...",
    "level": "L1" | "L2" | "L3",
    "category": "count" | "depth" | "relational" | "egocentric" | "metric" | "general",
    "teacher_answer": "<answer> 内文本",
    "teacher_full_response": "32B 原始回答（去 <confidence>）",
    "confidence": 5
  },
  ...
]

用法：
  # 基础用法（保留原始图像路径）
  python demo/build_teacher_cache.py \\
      --input ../teacher_responses.jsonl \\
      --out demo/teacher_cache.json \\
      --per_category 10

  # 路径重写：集群路径 /ocean/.../train2014/X.jpg → ./data/coco/X.jpg
  python demo/build_teacher_cache.py \\
      --input ../teacher_responses.jsonl \\
      --out demo/teacher_cache.json \\
      --image_root ./data/coco
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.spatial_vocab import classify_level  # noqa: E402
from src.teacher_responses import _extract_answer_tag  # noqa: E402

logger = logging.getLogger("build_teacher_cache")

COUNT_PAT = re.compile(r"\bhow many\b|\bcount\b|\bnumber of\b", flags=re.IGNORECASE)
DEPTH_PAT = re.compile(
    r"\bcloser\b|\bfarther\b|\bdepth\b|\bnearest\b|\bfarthest\b|\bin front of\b",
    flags=re.IGNORECASE,
)


def infer_category(obj: dict) -> str:
    t = (obj.get("type") or "").strip().lower()
    if t in ("metric", "relational", "egocentric"):
        if t == "metric" and DEPTH_PAT.search(obj.get("question") or ""):
            return "depth"
        return t
    q = obj.get("question") or ""
    if COUNT_PAT.search(q):
        return "count"
    if DEPTH_PAT.search(q):
        return "depth"
    if classify_level(q) in ("L2", "L3"):
        return "relational"
    return "general"


def extract_answer(response: str) -> tuple[str, str]:
    """Return (answer_only, full_without_confidence).

    Reuses src.teacher_responses._extract_answer_tag for the <answer>...</answer>
    extraction (R2 refactor).
    """
    tagged = _extract_answer_tag(response or "")
    ans = tagged if tagged is not None else (response or "").strip()
    full = re.sub(
        r"<confidence>\s*\d\s*</confidence>\s*$",
        "",
        response or "",
        flags=re.IGNORECASE,
    ).strip()
    return ans, full


def rewrite_image_path(original: str | None, image_root: str | None) -> str:
    """Rewrite cluster-absolute paths to a local root.

    - None → ""
    - image_root None → original unchanged
    - cluster path (e.g. /ocean/.../train2014/X.jpg) → join basename with image_root
    - other absolute or relative paths → if image_root set, join basename
    """
    if not original:
        return ""
    if not image_root:
        return original
    # Any path (absolute or relative) gets its basename joined with image_root.
    base = os.path.basename(original)
    if not base:
        return original
    return os.path.join(image_root, base).replace("\\", "/")


DEMO_CATEGORIES = ("count", "depth", "relational", "egocentric", "metric")


def build_cache(
    rows: list[dict],
    per_category: int,
    seed: int = 42,
    image_root: str | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    skipped_low_conf = 0
    skipped_no_confidence = 0
    for r in rows:
        conf = r.get("confidence")
        if conf is None:
            skipped_no_confidence += 1
            continue
        if isinstance(conf, (int, float)) and int(conf) < 5:
            skipped_low_conf += 1
            continue
        cat = infer_category(r)
        buckets[cat].append(r)

    out: list[dict] = []
    for cat in DEMO_CATEGORIES:
        pool = list(buckets.get(cat, []))  # copy before shuffle
        rng.shuffle(pool)
        for r in pool[:per_category]:
            ans, full = extract_answer(r.get("response") or "")
            if not ans:
                continue
            entry = {
                "id": r.get("id"),
                "image": rewrite_image_path(r.get("image"), image_root),
                "question": (r.get("question") or "").strip(),
                "level": classify_level(r.get("question") or ""),
                "category": cat,
                "teacher_answer": ans,
                "teacher_full_response": full,
                "confidence": r.get("confidence"),
            }
            out.append(entry)
    logger.info(
        "built %d entries (skipped_no_confidence=%d, skipped_low_conf=%d)",
        len(out),
        skipped_no_confidence,
        skipped_low_conf,
    )
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Curate 32B teacher responses for Gradio demo.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per_category", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="若提供，将图像路径重写为 <image_root>/<basename>（剥离集群绝对路径）",
    )
    return ap


def _main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = build_parser()
    args = ap.parse_args(argv)

    rows = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    cache = build_cache(
        rows,
        per_category=args.per_category,
        seed=args.seed,
        image_root=args.image_root,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    cat_counts = Counter(e["category"] for e in cache)
    level_counts = Counter(e["level"] for e in cache)
    print(f"wrote {len(cache)} entries -> {args.out}")
    print(f"by_category: {dict(cat_counts)}")
    print(f"by_level:    {dict(level_counts)}")
    return 0


def main() -> None:
    sys.exit(_main())


if __name__ == "__main__":
    main()
