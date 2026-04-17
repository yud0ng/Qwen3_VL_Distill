#!/usr/bin/env python3
"""
从 teacher_responses.jsonl 选出 ~10k 空间推理样本 ID，供 gen_teacher_topk.py 消费。

选择优先级（依次降级填配额）：
  1) coco_spatial 样本，confidence=5（最稳）
  2) coco_spatial 样本，confidence=4
  3) llava_general 样本中问句触发 L2/L3 空间关键词的，confidence=5
  4) 前三类不足时补 coco_spatial 其他 confidence

后端：src/spatial_vocab.classify_level

用法：
  python scripts/select_logit_subset.py \\
      --input ../teacher_responses.jsonl \\
      --n 10000 \\
      --out_ids data/logit_subset_ids.txt \\
      --out_manifest data/logit_subset_manifest.json

输出：
  - out_ids：每行一个 sample id（文本，便于下游 grep / --id_list）
  - out_manifest：JSON，含入选数量、来源分桶统计、skipped_no_id 计数
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.spatial_vocab import classify_level  # noqa: E402

logger = logging.getLogger("select_logit_subset")


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def bucket_key(obj: dict) -> str:
    src = (obj.get("source") or "").strip().lower()
    conf = obj.get("confidence")
    conf_i = int(conf) if isinstance(conf, (int, float)) else None

    if src == "coco_spatial" and conf_i == 5:
        return "B1_spatial_conf5"
    if src == "coco_spatial" and conf_i == 4:
        return "B2_spatial_conf4"
    if src == "llava_general" and conf_i == 5:
        lvl = classify_level(obj.get("question") or "")
        if lvl in ("L2", "L3"):
            return "B3_general_spatial_question_conf5"
    if src == "coco_spatial":
        return "B4_spatial_other"
    return "B5_other"


PRIORITY: tuple[str, ...] = (
    "B1_spatial_conf5",
    "B2_spatial_conf4",
    "B3_general_spatial_question_conf5",
    "B4_spatial_other",
)


def select(
    rows: list[dict],
    n: int,
    seed: int = 42,
) -> tuple[list[str], dict]:
    """Deterministic priority-based selection.

    Does not mutate ``rows``. Returns (ids, manifest).
    """
    rng = random.Random(seed)
    buckets: dict[str, list[str]] = {b: [] for b in PRIORITY}
    total_by_bucket: Counter[str] = Counter()
    skipped_no_id = 0

    for r in rows:
        bk = bucket_key(r)
        total_by_bucket[bk] += 1
        sid = r.get("id")
        if not sid:
            skipped_no_id += 1
            continue
        if bk in buckets:
            buckets[bk].append(str(sid))

    # Shuffle copies — never mutate the buckets we store in manifest
    shuffled = {bk: list(lst) for bk, lst in buckets.items()}
    for bk in shuffled:
        rng.shuffle(shuffled[bk])

    chosen: list[str] = []
    chosen_by_bucket: Counter[str] = Counter()
    for bk in PRIORITY:
        remaining = n - len(chosen)
        if remaining <= 0:
            break
        take = shuffled[bk][:remaining]
        chosen.extend(take)
        chosen_by_bucket[bk] = len(take)

    manifest = {
        "requested": n,
        "selected": len(chosen),
        "seed": seed,
        "skipped_no_id": skipped_no_id,
        "total_by_bucket": dict(total_by_bucket),
        "chosen_by_bucket": dict(chosen_by_bucket),
    }
    return chosen, manifest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Pick logit-generation subset ids.")
    ap.add_argument("--input", type=Path, required=True, help="teacher_responses.jsonl")
    ap.add_argument("--n", type=int, default=10000, help="目标 ID 数（技术方案约定 10k）")
    ap.add_argument("--out_ids", type=Path, required=True, help="输出 ID 列表（每行一条）")
    ap.add_argument("--out_manifest", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def _main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = build_parser()
    args = ap.parse_args(argv)

    if not args.input.is_file():
        logger.error("input not found: %s", args.input)
        return 1

    rows = list(iter_jsonl(args.input))
    ids, manifest = select(rows, args.n, seed=args.seed)

    args.out_ids.parent.mkdir(parents=True, exist_ok=True)
    args.out_ids.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")

    if args.out_manifest is None:
        args.out_manifest = args.out_ids.with_suffix(".manifest.json")
    args.out_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nwrote {len(ids)} ids -> {args.out_ids}")
    print(f"manifest -> {args.out_manifest}")

    if args.n > 0 and manifest["selected"] < args.n:
        return 2
    return 0


def main() -> None:
    sys.exit(_main())


if __name__ == "__main__":
    main()
