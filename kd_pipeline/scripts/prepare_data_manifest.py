#!/usr/bin/env python3
"""
统计 JSONL：总行数、按 data_source 分组（50k = cv_bench 25k + llava_instruct 25k 的验收用）。

  python scripts/prepare_data_manifest.py --jsonl data/clean_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True)
    args = ap.parse_args()
    path = Path(args.jsonl)
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(lines)
    src = Counter()
    missing_id = 0
    for line in lines:
        o = json.loads(line)
        if not o.get("id"):
            missing_id += 1
        src[o.get("data_source", "(missing)")] += 1
    print(f"file: {path}")
    print(f"total_rows: {n}")
    print(f"missing_id: {missing_id}")
    for k, v in sorted(src.items(), key=lambda x: -x[1]):
        print(f"  data_source[{k}]: {v}")


if __name__ == "__main__":
    main()
