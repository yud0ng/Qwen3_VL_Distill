#!/usr/bin/env python3
"""
合并多个 JSONL 为 clean_train.jsonl，并强制/补全 data_source 字段（50k 验收用）。

示例：
  python scripts/merge_jsonl.py \\
    --out data/clean_train.jsonl \\
    --inputs data/cv_bench_25k.jsonl:cv_bench data/llava_25k.jsonl:llava_instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="path:tag 形式，tag 写入 data_source（若行内已有则覆盖可选）",
    )
    ap.add_argument("--skip_existing_source", action="store_true", help="行内已有 data_source 时不覆盖")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for spec in args.inputs:
            if ":" not in spec:
                raise SystemExit(f"Bad --inputs entry (need path:tag): {spec}")
            path_s, tag = spec.rsplit(":", 1)
            path = Path(path_s)
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                o = json.loads(line)
                if not args.skip_existing_source or "data_source" not in o:
                    o["data_source"] = tag
                fout.write(json.dumps(o, ensure_ascii=False) + "\n")
                n += 1
    print(f"Wrote {n} lines -> {out_path}")


if __name__ == "__main__":
    main()
