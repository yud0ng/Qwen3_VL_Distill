#!/usr/bin/env python3
"""
按 teacher_responses.jsonl 的 `source` 字段拆分为两路数据，用于 Variant A（通用）与 Variant A+（空间聚焦）对照。

- llava_general -> teacher_general.jsonl
- coco_spatial -> teacher_spatial.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="teacher_responses.jsonl")
    p.add_argument("--out-dir", type=Path, required=True, help="输出目录（会创建）")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    general_p = args.out_dir / "teacher_general.jsonl"
    spatial_p = args.out_dir / "teacher_spatial.jsonl"

    c_gen: Counter[str] = Counter()
    n_gen = n_spa = 0
    other: list[tuple[str, dict]] = []

    with (
        args.input.open(encoding="utf-8") as fin,
        general_p.open("w", encoding="utf-8") as fg,
        spatial_p.open("w", encoding="utf-8") as fs,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            src = (o.get("source") or "").strip().lower()
            c_gen[src] += 1
            if src == "llava_general":
                fg.write(json.dumps(o, ensure_ascii=False) + "\n")
                n_gen += 1
            elif src == "coco_spatial":
                fs.write(json.dumps(o, ensure_ascii=False) + "\n")
                n_spa += 1
            else:
                other.append((src or "(empty)", o))

    manifest = {
        "input": str(args.input.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "files": {
            "teacher_general.jsonl": str(general_p.resolve()),
            "teacher_spatial.jsonl": str(spatial_p.resolve()),
        },
        "counts_by_source": dict(sorted(c_gen.items(), key=lambda x: -x[1])),
        "lines_written": {
            "llava_general": n_gen,
            "coco_spatial": n_spa,
        },
        "unmatched_source_count": len(other),
        "unmatched_sources_sample": list({s for s, _ in other[:50]}),
    }
    mpath = args.out_dir / "split_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if other:
        print(
            f"WARNING: {len(other)} lines did not match llava_general / coco_spatial "
            f"(sources: {manifest['unmatched_sources_sample']})",
            flush=True,
        )


if __name__ == "__main__":
    main()
