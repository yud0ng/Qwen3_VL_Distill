#!/usr/bin/env python3
"""把 teacher jsonl 里的集群 image 前缀换成本地 datasets（与 convert_teacher_all_to_topk 默认一致）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--from-prefix", default="/ocean/projects/cis220039p/yluo22/datasets")
    ap.add_argument("--to-prefix", default=str((ROOT.parent / "datasets").resolve()))
    args = ap.parse_args()

    lines_out = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            img = o.get("image")
            if isinstance(img, str) and img.startswith(args.from_prefix):
                o = {**o, "image": args.to_prefix + img[len(args.from_prefix) :]}
            lines_out.append(json.dumps(o, ensure_ascii=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines_out)} lines -> {args.out}")


if __name__ == "__main__":
    main()
