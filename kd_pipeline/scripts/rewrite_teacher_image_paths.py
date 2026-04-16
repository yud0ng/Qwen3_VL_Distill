#!/usr/bin/env python3
"""
将 **`distill/distill/teacher_responses.jsonl`** 等文件中集群路径 `/ocean/.../coco/train2014/`
或其它机器上的绝对路径，改为你本机 **`distill/distill/datasets/coco/train2014`**（或其它目录）下的同名文件路径。

两种方式（二选一）：
  1) --coco-root  指向本机 train2014 目录（按文件名拼接）
  2) --from-prefix / --to-prefix  整段字符串替换

原地修改（推荐本地只留一份数据）：
  python scripts/rewrite_teacher_image_paths.py \\
    --input ../teacher_responses.jsonl --in-place \\
    --coco-root ../datasets/coco/train2014

另存为新文件：
  ... --output ../teacher_responses.local.jsonl --coco-root ...
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEFAULT_OCEAN_PREFIX = "/ocean/projects/cis220039p/yluo22/datasets/coco/train2014"


def main() -> None:
    p = argparse.ArgumentParser(description="Rewrite image paths in teacher_responses.jsonl")
    p.add_argument("--input", "-i", type=Path, required=True)
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="输出路径；与 --in-place 二选一",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="直接覆盖 --input；原文件备份为同目录 .bak（一次）",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="与 --in-place 同用时，不保留 .bak（危险）",
    )
    p.add_argument(
        "--coco-root",
        type=Path,
        default=None,
        help="本机 COCO train2014 目录（内含 COCO_train2014_*.jpg）",
    )
    p.add_argument(
        "--from-prefix",
        type=str,
        default=DEFAULT_OCEAN_PREFIX,
        help="旧路径前缀（默认集群 train2014 父路径）",
    )
    p.add_argument(
        "--to-prefix",
        type=str,
        default=None,
        help="与 --from-prefix 配对做 str.replace（不要与 --coco-root 同用）",
    )
    p.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = p.parse_args()

    if args.in_place and args.output is not None:
        p.error("--in-place 时不要传 --output")
    if not args.in_place and args.output is None:
        p.error("请指定 --output，或使用 --in-place")

    if args.coco_root is not None and args.to_prefix is not None:
        p.error("请只使用 --coco-root 或 (--from-prefix + --to-prefix)，不要混用")
    if args.coco_root is None and args.to_prefix is None:
        p.error("请提供 --coco-root，或同时提供 --to-prefix（可与默认 --from-prefix 搭配）")

    n = 0
    n_missing = 0
    out_lines: list[str] = []

    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        img = o.get("image")
        if not isinstance(img, str) or not img:
            out_lines.append(json.dumps(o, ensure_ascii=False))
            n += 1
            continue

        if args.coco_root is not None:
            name = Path(img).name
            new_p = (args.coco_root.expanduser().resolve() / name).as_posix()
        else:
            new_p = img.replace(args.from_prefix, args.to_prefix or "")

        o["image"] = new_p
        out_lines.append(json.dumps(o, ensure_ascii=False))
        n += 1
        if args.coco_root is not None and not Path(new_p).is_file():
            n_missing += 1

    if args.dry_run:
        print(f"lines={n}, sample_new_path={out_lines[0][:200] if out_lines else ''}...")
        if args.coco_root is not None:
            print(f"after_rewrite_file_missing_estimate={n_missing}")
        return

    text = "\n".join(out_lines) + "\n"
    out_path = args.input if args.in_place else args.output
    assert out_path is not None

    if args.in_place:
        if not args.no_backup:
            bak = args.input.with_suffix(args.input.suffix + ".bak")
            shutil.copy2(args.input, bak)
            print(f"Backup: {bak}")
        tmp = args.input.with_suffix(".jsonl.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(args.input)
        print(f"In-place updated {args.input} ({n} lines)")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote {out_path} ({n} lines)")

    if args.coco_root is not None:
        print(f"Note: {n_missing} paths do not exist as files (fix --coco-root or download COCO)")


if __name__ == "__main__":
    main()
