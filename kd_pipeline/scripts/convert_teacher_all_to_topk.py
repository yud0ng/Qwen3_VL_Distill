#!/usr/bin/env python3
"""
将 gen_all / Drive 的 teacher_all.jsonl（token_ids + logit_probs）转为 train_distill 所需的 teacher_topk.jsonl（kl_steps）。

对齐方式：与训练时相同 —— row_teacher_responses + Qwen3VLChatCollator.build_one，
用 labels 的 shift 监督位与 teacher 每条 answer 位置的 top-k 逐一对齐。

teacher 侧存的是 softmax 概率；本脚本按 T*log(p+eps) 转成伪 logits，使 softmax(伪logits/T) 在 top-k 上与 p 成比例（与 src.losses.topk_kl_loss 一致）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from tqdm import tqdm
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.qwen3_vl_collator import Qwen3VLChatCollator
from src.teacher_responses import row_teacher_responses


def probs_to_pseudo_logits(probs: list[float], temperature: float, eps: float = 1e-8) -> list[float]:
    """与 topk_kl_loss 中 softmax(teacher_logits/T) 在 top-k 上匹配教师概率。"""
    t = float(temperature)
    return [t * math.log(float(p) + eps) for p in probs]


def _rewrite_image(obj: dict, from_prefix: str, to_prefix: str) -> dict:
    img = obj.get("image")
    if not isinstance(img, str) or not (from_prefix and to_prefix):
        return obj
    if img.startswith(from_prefix):
        return {**obj, "image": to_prefix + img[len(from_prefix) :]}
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_all", type=Path, required=True, help="teacher_all.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="输出 teacher_topk.jsonl")
    ap.add_argument("--student_model_id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=4.0, help="与训练 --temperature 一致")
    ap.add_argument("--max_samples", type=int, default=0, help="0=全部含 logits 的样本")
    ap.add_argument(
        "--rewrite-image-prefix",
        nargs=2,
        metavar=("FROM", "TO"),
        default=None,
        help="image 路径前缀替换；默认见下方 --no-rewrite-image",
    )
    ap.add_argument(
        "--no-rewrite-image",
        action="store_true",
        help="不做路径替换（需保证 jsonl 内 image 本机可读）",
    )
    args = ap.parse_args()

    if args.no_rewrite_image:
        rw_from, rw_to = "", ""
    elif args.rewrite_image_prefix is not None:
        rw_from, rw_to = args.rewrite_image_prefix
    else:
        rw_from = "/ocean/projects/cis220039p/yluo22/datasets"
        rw_to = str((ROOT.parent / "datasets").resolve())

    processor = AutoProcessor.from_pretrained(args.student_model_id, trust_remote_code=True)
    collator = Qwen3VLChatCollator(
        processor=processor,
        max_length=int(args.max_length),
        max_image_side=448,
    )

    n_ok = n_skip = 0
    out_lines: list[str] = []

    with args.teacher_all.open(encoding="utf-8") as fin:
        lines = [l for l in fin if l.strip()]

    for line in tqdm(lines, desc="convert"):
        obj = _rewrite_image(json.loads(line), rw_from, rw_to)
        tid = obj.get("token_ids")
        lps = obj.get("logit_probs")
        if tid is None or lps is None:
            continue

        user_txt, asst_a, img, meta = row_teacher_responses(obj, target="answer_only", min_confidence=0)
        if meta.get("skip") or not user_txt or not asst_a:
            n_skip += 1
            continue

        try:
            batch = collator.build_one(user_text=user_txt, assistant_text=asst_a, image_path=img)
        except FileNotFoundError:
            n_skip += 1
            continue

        labels = batch["labels"]
        shift_labels = labels[:, 1:].contiguous()
        mask = shift_labels != -100
        sup_t = [t for t in range(shift_labels.shape[1]) if bool(mask[0, t].item())]

        if len(sup_t) != len(tid) or len(tid) != len(lps):
            n_skip += 1
            continue

        kl_steps = []
        bad = False
        for t, ids_row, pr_row in zip(sup_t, tid, lps):
            if len(ids_row) != len(pr_row):
                bad = True
                break
            logits_row = probs_to_pseudo_logits(pr_row, args.temperature)
            kl_steps.append(
                {
                    "t": int(t),
                    "ids": [int(x) for x in ids_row],
                    "logits": logits_row,
                }
            )

        if bad or not kl_steps:
            n_skip += 1
            continue

        rec = {
            "id": obj.get("id", ""),
            "teacher_model_id": args.student_model_id,
            "topk": len(kl_steps[0]["ids"]) if kl_steps else 0,
            "kl_steps": kl_steps,
        }
        out_lines.append(json.dumps(rec, ensure_ascii=False))
        n_ok += 1
        if args.max_samples > 0 and n_ok >= args.max_samples:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    print(f"Wrote {n_ok} records -> {args.out} (skipped {n_skip})")


if __name__ == "__main__":
    main()
