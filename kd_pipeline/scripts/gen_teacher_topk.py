#!/usr/bin/env python3
"""
Offline teacher top-k logits for Variant C (§4.2).
Reads JSONL rows (see sample_train.jsonl), runs teacher forward, writes JSONL + optional .jsonl.gz per row id.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/gen_teacher_topk.py \\
    --in_jsonl data/sample_train.jsonl --out_jsonl data/teacher_topk.jsonl \\
    --teacher_model_id Qwen/Qwen3-VL-2B-Instruct --topk 20 --max_samples 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config_utils import load_yaml
from src.qwen3_vl_collator import Qwen3VLChatCollator, row_from_jsonl


def topk_logits_vector(logits_row: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """logits_row: [V]"""
    vals, idx = torch.topk(logits_row, k=min(k, logits_row.shape[-1]))
    return idx, vals


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Offline teacher top-k logits")
    ap.add_argument("--config", type=str, default="", help="YAML with in_jsonl, out_jsonl, ...")
    ap.add_argument("--in_jsonl", type=str, default=None)
    ap.add_argument("--out_jsonl", type=str, default=None)
    ap.add_argument("--teacher_model_id", type=str, default=None)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--max_samples", type=int, default=None, help="0 = all")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    args = ap.parse_args()
    y = load_yaml(args.config) if args.config else {}

    def pick(k: str, fb):
        v = getattr(args, k, None)
        return v if v is not None else y.get(k, fb)

    args.in_jsonl = pick("in_jsonl", None)
    args.out_jsonl = pick("out_jsonl", None)
    args.teacher_model_id = pick("teacher_model_id", "Qwen/Qwen3-VL-2B-Instruct")
    args.topk = pick("topk", 50)
    args.max_samples = pick("max_samples", 0)
    if args.bf16 is None:
        args.bf16 = bool(y.get("bf16", False))
    if not args.in_jsonl or not args.out_jsonl:
        ap.error("需要 --in_jsonl / --out_jsonl 或在 YAML 中配置")
    args.in_jsonl = str(Path(args.in_jsonl).expanduser())
    args.out_jsonl = str(Path(args.out_jsonl).expanduser())
    return args


def main() -> None:
    args = parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in in_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_bf16_supported() else torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(args.teacher_model_id, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.teacher_model_id,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    collator = Qwen3VLChatCollator(processor=processor, max_length=512, max_image_side=448)

    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])

    out_lines = []
    for obj in tqdm(rows, desc="teacher_topk"):
        sid = obj.get("id", "")
        if sid in done:
            continue
        user_txt, asst, img = row_from_jsonl(obj)
        if not asst:
            continue
        batch = collator.build_one(user_text=user_txt, assistant_text=asst, image_path=img)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if "pixel_values" in batch:
            kwargs["pixel_values"] = batch["pixel_values"].to(device)
        if "image_grid_thw" in batch:
            kwargs["image_grid_thw"] = batch["image_grid_thw"].to(device)
        if "mm_token_type_ids" in batch:
            kwargs["mm_token_type_ids"] = batch["mm_token_type_ids"].to(device)

        with torch.no_grad():
            out = model(**kwargs)
        logits = out.logits.float()  # [1, L, V]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        mask = shift_labels != -100

        kl_steps = []
        for t in range(shift_logits.shape[1]):
            if not mask[0, t]:
                continue
            lg = shift_logits[0, t]
            idx, vals = topk_logits_vector(lg, args.topk)
            kl_steps.append(
                {
                    "t": int(t),
                    "ids": idx.cpu().tolist(),
                    "logits": vals.cpu().tolist(),
                }
            )

        record = {
            "id": sid or f"row_{len(out_lines)}",
            "teacher_model_id": args.teacher_model_id,
            "topk": args.topk,
            "kl_steps": kl_steps,
        }
        out_lines.append(json.dumps(record, ensure_ascii=False))

    if out_lines:
        with out_path.open("a", encoding="utf-8") as f:
            for line in out_lines:
                f.write(line + "\n")
    print(f"Appended {len(out_lines)} records -> {out_path}")


if __name__ == "__main__":
    main()
