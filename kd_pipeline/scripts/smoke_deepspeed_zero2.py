#!/usr/bin/env python3
"""
DeepSpeed ZeRO-2 + bf16 + gradient checkpointing：双卡各 1 个 micro-batch，全局 batch=2，跑 1 个 optimizer step。
需占位数据：先执行 python scripts/make_dummy_assets.py

启动（默认 GPU 0,1）:
  cd kd_pipeline && bash scripts/smoke_deepspeed_zero2.sh

或:
  CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 scripts/smoke_deepspeed_zero2.py \\
    --deepspeed_config configs/deepspeed_zero2_bf16.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import deepspeed
import torch
import torch.nn.functional as F
import torch.optim as optim
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.qwen3_vl_collator import Qwen3VLChatCollator, row_from_jsonl


def forward_kwargs(batch: dict, device: torch.device) -> dict:
    """与 train_distill.forward_kwargs 一致（不把 labels 传入 forward，在外部算 CE）。"""
    kwargs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "use_cache": False,
    }
    if "pixel_values" in batch:
        kwargs["pixel_values"] = batch["pixel_values"].to(device)
    if "image_grid_thw" in batch:
        kwargs["image_grid_thw"] = batch["image_grid_thw"].to(device)
    if "mm_token_type_ids" in batch:
        kwargs["mm_token_type_ids"] = batch["mm_token_type_ids"].to(device)
    return kwargs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--local_rank", type=int, default=-1, help="distributed local rank（deepspeed 启动器传入）")
    p.add_argument("--student_model_id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument(
        "--train_jsonl",
        type=str,
        default=str(ROOT / "data" / "sample_train.jsonl"),
        help="含至少 1 条样本；双卡会取前 2 条（不足则复制）",
    )
    p = deepspeed.add_config_arguments(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    local_rank = args.local_rank if args.local_rank >= 0 else int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    train_path = Path(args.train_jsonl)
    if not train_path.is_file():
        raise SystemExit(f"Missing {train_path}; run: python scripts/make_dummy_assets.py")

    rows = [
        json.loads(l)
        for l in train_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    if not rows:
        raise SystemExit("train_jsonl is empty")
    while len(rows) < 2:
        rows.append(rows[0])

    processor = AutoProcessor.from_pretrained(args.student_model_id, trust_remote_code=True)
    dtype = torch.bfloat16
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.student_model_id,
        dtype=dtype,
        device_map=None,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    base.gradient_checkpointing_enable()
    base.config.use_cache = False
    base = base.to(device)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(base, lora_config)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=2e-5)

    # 配置仅通过 CLI --deepspeed_config；ZeRO-2 需要显式 optimizer
    model_engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=params,
    )

    collator = Qwen3VLChatCollator(processor=processor, max_length=512, max_image_side=448)
    # 双卡数据并行：rank i 使用第 i 条
    row = rows[local_rank]
    user_txt, asst, img = row_from_jsonl(row)
    if not asst:
        raise SystemExit("assistant_text empty")
    batch = collator.build_one(user_text=user_txt, assistant_text=asst, image_path=img)
    labels = batch["labels"].to(device)
    kwargs = forward_kwargs(batch, device)

    model_engine.train()
    # bf16 由 DeepSpeed config 与引擎处理；与 train_distill 一致用手动 CE
    out = model_engine(**kwargs)
    logits = out.logits.float()
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )

    model_engine.backward(loss)
    model_engine.step()

    if local_rank == 0:
        print(f"smoke_deepspeed_zero2 OK: loss={float(loss.item()):.4f}")


if __name__ == "__main__":
    main()
