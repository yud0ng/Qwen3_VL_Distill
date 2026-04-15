#!/usr/bin/env python3
"""
将 LoRA adapter 合并进基座并导出为完整权重目录，供 lmms-eval 等只接受单路径的工具使用。

  CUDA_VISIBLE_DEVICES=0 python scripts/export_merged_model.py \\
    --base_model_id Qwen/Qwen3-VL-2B-Instruct \\
    --adapter_dir runs/e2e_smoke/adapter_final \\
    --out_dir exports/merged_2b_lora
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_id", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--adapter_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_bf16_supported() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(args.base_model_id, trust_remote_code=True)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model_id,
        dtype=dtype if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    merged = model.merge_and_unload()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    processor.save_pretrained(out)
    print(f"Merged model saved -> {out}")


if __name__ == "__main__":
    main()
