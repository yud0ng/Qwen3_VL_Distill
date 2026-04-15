#!/usr/bin/env python3
"""
Variant A (CE only) vs Variant C (CE + top-k KL). Single-GPU, batch_size=1 for robustness.

Usage (smoke):
  CUDA_VISIBLE_DEVICES=0 python scripts/train_distill.py \\
    --train_jsonl data/sample_train.jsonl --teacher_topk_jsonl data/teacher_topk.jsonl \\
    --variant C --max_steps 5 --student_model_id Qwen/Qwen3-VL-2B-Instruct

YAML（见 configs/variant_C.yaml）：
  python scripts/train_distill.py --config configs/variant_C.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config_utils import load_yaml
from src.losses import topk_kl_loss
from src.qwen3_vl_collator import Qwen3VLChatCollator, row_from_jsonl


def load_topk_map(path: Path) -> dict[str, dict]:
    m: dict[str, dict] = {}
    if not path.is_file():
        return m
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        m[o["id"]] = o
    return m


def forward_kwargs(batch: dict, device: torch.device) -> dict:
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
    p = argparse.ArgumentParser(description="Qwen3-VL distill: Variant A / C")
    p.add_argument("--config", type=str, default="", help="YAML defaults (CLI overrides)")
    p.add_argument("--train_jsonl", type=str, default=None)
    p.add_argument("--teacher_topk_jsonl", type=str, default=None)
    p.add_argument("--student_model_id", type=str, default=None)
    p.add_argument("--variant", choices=["A", "C"], default=None)
    p.add_argument("--lam1", type=float, default=None)
    p.add_argument("--lam3", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_steps", type=int, default=None, help="0 = full epoch(s); across epochs")
    p.add_argument("--num_epochs", type=int, default=None, help="repeat JSONL this many times")
    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument(
        "--save_every",
        type=int,
        default=None,
        help="每 N 个 optimizer step 存一次 LoRA（0=仅最终 adapter_final）",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--resume_adapter",
        type=str,
        default=None,
        help="已有 LoRA 目录（如 checkpoint-500），在此权重上继续训练",
    )
    args = p.parse_args()

    y = load_yaml(args.config) if args.config else {}

    def pick(key: str, fallback):
        v = getattr(args, key, None)
        if v is not None:
            return v
        if key in y:
            return y[key]
        return fallback

    args.train_jsonl = pick("train_jsonl", None)
    args.teacher_topk_jsonl = pick("teacher_topk_jsonl", "")
    args.student_model_id = pick("student_model_id", "Qwen/Qwen3-VL-2B-Instruct")
    args.variant = pick("variant", "C")
    args.lam1 = pick("lam1", 0.5)
    args.lam3 = pick("lam3", 0.5)
    args.temperature = pick("temperature", 4.0)
    args.lr = pick("lr", 2e-5)
    args.max_steps = pick("max_steps", 0)
    args.num_epochs = pick("num_epochs", 1)
    args.lora_r = pick("lora_r", 64)
    args.lora_alpha = pick("lora_alpha", 128)
    if getattr(args, "bf16", None) is None:
        args.bf16 = bool(y.get("bf16", False))
    args.out_dir = pick("out_dir", "")

    if not args.train_jsonl:
        p.error("必须提供 --train_jsonl 或在 YAML 中设置 train_jsonl")

    args.train_jsonl = str(Path(args.train_jsonl).expanduser())
    if args.teacher_topk_jsonl:
        args.teacher_topk_jsonl = str(Path(args.teacher_topk_jsonl).expanduser())
    if args.out_dir:
        args.out_dir = str(Path(args.out_dir).expanduser())

    args.save_every = pick("save_every", 0)
    args.seed = pick("seed", 42)
    args.resume_adapter = pick("resume_adapter", None)
    if args.resume_adapter:
        args.resume_adapter = str(Path(args.resume_adapter).expanduser())

    return args


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_root = Path(args.out_dir) if args.out_dir else ROOT / "runs" / "distill_smoke"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "resolved_args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    rows = [
        json.loads(l)
        for l in Path(args.train_jsonl).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    topk_map = load_topk_map(Path(args.teacher_topk_jsonl)) if args.variant == "C" else {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(args.bf16) and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    processor = AutoProcessor.from_pretrained(args.student_model_id, trust_remote_code=True)
    load_kw = dict(
        dtype=dtype if device.type == "cuda" else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    base = Qwen3VLForConditionalGeneration.from_pretrained(args.student_model_id, **load_kw)
    base.gradient_checkpointing_enable()
    base.config.use_cache = False

    if args.resume_adapter:
        model = PeftModel.from_pretrained(base, args.resume_adapter, is_trainable=True)
        print(f"Resumed LoRA from {args.resume_adapter}")
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
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
    model.print_trainable_parameters()

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    collator = Qwen3VLChatCollator(processor=processor, max_length=512, max_image_side=448)

    global_step = 0
    log_path = out_root / "train_log.jsonl"

    def one_epoch() -> bool:
        nonlocal global_step
        pbar = tqdm(rows, desc="train")
        for obj in pbar:
            sid = obj.get("id", "")
            user_txt, asst, img = row_from_jsonl(obj)
            if not asst:
                continue
            batch = collator.build_one(user_text=user_txt, assistant_text=asst, image_path=img)
            labels = batch["labels"]
            kwargs = forward_kwargs(batch, device)
            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if use_bf16 else torch.float32):
                out = model(**kwargs)
                logits = out.logits.float()

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous().to(logits.device)
                loss_ce = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="mean",
                )

                loss_kl = torch.tensor(0.0, device=logits.device, dtype=loss_ce.dtype)
                if args.variant == "C" and sid in topk_map:
                    steps = topk_map[sid]["kl_steps"]
                    if steps:
                        ts = [s["t"] for s in steps]
                        tid_list = [s["ids"] for s in steps]
                        tlog_list = [s["logits"] for s in steps]
                        t_idx = torch.tensor(ts, device=logits.device, dtype=torch.long)
                        st = shift_logits[0].index_select(0, t_idx)
                        tid = torch.tensor(tid_list, device=logits.device, dtype=torch.long)
                        tlog = torch.tensor(tlog_list, device=logits.device, dtype=torch.float32)
                        loss_kl = topk_kl_loss(st, tid, tlog, temperature=args.temperature)

                if args.variant == "A":
                    loss = loss_ce
                else:
                    loss = args.lam1 * loss_ce + args.lam3 * loss_kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()

            rec = {
                "step": global_step,
                "loss": float(loss.item()),
                "loss_ce": float(loss_ce.item()),
                "loss_kl": float(loss_kl.item()) if isinstance(loss_kl, torch.Tensor) else loss_kl,
                "id": sid,
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            pbar.set_postfix(loss=float(loss.item()), ce=float(loss_ce.item()))
            global_step += 1
            if args.save_every and global_step % args.save_every == 0:
                ckpt_dir = out_root / f"checkpoint-{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)
            if args.max_steps > 0 and global_step >= args.max_steps:
                return True
        return False

    for ep in range(args.num_epochs):
        if one_epoch():
            break

    save_p = out_root / "adapter_final"
    save_p.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_p)
    processor.save_pretrained(save_p)
    print(f"Saved -> {save_p}")


if __name__ == "__main__":
    main()
