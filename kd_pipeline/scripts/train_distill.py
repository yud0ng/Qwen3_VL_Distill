#!/usr/bin/env python3
"""
统一训练脚本：Variant A / B / C / BC（λ 置 0 可关对应项）；可选 DeepSpeed ZeRO-2。
LoRA：--lora_r / --lora_rank 为 0 时全参数微调（显存大）。

单卡 Variant A:
  python scripts/train_distill.py --train_jsonl data/sample_train.jsonl --variant A --max_steps 10

teacher_responses + DeepSpeed 双卡:
  deepspeed --num_gpus=2 scripts/train_distill.py --config configs/train_teacher_responses.yaml \\
    --deepspeed_config configs/deepspeed_zero2_bf16.json --variant A
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import deepspeed
import torch
import torch.optim as optim
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config_utils import load_yaml
from src.losses import ce_shift_supervised_mean, ce_shift_trace_answer, topk_kl_loss
from src.qwen3_vl_collator import Qwen3VLChatCollator, row_from_jsonl
from src.teacher_responses import extract_trace_and_answer, row_teacher_responses


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


def _shard_rows_padded(rows: list, rank: int, world_size: int) -> list:
    if world_size <= 1:
        return rows
    pad = (-len(rows)) % world_size
    rp = rows + (rows[:pad] if pad else [])
    return rp[rank::world_size]


def _load_ds_config_json(path: Path, world_size: int) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    micro = int(cfg.get("train_micro_batch_size_per_gpu", 1))
    ga = int(cfg.get("gradient_accumulation_steps", 1))
    cfg["train_batch_size"] = micro * world_size * ga
    return cfg


def _strip_deepspeed_cli_for_init(ns: argparse.Namespace) -> argparse.Namespace:
    o = copy.copy(ns)
    for k in ("deepspeed_config", "deepscale_config"):
        if hasattr(o, k):
            setattr(o, k, None)
    if hasattr(o, "deepspeed"):
        setattr(o, "deepspeed", False)
    if hasattr(o, "deepscale"):
        setattr(o, "deepscale", False)
    return o


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Qwen3-VL distill: Variant A / B / C / BC")
    p.add_argument("--local_rank", type=int, default=-1, help="分布式 local rank（deepspeed 传入）")
    p.add_argument("--config", type=str, default="", help="YAML defaults (CLI overrides)")
    p.add_argument("--train_jsonl", type=str, default=None)
    p.add_argument("--teacher_topk_jsonl", type=str, default=None)
    p.add_argument("--student_model_id", type=str, default=None)
    p.add_argument("--variant", choices=["A", "B", "C", "BC"], default=None)
    p.add_argument("--lam1", type=float, default=None, help="answer CE 权重（B/C/BC）")
    p.add_argument("--lam2", type=float, default=None, help="trace CE 权重（B/BC）；0=关闭")
    p.add_argument("--lam3", type=float, default=None, help="KL 权重（C/BC）；0=关闭")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_steps", type=int, default=None, help="0 = full epoch(s)")
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument(
        "--lora_r",
        "--lora_rank",
        type=int,
        default=None,
        dest="lora_r",
        help="LoRA rank；0=全参数微调；16/64 等",
    )
    p.add_argument("--lora_alpha", type=int, default=None)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument(
        "--save_every",
        type=int,
        default=None,
        help="每 N 个 optimizer step 存 checkpoint（0=仅 adapter_final）",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume_adapter", type=str, default=None)
    p.add_argument(
        "--input_format",
        type=str,
        choices=["chat", "teacher_responses"],
        default=None,
    )
    p.add_argument(
        "--teacher_target",
        type=str,
        choices=["answer_only", "full"],
        default=None,
    )
    p.add_argument("--min_confidence", type=int, default=None)
    p.add_argument(
        "--skip_missing_image",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p = deepspeed.add_config_arguments(p)
    return p


def parse_args() -> argparse.Namespace:
    p = build_parser()
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
    args.variant = pick("variant", "A")
    args.lam1 = pick("lam1", 1.0)
    args.lam2 = pick("lam2", 0.4)
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
    args.deepspeed_config = pick("deepspeed_config", None)

    if not args.train_jsonl:
        p.error("必须提供 --train_jsonl 或在 YAML 中设置 train_jsonl")

    args.train_jsonl = str(Path(args.train_jsonl).expanduser())
    if args.teacher_topk_jsonl:
        args.teacher_topk_jsonl = str(Path(args.teacher_topk_jsonl).expanduser())
    if args.out_dir:
        args.out_dir = str(Path(args.out_dir).expanduser())
    if args.deepspeed_config:
        args.deepspeed_config = str(Path(args.deepspeed_config).expanduser())

    args.save_every = pick("save_every", 500)
    args.seed = pick("seed", 42)
    args.resume_adapter = pick("resume_adapter", None)
    if args.resume_adapter:
        args.resume_adapter = str(Path(args.resume_adapter).expanduser())

    args.input_format = pick("input_format", "chat")
    args.teacher_target = pick("teacher_target", "answer_only")
    args.min_confidence = pick("min_confidence", 0)
    if args.skip_missing_image is None:
        args.skip_missing_image = bool(y.get("skip_missing_image", True))

    if args.variant in ("B", "BC") and args.input_format != "teacher_responses":
        p.error("Variant B / BC 需要 --input_format teacher_responses（含 response 中的 trace/answer）")

    return args


def main() -> None:
    args = parse_args()

    use_ds = bool(args.deepspeed_config)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = args.local_rank if args.local_rank >= 0 else int(os.environ.get("LOCAL_RANK", "0"))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if use_ds:
        if not torch.cuda.is_available():
            raise SystemExit("DeepSpeed 训练需要 CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root = Path(args.out_dir) if args.out_dir else ROOT / "runs" / "distill_smoke"
    if (not use_ds) or rank == 0:
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
    topk_map = load_topk_map(Path(args.teacher_topk_jsonl)) if args.variant in ("C", "BC") else {}

    use_bf16 = bool(args.bf16) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    processor = AutoProcessor.from_pretrained(args.student_model_id, trust_remote_code=True)
    load_kw = dict(
        dtype=dtype if device.type == "cuda" else torch.float32,
        device_map=None if use_ds else ("auto" if torch.cuda.is_available() else None),
        trust_remote_code=True,
        attn_implementation="eager",
    )
    base = Qwen3VLForConditionalGeneration.from_pretrained(args.student_model_id, **load_kw)
    base.gradient_checkpointing_enable()
    base.config.use_cache = False
    if use_ds:
        base = base.to(device)

    if args.resume_adapter:
        if args.lora_r == 0:
            raise SystemExit("全参数模式下请使用 HuggingFace 全量 checkpoint 恢复，勿用 --resume_adapter（LoRA）")
        model = PeftModel.from_pretrained(base, args.resume_adapter, is_trainable=True)
        if rank == 0:
            print(f"Resumed LoRA from {args.resume_adapter}")
    elif args.lora_r == 0:
        model = base
        for p in model.parameters():
            p.requires_grad = True
        if rank == 0:
            n = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Full finetune: {n} trainable params")
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
    if rank == 0 and args.lora_r != 0:
        model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(params, lr=args.lr)

    model_engine = None
    if use_ds:
        ds_cfg = _load_ds_config_json(Path(args.deepspeed_config), world_size)
        ds_args = _strip_deepspeed_cli_for_init(args)
        model_engine, opt, _, _ = deepspeed.initialize(
            args=ds_args,
            model=model,
            optimizer=opt,
            model_parameters=params,
            config=ds_cfg,
        )
        forward_mod = model_engine
    else:
        forward_mod = model

    collator = Qwen3VLChatCollator(processor=processor, max_length=512, max_image_side=448)

    global_step = 0
    log_path = out_root / "train_log.jsonl"

    def _save_ckpt(ckpt_dir: Path) -> None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_m = model_engine.module if use_ds else model
        save_m.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)
        hint = ckpt_dir / "for_eval.txt"
        hint.write_text(
            f"checkpoint_dir={ckpt_dir.resolve()}\n"
            f"eval: bash {ROOT / 'scripts' / 'eval_checkpoint.sh'} {ckpt_dir}\n",
            encoding="utf-8",
        )

    def one_epoch() -> bool:
        nonlocal global_step
        my_rows = _shard_rows_padded(rows, rank, world_size if use_ds else 1)
        pbar = tqdm(my_rows, desc="train", disable=(use_ds and rank != 0))
        for obj in pbar:
            sid = obj.get("id", "")

            if args.input_format == "teacher_responses":
                user_txt, asst_a, img, meta = row_teacher_responses(
                    obj,
                    target="answer_only",
                    min_confidence=args.min_confidence,
                )
                if meta.get("skip"):
                    continue
                raw = obj.get("response") or ""
                trace, ans = extract_trace_and_answer(raw)
            else:
                user_txt, asst_a, img = row_from_jsonl(obj)
                trace, ans = "", ""
                meta = {"skip": False}

            if not user_txt:
                continue

            try:
                if args.variant in ("B", "BC"):
                    batch = collator.build_trace_answer(
                        user_text=user_txt,
                        trace=trace,
                        answer=ans,
                        image_path=img,
                    )
                else:
                    batch = collator.build_one(
                        user_text=user_txt,
                        assistant_text=asst_a,
                        image_path=img,
                    )
            except FileNotFoundError as e:
                if args.skip_missing_image:
                    continue
                raise e

            assistant_ok = args.variant in ("B", "BC") and bool(ans.strip())
            if args.variant in ("A", "C") and not asst_a:
                continue
            if args.variant in ("B", "BC") and not assistant_ok:
                continue

            labels = batch["labels"]
            kwargs = forward_kwargs(batch, device)
            if not use_ds:
                opt.zero_grad(set_to_none=True)

            if use_ds:
                out = forward_mod(**kwargs)
            else:
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16 if use_bf16 else torch.float32
                ):
                    out = forward_mod(**kwargs)

            logits = out.logits.float()
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous().to(logits.device)

            loss_ce = ce_shift_supervised_mean(shift_logits, shift_labels)
            loss_ce_tr = torch.tensor(0.0, device=logits.device, dtype=loss_ce.dtype)
            loss_ce_ans = loss_ce

            seg_ok = batch.get("segment_mask_valid")
            seg_ok = seg_ok[0].item() if seg_ok is not None else True
            if args.variant in ("B", "BC") and seg_ok:
                pl = int(batch["prompt_len"][0].item())
                tt = int(batch["trace_tok_len"][0].item())
                ta = int(batch["answer_tok_len"][0].item())
                loss_ce_tr, loss_ce_ans = ce_shift_trace_answer(
                    shift_logits,
                    shift_labels,
                    prompt_len=pl,
                    trace_tok_len=tt,
                    answer_tok_len=ta,
                )
            elif args.variant in ("B", "BC"):
                merged = ce_shift_supervised_mean(shift_logits, shift_labels)
                loss_ce_tr = merged
                loss_ce_ans = merged

            loss_kl = torch.tensor(0.0, device=logits.device, dtype=loss_ce.dtype)
            if args.variant in ("C", "BC") and sid in topk_map:
                steps = topk_map[sid].get("kl_steps") or []
                if steps and args.lam3 > 0:
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
            elif args.variant == "B":
                if not seg_ok:
                    loss = (args.lam1 + args.lam2) * loss_ce_ans
                else:
                    loss = args.lam1 * loss_ce_ans + args.lam2 * loss_ce_tr
            elif args.variant == "C":
                loss = args.lam1 * loss_ce + args.lam3 * loss_kl
            else:
                if not seg_ok:
                    loss = (args.lam1 + args.lam2) * loss_ce_ans + args.lam3 * loss_kl
                else:
                    loss = args.lam1 * loss_ce_ans + args.lam2 * loss_ce_tr + args.lam3 * loss_kl

            if use_ds:
                model_engine.backward(loss)
                model_engine.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

            rec = {
                "step": global_step,
                "loss": float(loss.item()),
                "loss_ce": float(loss_ce.item()),
                "loss_ce_trace": float(loss_ce_tr.item()),
                "loss_ce_answer": float(loss_ce_ans.item()),
                "loss_kl": float(loss_kl.item()) if isinstance(loss_kl, torch.Tensor) else loss_kl,
                "id": sid,
                "rank": rank,
            }
            if (not use_ds) or rank == 0:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if rank == 0 or not use_ds:
                pbar.set_postfix(loss=float(loss.item()))
            global_step += 1
            if args.save_every and global_step % args.save_every == 0 and ((not use_ds) or rank == 0):
                ckpt_dir = out_root / f"checkpoint-{global_step}"
                _save_ckpt(ckpt_dir)
            if args.max_steps > 0 and global_step >= args.max_steps:
                return True
        return False

    for ep in range(args.num_epochs):
        if one_epoch():
            break

    save_p = out_root / "adapter_final"
    if (not use_ds) or rank == 0:
        save_p.mkdir(parents=True, exist_ok=True)
        _save_ckpt(save_p)
        if rank == 0:
            print(f"Saved -> {save_p}")


if __name__ == "__main__":
    main()
