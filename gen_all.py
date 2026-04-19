"""
gen_data_all.py
===============
Unified teacher data generation: CoT text + top-k logits + hidden states.

One script, two phases, all distillation signals in one JSONL.
Training scripts select which signals to use via λ switches.

Pipeline:
  Phase 1 — vLLM generate()  : CoT thinking chain + answer  (streaming write)
  Phase 2 — HF forward()     : top-k logits + hidden state  (batch, in-place update)

Phase 1 writes immediately (same as gen_cot_data.py) so a Phase 2 crash
does not lose Phase 1 work.  Re-run with --resume to continue either phase.

Output schema (one JSON per line):
  {
    "id":             "coco_spatial_0000001",
    "source":         "coco_spatial" | "llava_general",
    "type":           "metric" | "relational" | "egocentric" | "general",
    "image":          "/absolute/path/to/image.jpg",
    "question":       "...",
    "thinking":       "<think>...</think>" | null,
    "response":       "<answer>...</answer><confidence>N</confidence>",
    "bbox_gt":        "left" | null,
    "confidence":     4,
    "think_len":      142 | null,
    "cot_quality":    true | false | null,
    "token_ids":      [[tok1, tok2, ...], ...]  | null,
    "logit_probs":    [[p1,  p2,  ...], ...]   | null,
    "teacher_hidden": [float16 …]              | null
  }

Usage:
  # Full run
  python gen_data_all.py --total 50000 --tp 4 --batch_size 16 --logit_k 20

  # Text only (skip logit/hidden extraction)
  python gen_data_all.py --total 50000 --tp 4 --batch_size 16 --skip_forward

  # Quick test
  python gen_data_all.py --total 100 --tp 2 --batch_size 8 --skip_forward

  # Resume after crash
  python gen_data_all.py --total 50000 --tp 4 --batch_size 16 --logit_k 20 --resume
"""

import argparse
import gc
import json
import os
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ── Reuse from gen_teacher_data.py ────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from gen_teacher_data import (
    SYSTEM_PROMPT,
    load_coco_samples,
    load_llava_samples,
    parse_confidence,
    build_prompt as _base_build_prompt,
)

# ── Reuse CoT quality filter from gen_cot_data.py ─────────────────
from gen_cot_data import is_cot_quality

# ── Constants ──────────────────────────────────────────────────────
FLUSH_INTERVAL = 500


# ── Prompt builders ───────────────────────────────────────────────

def build_prompt_cot(question: str) -> str:
    """Spatial: force CoT by appending <think>\\n (same as gen_cot_data)."""
    return _base_build_prompt(question) + "<think>\n"


def build_prompt_general(question: str) -> str:
    """General: standard prompt, no CoT."""
    return _base_build_prompt(question)


# ── Parse thinking + response ─────────────────────────────────────

def parse_thinking(raw: str) -> tuple[str, str]:
    full = "<think>\n" + raw
    m = re.search(r"(<think>.*?</think>)\s*(.*)", full, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return full.strip(), ""


# ── HF model (Phase 2 only) ───────────────────────────────────────

def load_hf_model(model_path: str):
    """Load Qwen3-VL via HuggingFace for forward pass (after vLLM is freed).

    Uses Qwen3VLForConditionalGeneration (built into transformers) — the correct
    class for qwen3_vl config, avoids the Qwen2_5_VL shape mismatch.
    """
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"\n[HF] Loading model for forward pass: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def extract_batch_logits_and_hidden(
    model,
    processor,
    items: list[dict],
    top_k: int = 50,
    hidden_layer: int = -1,
) -> list[dict]:
    """
    Batch forward pass on (image, question, response) tuples.

    Returns one dict per item with keys: token_ids, logit_probs, teacher_hidden.
    Failed items return all-None.
    """
    empty = {"token_ids": None, "logit_probs": None, "teacher_hidden": None}
    results = [dict(empty) for _ in items]

    batch_texts, batch_images, valid_idx = [], [], []
    for i, item in enumerate(items):
        try:
            img = Image.open(item["image"]).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot open {item['image']}: {e}", file=sys.stderr)
            continue
        answer_content = (item.get("response")
                          or re.sub(r"</?think>", "", item.get("thinking") or "").strip())
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": item["question"]},
            ]},
            {"role": "assistant", "content": answer_content},
        ]
        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        except Exception as e:
            print(f"[WARN] apply_chat_template failed: {e}", file=sys.stderr)
            continue
        batch_texts.append(text)
        batch_images.append(img)
        valid_idx.append(i)

    if not batch_texts:
        return results

    try:
        inputs = processor(
            text=batch_texts, images=batch_images,
            return_tensors="pt", padding=True,
        )
    except Exception as e:
        print(f"[WARN] Batch processor failed: {e}", file=sys.stderr)
        return results

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    try:
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
    except Exception as e:
        print(f"[WARN] Batch forward failed: {e}", file=sys.stderr)
        return results

    attn = inputs.get("attention_mask")  # (B, seq_len)

    for b, orig_i in enumerate(valid_idx):
        item = items[orig_i]
        # Fallback: CoT models often put the full answer inside <think>,
        # leaving response empty.  Use thinking content in that case.
        answer_text = item.get("response") or re.sub(r"</?think>", "", item.get("thinking") or "").strip()
        if not answer_text:
            continue

        answer_ids = processor.tokenizer.encode(answer_text, add_special_tokens=False)
        n_answer = len(answer_ids)

        seq_len = int(attn[b].sum().item()) if attn is not None else outputs.logits.shape[1]

        if n_answer == 0 or n_answer >= seq_len:
            continue

        # Logits at positions [answer_start-1 : seq_len-1] predict answer tokens
        answer_start = seq_len - n_answer
        ans_logits = outputs.logits[b, answer_start - 1 : seq_len - 1, :]  # (n_ans, vocab)

        probs = F.softmax(ans_logits.float(), dim=-1)
        topk_probs, topk_ids = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)

        # Last layer hidden state at final answer token position
        hidden_vec = outputs.hidden_states[hidden_layer][b, seq_len - 1, :]

        results[orig_i] = {
            "token_ids":      topk_ids.cpu().tolist(),
            "logit_probs":    topk_probs.cpu().to(torch.float16).tolist(),
            "teacher_hidden": hidden_vec.cpu().to(torch.float16).tolist(),
        }

    return results


# ── vLLM batch inference ──────────────────────────────────────────

def run_batch_spatial(llm, batch, sampling_params):
    inputs, valid = [], []
    for item in batch:
        try:
            img = Image.open(item["image"]).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot open {item['image']}: {e}", file=sys.stderr)
            continue
        inputs.append({
            "prompt":           build_prompt_cot(item["question"]),
            "multi_modal_data": {"image": img},
        })
        valid.append(item)
    if not inputs:
        return []
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    results = []
    for item, out in zip(valid, outputs):
        raw = out.outputs[0].text.strip()
        thinking, response = parse_thinking(raw)
        results.append({**item, "thinking": thinking, "response": response})
    return results


def run_batch_general(llm, batch, sampling_params):
    inputs, valid = [], []
    for item in batch:
        try:
            img = Image.open(item["image"]).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot open {item['image']}: {e}", file=sys.stderr)
            continue
        inputs.append({
            "prompt":           build_prompt_general(item["question"]),
            "multi_modal_data": {"image": img},
        })
        valid.append(item)
    if not inputs:
        return []
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    return [{**item, "response": out.outputs[0].text.strip(), "thinking": None}
            for item, out in zip(valid, outputs)]


# ── Args ──────────────────────────────────────────────────────────

def parse_args():
    BASE = "/ocean/projects/cis220039p/yluo22"
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--model_path",        default=f"{BASE}/models/qwen3-vl-32b")
    p.add_argument("--coco_dir",          default=f"{BASE}/datasets/coco/train2014")
    p.add_argument("--coco_ann",          default=f"{BASE}/datasets/coco/annotations/instances_train2014.json")
    p.add_argument("--llava_json",        default=f"{BASE}/datasets/LLaVA-Instruct-150K/llava_instruct_150k.json")
    p.add_argument("--llava_image_dir",   default=f"{BASE}/datasets/coco/train2014")
    p.add_argument("--output",            default=f"{BASE}/data/teacher_all.jsonl")

    p.add_argument("--total",             type=int,   default=50_000)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--resume",            action="store_true")

    p.add_argument("--tp",                type=int,   default=4)
    p.add_argument("--batch_size",        type=int,   default=16,
                   help="vLLM generation batch (keep small for CoT)")
    p.add_argument("--max_tokens_spatial",type=int,   default=3000)
    p.add_argument("--max_tokens_general",type=int,   default=512)

    p.add_argument("--logit_k",           type=int,   default=50,
                   help="Top-k logits to save (20 saves ~60%% storage)")
    p.add_argument("--hidden_layer",      type=int,   default=-1,
                   help="Which transformer layer's hidden state to save (-1 = last)")
    p.add_argument("--forward_batch_size",type=int,   default=4,
                   help="HF forward pass batch size (smaller = less VRAM after vLLM freed)")
    p.add_argument("--skip_forward",      action="store_true",
                   help="Phase 1 text only, skip logit/hidden extraction")
    p.add_argument("--skip_phase1",       action="store_true",
                   help="Skip vLLM generation entirely; go straight to Phase 2 "
                        "(use after convert_cot_to_all.py pre-fills the output file)")

    p.add_argument("--min_think_tokens",  type=int,   default=30)
    p.add_argument("--min_density",       type=float, default=0.01)

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)

    for label, path in [
        ("model_path",      args.model_path),
        ("coco_dir",        args.coco_dir),
        ("coco_ann",        args.coco_ann),
        ("llava_json",      args.llava_json),
        ("llava_image_dir", args.llava_image_dir),
    ]:
        if not os.path.exists(path):
            sys.exit(f"[ERROR] {label} not found: {path}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Load samples ───────────────────────────────────────────
    n_spatial = args.total // 2
    n_general = args.total - n_spatial

    spatial_samples = load_coco_samples(args.coco_ann, args.coco_dir, n_spatial)
    general_samples = load_llava_samples(args.llava_json, args.llava_image_dir, n_general)
    print(f"\nLoaded: {len(spatial_samples)} spatial + {len(general_samples)} general")

    # ── 1b. Resume ────────────────────────────────────────────────
    done_images: set[str] = set()
    resume_count = 0
    if args.resume and os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_images.add(r["image"])
                    resume_count += 1
                except Exception:
                    pass
        spatial_samples = [s for s in spatial_samples if s["image"] not in done_images]
        general_samples = [s for s in general_samples if s["image"] not in done_images]
        print(f"[RESUME] {resume_count} done, "
              f"{len(spatial_samples)} spatial + {len(general_samples)} general remaining")

    # ── 2. Phase 1: vLLM text generation (streaming write) ────────
    # Skipped when --skip_phase1 is set (e.g. after convert_cot_to_all.py).
    written        = resume_count
    sample_id      = resume_count
    n_rejected_cot = 0
    reject_reasons = {"fail_len": 0, "fail_density": 0, "fail_pivot": 0}

    if args.skip_phase1:
        print("\n[SKIP] Phase 1 skipped (--skip_phase1) — going straight to Phase 2")
    else:
        print(f"\n{'='*60}")
        print(f"PHASE 1: vLLM text generation")
        print(f"{'='*60}")

        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tp,
            max_model_len=4096,
            max_num_seqs=args.batch_size,
            mm_processor_kwargs={"min_pixels": 28*28, "max_pixels": 1280*28*28},
            limit_mm_per_prompt={"image": 1},
            gpu_memory_utilization=0.85,
            dtype="bfloat16",
            trust_remote_code=True,
            disable_custom_all_reduce=True,
        )
        sp_spatial = SamplingParams(
            temperature=0.7, top_p=0.9, max_tokens=args.max_tokens_spatial, stop=["<|im_end|>"],
        )
        sp_general = SamplingParams(
            temperature=0.7, top_p=0.9, max_tokens=args.max_tokens_general, stop=["<|im_end|>"],
        )

        out_f = open(args.output, "a" if args.resume else "w", encoding="utf-8")

        def write_item(item: dict):
            nonlocal written, sample_id, n_rejected_cot
            thinking = item.get("thinking") or ""
            response = item.get("response", "")
            cot_quality = None
            think_len   = None
            if item["source"] == "coco_spatial":
                passes, stats = is_cot_quality(thinking, args.min_think_tokens, args.min_density)
                cot_quality = passes
                think_len   = stats["think_len"]
                if not passes:
                    n_rejected_cot += 1
                    for k in reject_reasons:
                        if stats.get(k):
                            reject_reasons[k] += 1
            out_f.write(json.dumps({
                "id":             f"{item['source']}_{sample_id:07d}",
                "source":         item["source"],
                "type":           item["type"],
                "image":          item["image"],
                "question":       item["question"],
                "thinking":       thinking or None,
                "response":       response,
                "bbox_gt":        item.get("bbox_gt"),
                "confidence":     (parse_confidence(response)
                                   or parse_confidence(thinking)),
                "think_len":      think_len,
                "cot_quality":    cot_quality,
                "token_ids":      None,
                "logit_probs":    None,
                "teacher_hidden": None,
            }, ensure_ascii=False) + "\n")
            written   += 1
            sample_id += 1

        # -- Spatial (CoT) --
        print(f"\n[Spatial] {len(spatial_samples)} samples with CoT...")
        pbar = tqdm(total=len(spatial_samples), desc="Spatial CoT", dynamic_ncols=True)
        consecutive_failures = 0
        for i in range(0, len(spatial_samples), args.batch_size):
            batch = spatial_samples[i : i + args.batch_size]
            try:
                results = run_batch_spatial(llm, batch, sp_spatial)
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                print(f"\n[WARN] Spatial batch {i} failed: {e}", file=sys.stderr)
                pbar.update(len(batch))
                if consecutive_failures >= 3:
                    print(f"[FATAL] Engine dead. Re-run with --resume.", file=sys.stderr)
                    break
                continue
            for item in results:
                write_item(item)
            pbar.update(len(batch))
            pbar.set_postfix(written=written, rejected=n_rejected_cot)
            if written % FLUSH_INTERVAL == 0 and written > resume_count:
                out_f.flush()
        pbar.close()

        # -- General (no CoT) --
        print(f"\n[General] {len(general_samples)} samples...")
        pbar = tqdm(total=len(general_samples), desc="General SFT", dynamic_ncols=True)
        consecutive_failures = 0
        for i in range(0, len(general_samples), args.batch_size):
            batch = general_samples[i : i + args.batch_size]
            try:
                results = run_batch_general(llm, batch, sp_general)
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                print(f"\n[WARN] General batch {i} failed: {e}", file=sys.stderr)
                pbar.update(len(batch))
                if consecutive_failures >= 3:
                    print(f"[FATAL] Engine dead. Re-run with --resume.", file=sys.stderr)
                    break
                continue
            for item in results:
                write_item(item)
            pbar.update(len(batch))
            pbar.set_postfix(written=written)
            if written % FLUSH_INTERVAL == 0 and written > resume_count:
                out_f.flush()
        pbar.close()

        out_f.flush()
        out_f.close()
        print(f"\nPhase 1 done: {written - resume_count} new samples ({written} total)")
        print(f"CoT rejected: {n_rejected_cot}  reasons: {reject_reasons}")

        del llm
        torch.cuda.empty_cache()
        gc.collect()

    # ── 3. Phase 2: HF forward pass (logits + hidden) ─────────────
    # Reads the written file, enriches spatial records with null token_ids,
    # then rewrites the file once.  Re-run with --resume to retry Phase 2
    # without re-generating text.
    if args.skip_forward:
        print("\n[SKIP] Phase 2 skipped (--skip_forward)")
    else:
        print(f"\n{'='*60}")
        print(f"PHASE 2: HF forward pass — logits + hidden states")
        print(f"  top_k={args.logit_k}  hidden_layer={args.hidden_layer}"
              f"  forward_batch={args.forward_batch_size}")
        print(f"{'='*60}")

        # Load all written records
        all_records = []
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                try:
                    all_records.append(json.loads(line))
                except Exception:
                    pass

        # Find spatial records that still need forward pass
        # Include records where response is empty but thinking is non-empty:
        # in CoT mode the model often embeds the full answer inside <think>,
        # leaving response as "".  We fall back to thinking for those.
        needs_forward = [
            (i, r) for i, r in enumerate(all_records)
            if r["source"] == "coco_spatial"
            and r.get("token_ids") is None
            and (r.get("response") or r.get("thinking"))
        ]
        print(f"  Records needing forward pass: {len(needs_forward)}")

        if needs_forward:
            hf_model, hf_processor = load_hf_model(args.model_path)
            n_extracted = 0
            pbar = tqdm(
                range(0, len(needs_forward), args.forward_batch_size),
                desc="Forward pass", dynamic_ncols=True,
            )
            for batch_start in pbar:
                batch = needs_forward[batch_start : batch_start + args.forward_batch_size]
                indices = [idx for idx, _ in batch]
                items   = [r   for _, r  in batch]

                extracted = extract_batch_logits_and_hidden(
                    hf_model, hf_processor, items,
                    top_k=args.logit_k,
                    hidden_layer=args.hidden_layer,
                )
                for idx, result in zip(indices, extracted):
                    all_records[idx].update(result)
                    if result["token_ids"] is not None:
                        n_extracted += 1
                pbar.set_postfix(extracted=n_extracted)
            pbar.close()

            del hf_model, hf_processor
            torch.cuda.empty_cache()
            gc.collect()
            print(f"  Extracted: {n_extracted}/{len(needs_forward)}")

            # Rewrite file with enriched records
            print(f"  Rewriting {len(all_records)} records...")
            with open(args.output, "w", encoding="utf-8") as f:
                for r in all_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  Done.")
        else:
            print("  All spatial records already have logits/hidden — nothing to do.")

    # ── 4. Summary ────────────────────────────────────────────────
    total_written = written  # may be 0 if --skip_phase1; recount from file below
    n_spatial_tot = n_general_tot = n_with_cot = n_cot_pass = 0
    n_with_logits = n_with_hidden = 0
    conf_dist = {}

    with open(args.output, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            src = r.get("source", "")
            if src == "coco_spatial":  n_spatial_tot += 1
            if src == "llava_general": n_general_tot += 1
            if r.get("thinking"):      n_with_cot    += 1
            if r.get("cot_quality"):   n_cot_pass    += 1
            if r.get("token_ids") is not None:      n_with_logits += 1
            if r.get("teacher_hidden") is not None: n_with_hidden += 1
            c = str(r.get("confidence") or "missing")
            conf_dist[c] = conf_dist.get(c, 0) + 1

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    total_written = n_spatial_tot + n_general_tot  # count from file, not written counter
    print(f"Output            : {args.output}")
    print(f"Total written     : {total_written}")
    print(f"")
    print(f"── Data split ──────────────────────────────────")
    print(f"  Spatial (COCO)  : {n_spatial_tot}")
    print(f"  General (LLaVA) : {n_general_tot}")
    print(f"")
    print(f"── Distillation signals ────────────────────────")
    print(f"  With CoT        : {n_with_cot}  (pass filter: {n_cot_pass})")
    print(f"  With top-k logit: {n_with_logits}  (k={args.logit_k})")
    print(f"  With hidden     : {n_with_hidden}  (layer={args.hidden_layer})")
    print(f"")
    print(f"── CoT rejection (this run) ────────────────────")
    print(f"  Rejected total  : {n_rejected_cot}")
    print(f"    too short     : {reject_reasons['fail_len']}")
    print(f"    low density   : {reject_reasons['fail_density']}")
    print(f"    no pivot word : {reject_reasons['fail_pivot']}")
    print(f"")
    print(f"── Confidence distribution ─────────────────────")
    for k in ["5", "4", "3", "2", "1", "missing"]:
        v   = conf_dist.get(k, 0)
        pct = v / total_written * 100 if total_written else 0
        print(f"  conf={k:<7}  {v:>7}  ({pct:.1f}%)")
    hi = conf_dist.get("5", 0) + conf_dist.get("4", 0)
    if total_written:
        print(f"\n  conf>=4         : {hi}  ({hi/total_written*100:.1f}%)")
    print(f"{'='*60}")
    print(f"""
┌─────────────────────────────────────────────────────────┐
│  Downstream λ switches (train.py):                      │
│                                                         │
│  Variant A  SFT only   : λ_ce=1.0                       │
│  Variant B  + CoT      : λ_ce=0.6  λ_cot=0.4           │
│  Variant C  + KL/feat  : λ_ce=0.5  λ_kl=0.3  λ_f=0.2  │
│  Variant B+C full      : λ_ce=0.4  λ_cot=0.25          │
│                          λ_kl=0.2  λ_f=0.15            │
└─────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
